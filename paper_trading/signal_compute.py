"""
Daily signal computation pipeline for paper trading.

Reads latest market data from DB, computes indicators for 3 confirmed signals,
checks entry/exit conditions against open positions, and logs fired signals.

Designed to run daily after market close (3:35 PM IST) or on EOD data update.

Usage:
    python -m paper_trading.signal_compute          # run once
    python -m paper_trading.signal_compute --dry-run # check without DB writes
"""

import json
import logging
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import (
    add_all_indicators, historical_volatility,
)
from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

# ================================================================
# SIGNAL DEFINITIONS — 3 confirmed + 1 shadow
# ================================================================

SHADOW_SIGNALS = {
    'GUJRAL_DRY_7': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_gujral7',
        'check_exit': '_check_exit_gujral7',
        'trade_type': 'SHADOW',
    },
}

SIGNALS = {
    'KAUFMAN_DRY_20': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,  # no limit
        'check_entry': '_check_entry_dry20',
        'check_exit': '_check_exit_dry20',
    },
    'KAUFMAN_DRY_16': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 0,
        'check_entry': '_check_entry_dry16',
        'check_exit': '_check_exit_dry16',
    },
    'KAUFMAN_DRY_12': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 7,
        'check_entry': '_check_entry_dry12',
        'check_exit': '_check_exit_dry12',
    },
}


class SignalComputer:
    """Computes daily signals for paper trading."""

    def __init__(self, db_conn=None):
        self.conn = db_conn or psycopg2.connect(DATABASE_DSN)

    def run(self, as_of_date: date = None, dry_run: bool = False) -> Dict:
        """
        Run signal computation for a given date.

        Returns dict with:
            {
                'date': ...,
                'entries': [{'signal_id': ..., 'direction': ..., 'price': ...}, ...],
                'exits': [{'signal_id': ..., 'reason': ..., 'price': ...}, ...],
                'indicators': {'vix': ..., 'stoch_k_5': ..., ...},
            }
        """
        as_of_date = as_of_date or date.today()

        # Load market data (last 250 days is enough for all indicators)
        df = self._load_market_data(as_of_date)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient market data for {as_of_date}")
            return {'date': str(as_of_date), 'entries': [], 'exits': [], 'error': 'insufficient_data'}

        # Compute all indicators
        df = add_all_indicators(df)
        df['hvol_6'] = historical_volatility(df['close'], period=6)
        df['hvol_100'] = historical_volatility(df['close'], period=100)

        # Get today's and yesterday's rows
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        # Extract key indicator values for logging
        indicators = {
            'date': str(as_of_date),
            'close': float(today['close']),
            'prev_close': float(yesterday['close']),
            'volume': float(today['volume']),
            'prev_volume': float(yesterday['volume']),
            'india_vix': float(today.get('india_vix', 0)),
            'sma_10': float(today['sma_10']),
            'stoch_k_5': float(today['stoch_k_5']),
            'stoch_d_5': float(today['stoch_d_5']),
            'pivot': float(today['pivot']),
            'r1': float(today['r1']),
            's1': float(today['s1']),
            'hvol_6': float(today['hvol_6']) if pd.notna(today['hvol_6']) else None,
            'hvol_100': float(today['hvol_100']) if pd.notna(today['hvol_100']) else None,
            'vol_ratio_20': float(today['vol_ratio_20']) if pd.notna(today['vol_ratio_20']) else None,
            'adx_14': float(today['adx_14']) if pd.notna(today['adx_14']) else None,
        }

        # Load open positions (PAPER + SHADOW)
        open_positions = self._load_open_positions()
        open_shadow = self._load_open_positions(trade_type='SHADOW')

        # Check exits for PAPER positions
        exits = []
        for pos in open_positions:
            exit_signal = self._check_exit(pos, today, yesterday, as_of_date)
            if exit_signal:
                exits.append(exit_signal)
                if not dry_run:
                    self._record_exit(pos, exit_signal, today)

        # Check exits for SHADOW positions
        shadow_exits = []
        for pos in open_shadow:
            exit_signal = self._check_exit(pos, today, yesterday, as_of_date)
            if exit_signal:
                shadow_exits.append(exit_signal)
                if not dry_run:
                    self._record_exit(pos, exit_signal, today)

        # Check PAPER entries
        entries = []
        active_signal_ids = {pos['signal_id'] for pos in open_positions
                            if pos['signal_id'] not in {e['signal_id'] for e in exits}}

        for signal_id, config in SIGNALS.items():
            if signal_id in active_signal_ids:
                continue
            entry = self._check_entry(signal_id, config, today, yesterday)
            if entry:
                entries.append(entry)
                if not dry_run:
                    self._record_entry(signal_id, entry, today, as_of_date, indicators)

        # Check SHADOW entries
        shadow_entries = []
        active_shadow_ids = {pos['signal_id'] for pos in open_shadow
                            if pos['signal_id'] not in {e['signal_id'] for e in shadow_exits}}

        for signal_id, config in SHADOW_SIGNALS.items():
            if signal_id in active_shadow_ids:
                continue
            entry = self._check_entry(signal_id, config, today, yesterday)
            if entry:
                shadow_entries.append(entry)
                if not dry_run:
                    self._record_entry(signal_id, entry, today, as_of_date, indicators,
                                       trade_type='SHADOW')

        # Build signal actions dict for scoring engine
        signal_actions = {}
        for signal_id in list(SIGNALS.keys()) + list(SHADOW_SIGNALS.keys()):
            action = None
            for e in entries + shadow_entries:
                if e['signal_id'] == signal_id:
                    action = f"ENTER_{e['direction']}"
            for x in exits + shadow_exits:
                if x['signal_id'] == signal_id:
                    action = 'EXIT'
            signal_actions[signal_id] = {'action': action}

        result = {
            'date': str(as_of_date),
            'entries': entries,
            'exits': exits,
            'shadow_entries': shadow_entries,
            'shadow_exits': shadow_exits,
            'signal_actions': signal_actions,
            'indicators': indicators,
            'open_positions': len(open_positions) - len(exits) + len(entries),
            'open_shadow': len(open_shadow) - len(shadow_exits) + len(shadow_entries),
        }

        # Log summary
        logger.info(f"Signal compute {as_of_date}: "
                    f"{len(entries)} entries, {len(exits)} exits, "
                    f"{len(shadow_entries)} shadow entries, "
                    f"VIX={indicators['india_vix']:.1f}")
        for e in entries:
            logger.info(f"  ENTRY: {e['signal_id']} {e['direction']} @ {e['price']:.0f}")
        for x in exits:
            logger.info(f"  EXIT:  {x['signal_id']} reason={x['reason']} @ {x['price']:.0f}")
        for e in shadow_entries:
            logger.info(f"  SHADOW ENTRY: {e['signal_id']} {e['direction']} @ {e['price']:.0f}")

        return result

    # ================================================================
    # ENTRY CHECKS
    # ================================================================

    def _check_entry(self, signal_id: str, config: dict,
                     today, yesterday) -> Optional[Dict]:
        """Dispatch to signal-specific entry check."""
        method = getattr(self, config['check_entry'])
        return method(signal_id, config, today, yesterday)

    def _check_entry_dry20(self, signal_id, config, today, yesterday):
        """KAUFMAN_DRY_20: sma_10 < prev_close AND stoch_k_5 > 50"""
        if pd.isna(today['sma_10']) or pd.isna(today['stoch_k_5']):
            return None

        prev_close = float(yesterday['close'])
        sma_10 = float(today['sma_10'])
        stoch_k_5 = float(today['stoch_k_5'])

        if sma_10 < prev_close and stoch_k_5 > 50:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': float(today['close']),
                'reason': f'sma_10={sma_10:.0f} < prev_close={prev_close:.0f} '
                          f'AND stoch_k_5={stoch_k_5:.1f} > 50',
            }
        return None

    def _check_entry_dry16(self, signal_id, config, today, yesterday):
        """KAUFMAN_DRY_16: pivot breakout + hvol_6 < hvol_100"""
        if pd.isna(today['hvol_6']) or pd.isna(today['hvol_100']):
            return None

        close = float(today['close'])
        low = float(today['low'])
        high = float(today['high'])
        r1 = float(today['r1'])
        s1 = float(today['s1'])
        pivot = float(today['pivot'])
        hvol_6 = float(today['hvol_6'])
        hvol_100 = float(today['hvol_100'])

        if hvol_6 >= hvol_100:
            return None  # volatility filter blocks

        # Long entry: close > r1 AND low >= pivot
        if close > r1 and low >= pivot:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} > r1={r1:.0f} AND low={low:.0f} >= pivot={pivot:.0f} '
                          f'AND hvol_6={hvol_6:.4f} < hvol_100={hvol_100:.4f}',
            }

        # Short entry: close < s1
        if close < s1:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'close={close:.0f} < s1={s1:.0f} '
                          f'AND hvol_6={hvol_6:.4f} < hvol_100={hvol_100:.4f}',
            }

        return None

    def _check_entry_dry12(self, signal_id, config, today, yesterday):
        """KAUFMAN_DRY_12: price/volume divergence (close vs prev_close, vol vs prev_vol)"""
        close = float(today['close'])
        prev_close = float(yesterday['close'])
        volume = float(today['volume'])
        prev_volume = float(yesterday['volume'])

        # Long: close > prev_close AND volume < prev_volume
        if close > prev_close and volume < prev_volume:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} > prev_close={prev_close:.0f} '
                          f'AND vol={volume:.0f} < prev_vol={prev_volume:.0f}',
            }

        # Short: close < prev_close AND volume < prev_volume
        if close < prev_close and volume < prev_volume:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'close={close:.0f} < prev_close={prev_close:.0f} '
                          f'AND vol={volume:.0f} < prev_vol={prev_volume:.0f}',
            }

        return None

    def _check_entry_gujral7(self, signal_id, config, today, yesterday):
        """GUJRAL_DRY_7: pivot crossover (shadow signal for regime detection)"""
        close = float(today['close'])
        pivot = float(today['pivot'])
        prev_close = float(yesterday['close'])

        # Long: close > pivot AND prev_close <= pivot (crosses above)
        if close > pivot and prev_close <= pivot:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} > pivot={pivot:.0f} '
                          f'AND prev_close={prev_close:.0f} <= pivot',
            }

        # Short: close < pivot AND prev_close >= pivot (crosses below)
        if close < pivot and prev_close >= pivot:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'close={close:.0f} < pivot={pivot:.0f} '
                          f'AND prev_close={prev_close:.0f} >= pivot',
            }

        return None

    def _check_exit_gujral7(self, pos, today, yesterday):
        """GUJRAL_DRY_7 exit: close crosses back through pivot"""
        price = float(today['close'])
        pivot = float(today['pivot'])
        if pos['direction'] == 'LONG' and price < pivot:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'LONG',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < pivot={pivot:.0f})',
                'pnl': price - pos['entry_price'],
            }
        elif pos['direction'] == 'SHORT' and price > pivot:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'SHORT',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} > pivot={pivot:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    # ================================================================
    # EXIT CHECKS
    # ================================================================

    def _check_exit(self, pos: dict, today, yesterday,
                    as_of_date: date) -> Optional[Dict]:
        """Check if an open position should be exited."""
        signal_id = pos['signal_id']
        config = SIGNALS.get(signal_id) or SHADOW_SIGNALS.get(signal_id)
        if not config:
            return None

        entry_price = pos['entry_price']
        current_price = float(today['close'])
        direction = pos['direction']

        # Stop loss
        if direction == 'LONG':
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price

        if loss_pct >= config['stop_loss_pct']:
            return {
                'signal_id': signal_id,
                'direction': direction,
                'price': current_price,
                'reason': 'stop_loss',
                'pnl': -loss_pct * entry_price if direction == 'LONG' else loss_pct * entry_price,
            }

        # Take profit
        if config['take_profit_pct'] > 0:
            if direction == 'LONG':
                gain_pct = (current_price - entry_price) / entry_price
            else:
                gain_pct = (entry_price - current_price) / entry_price

            if gain_pct >= config['take_profit_pct']:
                return {
                    'signal_id': signal_id,
                    'direction': direction,
                    'price': current_price,
                    'reason': 'take_profit',
                    'pnl': gain_pct * entry_price,
                }

        # Hold days max
        if config['hold_days_max'] > 0:
            entry_date = pos['entry_date']
            if isinstance(entry_date, str):
                entry_date = datetime.strptime(entry_date, '%Y-%m-%d').date()
            days_held = (as_of_date - entry_date).days
            if days_held >= config['hold_days_max']:
                if direction == 'LONG':
                    pnl = current_price - entry_price
                else:
                    pnl = entry_price - current_price
                return {
                    'signal_id': signal_id,
                    'direction': direction,
                    'price': current_price,
                    'reason': 'hold_days_max',
                    'pnl': pnl,
                }

        # Signal-specific exit
        method = getattr(self, config['check_exit'])
        return method(pos, today, yesterday)

    def _check_exit_dry20(self, pos, today, yesterday):
        """KAUFMAN_DRY_20 exit: stoch_k_5 <= 50"""
        if pd.isna(today['stoch_k_5']):
            return None
        stoch_k_5 = float(today['stoch_k_5'])
        if stoch_k_5 <= 50:
            price = float(today['close'])
            pnl = price - pos['entry_price']  # always LONG
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (stoch_k_5={stoch_k_5:.1f} <= 50)',
                'pnl': pnl,
            }
        return None

    def _check_exit_dry16(self, pos, today, yesterday):
        """KAUFMAN_DRY_16 exit: low < pivot (long) or high > r1 (short)"""
        price = float(today['close'])
        if pos['direction'] == 'LONG':
            if float(today['low']) < float(today['pivot']):
                return {
                    'signal_id': pos['signal_id'],
                    'direction': 'LONG',
                    'price': price,
                    'reason': f'signal_exit (low={today["low"]:.0f} < pivot={today["pivot"]:.0f})',
                    'pnl': price - pos['entry_price'],
                }
        else:
            if float(today['high']) > float(today['r1']):
                return {
                    'signal_id': pos['signal_id'],
                    'direction': 'SHORT',
                    'price': price,
                    'reason': f'signal_exit (high={today["high"]:.0f} > r1={today["r1"]:.0f})',
                    'pnl': pos['entry_price'] - price,
                }
        return None

    def _check_exit_dry12(self, pos, today, yesterday):
        """KAUFMAN_DRY_12 exit: close reversal vs prev_close"""
        price = float(today['close'])
        prev_close = float(yesterday['close'])
        if pos['direction'] == 'LONG' and price < prev_close:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'LONG',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < prev_close={prev_close:.0f})',
                'pnl': price - pos['entry_price'],
            }
        elif pos['direction'] == 'SHORT' and price > prev_close:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'SHORT',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} > prev_close={prev_close:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    # ================================================================
    # DATABASE OPERATIONS
    # ================================================================

    def _load_market_data(self, as_of_date: date) -> Optional[pd.DataFrame]:
        """Load last 250 trading days of OHLCV data."""
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume, india_vix "
                "FROM nifty_daily WHERE date <= %s ORDER BY date DESC LIMIT 250",
                self.conn, params=(as_of_date,)
            )
            if df.empty:
                return None
            df = df.sort_values('date').reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return None

    def _load_open_positions(self, trade_type: str = 'PAPER') -> List[Dict]:
        """Load open positions by trade_type (no exit_date yet)."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT trade_id, signal_id, direction, entry_price, entry_date
                FROM trades
                WHERE trade_type = %s
                  AND exit_date IS NULL
                ORDER BY entry_date
            """, (trade_type,))
            rows = cur.fetchall()
            return [
                {
                    'trade_id': r[0],
                    'signal_id': r[1],
                    'direction': r[2],
                    'entry_price': r[3],
                    'entry_date': r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to load open positions: {e}")
            return []

    def _record_entry(self, signal_id: str, entry: dict,
                      today, as_of_date: date, indicators: dict,
                      trade_type: str = 'PAPER',
                      size_multiplier: float = 1.0,
                      confidence_label: str = None):
        """Record a new trade entry."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO trades (
                    signal_id, trade_type, entry_date, entry_time,
                    instrument, direction, lots, entry_price,
                    entry_regime, entry_vix,
                    intended_lots, fill_quality,
                    size_multiplier, confidence_label, created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    'FUTURES', %s, 1, %s,
                    %s, %s,
                    1, 'SIMULATED',
                    %s, %s, NOW()
                )
            """, (
                signal_id, trade_type, as_of_date, datetime.now().time(),
                entry['direction'], entry['price'],
                indicators.get('regime'),
                indicators.get('india_vix'),
                size_multiplier, confidence_label,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to record entry: {e}")
            self.conn.rollback()

    def _record_exit(self, pos: dict, exit_signal: dict, today):
        """Record exit for an open paper trade."""
        try:
            entry_price = pos['entry_price']
            exit_price = exit_signal['price']
            direction = pos['direction']

            if direction == 'LONG':
                gross_pnl = exit_price - entry_price
            else:
                gross_pnl = entry_price - exit_price

            return_pct = gross_pnl / entry_price

            cur = self.conn.cursor()
            cur.execute("""
                UPDATE trades SET
                    exit_date = CURRENT_DATE,
                    exit_time = CURRENT_TIME,
                    exit_price = %s,
                    exit_reason = %s,
                    gross_pnl = %s,
                    costs = 0,
                    net_pnl = %s,
                    return_pct = %s
                WHERE trade_id = %s
            """, (
                exit_price,
                exit_signal['reason'],
                gross_pnl,
                gross_pnl,  # no costs in paper trading
                return_pct,
                pos['trade_id'],
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to record exit: {e}")
            self.conn.rollback()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Daily signal computation')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--date', type=str, help='YYYY-MM-DD (default: today)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    as_of = date.fromisoformat(args.date) if args.date else date.today()

    computer = SignalComputer()
    result = computer.run(as_of_date=as_of, dry_run=args.dry_run)

    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
