"""
Portfolio backtest — all 16 signals, 10 years.
SCORING at full size, SHADOW at 0.25x, OVERLAY as regime modifier only.

Usage:
    python run_portfolio_all_signals.py
    python run_portfolio_all_signals.py --capital 2500000
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import _eval_conditions
from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

# ── CONFIG ────────────────────────────────────────────────────────────────────
RISK_PCT        = 0.01
SHADOW_SCALE    = 0.25
MAX_POSITIONS   = 8       # total open positions (raised from 4)
MAX_SAME_DIR    = 4       # max same direction (raised from 2)
MAX_PER_FAMILY  = 1       # max 1 position per signal family (diversification)

# ── ALL SIGNAL RULES ─────────────────────────────────────────────────────────
# Category: SCORING (full size), SHADOW (0.25x), OVERLAY (no trades)

ALL_SIGNALS = {
    # ── SCORING ───────────────────────────────────────────────────────────────
    'KAUFMAN_DRY_20': {
        'category': 'SCORING', 'family': 'KAUFMAN_TREND', 'direction': 'LONG',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'sma_10', 'op': '<', 'value': 'prev_close'},
            {'indicator': 'stoch_k_5', 'op': '>', 'value': 50},
        ],
        'exit_long': [{'indicator': 'stoch_k_5', 'op': '<=', 'value': 50}],
    },
    'KAUFMAN_DRY_16': {
        'category': 'SCORING', 'family': 'KAUFMAN_PIVOT', 'direction': 'BOTH',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.03, 'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'r1'},
            {'indicator': 'low', 'op': '>=', 'value': 'pivot'},
            {'indicator': 'hvol_6', 'op': '<', 'value': 'hvol_100'},
        ],
        'entry_short': [
            {'indicator': 'close', 'op': '<', 'value': 's1'},
            {'indicator': 'hvol_6', 'op': '<', 'value': 'hvol_100'},
        ],
        'exit_long': [{'indicator': 'low', 'op': '<', 'value': 'pivot'}],
        'exit_short': [{'indicator': 'high', 'op': '>', 'value': 'r1'}],
    },
    'KAUFMAN_DRY_12': {
        'category': 'SCORING', 'family': 'KAUFMAN_DIVERGENCE', 'direction': 'BOTH',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.03, 'hold_days_max': 7,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'prev_close'},
            {'indicator': 'volume', 'op': '<', 'value': 'prev_volume'},
        ],
        'entry_short': [
            {'indicator': 'close', 'op': '<', 'value': 'prev_close'},
            {'indicator': 'volume', 'op': '<', 'value': 'prev_volume'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'prev_close'}],
        'exit_short': [{'indicator': 'close', 'op': '>', 'value': 'prev_close'}],
    },
    'GUJRAL_DRY_8': {
        'category': 'SCORING', 'family': 'GUJRAL_PIVOT', 'direction': 'LONG',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'pivot'},
            {'indicator': 'open', 'op': '>', 'value': 'pivot'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'pivot'}],
    },
    'GUJRAL_DRY_13': {
        'category': 'SCORING', 'family': 'GUJRAL_BREAKOUT', 'direction': 'LONG',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 10,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'prev_high'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'prev_low'}],
    },

    # ── SHADOW (0.25x size) ──────────────────────────────────────────────────
    'CHAN_AT_DRY_4': {
        'category': 'SHADOW', 'family': 'CHAN_MEANREV', 'direction': 'BOTH',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 10,
        'entry_long': [
            {'indicator': 'close', 'op': '<', 'value': 'sma_20'},
            {'indicator': 'bb_pct_b', 'op': '<', 'value': 0.0},
        ],
        'entry_short': [
            {'indicator': 'close', 'op': '>', 'value': 'sma_20'},
            {'indicator': 'bb_pct_b', 'op': '>', 'value': 1.0},
        ],
        'exit_long': [{'indicator': 'close', 'op': '>=', 'value': 'sma_20'}],
        'exit_short': [{'indicator': 'close', 'op': '<=', 'value': 'sma_20'}],
    },
    'CANDLESTICK_DRY_0': {
        'category': 'SHADOW', 'family': 'CANDLESTICK', 'direction': 'LONG',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 10,
        'entry_long': [
            {'indicator': 'body', 'op': '>', 'value': 0.0},
            {'indicator': 'close', 'op': '>', 'value': 'open'},
            {'indicator': 'upper_wick', 'op': '<', 'value': 'body'},
            {'indicator': 'lower_wick', 'op': '<', 'value': 'body'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'open'}],
    },
    'KAUFMAN_DRY_7': {
        'category': 'SHADOW', 'family': 'KAUFMAN_SMA', 'direction': 'LONG',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'close', 'op': 'crosses_above', 'value': 'sma_5'},
            {'indicator': 'close', 'op': '>', 'value': 'sma_5'},
        ],
        'exit_long': [{'indicator': 'close', 'op': 'crosses_below', 'value': 'sma_5'}],
    },
    'GUJRAL_DRY_9': {
        'category': 'SHADOW', 'family': 'GUJRAL_RETRACE', 'direction': 'LONG',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'open', 'op': '<', 'value': 'pivot'},
            {'indicator': 'low', 'op': '>', 'value': 's1'},
            {'indicator': 'close', 'op': '>', 'value': 'open'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 's1'}],
    },
    'BULKOWSKI_CUP_HANDLE': {
        'category': 'SHADOW', 'family': 'BULKOWSKI_CUP', 'direction': 'BOTH',
        'stop_loss_pct': 0.03, 'take_profit_pct': 0.0, 'hold_days_max': 30,
        'entry_long': [
            {'indicator': 'price_pos_20', 'op': '<', 'value': 0.2},
            {'indicator': 'vol_ratio_20', 'op': '>', 'value': 1.2},
            {'indicator': 'adx_14', 'op': '>', 'value': 20.0},
        ],
        'entry_short': [
            {'indicator': 'price_pos_20', 'op': '>', 'value': 0.8},
            {'indicator': 'vol_ratio_20', 'op': '>', 'value': 1.2},
            {'indicator': 'adx_14', 'op': '>', 'value': 20.0},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_20'}],
        'exit_short': [{'indicator': 'close', 'op': '>', 'value': 'sma_20'}],
    },
    'BULKOWSKI_ADAM_AND_EVE_OR': {
        'category': 'SHADOW', 'family': 'BULKOWSKI_REVERSAL', 'direction': 'LONG',
        'stop_loss_pct': 0.03, 'take_profit_pct': 0.0, 'hold_days_max': 5,
        'entry_long': [
            {'indicator': 'range', 'op': '>', 'value': 'atr_14'},
            {'indicator': 'close', 'op': '>', 'value': 'low'},
            {'indicator': 'price_pos_20', 'op': '<', 'value': 0.2},
            {'indicator': 'lower_wick', 'op': '>', 'value': 'body'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '>', 'value': 'open'}],
    },
    'BULKOWSKI_ROUND_BOTTOM_RDB_PATTERN': {
        'category': 'SHADOW', 'family': 'BULKOWSKI_BOTTOM', 'direction': 'LONG',
        'stop_loss_pct': 0.03, 'take_profit_pct': 0.0, 'hold_days_max': 30,
        'entry_long': [
            {'indicator': 'price_pos_20', 'op': '<', 'value': 0.3},
            {'indicator': 'adx_14', 'op': '>', 'value': 15.0},
            {'indicator': 'open', 'op': '<=', 'value': 'prev_close'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_20'}],
    },
    'BULKOWSKI_EADT_BUSTED_PATTERN': {
        'category': 'SHADOW', 'family': 'BULKOWSKI_BUSTED', 'direction': 'LONG',
        'stop_loss_pct': 0.05, 'take_profit_pct': 0.0, 'hold_days_max': 20,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'prev_close'},
            {'indicator': 'returns', 'op': '>', 'value': -0.05},
            {'indicator': 'adx_14', 'op': '>', 'value': 20.0},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_20'}],
    },
    'BULKOWSKI_FALLING_VOLUME_TREND_IN': {
        'category': 'SHADOW', 'family': 'BULKOWSKI_VOLUME', 'direction': 'LONG',
        'stop_loss_pct': 0.03, 'take_profit_pct': 0.0, 'hold_days_max': 20,
        'entry_long': [
            {'indicator': 'volume', 'op': '<', 'value': 'prev_volume'},
            {'indicator': 'vol_ratio_20', 'op': '<', 'value': 1.0},
        ],
        'exit_long': [{'indicator': 'volume', 'op': '>', 'value': 'prev_volume'}],
    },
    'BULKOWSKI_EADB_EARLY_ATTEMPT_TO': {
        'category': 'SHADOW', 'family': 'BULKOWSKI_SHORT', 'direction': 'SHORT',
        'stop_loss_pct': 0.03, 'take_profit_pct': 0.0, 'hold_days_max': 10,
        'regime_gate_high_vol': True,
        'entry_short': [
            {'indicator': 'close', 'op': '<', 'value': 'prev_close'},
            {'indicator': 'volume', 'op': '>', 'value': 'prev_volume'},
            {'indicator': 'price_pos_20', 'op': '>', 'value': 0.6},
        ],
        'exit_short': [{'indicator': 'close', 'op': '>', 'value': 'sma_20'}],
    },

    # ── OVERLAY (no trades, regime modifier) ─────────────────────────────────
    'CRISIS_SHORT': {
        'category': 'OVERLAY', 'family': 'OVERLAY_CRISIS', 'direction': 'SHORT',
        'overlay_type': 'CRISIS_MODE',
        'entry_short': [
            {'indicator': 'india_vix', 'op': '>', 'value': 25},
            {'indicator': 'close', 'op': '<', 'value': 'sma_50'},
            {'indicator': 'adx_14', 'op': '>', 'value': 25},
        ],
    },
    'GUJRAL_DRY_7': {
        'category': 'OVERLAY', 'family': 'OVERLAY_REGIME', 'direction': 'BOTH',
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'pivot'},
        ],
        'entry_short': [
            {'indicator': 'close', 'op': '<', 'value': 'pivot'},
        ],
    },
    'SSRN_JANUARY_EFFECT': {
        'category': 'OVERLAY', 'family': 'OVERLAY_CALENDAR', 'direction': 'LONG',
        'overlay_type': 'CALENDAR_BOOST',
        # Month check done in code below, not via indicator conditions
    },

    # ── SHADOW — SSRN ────────────────────────────────────────────────────────
    'SSRN_WEEKLY_MOM': {
        'category': 'SHADOW', 'family': 'SSRN_MOMENTUM', 'direction': 'LONG',
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 5,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'sma_5'},
            {'indicator': 'returns', 'op': '>', 'value': 0},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_5'}],
    },
}


# ── REGIME SCENARIOS ──────────────────────────────────────────────────────

# ADX-based regime weights (applied regardless of scenario)
ADX_WEIGHTS = {
    'KAUFMAN_DRY_20': {'TRENDING': 1.5, 'RANGING': 0.5},
    'KAUFMAN_DRY_16': {'TRENDING': 1.5, 'RANGING': 0.5},
    'KAUFMAN_DRY_12': {'TRENDING': 1.5, 'RANGING': 0.5},
    'GUJRAL_DRY_8':   {'TRENDING': 1.5, 'RANGING': 1.0},
    'GUJRAL_DRY_13':  {'TRENDING': 1.5, 'RANGING': 0.8},
    'CHAN_AT_DRY_4':   {'TRENDING': 0.5, 'RANGING': 1.5},
    'KAUFMAN_DRY_7':  {'TRENDING': 1.5, 'RANGING': 0.5},
    'GUJRAL_DRY_9':   {'TRENDING': 1.5, 'RANGING': 1.0},
    'BULKOWSKI_CUP_HANDLE':  {'TRENDING': 0.8, 'RANGING': 1.5},
    'BULKOWSKI_ADAM_AND_EVE_OR': {'TRENDING': 0.8, 'RANGING': 1.5},
}


def vix_multiplier_graduated(vix):
    """Graduated VIX scale — Scenario C. No cliff edges."""
    if vix < 15:  return 1.2
    if vix < 20:  return 1.0
    if vix < 25:  return 0.7
    if vix < 30:  return 0.3
    return 0.0


def get_regime_mult(signal_id, row, scenario='C'):
    """Compute regime multiplier for a signal given current market state."""
    vix = float(row['india_vix']) if pd.notna(row.get('india_vix')) else 15.0
    adx = float(row['adx_14']) if pd.notna(row.get('adx_14')) else 20.0

    # ADX-based weight (trend vs range)
    adx_regime = 'TRENDING' if adx > 25 else 'RANGING'
    adx_mult = ADX_WEIGHTS.get(signal_id, {}).get(adx_regime, 1.0)

    # SHORT signals (EADB) get boosted in crisis, not penalized
    if signal_id == 'BULKOWSKI_EADB_EARLY_ATTEMPT_TO':
        if vix >= 25: return adx_mult * 2.0
        return adx_mult * 1.0

    # VIX-based weight (scenario-dependent)
    if scenario == 'A':
        # Scenario A: VIX≥18 = 0x (original, too aggressive)
        if vix >= 25: vix_mult = 0.0
        elif vix >= 18: vix_mult = 0.0
        else: vix_mult = 1.0

    elif scenario == 'B':
        # Scenario B: VIX≥25 = 0x (raised threshold)
        if vix >= 25: vix_mult = 0.0
        else: vix_mult = 1.0

    elif scenario == 'C':
        # Scenario C: graduated VIX scale
        vix_mult = vix_multiplier_graduated(vix)

    else:
        # No regime (baseline)
        vix_mult = 1.0

    return adx_mult * vix_mult


def run_backtest(capital, df, scenario='C'):
    n_bars = len(df)
    equity = capital
    peak_equity = capital
    positions = []
    closed_trades = []
    daily_equity = []

    # Overlay state
    overlay_bullish = False

    for i in range(1, n_bars):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_date = row['date']
        close = float(row['close'])
        vix = float(row['india_vix']) if pd.notna(row.get('india_vix')) else 15.0

        day_pnl = 0.0

        # Update overlays
        ov = ALL_SIGNALS['GUJRAL_DRY_7']
        overlay_bullish = _eval_conditions(row, prev, ov.get('entry_long', []))

        # Crisis mode: VIX>25 + close<sma_50 + ADX>25
        crisis = ALL_SIGNALS.get('CRISIS_SHORT', {})
        crisis_mode = _eval_conditions(row, prev, crisis.get('entry_short', []))

        # January effect: boost LONG sizing 1.25x in January
        january_boost = bar_date.month == 1 if hasattr(bar_date, 'month') else False

        # ── EXITS ─────────────────────────────────────────────────────────
        still_open = []
        for pos in positions:
            config = ALL_SIGNALS[pos['signal_id']]
            direction = pos['direction']
            entry_price = pos['entry_price']
            days_held = i - pos['entry_idx']

            if direction == 'LONG':
                loss_pct = (entry_price - close) / entry_price
            else:
                loss_pct = (close - entry_price) / entry_price

            exit_reason = None
            if loss_pct >= config['stop_loss_pct']:
                exit_reason = 'stop_loss'
            elif config.get('take_profit_pct', 0) > 0:
                gain = (close - entry_price) / entry_price if direction == 'LONG' else (entry_price - close) / entry_price
                if gain >= config['take_profit_pct']:
                    exit_reason = 'take_profit'

            if not exit_reason and config.get('hold_days_max', 0) > 0 and days_held >= config['hold_days_max']:
                exit_reason = 'hold_days_max'

            if not exit_reason:
                exit_key = 'exit_long' if direction == 'LONG' else 'exit_short'
                exit_conds = config.get(exit_key, [])
                if exit_conds and _eval_conditions(row, prev, exit_conds):
                    exit_reason = 'signal_exit'

            if exit_reason:
                pnl_pts = (close - entry_price) if direction == 'LONG' else (entry_price - close)
                pnl_rs = pnl_pts * NIFTY_LOT_SIZE * pos['lots']
                day_pnl += pnl_rs
                closed_trades.append({
                    'signal_id': pos['signal_id'],
                    'category': pos['category'],
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': close,
                    'pnl_pts': round(pnl_pts, 2),
                    'pnl_rs': round(pnl_rs, 2),
                    'lots': pos['lots'],
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'date': bar_date,
                })
            else:
                still_open.append(pos)

        positions = still_open

        # ── ENTRIES ───────────────────────────────────────────────────────
        active_sids = {p['signal_id'] for p in positions}
        active_families = {p['family'] for p in positions}
        n_long = sum(1 for p in positions if p['direction'] == 'LONG')
        n_short = sum(1 for p in positions if p['direction'] == 'SHORT')

        for signal_id, config in ALL_SIGNALS.items():
            if config['category'] == 'OVERLAY':
                continue
            if signal_id in active_sids:
                continue
            if len(positions) >= MAX_POSITIONS:
                break

            # Per-family limit
            family = config.get('family', signal_id)
            family_count = sum(1 for p in positions if p['family'] == family)
            if family_count >= MAX_PER_FAMILY:
                continue

            # HIGH_VOL gate
            if config.get('regime_gate_high_vol') and vix > 22:
                continue

            direction = config['direction']
            fired_direction = None

            # CRISIS MODE: block new LONG entries
            if crisis_mode and direction == 'LONG':
                continue

            if direction in ('LONG', 'BOTH'):
                # Block LONG side during crisis
                if not crisis_mode:
                    conds = config.get('entry_long', [])
                    if conds and _eval_conditions(row, prev, conds):
                        if n_long < MAX_SAME_DIR:
                            fired_direction = 'LONG'

            if not fired_direction and direction in ('SHORT', 'BOTH'):
                conds = config.get('entry_short', [])
                if conds and _eval_conditions(row, prev, conds):
                    if n_short < MAX_SAME_DIR:
                        fired_direction = 'SHORT'

            if fired_direction:
                is_shadow = config['category'] == 'SHADOW'
                scale = SHADOW_SCALE if is_shadow else 1.0

                # Apply regime weight (scenario-dependent)
                regime_mult = get_regime_mult(signal_id, row, scenario)
                scale *= regime_mult

                # If regime weight is 0, skip entry entirely
                if regime_mult == 0:
                    continue

                # CRISIS MODE: additional boost for SHORT signals
                if crisis_mode and fired_direction == 'SHORT':
                    scale *= 2.0

                # Overlay confidence boost: +25% if overlay agrees with direction
                if overlay_bullish and fired_direction == 'LONG':
                    scale *= 1.25
                elif not overlay_bullish and fired_direction == 'SHORT':
                    scale *= 1.25

                # January effect: +25% LONG sizing in January
                if january_boost and fired_direction == 'LONG':
                    scale *= 1.25

                risk_amount = equity * RISK_PCT * scale
                stop_pts = close * config.get('stop_loss_pct', 0.02)
                risk_per_lot = stop_pts * NIFTY_LOT_SIZE
                lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
                lots = min(lots, 50)

                positions.append({
                    'signal_id': signal_id,
                    'category': config['category'],
                    'family': family,
                    'direction': fired_direction,
                    'entry_price': close,
                    'entry_idx': i,
                    'lots': lots,
                })

                if fired_direction == 'LONG':
                    n_long += 1
                else:
                    n_short += 1

        equity += day_pnl
        peak_equity = max(peak_equity, equity)
        dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        daily_equity.append({'date': bar_date, 'equity': equity, 'dd_pct': dd_pct})

    return equity, closed_trades, daily_equity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=1_000_000)
    args = parser.parse_args()
    capital = args.capital

    scoring = [k for k, v in ALL_SIGNALS.items() if v['category'] == 'SCORING']
    shadow = [k for k, v in ALL_SIGNALS.items() if v['category'] == 'SHADOW']
    overlay = [k for k, v in ALL_SIGNALS.items() if v['category'] == 'OVERLAY']

    print(f"Portfolio backtest — all {len(ALL_SIGNALS)} signals")
    print(f"Capital: ₹{capital/1e5:.0f}L | Risk: {RISK_PCT*100:.0f}% | SHADOW scale: {SHADOW_SCALE}x")
    print(f"SCORING ({len(scoring)}): {', '.join(scoring)}")
    print(f"SHADOW  ({len(shadow)}): {', '.join(shadow)}")
    print(f"OVERLAY ({len(overlay)}): {', '.join(overlay)}")

    print("\nLoading market data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date", conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f"  {len(df)} trading days ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")

    print("Computing indicators...", flush=True)
    df = add_all_indicators(df)
    df['hvol_6'] = historical_volatility(df['close'], period=6)
    df['hvol_100'] = historical_volatility(df['close'], period=100)

    # ── RUN ALL SCENARIOS ────────────────────────────────────────────────
    scenarios = {
        'NONE':  ('No regime weights (baseline)',   'NONE'),
        'A':     ('VIX≥18 = 0x (original)',         'A'),
        'B':     ('VIX≥25 = 0x (raised threshold)', 'B'),
        'C':     ('Graduated VIX scale',            'C'),
    }

    years = len(df) / 252
    results = {}

    for key, (label, sc) in scenarios.items():
        print(f"\n  Running scenario {key}: {label}...", flush=True)
        final_eq, trades, daily_eq = run_backtest(capital, df, scenario=sc)

        eq_series = pd.Series([capital] + [d['equity'] for d in daily_eq])
        returns = eq_series.pct_change().dropna()
        cagr = (final_eq / capital) ** (1 / years) - 1
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        rolling_max = eq_series.cummax()
        max_dd = abs(((eq_series - rolling_max) / rolling_max).min())
        calmar = cagr / max_dd if max_dd > 0 else 0

        eq_df = pd.DataFrame(daily_eq)
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df['month'] = eq_df['date'].dt.to_period('M')
        monthly = eq_df.groupby('month')['equity'].last().pct_change().dropna()
        monthly_wr = (monthly > 0).mean()
        worst_mo = monthly.min()

        # Annual returns
        eq_df['year'] = eq_df['date'].dt.year
        annual = {}
        yearly = eq_df.groupby('year')['equity'].agg(['first', 'last'])
        for yr, row in yearly.iterrows():
            annual[yr] = (row['last'] / row['first'] - 1) * 100

        results[key] = {
            'label': label, 'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd,
            'calmar': calmar, 'final_eq': final_eq, 'trades': len(trades),
            'monthly_wr': monthly_wr, 'worst_mo': worst_mo, 'annual': annual,
            'all_trades': trades, 'daily_eq': daily_eq,
        }

    # ── SCENARIO COMPARISON TABLE ─────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  THREE-SCENARIO COMPARISON — ₹{capital/1e5:.0f}L capital")
    print(f"{'='*90}")
    hdr = f"{'Metric':<18s}"
    for key in scenarios:
        hdr += f" {key:>14s}"
    print(hdr)
    print("-" * 76)

    for metric, fmt, getter in [
        ('CAGR',         '{:>13.1f}%', lambda r: r['cagr']*100),
        ('Sharpe',       '{:>14.2f}',  lambda r: r['sharpe']),
        ('Max DD',       '{:>13.1f}%', lambda r: r['max_dd']*100),
        ('Calmar',       '{:>14.2f}',  lambda r: r['calmar']),
        ('Final equity', '{:>14s}',    lambda r: f"₹{r['final_eq']/1e5:.0f}L"),
        ('Trades',       '{:>14,}',    lambda r: r['trades']),
        ('Monthly WR',   '{:>13.1f}%', lambda r: r['monthly_wr']*100),
        ('Worst month',  '{:>13.1f}%', lambda r: r['worst_mo']*100),
    ]:
        line = f"{metric:<18s}"
        for key in scenarios:
            val = getter(results[key])
            if isinstance(val, str):
                line += f" {val:>14s}"
            else:
                line += fmt.format(val)
        print(line)

    # ── ANNUAL COMPARISON ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  ANNUAL RETURNS BY SCENARIO")
    print(f"{'='*90}")
    all_years = sorted(set(y for r in results.values() for y in r['annual']))
    hdr = f"{'Year':<6s}"
    for key in scenarios:
        hdr += f" {key:>14s}"
    print(hdr)
    print("-" * 66)
    for yr in all_years:
        line = f"{yr:<6d}"
        for key in scenarios:
            val = results[key]['annual'].get(yr, 0)
            line += f" {val:>13.1f}%"
        print(line)

    # ── BEST SCENARIO DETAIL ─────────────────────────────────────────────
    best = max(results.items(), key=lambda x: x[1]['calmar'])
    bk, br = best
    trades = br['all_trades']
    daily_eq = br['daily_eq']

    signal_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        sid = t['signal_id']
        signal_stats[sid]['trades'] += 1
        signal_stats[sid]['pnl'] += t['pnl_rs']
        if t['pnl_rs'] > 0:
            signal_stats[sid]['wins'] += 1

    print(f"\n{'='*90}")
    print(f"  BEST SCENARIO: {bk} — {br['label']}")
    print(f"  CAGR: {br['cagr']*100:.1f}% | Sharpe: {br['sharpe']:.2f} | Max DD: {br['max_dd']*100:.1f}% | Calmar: {br['calmar']:.2f}")
    print(f"{'='*90}")
    print(f"\n{'Signal':<40s} {'Cat':<8s} {'Trades':>6s} {'WR':>6s} {'Total P&L':>12s} {'Avg':>10s}")
    print("-" * 85)

    rows = []
    for sid in ALL_SIGNALS:
        if ALL_SIGNALS[sid]['category'] == 'OVERLAY': continue
        ss = signal_stats.get(sid, {'trades': 0, 'wins': 0, 'pnl': 0.0})
        cat = ALL_SIGNALS[sid]['category']
        if ss['trades'] > 0:
            wr = ss['wins'] / ss['trades'] * 100
            avg = ss['pnl'] / ss['trades']
            rows.append((ss['pnl'], f"{sid:<40s} {cat:<8s} {ss['trades']:>6d} {wr:>5.1f}% ₹{ss['pnl']:>11,.0f} ₹{avg:>9,.0f}"))
        else:
            rows.append((0, f"{sid:<40s} {cat:<8s}      0     -            -          -"))
    for _, line in sorted(rows, key=lambda x: -x[0]):
        print(line)

    # ── SAVE BEST SCENARIO ────────────────────────────────────────────────
    os.makedirs('backtest_results', exist_ok=True)
    save_data = {
        'best_scenario': bk,
        'capital': capital,
        'scenarios': {
            k: {'cagr': round(r['cagr']*100,1), 'sharpe': round(r['sharpe'],2),
                'max_dd': round(r['max_dd']*100,1), 'calmar': round(r['calmar'],2),
                'trades': r['trades'], 'final_equity': round(r['final_eq'])}
            for k, r in results.items()
        },
        'signal_stats': {sid: {'trades': s['trades'], 'wins': s['wins'],
                               'total_pnl': round(s['pnl'])}
                         for sid, s in signal_stats.items()},
    }
    with open('backtest_results/_PORTFOLIO_ALL16.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    eq_df = pd.DataFrame(br['daily_eq'])
    eq_df[['date', 'equity']].to_csv('backtest_results/portfolio_equity_all16.csv', index=False)
    print(f"\n  Saved: backtest_results/_PORTFOLIO_ALL16.json")
    print(f"  Saved: backtest_results/portfolio_equity_all16.csv")


if __name__ == '__main__':
    main()
