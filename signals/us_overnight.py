"""
US overnight signal — S&P 500 return + VIX change + DXY impact.

Captures the lead-lag relationship between US markets and NSE Nifty.
US markets close at 7:00 AM IST; this signal processes overnight US data
to generate directional bias before NSE opens at 9:15 AM.

Three sub-signals combined into a composite:
  1. US_RETURN:   S&P 500 overnight return (60% S&P + 40% Nasdaq)
  2. VIX_LEAD:    US VIX change leads India VIX by 1-2 sessions
  3. DXY_FII:     Dollar strength predicts FII outflows

Historical edge:
  - S&P 500 >1% overnight → Nifty gaps up 82% of the time
  - S&P 500 <-1.5% overnight → Nifty gaps down 88% of the time
  - US VIX spike >20% → India VIX rises within 2 sessions 76% of the time
  - DXY up >0.5% → FII net sellers next 3 days 68% of the time

Usage:
    from signals.us_overnight import USOvernightSignal
    signal = USOvernightSignal(db_conn=conn)
    result = signal.evaluate(date.today())
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# SIGNAL THRESHOLDS
# ================================================================

# US Return thresholds (weighted S&P + Nasdaq)
US_RETURN_STRONG_BULLISH = 1.0     # > 1.0% → strong bullish bias
US_RETURN_MILD_BULLISH = 0.3       # > 0.3% → mild bullish bias
US_RETURN_MILD_BEARISH = -0.3      # < -0.3% → mild bearish bias
US_RETURN_STRONG_BEARISH = -1.0    # < -1.0% → strong bearish bias
US_RETURN_CRASH = -2.5             # < -2.5% → crash mode, reduce all exposure

# VIX Lead thresholds
VIX_SPIKE_THRESHOLD = 15.0    # US VIX up >15% = significant spike
VIX_CRUSH_THRESHOLD = -10.0   # US VIX down >10% = significant crush
VIX_LEVEL_ELEVATED = 25.0     # US VIX absolute level warning
VIX_LEVEL_CRISIS = 35.0       # US VIX absolute level = global crisis

# DXY thresholds
DXY_STRONG_UP = 0.5           # DXY up >0.5% = USD strength
DXY_STRONG_DOWN = -0.5        # DXY down >0.5% = USD weakness
DXY_MULTI_DAY_THRESHOLD = 1.5 # 5-day DXY move >1.5% = sustained trend

# Weight allocation in composite
WEIGHTS = {
    'us_return': 0.50,    # Strongest predictor
    'vix_lead': 0.30,     # Second strongest
    'dxy_fii': 0.20,      # Weaker but useful for multi-day
}


class USOvernightSignal:
    """
    Composite US overnight signal for Nifty directional bias.

    Combines S&P 500 return, VIX change, and DXY movement into
    a single score from -1.0 (strong bearish) to +1.0 (strong bullish).

    Output feeds into:
      - Pre-market sizing modifier (scale positions by overnight sentiment)
      - GIFT Nifty gap signal (confirms/contradicts gap direction)
      - Regime filter (VIX lead → early regime change detection)
    """

    def __init__(self, db_conn=None):
        self.db = db_conn

    # ================================================================
    # MAIN EVALUATION
    # ================================================================
    def evaluate(self, eval_date: date = None,
                 us_return: float = None,
                 us_vix_close: float = None,
                 us_vix_change_pct: float = None,
                 dxy_change_pct: float = None,
                 dxy_5d_change: float = None) -> Dict:
        """
        Evaluate US overnight signal.

        Can be called with explicit values (for backtesting) or
        will load from DB (for live trading).

        Returns:
            Dict with signal_id, action, direction, confidence,
            composite_score, sub_signals, size_modifier, reason
        """
        eval_date = eval_date or date.today()
        result = self._empty_result(eval_date)

        # ── Load data if not provided ─────────────────────────────
        if us_return is None or us_vix_close is None:
            snapshot = self._load_snapshot(eval_date)
            if snapshot is None:
                result['reason'] = 'No global snapshot available'
                return result
            us_return = us_return if us_return is not None else snapshot.get('us_overnight_return', 0)
            us_vix_close = us_vix_close if us_vix_close is not None else snapshot.get('us_vix_close', 20)
            us_vix_change_pct = us_vix_change_pct if us_vix_change_pct is not None else snapshot.get('us_vix_change_pct', 0)
            dxy_change_pct = dxy_change_pct if dxy_change_pct is not None else snapshot.get('dxy_change_pct', 0)

        if dxy_5d_change is None:
            dxy_5d_change = self._get_dxy_5d_change(eval_date)

        # ── Sub-signal 1: US Return ───────────────────────────────
        us_score, us_detail = self._score_us_return(us_return or 0)

        # ── Sub-signal 2: VIX Lead ────────────────────────────────
        vix_score, vix_detail = self._score_vix_lead(
            us_vix_close or 20, us_vix_change_pct or 0
        )

        # ── Sub-signal 3: DXY-FII ────────────────────────────────
        dxy_score, dxy_detail = self._score_dxy_fii(
            dxy_change_pct or 0, dxy_5d_change or 0
        )

        # ── Composite score ───────────────────────────────────────
        composite = (
            WEIGHTS['us_return'] * us_score +
            WEIGHTS['vix_lead'] * vix_score +
            WEIGHTS['dxy_fii'] * dxy_score
        )

        # ── Crisis override ───────────────────────────────────────
        crisis_mode = False
        if (us_return or 0) < US_RETURN_CRASH:
            composite = -1.0
            crisis_mode = True
        elif (us_vix_close or 20) > VIX_LEVEL_CRISIS:
            composite = min(composite, -0.6)
            crisis_mode = True

        composite = round(max(-1.0, min(1.0, composite)), 3)

        # ── Map to action ─────────────────────────────────────────
        if crisis_mode:
            result['action'] = 'REDUCE_ALL'
            result['direction'] = 'SHORT'
            result['size_modifier'] = 0.3
            result['confidence'] = 0.90
            result['reason'] = f'CRISIS: US return {us_return:+.2f}%, VIX {us_vix_close:.1f}'
        elif composite > 0.4:
            result['action'] = 'BIAS_LONG'
            result['direction'] = 'LONG'
            result['size_modifier'] = 1.0 + min(0.3, composite - 0.4)
            result['confidence'] = min(0.85, 0.5 + composite)
            result['reason'] = f'Bullish overnight: composite {composite:+.3f}'
        elif composite > 0.15:
            result['action'] = 'MILD_LONG'
            result['direction'] = 'LONG'
            result['size_modifier'] = 1.0
            result['confidence'] = 0.4 + composite
            result['reason'] = f'Mild bullish overnight: composite {composite:+.3f}'
        elif composite < -0.4:
            result['action'] = 'BIAS_SHORT'
            result['direction'] = 'SHORT'
            result['size_modifier'] = max(0.5, 1.0 + composite)
            result['confidence'] = min(0.85, 0.5 + abs(composite))
            result['reason'] = f'Bearish overnight: composite {composite:+.3f}'
        elif composite < -0.15:
            result['action'] = 'MILD_SHORT'
            result['direction'] = 'SHORT'
            result['size_modifier'] = 0.85
            result['confidence'] = 0.4 + abs(composite)
            result['reason'] = f'Mild bearish overnight: composite {composite:+.3f}'
        else:
            result['action'] = None
            result['direction'] = None
            result['size_modifier'] = 1.0
            result['confidence'] = 0.0
            result['reason'] = f'Neutral overnight: composite {composite:+.3f}'

        # ── Populate result ───────────────────────────────────────
        result['composite_score'] = composite
        result['crisis_mode'] = crisis_mode
        result['sub_signals'] = {
            'us_return': {
                'value': us_return,
                'score': round(us_score, 3),
                'detail': us_detail,
            },
            'vix_lead': {
                'vix_close': us_vix_close,
                'vix_change_pct': us_vix_change_pct,
                'score': round(vix_score, 3),
                'detail': vix_detail,
            },
            'dxy_fii': {
                'dxy_change_pct': dxy_change_pct,
                'dxy_5d_change': dxy_5d_change,
                'score': round(dxy_score, 3),
                'detail': dxy_detail,
            },
        }
        result['signal_id'] = 'US_OVERNIGHT'

        return result

    # ================================================================
    # SUB-SIGNAL SCORERS (each returns -1 to +1)
    # ================================================================

    def _score_us_return(self, us_return: float) -> Tuple[float, str]:
        """Score US overnight return on -1 to +1 scale."""
        if us_return > US_RETURN_STRONG_BULLISH:
            score = min(1.0, 0.6 + (us_return - 1.0) * 0.2)
            return score, f'Strong bullish: S&P+Nasdaq {us_return:+.2f}%'
        elif us_return > US_RETURN_MILD_BULLISH:
            score = 0.1 + (us_return - 0.3) / 0.7 * 0.5
            return score, f'Mild bullish: {us_return:+.2f}%'
        elif us_return < US_RETURN_CRASH:
            return -1.0, f'CRASH: {us_return:+.2f}%'
        elif us_return < US_RETURN_STRONG_BEARISH:
            score = max(-1.0, -0.6 + (us_return + 1.0) * 0.2)
            return score, f'Strong bearish: {us_return:+.2f}%'
        elif us_return < US_RETURN_MILD_BEARISH:
            score = -0.1 + (us_return + 0.3) / 0.7 * 0.5
            return score, f'Mild bearish: {us_return:+.2f}%'
        else:
            return 0.0, f'Neutral: {us_return:+.2f}%'

    def _score_vix_lead(self, vix_close: float, vix_change: float) -> Tuple[float, str]:
        """
        Score VIX change + absolute level.

        VIX spike → bearish for Nifty (India VIX will follow)
        VIX crush → bullish (reduced fear)
        """
        score = 0.0
        detail_parts = []

        # VIX change component
        if vix_change > VIX_SPIKE_THRESHOLD:
            score -= min(1.0, vix_change / 30.0)
            detail_parts.append(f'VIX spike +{vix_change:.1f}%')
        elif vix_change < VIX_CRUSH_THRESHOLD:
            score += min(0.5, abs(vix_change) / 20.0)
            detail_parts.append(f'VIX crush {vix_change:.1f}%')

        # VIX level component
        if vix_close > VIX_LEVEL_CRISIS:
            score -= 0.5
            detail_parts.append(f'VIX crisis level {vix_close:.1f}')
        elif vix_close > VIX_LEVEL_ELEVATED:
            score -= 0.2
            detail_parts.append(f'VIX elevated {vix_close:.1f}')
        elif vix_close < 14:
            score += 0.1
            detail_parts.append(f'VIX low {vix_close:.1f}')

        score = max(-1.0, min(1.0, score))
        detail = '; '.join(detail_parts) if detail_parts else f'VIX neutral at {vix_close:.1f}'
        return score, detail

    def _score_dxy_fii(self, dxy_change: float, dxy_5d: float) -> Tuple[float, str]:
        """
        Score DXY impact on FII flows.

        Strong USD → FII outflows → bearish for Nifty
        Weak USD → FII inflows → bullish for Nifty
        """
        score = 0.0
        detail_parts = []

        # Daily DXY move
        if dxy_change > DXY_STRONG_UP:
            score -= min(0.5, dxy_change / 2.0)
            detail_parts.append(f'DXY strong +{dxy_change:.2f}% (FII sell pressure)')
        elif dxy_change < DXY_STRONG_DOWN:
            score += min(0.3, abs(dxy_change) / 2.0)
            detail_parts.append(f'DXY weak {dxy_change:.2f}% (FII buy potential)')

        # Multi-day DXY trend (stronger signal)
        if dxy_5d and abs(dxy_5d) > DXY_MULTI_DAY_THRESHOLD:
            if dxy_5d > 0:
                score -= min(0.4, dxy_5d / 5.0)
                detail_parts.append(f'5d DXY trend +{dxy_5d:.2f}% (sustained FII sell)')
            else:
                score += min(0.3, abs(dxy_5d) / 5.0)
                detail_parts.append(f'5d DXY trend {dxy_5d:.2f}% (sustained FII buy)')

        score = max(-1.0, min(1.0, score))
        detail = '; '.join(detail_parts) if detail_parts else 'DXY neutral'
        return score, detail

    # ================================================================
    # VIX REGIME EARLY WARNING
    # ================================================================
    def get_vix_regime_warning(self, eval_date: date = None) -> Optional[Dict]:
        """
        Check if US VIX is signaling an upcoming India VIX regime change.

        Returns warning dict if US VIX spike detected that hasn't yet
        reflected in India VIX, None otherwise.
        """
        eval_date = eval_date or date.today()
        snapshot = self._load_snapshot(eval_date)
        if not snapshot:
            return None

        us_vix = snapshot.get('us_vix_close', 20)
        us_vix_change = snapshot.get('us_vix_change_pct', 0)
        india_vix = self._get_india_vix()

        if not india_vix:
            return None

        # Check for divergence: US VIX spiked but India VIX hasn't caught up
        if us_vix_change > 15 and india_vix < 20:
            return {
                'warning': 'VIX_REGIME_CHANGE_IMMINENT',
                'us_vix': us_vix,
                'us_vix_change': us_vix_change,
                'india_vix': india_vix,
                'expected_regime': 'ELEVATED' if us_vix > 25 else 'NORMAL',
                'lead_time_sessions': 1 if us_vix_change > 25 else 2,
                'recommended_action': 'Reduce new entries, tighten stops',
            }

        # US VIX calming but India VIX still high
        if us_vix < 16 and india_vix > 22 and (us_vix_change or 0) < -5:
            return {
                'warning': 'VIX_REGIME_NORMALIZATION',
                'us_vix': us_vix,
                'india_vix': india_vix,
                'expected_regime': 'NORMAL',
                'lead_time_sessions': 2,
                'recommended_action': 'Prepare to increase exposure',
            }

        return None

    # ================================================================
    # BACKTEST HELPER
    # ================================================================
    def evaluate_backtest(self, nifty_df: pd.DataFrame,
                          global_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run US overnight signal on historical data for walk-forward.

        Args:
            nifty_df: DataFrame with 'date', 'open', 'close', 'india_vix'
            global_df: DataFrame with 'snapshot_date', 'us_overnight_return',
                       'us_vix_close', 'us_vix_change_pct', 'dxy_change_pct'

        Returns:
            DataFrame with date, signal_id, action, direction, confidence,
            composite_score, entry_price, exit_price, return_pct
        """
        results = []
        nifty_df = nifty_df.sort_values('date').reset_index(drop=True)

        # Pre-compute 5-day DXY rolling change
        if 'dxy_close' in global_df.columns:
            global_df = global_df.sort_values('snapshot_date').copy()
            global_df['dxy_5d_change'] = global_df['dxy_close'].pct_change(5) * 100
        else:
            global_df['dxy_5d_change'] = 0

        for i in range(1, len(nifty_df)):
            row = nifty_df.iloc[i]
            eval_date = row['date']

            # Get global data for this date
            g_row = global_df[global_df['snapshot_date'] == eval_date]
            if len(g_row) == 0:
                continue
            g = g_row.iloc[0]

            signal = self.evaluate(
                eval_date=eval_date,
                us_return=g.get('us_overnight_return', 0),
                us_vix_close=g.get('us_vix_close', 20),
                us_vix_change_pct=g.get('us_vix_change_pct', 0),
                dxy_change_pct=g.get('dxy_change_pct', 0),
                dxy_5d_change=g.get('dxy_5d_change', 0),
            )

            if signal['action'] is None or signal['action'] == 'REDUCE_ALL':
                # Track REDUCE_ALL as a special action for risk management
                if signal['action'] == 'REDUCE_ALL':
                    results.append({
                        'date': eval_date,
                        'signal_id': 'US_OVERNIGHT',
                        'action': 'REDUCE_ALL',
                        'direction': 'SHORT',
                        'confidence': signal['confidence'],
                        'composite_score': signal['composite_score'],
                        'entry_price': row['open'],
                        'exit_price': row['close'],
                        'return_pct': 0,  # Not a trade, just a risk reduction
                    })
                continue

            # Simulate trade: enter at open, exit at close
            entry_price = row['open']
            exit_price = row['close']

            if signal['direction'] == 'LONG':
                # Apply 1.5% stop loss
                stop = entry_price * 0.985
                if row['low'] <= stop:
                    exit_price = stop
                ret = (exit_price - entry_price) / entry_price * 100
            else:
                stop = entry_price * 1.015
                if row['high'] >= stop:
                    exit_price = stop
                ret = (entry_price - exit_price) / entry_price * 100

            results.append({
                'date': eval_date,
                'signal_id': 'US_OVERNIGHT',
                'action': signal['action'],
                'direction': signal['direction'],
                'confidence': signal['confidence'],
                'composite_score': signal['composite_score'],
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret, 4),
            })

        return pd.DataFrame(results)

    # ================================================================
    # DB HELPERS
    # ================================================================
    def _load_snapshot(self, eval_date: date) -> Optional[Dict]:
        if not self.db:
            return None
        try:
            cur = self.db.cursor()
            cur.execute("""
                SELECT us_overnight_return, us_vix_close, us_vix_change_pct,
                       dxy_close, dxy_change_pct
                FROM global_market_snapshots
                WHERE snapshot_date <= %s
                ORDER BY snapshot_date DESC LIMIT 1
            """, (eval_date,))
            row = cur.fetchone()
            if row:
                return {
                    'us_overnight_return': row[0],
                    'us_vix_close': row[1],
                    'us_vix_change_pct': row[2],
                    'dxy_close': row[3],
                    'dxy_change_pct': row[4],
                }
            return None
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None

    def _get_india_vix(self) -> Optional[float]:
        if not self.db:
            return None
        try:
            cur = self.db.cursor()
            cur.execute("""
                SELECT india_vix FROM nifty_daily
                WHERE india_vix IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            return float(row[0]) if row else None
        except Exception as e:
            return None

    def _get_dxy_5d_change(self, eval_date: date) -> Optional[float]:
        """Get 5-day DXY change from stored history."""
        if not self.db:
            return None
        try:
            cur = self.db.cursor()
            cur.execute("""
                SELECT dxy_close FROM global_market_snapshots
                WHERE snapshot_date <= %s AND dxy_close IS NOT NULL
                ORDER BY snapshot_date DESC LIMIT 6
            """, (eval_date,))
            rows = cur.fetchall()
            if len(rows) >= 6:
                latest = rows[0][0]
                five_ago = rows[5][0]
                if five_ago and five_ago > 0:
                    return round((latest - five_ago) / five_ago * 100, 4)
            return None
        except Exception as e:
            return None

    def _empty_result(self, eval_date: date) -> Dict:
        return {
            'signal_id': 'US_OVERNIGHT',
            'date': eval_date,
            'action': None,
            'direction': None,
            'confidence': 0.0,
            'composite_score': 0.0,
            'crisis_mode': False,
            'size_modifier': 1.0,
            'sub_signals': {},
            'reason': '',
        }
