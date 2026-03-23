"""
Global overnight composite scorer — unified pre-market signal.

Combines GIFT Nifty gap, US overnight return, VIX lead, DXY-FII,
and crude oil impact into a single regime-weighted pre-market score.

This is the "one signal to rule them all" for global market context.
It runs at 8:45 AM IST (pre-market cron job) and produces:
  1. Directional bias: LONG / SHORT / NEUTRAL
  2. Size modifier: 0.3x to 1.3x multiplier on normal position size
  3. Confidence: 0-1 probability of directional accuracy
  4. Regime warning: Early VIX regime change detection
  5. Risk-off flag: When global conditions warrant reduced exposure

Architecture:
  - Reads from global_market_snapshots table (populated by GlobalMarketFetcher)
  - Calls GiftNiftyGapSignal and USOvernightSignal as sub-components
  - Outputs a GlobalPreMarketContext that integrates into DailyRun scoring

Usage:
    from signals.global_composite import GlobalCompositeSignal
    signal = GlobalCompositeSignal(db_conn=conn)
    context = signal.evaluate(date.today())
    # context.direction = 'LONG'
    # context.size_modifier = 1.15
    # context.confidence = 0.72
"""

import logging
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from signals.gift_nifty_gap import GiftNiftyGapSignal
from signals.us_overnight import USOvernightSignal

logger = logging.getLogger(__name__)

# ================================================================
# OUTPUT DATA CLASS
# ================================================================

@dataclass
class GlobalPreMarketContext:
    """Pre-market context from global signals, fed into daily scoring."""
    date: date
    direction: Optional[str] = None          # LONG, SHORT, or None
    bias_strength: float = 0.0               # -1 to +1
    size_modifier: float = 1.0               # 0.3 to 1.3
    confidence: float = 0.0                  # 0 to 1
    risk_off: bool = False                   # True = reduce all exposure
    regime_warning: Optional[Dict] = None    # Early VIX regime change
    gift_gap_signal: Optional[Dict] = None   # GIFT Nifty gap details
    us_overnight_signal: Optional[Dict] = None  # US overnight details
    crude_impact: float = 0.0                # -1 to +1
    composite_score: float = 0.0             # Final weighted score
    components: Dict = field(default_factory=dict)  # All sub-scores
    reason: str = ''

    def to_dict(self) -> Dict:
        return {
            'date': self.date,
            'direction': self.direction,
            'bias_strength': self.bias_strength,
            'size_modifier': self.size_modifier,
            'confidence': self.confidence,
            'risk_off': self.risk_off,
            'composite_score': self.composite_score,
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        """Format for Telegram alert."""
        emoji = {'LONG': '🟢', 'SHORT': '🔴'}.get(self.direction, '⚪')
        risk = '🚨 RISK-OFF' if self.risk_off else ''
        lines = [
            f"{emoji} Global Pre-Market: {self.direction or 'NEUTRAL'} {risk}",
            f"Score: {self.composite_score:+.3f} | Confidence: {self.confidence:.0%}",
            f"Size: {self.size_modifier:.2f}x",
        ]
        if self.gift_gap_signal and self.gift_gap_signal.get('gap_pct'):
            lines.append(f"GIFT Gap: {self.gift_gap_signal['gap_pct']:+.2f}% ({self.gift_gap_signal.get('gap_type', '?')})")
        if self.us_overnight_signal:
            us = self.us_overnight_signal.get('sub_signals', {})
            us_ret = us.get('us_return', {}).get('value', 0)
            lines.append(f"US Return: {us_ret:+.2f}%")
        if self.regime_warning:
            lines.append(f"⚠️ {self.regime_warning.get('warning', '')}")
        return '\n'.join(lines)


# ================================================================
# COMPOSITE WEIGHTS (sum to 1.0)
# ================================================================
COMPONENT_WEIGHTS = {
    'gift_gap': 0.35,       # Strongest: direct Nifty price discovery
    'us_overnight': 0.35,   # Strong: S&P + VIX + DXY composite
    'crude_oil': 0.15,      # Moderate: India crude dependency
    'asian_session': 0.15,  # Moderate: regional sentiment
}

# ================================================================
# SIZE MODIFIER BOUNDS
# ================================================================
SIZE_MIN = 0.3
SIZE_MAX = 1.3
SIZE_NEUTRAL = 1.0

# Minimum composite magnitude to generate a signal
SIGNAL_THRESHOLD = 0.15


class GlobalCompositeSignal:
    """
    Master pre-market signal combining all global market inputs.

    Evaluation order:
      1. Fetch global snapshot from DB
      2. Evaluate GIFT Nifty gap (independent signal)
      3. Evaluate US overnight composite (S&P + VIX + DXY)
      4. Score crude oil impact
      5. Score Asian session (Hang Seng, Nikkei)
      6. Check for regime warnings
      7. Combine with regime-weighted scores
      8. Apply crisis overrides
      9. Output GlobalPreMarketContext
    """

    def __init__(self, db_conn=None):
        self.db = db_conn
        self.gift_signal = GiftNiftyGapSignal(db_conn=db_conn)
        self.us_signal = USOvernightSignal(db_conn=db_conn)

    # ================================================================
    # MAIN EVALUATION
    # ================================================================
    def evaluate(self, eval_date: date = None,
                 snapshot: Dict = None) -> GlobalPreMarketContext:
        """
        Evaluate all global signals and produce unified pre-market context.

        Args:
            eval_date: Date to evaluate
            snapshot: Pre-fetched snapshot dict (optional, for backtesting)

        Returns:
            GlobalPreMarketContext with all signals and composite score
        """
        eval_date = eval_date or date.today()
        ctx = GlobalPreMarketContext(date=eval_date)

        # ── Load snapshot ─────────────────────────────────────────
        if snapshot is None:
            snapshot = self._load_full_snapshot(eval_date)
        if not snapshot:
            ctx.reason = 'No global market data available'
            return ctx

        # ── 1. GIFT Nifty Gap ─────────────────────────────────────
        gift_result = self.gift_signal.evaluate(
            eval_date=eval_date,
            gift_nifty_price=snapshot.get('gift_nifty_last'),
            nifty_prev_close=snapshot.get('nifty_prev_close'),
            india_vix=snapshot.get('india_vix'),
        )
        ctx.gift_gap_signal = gift_result
        gift_score = self._gift_to_score(gift_result)

        # ── 2. US Overnight ───────────────────────────────────────
        us_result = self.us_signal.evaluate(
            eval_date=eval_date,
            us_return=snapshot.get('us_overnight_return'),
            us_vix_close=snapshot.get('us_vix_close'),
            us_vix_change_pct=snapshot.get('us_vix_change_pct'),
            dxy_change_pct=snapshot.get('dxy_change_pct'),
        )
        ctx.us_overnight_signal = us_result
        us_score = us_result.get('composite_score', 0)

        # ── 3. Crude Oil Impact ───────────────────────────────────
        crude_score = self._score_crude(
            snapshot.get('brent_change_pct', 0),
            snapshot.get('brent_close', 75),
        )
        ctx.crude_impact = crude_score

        # ── 4. Asian Session ──────────────────────────────────────
        asian_score = self._score_asian(
            snapshot.get('hang_seng_change_pct', 0),
            snapshot.get('nikkei_change_pct', 0),
        )

        # ── 5. Weighted Composite ─────────────────────────────────
        composite = (
            COMPONENT_WEIGHTS['gift_gap'] * gift_score +
            COMPONENT_WEIGHTS['us_overnight'] * us_score +
            COMPONENT_WEIGHTS['crude_oil'] * crude_score +
            COMPONENT_WEIGHTS['asian_session'] * asian_score
        )

        # ── 6. Agreement Bonus ────────────────────────────────────
        # When multiple signals agree, boost confidence
        signs = [
            np.sign(gift_score) if abs(gift_score) > 0.1 else 0,
            np.sign(us_score) if abs(us_score) > 0.1 else 0,
            np.sign(crude_score) if abs(crude_score) > 0.1 else 0,
            np.sign(asian_score) if abs(asian_score) > 0.1 else 0,
        ]
        active_signs = [s for s in signs if s != 0]
        if len(active_signs) >= 3 and len(set(active_signs)) == 1:
            # 3+ signals agree → boost 15%
            composite *= 1.15
            ctx.reason = 'Multi-signal agreement (3+ components align)'
        elif len(active_signs) >= 2 and len(set(active_signs)) > 1:
            # Conflicting signals → dampen
            composite *= 0.8

        # ── 7. Crisis Override ────────────────────────────────────
        if us_result.get('crisis_mode', False):
            composite = min(composite, -0.6)
            ctx.risk_off = True

        composite = round(max(-1.0, min(1.0, composite)), 3)
        ctx.composite_score = composite

        # ── 8. Map to Direction ───────────────────────────────────
        if abs(composite) < SIGNAL_THRESHOLD:
            ctx.direction = None
            ctx.bias_strength = composite
            ctx.size_modifier = SIZE_NEUTRAL
            ctx.confidence = 0.0
            if not ctx.reason:
                ctx.reason = f'Neutral global context: {composite:+.3f}'
        elif composite > 0:
            ctx.direction = 'LONG'
            ctx.bias_strength = composite
            ctx.size_modifier = round(
                min(SIZE_MAX, SIZE_NEUTRAL + composite * 0.4), 3
            )
            ctx.confidence = round(min(0.90, 0.4 + composite * 0.6), 3)
            if not ctx.reason:
                ctx.reason = f'Bullish global: {composite:+.3f}'
        else:
            ctx.direction = 'SHORT'
            ctx.bias_strength = composite
            ctx.size_modifier = round(
                max(SIZE_MIN, SIZE_NEUTRAL + composite * 0.5), 3
            )
            ctx.confidence = round(min(0.90, 0.4 + abs(composite) * 0.6), 3)
            if not ctx.reason:
                ctx.reason = f'Bearish global: {composite:+.3f}'

        # ── 9. Regime Warning ─────────────────────────────────────
        regime_warning = self.us_signal.get_vix_regime_warning(eval_date)
        if regime_warning:
            ctx.regime_warning = regime_warning
            # If regime change imminent, reduce size
            if regime_warning.get('warning') == 'VIX_REGIME_CHANGE_IMMINENT':
                ctx.size_modifier = min(ctx.size_modifier, 0.7)

        # ── Store components for analysis ─────────────────────────
        ctx.components = {
            'gift_gap': round(gift_score, 3),
            'us_overnight': round(us_score, 3),
            'crude_oil': round(crude_score, 3),
            'asian_session': round(asian_score, 3),
            'agreement_count': len([s for s in active_signs if s != 0]),
        }

        return ctx

    # ================================================================
    # COMPONENT SCORERS
    # ================================================================

    def _gift_to_score(self, gift_result: Dict) -> float:
        """Convert GIFT Nifty gap signal to -1/+1 score."""
        if gift_result.get('action') is None:
            return 0.0

        direction_sign = 1.0 if gift_result.get('direction') == 'LONG' else -1.0
        confidence = gift_result.get('confidence', 0.5)
        return round(direction_sign * confidence, 3)

    def _score_crude(self, brent_change: float, brent_price: float) -> float:
        """
        Score crude oil impact on Nifty.

        India imports ~85% of crude oil. Sharp rises hurt:
          - Fiscal deficit widens
          - INR weakens → FII sell
          - Inflation rises → RBI hawkish

        Conversely, crude drops are bullish for India.
        """
        if brent_change is None:
            return 0.0

        score = 0.0

        # Daily change impact
        if brent_change > 5.0:
            score = -0.8   # Major crude spike
        elif brent_change > 3.0:
            score = -0.5   # Significant spike
        elif brent_change > 1.5:
            score = -0.2
        elif brent_change < -5.0:
            score = 0.5    # Major crude drop (bullish for India)
        elif brent_change < -3.0:
            score = 0.3
        elif brent_change < -1.5:
            score = 0.15

        # Absolute level impact (sustained high crude is bearish)
        if brent_price and brent_price > 100:
            score -= 0.1
        elif brent_price and brent_price < 60:
            score += 0.1

        return round(max(-1.0, min(1.0, score)), 3)

    def _score_asian(self, hang_seng_change: float, nikkei_change: float) -> float:
        """
        Score Asian market sentiment.

        Hang Seng and Nikkei trade during overlapping hours with NSE,
        but their pre-open moves (7-9 AM IST) provide directional context.
        Lower weight since limited lead time.
        """
        if hang_seng_change is None and nikkei_change is None:
            return 0.0

        hs = hang_seng_change or 0
        nk = nikkei_change or 0

        # Weighted average (Hang Seng more relevant for EM sentiment)
        asian_return = 0.6 * hs + 0.4 * nk

        if asian_return > 1.5:
            return 0.5
        elif asian_return > 0.5:
            return 0.2
        elif asian_return < -1.5:
            return -0.5
        elif asian_return < -0.5:
            return -0.2
        return 0.0

    # ================================================================
    # BACKTEST: Run on historical data
    # ================================================================
    def evaluate_backtest(self, nifty_df: pd.DataFrame,
                          global_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run composite signal on historical data for walk-forward.

        Returns DataFrame with trade-level results.
        """
        results = []
        nifty_df = nifty_df.sort_values('date').reset_index(drop=True)

        for i in range(1, len(nifty_df)):
            row = nifty_df.iloc[i]
            prev = nifty_df.iloc[i - 1]
            eval_date = row['date']

            # Build snapshot from global_df
            g_row = global_df[global_df['snapshot_date'] == eval_date]
            if len(g_row) == 0:
                continue
            g = g_row.iloc[0].to_dict()
            g['nifty_prev_close'] = prev['close']
            g['india_vix'] = row.get('india_vix', 16.0)

            # If we have SP500 change but no GIFT data, synthesize gap
            if g.get('gift_nifty_last') is None and g.get('sp500_change_pct') is not None:
                g['gift_nifty_last'] = prev['close'] * (
                    1 + g['sp500_change_pct'] * 0.7 / 100
                )

            ctx = self.evaluate(eval_date=eval_date, snapshot=g)

            if ctx.direction is None:
                continue

            # Simulate trade
            entry_price = row['open']
            exit_price = row['close']
            stop_pct = 0.015  # 1.5% stop

            if ctx.direction == 'LONG':
                stop = entry_price * (1 - stop_pct)
                if row['low'] <= stop:
                    exit_price = stop
                ret = (exit_price - entry_price) / entry_price * 100
            else:
                stop = entry_price * (1 + stop_pct)
                if row['high'] >= stop:
                    exit_price = stop
                ret = (entry_price - exit_price) / entry_price * 100

            # Apply size modifier to return
            sized_ret = ret * ctx.size_modifier

            results.append({
                'date': eval_date,
                'signal_id': 'GLOBAL_COMPOSITE',
                'action': f'BIAS_{ctx.direction}',
                'direction': ctx.direction,
                'confidence': ctx.confidence,
                'composite_score': ctx.composite_score,
                'size_modifier': ctx.size_modifier,
                'risk_off': ctx.risk_off,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret, 4),
                'sized_return_pct': round(sized_ret, 4),
                'gift_gap_score': ctx.components.get('gift_gap', 0),
                'us_overnight_score': ctx.components.get('us_overnight', 0),
                'crude_score': ctx.components.get('crude_oil', 0),
                'asian_score': ctx.components.get('asian_session', 0),
            })

        return pd.DataFrame(results)

    # ================================================================
    # DB HELPERS
    # ================================================================
    def _load_full_snapshot(self, eval_date: date) -> Optional[Dict]:
        """Load full global snapshot for given date."""
        if not self.db:
            return None
        try:
            cur = self.db.cursor()
            cur.execute("""
                SELECT snapshot_date, gift_nifty_last, gift_nifty_gap_pct,
                       nifty_prev_close, sp500_close, sp500_change_pct,
                       nasdaq_close, nasdaq_change_pct,
                       us_vix_close, us_vix_change_pct,
                       dxy_close, dxy_change_pct,
                       brent_close, brent_change_pct,
                       us_overnight_return, global_risk_score
                FROM global_market_snapshots
                WHERE snapshot_date <= %s
                ORDER BY snapshot_date DESC LIMIT 1
            """, (eval_date,))
            row = cur.fetchone()
            if row:
                return {
                    'snapshot_date': row[0],
                    'gift_nifty_last': row[1],
                    'gift_nifty_gap_pct': row[2],
                    'nifty_prev_close': row[3],
                    'sp500_close': row[4],
                    'sp500_change_pct': row[5],
                    'nasdaq_close': row[6],
                    'nasdaq_change_pct': row[7],
                    'us_vix_close': row[8],
                    'us_vix_change_pct': row[9],
                    'dxy_close': row[10],
                    'dxy_change_pct': row[11],
                    'brent_close': row[12],
                    'brent_change_pct': row[13],
                    'us_overnight_return': row[14],
                    'global_risk_score': row[15],
                }
            return None
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None

    # ================================================================
    # STORE EVALUATION RESULT
    # ================================================================
    def store_evaluation(self, ctx: GlobalPreMarketContext) -> bool:
        """Store evaluation result to global_signal_evaluations table."""
        if not self.db:
            return False
        try:
            import json
            cur = self.db.cursor()
            cur.execute("""
                INSERT INTO global_signal_evaluations (
                    eval_date, direction, bias_strength, size_modifier,
                    confidence, risk_off, composite_score,
                    gift_gap_score, us_overnight_score, crude_score,
                    asian_score, regime_warning, reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (eval_date) DO UPDATE SET
                    direction = EXCLUDED.direction,
                    bias_strength = EXCLUDED.bias_strength,
                    size_modifier = EXCLUDED.size_modifier,
                    confidence = EXCLUDED.confidence,
                    risk_off = EXCLUDED.risk_off,
                    composite_score = EXCLUDED.composite_score,
                    reason = EXCLUDED.reason
            """, (
                ctx.date, ctx.direction, ctx.bias_strength,
                ctx.size_modifier, ctx.confidence, ctx.risk_off,
                ctx.composite_score,
                ctx.components.get('gift_gap', 0),
                ctx.components.get('us_overnight', 0),
                ctx.components.get('crude_oil', 0),
                ctx.components.get('asian_session', 0),
                json.dumps(ctx.regime_warning) if ctx.regime_warning else None,
                ctx.reason,
            ))
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store evaluation: {e}")
            self.db.rollback()
            return False
