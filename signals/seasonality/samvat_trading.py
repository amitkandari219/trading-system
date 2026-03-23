"""
Diwali Muhurat Trading Signal.

Muhurat trading is a special 1-hour session on Diwali evening, considered
auspicious by Indian traders.  Historically shows a strong bullish bias:
~70%+ positive closes on Muhurat day with a ~65% win-rate on a 5-day
buy-and-hold from Muhurat.

Signal logic:
    T+0 (Muhurat day):
        LONG signal — strong historical bullish bias.
        Strength scaled by recent market regime (trending up = stronger).

    T+1 to T+5 (next 5 trading days):
        Hold the LONG from Muhurat day.
        Exit after 5 trading days.

    Filters:
        - India VIX > 30 → skip (extreme fear overrides seasonal)
        - If market in severe downtrend (close < 200 DMA by >5%) → reduce strength

Walk-forward parameters exposed as class constants.

Usage:
    from signals.seasonality.samvat_trading import SamvatTradingSignal

    sig = SamvatTradingSignal()
    result = sig.evaluate(df, date(2026, 10, 20))
"""

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS / WF PARAMETERS
# ================================================================

SIGNAL_ID = 'SAMVAT_TRADING'

# Historical Diwali / Muhurat trading dates
# Muhurat trading happens on Diwali day (Amavasya in Kartik month)
DIWALI_DATES: List[date] = [
    date(2017, 10, 19),
    date(2018, 11, 7),
    date(2019, 10, 27),
    date(2020, 11, 14),
    date(2021, 11, 4),
    date(2022, 10, 24),
    date(2023, 11, 12),
    date(2024, 11, 1),
    date(2025, 10, 20),
    date(2026, 11, 8),   # Projected
]

# Hold period
HOLD_DAYS = 5                    # Exit after 5 trading days

# Historical stats
MUHURAT_WIN_RATE = 0.70          # 70%+ positive closes on Muhurat day
HOLD_WIN_RATE = 0.65             # 65% win rate for 5-day hold

# Strength
BASE_STRENGTH = 0.65
TREND_BOOST = 0.08               # Extra if above 200 DMA
TREND_PENALTY = -0.10            # Penalty if well below 200 DMA
MIN_STRENGTH = 0.10
MAX_STRENGTH = 0.90

# Filters
VIX_MAX = 30.0                   # Skip if VIX ≥ 30
DMA_200_DISCOUNT_PCT = 5.0       # Below 200 DMA by this much → penalise


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _compute_sma(series: pd.Series, window: int) -> float:
    """Compute SMA of a series; returns NaN if insufficient data."""
    if series is None or len(series) < window:
        return float('nan')
    return float(series.tail(window).mean())


# ================================================================
# SIGNAL CLASS
# ================================================================

class SamvatTradingSignal:
    """
    Diwali Muhurat trading signal for Nifty.

    LONG on Muhurat day, hold for 5 trading days.
    Exploits the strong historical bullish bias during the auspicious session.
    """

    SIGNAL_ID = SIGNAL_ID
    DIWALI_DATES = DIWALI_DATES

    # WF parameters
    HOLD_DAYS = HOLD_DAYS
    VIX_MAX = VIX_MAX
    BASE_STRENGTH = BASE_STRENGTH

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('SamvatTradingSignal initialised')

    # ----------------------------------------------------------
    # PUBLIC evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate Samvat (Muhurat) Trading signal.

        Parameters
        ----------
        df         : DataFrame with columns: date, open, high, low, close,
                     india_vix (optional).
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason, metadata
        or None if no signal.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('SamvatTradingSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None
        if 'date' not in df.columns:
            return None

        td = trade_date

        # Get current row
        row = df[df['date'] == pd.Timestamp(td)]
        if row.empty:
            return None
        row = row.iloc[-1]

        close = _safe_float(row.get('close'))
        if math.isnan(close) or close <= 0:
            return None

        # ── Check: Is this Muhurat day? ──────────────────────────
        if td in DIWALI_DATES:
            return self._muhurat_day_signal(df, row, td, close)

        # ── Check: Is this within 5 trading days after Muhurat? ──
        for diwali_date in DIWALI_DATES:
            if diwali_date >= td:
                continue
            cal_gap = (td - diwali_date).days
            if cal_gap > 10:  # Can't be more than 10 calendar days for 5 trading days
                continue
            # Count trading days between diwali_date and td
            trading_days_after = df[
                (df['date'] > pd.Timestamp(diwali_date)) &
                (df['date'] <= pd.Timestamp(td))
            ].shape[0]
            if 1 <= trading_days_after <= self.HOLD_DAYS:
                return self._hold_signal(df, row, td, close, diwali_date, trading_days_after)

        return None

    # ----------------------------------------------------------
    # Phase signals
    # ----------------------------------------------------------
    def _muhurat_day_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date, close: float,
    ) -> Optional[Dict]:
        """MUHURAT DAY: LONG with strong bullish bias."""
        # Filter: VIX
        vix = _safe_float(row.get('india_vix'))
        if not math.isnan(vix) and vix >= self.VIX_MAX:
            logger.debug('VIX %.1f >= %.1f — skip Muhurat signal', vix, self.VIX_MAX)
            return None

        strength = self.BASE_STRENGTH

        # Check 200 DMA regime
        hist_close = df[df['date'] <= pd.Timestamp(td)]['close']
        sma_200 = _compute_sma(hist_close, 200)
        regime_note = ''
        if not math.isnan(sma_200) and sma_200 > 0:
            pct_from_200 = ((close - sma_200) / sma_200) * 100.0
            if pct_from_200 > 0:
                strength = min(MAX_STRENGTH, strength + TREND_BOOST)
                regime_note = f'Above200DMA by {pct_from_200:.1f}%'
            elif pct_from_200 < -DMA_200_DISCOUNT_PCT:
                strength = max(MIN_STRENGTH, strength + TREND_PENALTY)
                regime_note = f'Below200DMA by {abs(pct_from_200):.1f}% (caution)'
            else:
                regime_note = f'Near200DMA ({pct_from_200:+.1f}%)'

        reason = (
            f"SAMVAT_MUHURAT | Diwali {td} | LONG bullish bias "
            f"(hist {MUHURAT_WIN_RATE*100:.0f}% win) | {regime_note}"
        )

        logger.info('%s MUHURAT_DAY: %s LONG strength=%.3f',
                     self.SIGNAL_ID, td, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'LONG',
            'strength': round(max(MIN_STRENGTH, strength), 3),
            'price': round(close, 2),
            'reason': reason.strip(),
            'metadata': {
                'phase': 'MUHURAT_DAY',
                'diwali_date': td.isoformat(),
                'hold_days': self.HOLD_DAYS,
                'historical_win_rate': MUHURAT_WIN_RATE,
            },
        }

    def _hold_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date,
        close: float, diwali_date: date, days_held: int,
    ) -> Optional[Dict]:
        """HOLD phase: continue LONG from Muhurat day."""
        # Decaying strength as we approach exit
        decay = 1.0 - (days_held / (self.HOLD_DAYS + 1)) * 0.3
        strength = self.BASE_STRENGTH * decay
        strength = max(MIN_STRENGTH, min(MAX_STRENGTH, strength))

        # Is this the exit day?
        is_exit = (days_held == self.HOLD_DAYS)

        reason = (
            f"SAMVAT_HOLD | Day {days_held}/{self.HOLD_DAYS} after Muhurat {diwali_date} | "
            f"{'EXIT today' if is_exit else 'Continue LONG'}"
        )

        logger.info('%s HOLD: %s day=%d/%d %s',
                     self.SIGNAL_ID, td, days_held, self.HOLD_DAYS,
                     'EXIT' if is_exit else 'HOLD')

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'LONG',
            'strength': round(strength, 3),
            'price': round(close, 2),
            'reason': reason,
            'metadata': {
                'phase': 'HOLD' if not is_exit else 'EXIT',
                'diwali_date': diwali_date.isoformat(),
                'days_held': days_held,
                'hold_days_total': self.HOLD_DAYS,
                'historical_hold_win_rate': HOLD_WIN_RATE,
            },
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"SamvatTradingSignal(signal_id='{self.SIGNAL_ID}')"
