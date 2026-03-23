"""
China / Hang Seng Correlation Break Signal.

Monitors the rolling correlation between Hang Seng and Nifty daily
returns.  Under normal conditions, both track global risk sentiment
with correlation > 0.5.  When correlation drops below 0.2 for 3+
consecutive days, a "decouple event" is detected.

Signal logic:
    corr_20d = rolling 20-day correlation of daily returns
    decouple = corr_20d < 0.2 for 3+ consecutive days

    If decouple AND Nifty outperforming Hang Seng (20d) -> LONG
        (India-specific domestic bull — independent of China drag)
    If decouple AND Nifty underperforming Hang Seng (20d) -> caution overlay
        (India-specific weakness — not just EM contagion)
    If no decouple -> no signal

Data requirements:
    - Column 'hangseng_close' in df
    - Column 'close' for Nifty price

Walk-forward parameters:
    CORR_WINDOW, CORR_DECOUPLE_THRESH, DECOUPLE_MIN_DAYS,
    PERF_WINDOW
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS / WF PARAMETERS
# ================================================================

SIGNAL_ID = 'CHINA_DECOUPLE'

# Correlation parameters
CORR_WINDOW = 20                # Rolling correlation window (days)
CORR_DECOUPLE_THRESH = 0.2     # Correlation below this = decouple
CORR_NORMAL_THRESH = 0.5       # Normal regime above this
DECOUPLE_MIN_DAYS = 3           # Must persist for 3+ days

# Performance comparison window
PERF_WINDOW = 20                # 20-day return comparison

# Strength scaling
BASE_STRENGTH_LONG = 0.55
BASE_STRENGTH_CAUTION = 0.45
OUTPERFORMANCE_BOOST = 0.02     # Per 1% outperformance
MIN_STRENGTH = 0.3
MAX_STRENGTH = 1.0

# Minimum data points
MIN_ROWS = CORR_WINDOW + DECOUPLE_MIN_DAYS + 5

# Column names
COL_HANGSENG = 'hangseng_close'
COL_NIFTY_CLOSE = 'close'


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


def _pct_return(series: pd.Series, period: int) -> Optional[float]:
    """Compute percentage return over `period` rows."""
    if len(series) < period + 1:
        return None
    current = _safe_float(series.iloc[-1])
    past = _safe_float(series.iloc[-(period + 1)])
    if math.isnan(current) or math.isnan(past) or past <= 0:
        return None
    return ((current / past) - 1.0) * 100.0


# ================================================================
# SIGNAL CLASS
# ================================================================

class ChinaDecoupleSignal:
    """
    China/Hang Seng correlation break signal for Nifty.

    Detects when the Nifty-Hang Seng correlation breaks down,
    indicating India-specific dynamics rather than global EM contagion.
    Nifty outperformance during decouple → LONG (domestic bull).
    Nifty underperformance during decouple → caution overlay.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_CORR_WINDOW = CORR_WINDOW
    WF_CORR_DECOUPLE_THRESH = CORR_DECOUPLE_THRESH
    WF_DECOUPLE_MIN_DAYS = DECOUPLE_MIN_DAYS
    WF_PERF_WINDOW = PERF_WINDOW

    def __init__(self) -> None:
        logger.info('ChinaDecoupleSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate China/Hang Seng decouple signal.

        Parameters
        ----------
        df   : DataFrame with columns 'hangseng_close' and 'close'.
        date : Evaluation date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None if no signal / missing data.
        """
        try:
            return self._evaluate_inner(df, date)
        except Exception as e:
            logger.error('%s.evaluate error: %s', self.SIGNAL_ID, e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, eval_date: date) -> Optional[Dict]:
        # ── Check required columns ───────────────────────────────
        if COL_HANGSENG not in df.columns:
            logger.debug('%s: column %s not found', self.SIGNAL_ID, COL_HANGSENG)
            return None

        if COL_NIFTY_CLOSE not in df.columns:
            logger.debug('%s: column %s not found', self.SIGNAL_ID, COL_NIFTY_CLOSE)
            return None

        # ── Slice data up to eval_date ───────────────────────────
        if hasattr(df.index, 'date'):
            mask = df.index.date <= eval_date
        elif 'date' in df.columns:
            mask = pd.to_datetime(df['date']).dt.date <= eval_date
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        subset = df.loc[mask].copy()

        if len(subset) < MIN_ROWS:
            logger.debug('%s: insufficient data (%d < %d)', self.SIGNAL_ID,
                         len(subset), MIN_ROWS)
            return None

        # ── Compute daily returns ────────────────────────────────
        nifty = subset[COL_NIFTY_CLOSE].dropna()
        hangseng = subset[COL_HANGSENG].dropna()

        # Align on common index
        common_idx = nifty.index.intersection(hangseng.index)
        if len(common_idx) < MIN_ROWS:
            logger.debug('%s: insufficient common data points', self.SIGNAL_ID)
            return None

        nifty = nifty.loc[common_idx]
        hangseng = hangseng.loc[common_idx]

        nifty_ret = nifty.pct_change().dropna()
        hs_ret = hangseng.pct_change().dropna()

        # Re-align after pct_change
        common_ret_idx = nifty_ret.index.intersection(hs_ret.index)
        nifty_ret = nifty_ret.loc[common_ret_idx]
        hs_ret = hs_ret.loc[common_ret_idx]

        if len(nifty_ret) < CORR_WINDOW + DECOUPLE_MIN_DAYS:
            return None

        # ── Rolling correlation ──────────────────────────────────
        rolling_corr = nifty_ret.rolling(window=CORR_WINDOW).corr(hs_ret)
        rolling_corr = rolling_corr.dropna()

        if len(rolling_corr) < DECOUPLE_MIN_DAYS:
            return None

        # ── Check decouple event: corr < threshold for N+ days ──
        recent_corr = rolling_corr.iloc[-DECOUPLE_MIN_DAYS:]
        current_corr = _safe_float(rolling_corr.iloc[-1])

        if math.isnan(current_corr):
            return None

        decouple_days = int((recent_corr < CORR_DECOUPLE_THRESH).sum())
        is_decoupled = decouple_days >= DECOUPLE_MIN_DAYS

        if not is_decoupled:
            logger.debug('%s: no decouple — corr=%.3f, decouple_days=%d/%d',
                         self.SIGNAL_ID, current_corr, decouple_days, DECOUPLE_MIN_DAYS)
            return None

        # ── Compare relative performance ─────────────────────────
        nifty_perf = _pct_return(nifty, PERF_WINDOW)
        hs_perf = _pct_return(hangseng, PERF_WINDOW)

        if nifty_perf is None or hs_perf is None:
            return None

        nifty_price = _safe_float(nifty.iloc[-1])
        if math.isnan(nifty_price) or nifty_price <= 0:
            return None

        outperformance = nifty_perf - hs_perf  # positive = Nifty winning

        # ── Direction and strength ───────────────────────────────
        reason_parts = [self.SIGNAL_ID]

        if outperformance > 0:
            # Nifty outperforming during decouple → India-specific bull
            direction = 'LONG'
            strength = BASE_STRENGTH_LONG + outperformance * OUTPERFORMANCE_BOOST
            reason_parts.extend([
                f"Correlation={current_corr:.3f} (decouple {decouple_days}d)",
                f"Nifty_20d={nifty_perf:+.2f}% vs HangSeng_20d={hs_perf:+.2f}%",
                "India outperforming during decouple -> domestic bull",
            ])
        else:
            # Nifty underperforming during decouple → India-specific risk
            direction = 'SHORT'
            strength = BASE_STRENGTH_CAUTION + abs(outperformance) * OUTPERFORMANCE_BOOST
            reason_parts.extend([
                f"Correlation={current_corr:.3f} (decouple {decouple_days}d)",
                f"Nifty_20d={nifty_perf:+.2f}% vs HangSeng_20d={hs_perf:+.2f}%",
                "India underperforming during decouple -> caution overlay",
            ])

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        # ── Build metadata ───────────────────────────────────────
        metadata = {
            'current_corr': round(current_corr, 4),
            'decouple_days': decouple_days,
            'nifty_20d_return': round(nifty_perf, 4),
            'hangseng_20d_return': round(hs_perf, 4),
            'outperformance': round(outperformance, 4),
            'corr_window': CORR_WINDOW,
            'corr_decouple_thresh': CORR_DECOUPLE_THRESH,
        }

        logger.info(
            '%s signal: %s on %s | strength=%.3f | corr=%.3f | outperf=%.2f%%',
            self.SIGNAL_ID, direction, eval_date, strength, current_corr, outperformance,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(nifty_price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': metadata,
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        pass

    def __repr__(self) -> str:
        return f"ChinaDecoupleSignal(signal_id='{self.SIGNAL_ID}')"
