"""
Quarter-End Mutual Fund Window Dressing Signal.

Mutual funds rebalance portfolios near quarter-end to show winners
and dump losers in their holdings reports.  This creates predictable
buying pressure in the last 5 trading days of each quarter (DRESSING)
and a reversal in the first 5 trading days of the new quarter (REVERSAL).

Signal logic:
    Phase 1 — DRESSING (last 5 trading days of quarter):
        LONG.  Ride institutional buying as MFs accumulate winners.

    Phase 2 — REVERSAL (first 5 trading days of new quarter):
        SHORT.  Fade the unwind / profit-booking once reporting pressure ends.

    Filters:
        - VIX < 20 (skip in volatile regimes — window dressing takes a back
          seat when markets are panicking).
        - Clear winner/loser separation when top_winners / bottom_losers
          data is available.

    Risk management:
        - SL: 1.5 % of Nifty spot.
        - TGT: 1.5 % of Nifty spot.
        - Max hold: 5 trading days per phase.

Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31.

Interface:
    evaluate(trade_date, nifty_spot, quarter_end_date, top_winners,
             bottom_losers, vix) → dict | None

    backtest_evaluate(trade_date, daily_df) → dict | None

Usage:
    from signals.structural.quarter_window import QuarterWindowSignal

    sig = QuarterWindowSignal()
    result = sig.evaluate(date(2026, 3, 27), 24500, date(2026, 3, 31))

Academic basis: Lakonishok et al. (1991) — "Window Dressing by Pension
Fund Managers"; Carhart et al. (2002) — "Leaning for the Tape".
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = "QUARTER_WINDOW"

# Quarter-end months and their last days
_QUARTER_END_MONTHS = {
    3: 31,   # Mar 31
    6: 30,   # Jun 30
    9: 30,   # Sep 30
    12: 31,  # Dec 31
}

# Window sizes (trading days, not calendar days)
DRESSING_WINDOW_DAYS = 5        # Last 5 trading days of quarter
REVERSAL_WINDOW_DAYS = 5        # First 5 trading days of new quarter

# Calendar buffer — we look back/forward this many calendar days
# to cover weekends and holidays when mapping trading days.
_CALENDAR_BUFFER = 10

# VIX filter
MAX_VIX = 20.0

# Winner / loser separation (min return gap in pct points)
MIN_WINNER_LOSER_GAP = 5.0      # top winners must outperform bottom losers by ≥ 5 pp

# Risk management (as fraction of Nifty spot)
SL_PCT = 0.015                   # 1.5 %
TGT_PCT = 0.015                  # 1.5 %
MAX_HOLD_DAYS = 5                # 5 trading days per phase

# Confidence
BASE_CONF_DRESSING = 0.55
BASE_CONF_REVERSAL = 0.50
STRONG_SEPARATION_BOOST = 0.05   # Clear winner/loser gap
VIX_LOW_BOOST = 0.03             # VIX < 14 → calmer market
VIX_MODERATE_PENALTY = -0.03     # VIX 17-20

# Phase labels
PHASE_DRESSING = "DRESSING"
PHASE_REVERSAL = "REVERSAL"


# ================================================================
# Helpers
# ================================================================

def _safe_float(val: Any, default: float = float("nan")) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_quarter_end_dates(year: int) -> List[date]:
    """
    Return the four quarter-end dates for the given year.

    Returns
    -------
    List of date objects: [Mar 31, Jun 30, Sep 30, Dec 31].
    """
    ends: List[date] = []
    for month, day in sorted(_QUARTER_END_MONTHS.items()):
        ends.append(date(year, month, day))
    return ends


def _nearest_quarter_end(trade_date: date) -> date:
    """
    Find the nearest quarter-end date relative to *trade_date*.

    Looks at the current year's quarter ends plus the first quarter end
    of the next year and the last quarter end of the previous year to
    handle edge cases around Dec-31 / Jan-1.
    """
    candidates: List[date] = []
    for y in (trade_date.year - 1, trade_date.year, trade_date.year + 1):
        candidates.extend(_get_quarter_end_dates(y))

    best = min(candidates, key=lambda d: abs((d - trade_date).days))
    return best


def _is_weekday(d: date) -> bool:
    """True if d is Mon-Fri."""
    return d.weekday() < 5


def _trading_days_before(end_date: date, count: int) -> List[date]:
    """
    Return the last *count* weekdays on or before *end_date*, in
    chronological order (earliest first).

    Note: this is a simple weekday heuristic.  It does not account for
    exchange holidays.  The caller may further filter with an actual
    trading-calendar if available.
    """
    days: List[date] = []
    d = end_date
    while len(days) < count:
        if _is_weekday(d):
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    return days


def _trading_days_after(start_date: date, count: int) -> List[date]:
    """
    Return the first *count* weekdays strictly after *start_date*,
    in chronological order.
    """
    days: List[date] = []
    d = start_date + timedelta(days=1)
    while len(days) < count:
        if _is_weekday(d):
            days.append(d)
        d += timedelta(days=1)
    return days


# ================================================================
# Signal Class
# ================================================================

class QuarterWindowSignal:
    """
    Quarter-end mutual-fund window-dressing signal.

    Detects whether the current trading day falls within the DRESSING
    window (last 5 days of quarter → LONG) or the REVERSAL window
    (first 5 days of new quarter → SHORT).
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(
        self,
        max_vix: float = MAX_VIX,
        sl_pct: float = SL_PCT,
        tgt_pct: float = TGT_PCT,
        max_hold_days: int = MAX_HOLD_DAYS,
    ) -> None:
        self.max_vix = max_vix
        self.sl_pct = sl_pct
        self.tgt_pct = tgt_pct
        self.max_hold_days = max_hold_days
        logger.info("QuarterWindowSignal initialised")

    # ----------------------------------------------------------------
    # Public: live evaluate
    # ----------------------------------------------------------------

    def evaluate(
        self,
        trade_date: date,
        nifty_spot: float,
        quarter_end_date: Optional[date] = None,
        top_winners: Optional[List[float]] = None,
        bottom_losers: Optional[List[float]] = None,
        vix: float = 15.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate quarter-window signal for a given trading day.

        Parameters
        ----------
        trade_date       : Current trading date.
        nifty_spot       : Current Nifty spot price.
        quarter_end_date : Override the quarter-end date (auto-detected if None).
        top_winners      : List of quarterly returns (%) for top-performing
                           stocks.  Used for winner/loser separation check.
        bottom_losers    : List of quarterly returns (%) for bottom-performing
                           stocks.
        vix              : India VIX value.

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(
                trade_date, nifty_spot, quarter_end_date,
                top_winners, bottom_losers, vix,
            )
        except Exception as e:
            logger.error(
                "QuarterWindowSignal.evaluate error: %s", e, exc_info=True
            )
            return None

    def _evaluate_inner(
        self,
        trade_date: date,
        nifty_spot: float,
        quarter_end_date: Optional[date],
        top_winners: Optional[List[float]],
        bottom_losers: Optional[List[float]],
        vix: float,
    ) -> Optional[Dict[str, Any]]:
        # ── Validate price ─────────────────────────────────────────
        nifty_spot = _safe_float(nifty_spot)
        if math.isnan(nifty_spot) or nifty_spot <= 0:
            logger.debug("Invalid nifty_spot: %s", nifty_spot)
            return None

        # ── VIX filter ─────────────────────────────────────────────
        vix = _safe_float(vix, default=99.0)
        if math.isnan(vix) or vix >= self.max_vix:
            logger.debug(
                "%s: VIX %.1f >= %.1f — skip", trade_date, vix, self.max_vix
            )
            return None

        # ── Check quarter window ───────────────────────────────────
        in_window, phase, qe_date = self._is_quarter_window(
            trade_date, quarter_end_date
        )
        if not in_window or phase is None:
            return None

        # ── Winner / loser separation (optional filter) ────────────
        separation_strong = False
        if top_winners is not None and bottom_losers is not None:
            avg_win = np.nanmean(top_winners) if len(top_winners) > 0 else 0.0
            avg_lose = np.nanmean(bottom_losers) if len(bottom_losers) > 0 else 0.0
            gap = avg_win - avg_lose
            if gap < MIN_WINNER_LOSER_GAP:
                logger.debug(
                    "%s: Winner/loser gap %.1f pp < %.1f — skip",
                    trade_date, gap, MIN_WINNER_LOSER_GAP,
                )
                return None
            if gap >= MIN_WINNER_LOSER_GAP * 1.5:
                separation_strong = True

        # ── Direction ──────────────────────────────────────────────
        if phase == PHASE_DRESSING:
            direction = "LONG"
        elif phase == PHASE_REVERSAL:
            direction = "SHORT"
        else:
            return None

        # ── Confidence ─────────────────────────────────────────────
        if phase == PHASE_DRESSING:
            confidence = BASE_CONF_DRESSING
        else:
            confidence = BASE_CONF_REVERSAL

        if separation_strong:
            confidence += STRONG_SEPARATION_BOOST
        if vix < 14.0:
            confidence += VIX_LOW_BOOST
        elif vix >= 17.0:
            confidence += VIX_MODERATE_PENALTY

        confidence = min(0.85, max(0.10, confidence))

        # ── SL / TGT ──────────────────────────────────────────────
        sl_pts = round(nifty_spot * self.sl_pct, 2)
        tgt_pts = round(nifty_spot * self.tgt_pct, 2)

        if direction == "LONG":
            stop_loss = round(nifty_spot - sl_pts, 2)
            target = round(nifty_spot + tgt_pts, 2)
        else:
            stop_loss = round(nifty_spot + sl_pts, 2)
            target = round(nifty_spot - tgt_pts, 2)

        # ── Reason ─────────────────────────────────────────────────
        reason_parts = [
            f"QUARTER_WINDOW ({phase})",
            f"Dir={direction}",
            f"QtrEnd={qe_date.isoformat()}",
            f"Spot={nifty_spot:.2f}",
            f"VIX={vix:.1f}",
            f"Hold≤{self.max_hold_days}d",
        ]

        logger.info(
            "%s signal: %s %s %s qtr_end=%s spot=%.1f vix=%.1f conf=%.3f",
            self.SIGNAL_ID, phase, direction, trade_date,
            qe_date, nifty_spot, vix, confidence,
        )

        return {
            "signal_id": self.SIGNAL_ID,
            "direction": direction,
            "phase": phase,
            "confidence": round(confidence, 3),
            "quarter_end_date": qe_date.isoformat(),
            "trade_date": trade_date.isoformat(),
            "entry_price": round(nifty_spot, 2),
            "stop_loss": stop_loss,
            "target": target,
            "sl_pct": self.sl_pct,
            "tgt_pct": self.tgt_pct,
            "max_hold_days": self.max_hold_days,
            "vix": round(vix, 2),
            "reason": " | ".join(reason_parts),
        }

    # ----------------------------------------------------------------
    # Quarter window detection
    # ----------------------------------------------------------------

    def _is_quarter_window(
        self,
        trade_date: date,
        quarter_end_override: Optional[date] = None,
    ) -> Tuple[bool, Optional[str], date]:
        """
        Determine whether *trade_date* falls in a quarter-window.

        Parameters
        ----------
        trade_date          : Date to check.
        quarter_end_override: Explicit quarter-end date.  If None, the
                              nearest quarter-end is auto-detected.

        Returns
        -------
        (is_in_window, phase, quarter_end_date)
            phase is 'DRESSING', 'REVERSAL', or None.
        """
        qe = quarter_end_override if quarter_end_override else _nearest_quarter_end(trade_date)

        # ── DRESSING window: last N trading days up to and including qe ──
        dressing_days = _trading_days_before(qe, DRESSING_WINDOW_DAYS)
        if trade_date in dressing_days:
            return True, PHASE_DRESSING, qe

        # ── REVERSAL window: first N trading days after qe ──────────────
        reversal_days = _trading_days_after(qe, REVERSAL_WINDOW_DAYS)
        if trade_date in reversal_days:
            return True, PHASE_REVERSAL, qe

        return False, None, qe

    # ----------------------------------------------------------------
    # Backtest evaluate
    # ----------------------------------------------------------------

    def backtest_evaluate(
        self,
        trade_date: date,
        daily_df=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Backtest variant — uses daily OHLCV DataFrame to evaluate.

        Parameters
        ----------
        trade_date : date
            Date being evaluated.
        daily_df   : pandas DataFrame, optional
            Daily Nifty OHLCV with columns: date, open, high, low, close,
            volume.  Used to confirm elevated volume near quarter-end
            and to extract nifty_spot.

        Returns
        -------
        dict with signal details, or None.
        """
        try:
            return self._backtest_inner(trade_date, daily_df)
        except Exception as e:
            logger.error(
                "QuarterWindowSignal.backtest_evaluate error: %s",
                e, exc_info=True,
            )
            return None

    def _backtest_inner(
        self,
        trade_date: date,
        daily_df,
    ) -> Optional[Dict[str, Any]]:
        # ── Check if in quarter window ─────────────────────────────
        in_window, phase, qe_date = self._is_quarter_window(trade_date)
        if not in_window or phase is None:
            return None

        # ── Extract Nifty spot and VIX proxy from DataFrame ────────
        nifty_spot = None
        vix_proxy = 15.0  # default
        volume_elevated = False

        if daily_df is not None:
            try:
                import pandas as pd

                df = daily_df.copy()

                # Normalise column names
                col_map = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl in ("date", "trade_date"):
                        col_map[c] = "date"
                    elif cl == "close":
                        col_map[c] = "close"
                    elif cl == "volume":
                        col_map[c] = "volume"
                    elif cl in ("vix", "india_vix"):
                        col_map[c] = "vix"
                df = df.rename(columns=col_map)

                # Convert date column
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.date

                # Get close price for trade_date
                if "date" in df.columns and "close" in df.columns:
                    row = df[df["date"] == trade_date]
                    if not row.empty:
                        nifty_spot = float(row.iloc[0]["close"])

                # Get VIX if available
                if "date" in df.columns and "vix" in df.columns:
                    row = df[df["date"] == trade_date]
                    if not row.empty:
                        v = row.iloc[0]["vix"]
                        if pd.notna(v):
                            vix_proxy = float(v)

                # ── Volume confirmation ────────────────────────────
                # Check if recent volume is elevated relative to the
                # 20-day average (proxy for MF activity).
                if "date" in df.columns and "volume" in df.columns:
                    df_sorted = df.sort_values("date")
                    df_sorted = df_sorted[df_sorted["date"] <= trade_date]
                    if len(df_sorted) >= 20:
                        recent_vol = df_sorted["volume"].iloc[-5:].mean()
                        avg_vol = df_sorted["volume"].iloc[-25:-5].mean()
                        if avg_vol > 0 and recent_vol > avg_vol * 1.15:
                            volume_elevated = True

            except Exception as exc:
                logger.warning(
                    "Error extracting data from daily_df: %s", exc
                )

        # ── Fallback: no spot available ────────────────────────────
        if nifty_spot is None or nifty_spot <= 0:
            logger.debug(
                "%s: no nifty_spot available for %s", self.SIGNAL_ID, trade_date
            )
            return None

        # ── VIX filter ─────────────────────────────────────────────
        if vix_proxy >= self.max_vix:
            logger.debug(
                "%s backtest: VIX %.1f >= %.1f — skip",
                self.SIGNAL_ID, vix_proxy, self.max_vix,
            )
            return None

        # ── Direction ──────────────────────────────────────────────
        if phase == PHASE_DRESSING:
            direction = "LONG"
        else:
            direction = "SHORT"

        # ── Confidence ─────────────────────────────────────────────
        if phase == PHASE_DRESSING:
            confidence = BASE_CONF_DRESSING
        else:
            confidence = BASE_CONF_REVERSAL

        if volume_elevated:
            confidence += STRONG_SEPARATION_BOOST
        if vix_proxy < 14.0:
            confidence += VIX_LOW_BOOST
        elif vix_proxy >= 17.0:
            confidence += VIX_MODERATE_PENALTY

        confidence = min(0.85, max(0.10, confidence))

        # ── SL / TGT ──────────────────────────────────────────────
        sl_pts = round(nifty_spot * self.sl_pct, 2)
        tgt_pts = round(nifty_spot * self.tgt_pct, 2)

        if direction == "LONG":
            stop_loss = round(nifty_spot - sl_pts, 2)
            target = round(nifty_spot + tgt_pts, 2)
        else:
            stop_loss = round(nifty_spot + sl_pts, 2)
            target = round(nifty_spot - tgt_pts, 2)

        reason_parts = [
            f"QUARTER_WINDOW ({phase})",
            f"Dir={direction}",
            f"QtrEnd={qe_date.isoformat()}",
            f"Spot={nifty_spot:.2f}",
            f"VIX={vix_proxy:.1f}",
            f"VolElevated={volume_elevated}",
        ]

        logger.info(
            "%s backtest: %s %s %s qtr_end=%s spot=%.1f conf=%.3f",
            self.SIGNAL_ID, phase, direction, trade_date,
            qe_date, nifty_spot, confidence,
        )

        return {
            "signal_id": self.SIGNAL_ID,
            "direction": direction,
            "phase": phase,
            "confidence": round(confidence, 3),
            "quarter_end_date": qe_date.isoformat(),
            "trade_date": trade_date.isoformat(),
            "entry_price": round(nifty_spot, 2),
            "stop_loss": stop_loss,
            "target": target,
            "sl_pct": self.sl_pct,
            "tgt_pct": self.tgt_pct,
            "max_hold_days": self.max_hold_days,
            "vix": round(vix_proxy, 2),
            "volume_elevated": volume_elevated,
            "reason": " | ".join(reason_parts),
        }

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    @staticmethod
    def _get_quarter_end_dates(year: int) -> List[date]:
        """Return the four quarter-end dates for the given year."""
        return _get_quarter_end_dates(year)

    def reset(self) -> None:
        """Reset internal state (no-op for this signal)."""
        pass

    def __repr__(self) -> str:
        return f"QuarterWindowSignal(signal_id='{self.SIGNAL_ID}')"
