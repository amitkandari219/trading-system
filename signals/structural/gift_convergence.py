"""
GIFT Nifty → NSE Convergence Signal.

Fades the basis between GIFT Nifty overnight price and the NSE opening
price.  The premise is that GIFT Nifty (SGX successor) establishes an
overnight fair-value; when NSE opens at a significant premium or discount
to that level, the first 30 minutes tend to converge toward the GIFT price.

Signal logic:
    basis = gift_price - nse_open

    basis > +50 pts  → LONG  Nifty (NSE is underpriced vs GIFT)
    basis < -50 pts  → SHORT Nifty (NSE is overpriced vs GIFT)
    |basis| < 50     → NO TRADE

    Target : basis * 0.7 convergence
    SL     : 0.5 * |basis| in the wrong direction
    Hard exit: 09:45 AM IST (30 min window from 09:15)

    Filters:
        - India VIX < 25 (too volatile = convergence unreliable)
        - Not an event day (RBI policy, budget, FOMC outcome)
        - Gap (open vs prev close) < 2% (extreme gaps break convergence)

Backtest proxy:
    Since GIFT Nifty historical data is unavailable in the DB, backtest
    uses (NSE open - prev NSE close) as the gap proxy for GIFT basis.
    The signal fires at 09:15:30 and exits by 09:45.

Usage:
    from signals.structural.gift_convergence import GiftConvergenceSignal

    sig = GiftConvergenceSignal()

    # Live
    result = sig.evaluate(trade_date, gift_price, nse_open,
                          prev_close, vix, is_event_day)

    # Backtest
    result = sig.backtest_evaluate(trade_date, day_open, prev_close,
                                    session_bars, vix)

Academic basis: SGX-NIFTY / GIFT-NIFTY basis convergence — overnight
price discovery leads NSE with ~70% convergence within the first 30 min
on non-event days (empirical study of gap-fill statistics).
"""

import logging
import math
from datetime import date, time, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'GIFT_CONVERGENCE'

# Basis thresholds (Nifty points)
BASIS_MIN_PTS = 50               # Minimum basis to act on
BASIS_MAX_PTS = 500              # Sanity cap — beyond this data is suspect

# Time window
SIGNAL_FIRE_TIME = time(9, 15, 30)   # Entry at 09:15:30 (first bar close)
HARD_EXIT_TIME = time(9, 45)         # Must be flat by 09:45
SIGNAL_WINDOW_START = time(9, 15)
SIGNAL_WINDOW_END = time(9, 45)

# Risk / reward
TGT_CONVERGENCE_RATIO = 0.70    # Expect 70% of basis to converge
SL_ADVERSE_RATIO = 0.50         # SL if basis expands 50% further
MAX_HOLD_BARS = 6               # 6 × 5-min bars = 30 min (for backtest)

# Filters
VIX_MAX = 25.0                  # Don't trade if VIX ≥ 25
GAP_MAX_PCT = 2.0               # Gap > 2% → skip (extreme dislocation)

# Confidence
BASE_CONFIDENCE = 0.58          # Historical convergence ~70% of the time
VIX_LOW_BOOST = 0.06            # Extra confidence when VIX < 15
LARGE_BASIS_BOOST = 0.05        # Extra when |basis| > 80 pts
SMALL_BASIS_PENALTY = -0.04     # Less confidence near threshold

# Size
BASE_SIZE_MODIFIER = 1.0
MAX_SIZE_MODIFIER = 1.5
MIN_SIZE_MODIFIER = 0.5


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


def _gap_pct(open_price: float, prev_close: float) -> float:
    """Compute gap as a percentage of previous close."""
    if prev_close <= 0:
        return float('nan')
    return ((open_price - prev_close) / prev_close) * 100.0


def _bar_close(bar: Dict) -> float:
    """Extract close price from a bar dict."""
    return _safe_float(bar.get('close'))


def _bar_high(bar: Dict) -> float:
    """Extract high price from a bar dict."""
    return _safe_float(bar.get('high'))


def _bar_low(bar: Dict) -> float:
    """Extract low price from a bar dict."""
    return _safe_float(bar.get('low'))


def _bar_time_obj(bar: Dict) -> Optional[time]:
    """Extract time from bar timestamp."""
    ts = bar.get('timestamp') or bar.get('time') or bar.get('datetime')
    if ts is None:
        return None
    if hasattr(ts, 'time'):
        return ts.time()
    if isinstance(ts, time):
        return ts
    try:
        from datetime import datetime as _dt
        return _dt.fromisoformat(str(ts)).time()
    except (ValueError, TypeError):
        return None


# ================================================================
# SIGNAL CLASS
# ================================================================

class GiftConvergenceSignal:
    """
    GIFT Nifty → NSE convergence signal.

    Fires at most once per day at 09:15:30 IST.  Fades the basis
    between GIFT Nifty and NSE open, targeting 70% convergence
    within a 30-minute hard-exit window.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('GiftConvergenceSignal initialised')

    # ----------------------------------------------------------
    # LIVE evaluate
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: date,
        gift_price: float,
        nse_open: float,
        prev_close: float,
        vix: float,
        is_event_day: bool = False,
    ) -> Optional[Dict]:
        """
        Evaluate GIFT-NSE convergence signal (live or paper trading).

        Parameters
        ----------
        trade_date   : Trading date.
        gift_price   : GIFT Nifty last traded price (pre-open).
        nse_open     : NSE Nifty opening price at 09:15.
        prev_close   : Previous day's NSE Nifty close.
        vix          : India VIX level.
        is_event_day : True if RBI policy / budget / FOMC etc.

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(
                trade_date, gift_price, nse_open, prev_close, vix, is_event_day
            )
        except Exception as e:
            logger.error('GiftConvergenceSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(
        self,
        trade_date: date,
        gift_price: float,
        nse_open: float,
        prev_close: float,
        vix: float,
        is_event_day: bool,
    ) -> Optional[Dict]:
        # ── Max 1 fire per day ────────────────────────────────────
        if self._last_fire_date == trade_date:
            logger.debug('Already fired for %s', trade_date)
            return None

        # ── Validate inputs ───────────────────────────────────────
        gift_price = _safe_float(gift_price)
        nse_open = _safe_float(nse_open)
        prev_close = _safe_float(prev_close)
        vix = _safe_float(vix)

        if math.isnan(gift_price) or gift_price <= 0:
            logger.debug('Invalid gift_price: %s', gift_price)
            return None
        if math.isnan(nse_open) or nse_open <= 0:
            logger.debug('Invalid nse_open: %s', nse_open)
            return None
        if math.isnan(prev_close) or prev_close <= 0:
            logger.debug('Invalid prev_close: %s', prev_close)
            return None
        if math.isnan(vix) or vix <= 0:
            logger.debug('Invalid vix: %s', vix)
            return None

        # ── Filter: VIX ───────────────────────────────────────────
        if vix >= VIX_MAX:
            logger.debug('VIX %.1f >= %.1f threshold — skip', vix, VIX_MAX)
            return None

        # ── Filter: Event day ─────────────────────────────────────
        if is_event_day:
            logger.debug('Event day — skip')
            return None

        # ── Filter: Gap size ──────────────────────────────────────
        gap = _gap_pct(nse_open, prev_close)
        if math.isnan(gap) or abs(gap) > GAP_MAX_PCT:
            logger.debug('Gap %.2f%% exceeds %.1f%% limit', gap, GAP_MAX_PCT)
            return None

        # ── Compute basis ─────────────────────────────────────────
        basis = gift_price - nse_open
        abs_basis = abs(basis)

        if abs_basis < BASIS_MIN_PTS:
            logger.debug('|basis| %.1f < %.1f min threshold', abs_basis, BASIS_MIN_PTS)
            return None

        if abs_basis > BASIS_MAX_PTS:
            logger.debug('|basis| %.1f > %.1f sanity cap — data suspect', abs_basis, BASIS_MAX_PTS)
            return None

        # ── Direction ─────────────────────────────────────────────
        if basis > 0:
            # GIFT > NSE → NSE underpriced → LONG
            direction = 'LONG'
            entry_price = nse_open
            target = nse_open + abs_basis * TGT_CONVERGENCE_RATIO
            stop_loss = nse_open - abs_basis * SL_ADVERSE_RATIO
        else:
            # GIFT < NSE → NSE overpriced → SHORT
            direction = 'SHORT'
            entry_price = nse_open
            target = nse_open - abs_basis * TGT_CONVERGENCE_RATIO
            stop_loss = nse_open + abs_basis * SL_ADVERSE_RATIO

        # ── Confidence ────────────────────────────────────────────
        confidence = BASE_CONFIDENCE

        if vix < 15:
            confidence += VIX_LOW_BOOST
        if abs_basis > 80:
            confidence += LARGE_BASIS_BOOST
        if abs_basis < 60:
            confidence += SMALL_BASIS_PENALTY

        confidence = min(0.90, max(0.10, confidence))

        # ── Size modifier ─────────────────────────────────────────
        # Scale size with basis magnitude: larger basis → more conviction
        size_modifier = BASE_SIZE_MODIFIER
        if abs_basis > 100:
            size_modifier = min(MAX_SIZE_MODIFIER, 1.0 + (abs_basis - 100) * 0.005)
        elif abs_basis < 60:
            size_modifier = max(MIN_SIZE_MODIFIER, 0.8)

        # ── Risk / reward ─────────────────────────────────────────
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        # ── Build reason ──────────────────────────────────────────
        reason_parts = [
            f"GIFT_CONVERGENCE",
            f"Basis={basis:+.1f} pts",
            f"GIFT={gift_price:.2f}",
            f"NSE_Open={nse_open:.2f}",
            f"Gap={gap:+.2f}%",
            f"VIX={vix:.1f}",
            f"R:R={rr:.1f}",
        ]

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s basis=%.1f conf=%.3f',
            self.SIGNAL_ID, direction, trade_date, basis, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'hard_exit_time': HARD_EXIT_TIME.isoformat(),
            'entry_time': SIGNAL_FIRE_TIME.isoformat(),
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # BACKTEST evaluate
    # ----------------------------------------------------------
    def backtest_evaluate(
        self,
        trade_date: date,
        day_open: float,
        prev_close: float,
        session_bars: List[Dict],
        vix: float,
        is_event_day: bool = False,
    ) -> Optional[Dict]:
        """
        Backtest variant — uses (open - prev_close) as GIFT basis proxy.

        Enters at bar 0 close, checks SL/TGT through bars 1-6,
        force-exits at bar 6 if neither SL nor TGT is hit.

        Parameters
        ----------
        trade_date   : Date being backtested.
        day_open     : NSE Nifty open for the day.
        prev_close   : Previous day's NSE Nifty close.
        session_bars : List of 5-min bar dicts (at least 7 bars from 09:15).
                       Each bar: {open, high, low, close, volume, timestamp}.
        vix          : India VIX on that date.
        is_event_day : True on event days.

        Returns
        -------
        dict with trade result, or None if no trade.
        """
        try:
            return self._backtest_inner(
                trade_date, day_open, prev_close, session_bars, vix, is_event_day
            )
        except Exception as e:
            logger.error(
                'GiftConvergenceSignal.backtest_evaluate error: %s', e, exc_info=True
            )
            return None

    def _backtest_inner(
        self,
        trade_date: date,
        day_open: float,
        prev_close: float,
        session_bars: List[Dict],
        vix: float,
        is_event_day: bool,
    ) -> Optional[Dict]:
        # ── Validate inputs ───────────────────────────────────────
        day_open = _safe_float(day_open)
        prev_close = _safe_float(prev_close)
        vix = _safe_float(vix)

        if math.isnan(day_open) or day_open <= 0:
            return None
        if math.isnan(prev_close) or prev_close <= 0:
            return None
        if math.isnan(vix) or vix <= 0:
            return None

        if not session_bars or len(session_bars) < 2:
            return None

        # ── Filters ───────────────────────────────────────────────
        if vix >= VIX_MAX:
            return None
        if is_event_day:
            return None

        gap = _gap_pct(day_open, prev_close)
        if math.isnan(gap) or abs(gap) > GAP_MAX_PCT:
            return None

        # ── Compute basis proxy ───────────────────────────────────
        # In backtest: use gap as GIFT basis proxy
        basis = day_open - prev_close
        abs_basis = abs(basis)

        if abs_basis < BASIS_MIN_PTS:
            return None
        if abs_basis > BASIS_MAX_PTS:
            return None

        # ── Direction ─────────────────────────────────────────────
        # Positive gap → NSE opened higher than prev close
        # We fade the gap → SHORT (expecting pull back toward prev close)
        # Negative gap → NSE opened lower → LONG (expecting bounce)
        #
        # Note: In live mode, basis = GIFT - NSE_open.
        # In backtest proxy, basis = NSE_open - prev_close (the gap itself).
        # We fade the gap: gap > 0 → SHORT, gap < 0 → LONG.
        if basis < 0:
            direction = 'LONG'
        else:
            direction = 'SHORT'

        # ── Entry at bar 0 close ──────────────────────────────────
        entry_bar = session_bars[0]
        entry_price = _bar_close(entry_bar)
        if math.isnan(entry_price) or entry_price <= 0:
            return None

        # ── Compute SL / TGT ─────────────────────────────────────
        if direction == 'LONG':
            target = entry_price + abs_basis * TGT_CONVERGENCE_RATIO
            stop_loss = entry_price - abs_basis * SL_ADVERSE_RATIO
        else:
            target = entry_price - abs_basis * TGT_CONVERGENCE_RATIO
            stop_loss = entry_price + abs_basis * SL_ADVERSE_RATIO

        # ── Confidence ────────────────────────────────────────────
        confidence = BASE_CONFIDENCE
        if vix < 15:
            confidence += VIX_LOW_BOOST
        if abs_basis > 80:
            confidence += LARGE_BASIS_BOOST
        if abs_basis < 60:
            confidence += SMALL_BASIS_PENALTY
        confidence = min(0.90, max(0.10, confidence))

        # ── Simulate through bars 1 to 6 ─────────────────────────
        max_bar_idx = min(MAX_HOLD_BARS, len(session_bars) - 1)
        exit_price = None
        exit_bar_idx = None
        exit_reason = None

        for i in range(1, max_bar_idx + 1):
            bar = session_bars[i]
            bar_high = _bar_high(bar)
            bar_low = _bar_low(bar)
            bar_cls = _bar_close(bar)

            if math.isnan(bar_high) or math.isnan(bar_low):
                continue

            if direction == 'LONG':
                # Check SL first (worst case)
                if bar_low <= stop_loss:
                    exit_price = stop_loss
                    exit_bar_idx = i
                    exit_reason = 'SL_HIT'
                    break
                # Check TGT
                if bar_high >= target:
                    exit_price = target
                    exit_bar_idx = i
                    exit_reason = 'TGT_HIT'
                    break
            else:  # SHORT
                # Check SL first
                if bar_high >= stop_loss:
                    exit_price = stop_loss
                    exit_bar_idx = i
                    exit_reason = 'SL_HIT'
                    break
                # Check TGT
                if bar_low <= target:
                    exit_price = target
                    exit_bar_idx = i
                    exit_reason = 'TGT_HIT'
                    break

        # ── Hard exit at last bar if no SL/TGT ────────────────────
        if exit_price is None:
            last_idx = max_bar_idx
            if last_idx > 0 and last_idx < len(session_bars):
                exit_bar = session_bars[last_idx]
                exit_price = _bar_close(exit_bar)
                exit_bar_idx = last_idx
                exit_reason = 'HARD_EXIT_TIME'
            else:
                return None

        if math.isnan(exit_price) or exit_price <= 0:
            return None

        # ── Compute PnL ──────────────────────────────────────────
        if direction == 'LONG':
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        reason_parts = [
            'GIFT_CONVERGENCE (backtest)',
            f"Basis_proxy={basis:+.1f} pts",
            f"Open={day_open:.2f}",
            f"PrevClose={prev_close:.2f}",
            f"Gap={gap:+.2f}%",
            f"VIX={vix:.1f}",
            f"Exit={exit_reason}",
            f"R:R={rr:.1f}",
        ]

        logger.info(
            '%s backtest: %s %s basis_proxy=%.1f pnl=%.1f exit=%s',
            self.SIGNAL_ID, direction, trade_date, basis, pnl_pts, exit_reason,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'trade_date': trade_date.isoformat(),
            'direction': direction,
            'confidence': round(confidence, 3),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'exit_price': round(exit_price, 2),
            'exit_bar_idx': exit_bar_idx,
            'exit_reason': exit_reason,
            'pnl_pts': round(pnl_pts, 2),
            'basis_proxy': round(basis, 2),
            'gap_pct': round(gap, 4),
            'vix': round(vix, 1),
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility: reset state (for multi-day backtest loops)
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"GiftConvergenceSignal(signal_id='{self.SIGNAL_ID}')"
