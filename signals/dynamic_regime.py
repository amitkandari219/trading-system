"""
Dynamic Regime Manager — VIX-driven regime transitions with hysteresis and safety.

Maps real-time VIX to regime labels (CALM/NORMAL/ELEVATED/HIGH_VOL/CRISIS),
with safety-first hysteresis: escalates quickly, de-escalates slowly.
Includes cooldown between transitions and session transition limits.

Usage:
    from signals.dynamic_regime import DynamicRegimeManager
    from signals.regime_filter import IntradayRegimeFilter

    rf = IntradayRegimeFilter()
    drm = DynamicRegimeManager(rf, telegram_alerter=alerter)

    result = drm.update_regime(vix=16.5, timestamp=now)
    # result = {regime, changed, old_regime, vix, size_factor, blocked_signals}

    if drm.should_emergency_exit(vix=22.0, vix_velocity=4.5):
        # VIX spiked >3 pts in <10 min — exit all positions
        pass
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from signals.regime_filter import (
    IntradayRegimeFilter, REGIMES, SIGNAL_REGIME_MATRIX, DEFAULT_REGIME_POLICY,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# VIX-TO-REGIME MAPPING
# ═══════════════════════════════════════════════════════════

# Ordered from most severe to least — classification checks top-down
VIX_REGIME_THRESHOLDS = [
    # (regime, vix_threshold) — VIX >= threshold maps to this regime
    ('CRISIS',   25.0),
    ('HIGH_VOL', 18.0),
    ('ELEVATED', 14.0),
    ('NORMAL',   10.0),
    ('CALM',      0.0),   # VIX < 10
]

# Severity ordering (higher = more dangerous)
REGIME_SEVERITY = {
    'CALM': 0,
    'NORMAL': 1,
    'ELEVATED': 2,
    'HIGH_VOL': 3,
    'CRISIS': 4,
}

# Hysteresis offsets (safety-first: escalate easily, de-escalate slowly)
ESCALATE_OFFSET = 0.5    # escalate at threshold + 0.5
DEESCALATE_OFFSET = 1.0  # de-escalate at threshold - 1.0

# Cooldown: minimum bars between regime changes (3 bars = 15 min)
MIN_BARS_BETWEEN_CHANGES = 3

# Session limit: max transitions per session
MAX_SESSION_TRANSITIONS = 5

# Emergency exit: VIX jump threshold
EMERGENCY_VIX_JUMP = 3.0        # pts
EMERGENCY_WINDOW_MINUTES = 10   # time window

# VIX velocity history for emergency detection
VIX_VELOCITY_HISTORY_LEN = 20


class DynamicRegimeManager:
    """
    Manages real-time regime based on VIX with hysteresis and safety controls.

    Safety mechanisms:
    - Hysteresis: escalate at threshold+0.5, de-escalate at threshold-1.0
    - Cooldown: minimum 3 bars (15 min) between changes
    - Max 5 transitions per session, then lock to most conservative seen
    - Emergency exit detection for VIX spikes
    """

    def __init__(self, regime_filter: IntradayRegimeFilter,
                 telegram_alerter=None):
        """
        Args:
            regime_filter: IntradayRegimeFilter instance to keep in sync
            telegram_alerter: TelegramAlerter for regime change notifications
        """
        self._regime_filter = regime_filter
        self._alerter = telegram_alerter

        # Current state
        self._current_regime: str = 'NORMAL'
        self._regime_set = False  # True after first update_regime call

        # Transition tracking
        self._transition_count: int = 0
        self._last_transition_bar: int = 0
        self._bar_counter: int = 0
        self._locked: bool = False
        self._locked_regime: Optional[str] = None

        # Session history
        self._transitions: List[Dict] = []
        self._regime_time: Dict[str, float] = {
            r: 0.0 for r in REGIME_SEVERITY
        }
        self._last_regime_timestamp: Optional[datetime] = None

        # VIX tracking
        self._vix_history: deque = deque(maxlen=VIX_VELOCITY_HISTORY_LEN)
        self._session_max_vix: Optional[float] = None
        self._session_min_vix: Optional[float] = None
        self._most_conservative_regime: str = 'NORMAL'

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════

    def update_regime(self, vix: float, timestamp: datetime) -> Dict:
        """
        Update regime based on current VIX value.

        Applies hysteresis, cooldown, and session transition limits.

        Args:
            vix: current India VIX value
            timestamp: current timestamp

        Returns:
            dict with:
                regime: str — current regime label
                changed: bool — whether regime changed
                old_regime: str — previous regime (or None)
                vix: float — input VIX value
                size_factor: float — position size multiplier
                blocked_signals: list — signals blocked in this regime
        """
        self._bar_counter += 1

        # Track VIX history
        self._vix_history.append((timestamp, vix))
        if self._session_max_vix is None or vix > self._session_max_vix:
            self._session_max_vix = vix
        if self._session_min_vix is None or vix < self._session_min_vix:
            self._session_min_vix = vix

        # Track time in current regime
        if self._last_regime_timestamp is not None:
            elapsed = (timestamp - self._last_regime_timestamp).total_seconds()
            self._regime_time[self._current_regime] = (
                self._regime_time.get(self._current_regime, 0.0) + elapsed
            )
        self._last_regime_timestamp = timestamp

        # Determine raw regime from VIX
        raw_regime = self._vix_to_regime(vix)

        # Apply hysteresis
        effective_regime = self._apply_hysteresis(vix, raw_regime)

        # Check if transition should happen
        old_regime = self._current_regime
        changed = False

        if effective_regime != self._current_regime:
            # Check cooldown
            bars_since_last = self._bar_counter - self._last_transition_bar
            if bars_since_last < MIN_BARS_BETWEEN_CHANGES and self._regime_set:
                logger.debug(
                    f"DynamicRegime: cooldown active "
                    f"({bars_since_last}/{MIN_BARS_BETWEEN_CHANGES} bars), "
                    f"staying in {self._current_regime}"
                )
                effective_regime = self._current_regime

            # Check session transition limit
            elif self._locked:
                logger.warning(
                    f"DynamicRegime: locked to {self._locked_regime} "
                    f"(max {MAX_SESSION_TRANSITIONS} transitions reached)"
                )
                effective_regime = self._locked_regime

            else:
                # Transition approved
                changed = True
                self._current_regime = effective_regime
                self._last_transition_bar = self._bar_counter
                self._transition_count += 1
                self._regime_set = True

                # Track most conservative regime seen
                if (REGIME_SEVERITY.get(effective_regime, 0)
                        > REGIME_SEVERITY.get(self._most_conservative_regime, 0)):
                    self._most_conservative_regime = effective_regime

                # Record transition
                self._transitions.append({
                    'timestamp': timestamp,
                    'from': old_regime,
                    'to': effective_regime,
                    'vix': vix,
                    'bar': self._bar_counter,
                })

                # Check if we hit max transitions
                if self._transition_count >= MAX_SESSION_TRANSITIONS:
                    self._locked = True
                    self._locked_regime = self._most_conservative_regime
                    self._current_regime = self._locked_regime
                    logger.warning(
                        f"DynamicRegime: max transitions "
                        f"({MAX_SESSION_TRANSITIONS}) reached. "
                        f"Locked to {self._locked_regime}"
                    )

                # Update the regime filter's live VIX
                self._regime_filter.update_live_vix(vix)

                # Send Telegram alert
                self._alert_transition(old_regime, effective_regime, vix, timestamp)

                logger.info(
                    f"DynamicRegime: {old_regime} -> {effective_regime} "
                    f"(VIX={vix:.2f}, transition #{self._transition_count})"
                )

        elif not self._regime_set:
            # First call — set initial regime without counting as transition
            self._current_regime = effective_regime
            self._regime_set = True
            self._regime_filter.update_live_vix(vix)
            changed = (effective_regime != old_regime)
            if changed:
                old_regime = 'NORMAL'  # was default
            logger.info(
                f"DynamicRegime: initial regime = {effective_regime} "
                f"(VIX={vix:.2f})"
            )

        # Build result
        regime = self._current_regime
        regime_config = REGIMES.get(regime, REGIMES['NORMAL'])
        size_factor = regime_config['size_factor']

        blocked = self._get_blocked_signals(regime)

        return {
            'regime': regime,
            'changed': changed,
            'old_regime': old_regime if changed else None,
            'vix': round(vix, 2),
            'size_factor': size_factor,
            'blocked_signals': blocked,
        }

    def get_regime(self) -> str:
        """Get current regime label."""
        return self._current_regime

    def get_session_summary(self) -> Dict:
        """
        Get session regime summary.

        Returns:
            dict with transitions, time_per_regime, max_vix, min_vix
        """
        return {
            'transitions': len(self._transitions),
            'transition_log': list(self._transitions),
            'time_per_regime': {
                r: round(secs / 60, 1)
                for r, secs in self._regime_time.items()
                if secs > 0
            },
            'max_vix': round(self._session_max_vix, 2) if self._session_max_vix else None,
            'min_vix': round(self._session_min_vix, 2) if self._session_min_vix else None,
            'current_regime': self._current_regime,
            'locked': self._locked,
        }

    def should_emergency_exit(self, vix: float,
                              vix_velocity: Optional[float] = None) -> bool:
        """
        Check if VIX spike warrants emergency exit.

        Emergency if VIX jumped >3 pts in <10 minutes.

        Args:
            vix: current VIX value
            vix_velocity: pre-computed VIX change rate (pts/10min).
                          If None, computed from history.

        Returns:
            True if emergency exit should be triggered
        """
        if vix_velocity is not None:
            if vix_velocity >= EMERGENCY_VIX_JUMP:
                logger.critical(
                    f"DynamicRegime: EMERGENCY — VIX velocity "
                    f"{vix_velocity:.1f} pts >= {EMERGENCY_VIX_JUMP} threshold"
                )
                return True

        # Compute from history
        if len(self._vix_history) >= 2:
            now_ts = self._vix_history[-1][0]
            window_start = now_ts - timedelta(minutes=EMERGENCY_WINDOW_MINUTES)

            # Find oldest VIX within the window
            for ts, hist_vix in self._vix_history:
                if ts >= window_start:
                    vix_jump = vix - hist_vix
                    if vix_jump >= EMERGENCY_VIX_JUMP:
                        logger.critical(
                            f"DynamicRegime: EMERGENCY — VIX jumped "
                            f"{vix_jump:.1f} pts in "
                            f"{(now_ts - ts).total_seconds() / 60:.0f} min "
                            f"({hist_vix:.1f} -> {vix:.1f})"
                        )
                        return True
                    break  # only need the oldest in window

        return False

    # ══════════════════════════════════════════════════════════
    # INTERNAL
    # ══════════════════════════════════════════════════════════

    def _vix_to_regime(self, vix: float) -> str:
        """Map raw VIX value to regime label (no hysteresis)."""
        for regime, threshold in VIX_REGIME_THRESHOLDS:
            if vix >= threshold:
                return regime
        return 'CALM'

    def _apply_hysteresis(self, vix: float, raw_regime: str) -> str:
        """
        Apply hysteresis to prevent oscillation at boundaries.

        Safety-first:
        - Escalate (more dangerous): requires VIX >= threshold + 0.5
          (but we accept it since raw_regime already crossed the threshold)
        - De-escalate (less dangerous): requires VIX < threshold - 1.0
        """
        current_severity = REGIME_SEVERITY.get(self._current_regime, 1)
        raw_severity = REGIME_SEVERITY.get(raw_regime, 1)

        if raw_severity > current_severity:
            # Escalating — check if VIX crossed threshold + offset
            target_threshold = self._get_threshold_for_regime(raw_regime)
            if target_threshold is not None:
                if vix >= target_threshold + ESCALATE_OFFSET:
                    return raw_regime
                else:
                    # Not enough above threshold — stay
                    return self._current_regime
            return raw_regime

        elif raw_severity < current_severity:
            # De-escalating — check if VIX dropped below threshold - offset
            current_threshold = self._get_threshold_for_regime(self._current_regime)
            if current_threshold is not None:
                if vix < current_threshold - DEESCALATE_OFFSET:
                    return raw_regime
                else:
                    # Not far enough below threshold — stay conservative
                    return self._current_regime
            return raw_regime

        # Same severity — no change
        return raw_regime

    def _get_threshold_for_regime(self, regime: str) -> Optional[float]:
        """Get the VIX threshold for a given regime."""
        for r, threshold in VIX_REGIME_THRESHOLDS:
            if r == regime:
                return threshold
        return None

    def _get_blocked_signals(self, regime: str) -> List[str]:
        """Get list of signal names blocked in the given regime."""
        blocked = []
        for signal_name, policy in SIGNAL_REGIME_MATRIX.items():
            if not policy.get(regime, DEFAULT_REGIME_POLICY.get(regime, True)):
                blocked.append(signal_name)
        return blocked

    def _alert_transition(self, old_regime: str, new_regime: str,
                          vix: float, timestamp: datetime):
        """Send Telegram alert on regime transition."""
        if self._alerter is None:
            return

        old_severity = REGIME_SEVERITY.get(old_regime, 1)
        new_severity = REGIME_SEVERITY.get(new_regime, 1)

        # Emoji based on direction and severity
        if new_severity > old_severity:
            direction = "ESCALATED"
            if new_regime == 'CRISIS':
                emoji_prefix = "\U0001f6a8"  # siren
            elif new_regime == 'HIGH_VOL':
                emoji_prefix = "\U0001f534"  # red circle
            else:
                emoji_prefix = "\u26a0\ufe0f"  # warning
            level = 'WARNING' if new_severity <= 2 else 'CRITICAL'
        else:
            direction = "DE-ESCALATED"
            if new_regime in ('CALM', 'NORMAL'):
                emoji_prefix = "\u2705"  # green check
            else:
                emoji_prefix = "\u2139\ufe0f"  # info
            level = 'INFO'

        blocked = self._get_blocked_signals(new_regime)
        size_factor = REGIMES.get(new_regime, {}).get('size_factor', 1.0)

        message = (
            f"{emoji_prefix} Regime {direction}\n"
            f"{old_regime} -> {new_regime}\n"
            f"VIX: {vix:.2f}\n"
            f"Size factor: {size_factor:.0%}\n"
            f"Transition #{self._transition_count}/{MAX_SESSION_TRANSITIONS}"
        )
        if blocked:
            message += f"\nBlocked: {', '.join(blocked)}"
        if self._locked:
            message += f"\nLOCKED to {self._locked_regime} (max transitions)"

        try:
            self._alerter.send(
                level, message, signal_id='REGIME_CHANGE'
            )
        except Exception as e:
            logger.debug(f"DynamicRegime: Telegram alert failed: {e}")
