"""
Scoring engine that combines 3 Kaufman signals into a weighted score
for position sizing decisions.

FIX 6: Added drawdown-based dynamic scaling. When the portfolio is in
drawdown, position sizes are reduced progressively:
  drawdown < 5%  -> 1.0x (no reduction)
  drawdown 5-10% -> 0.75x
  drawdown 10-15% -> 0.50x
  drawdown > 15% -> 0.25x
This prevents aggressive sizing during losing streaks.
"""

from __future__ import annotations


# Drawdown tiers: (max_drawdown_pct, size_multiplier)
DRAWDOWN_TIERS = [
    (5.0,  1.00),   # < 5% drawdown: full size
    (10.0, 0.75),   # 5-10% drawdown: 75% size
    (15.0, 0.50),   # 10-15% drawdown: 50% size
    (100.0, 0.25),  # > 15% drawdown: 25% size (floor)
]


def drawdown_scale(current_drawdown_pct: float) -> float:
    """
    Compute position size multiplier based on current portfolio drawdown.

    Args:
        current_drawdown_pct: Current drawdown as positive percentage (e.g. 8.5 for -8.5%)

    Returns:
        Multiplier between 0.25 and 1.0
    """
    dd = abs(current_drawdown_pct)
    for threshold, mult in DRAWDOWN_TIERS:
        if dd < threshold:
            return mult
    return DRAWDOWN_TIERS[-1][1]  # Floor


class ScoringEngine:
    """
    Combines 3 Kaufman signals into a weighted score for position sizing.

    Score rules:
      DRY_20 fires LONG  -> score += 2
      DRY_20 fires exit  -> score resets to 0
      DRY_12 fires LONG  -> score += 1
      DRY_12 fires SHORT -> score -= 1
      DRY_16 fires LONG  -> score += 1
      DRY_16 fires SHORT -> score -= 1

    Trading rules:
      score >= 3  -> LONG full size (1.0x)
      score >= 2  -> LONG half size (0.5x)
      score <= -2 -> SHORT half size (0.5x)
      |score| <= 1 -> no new position

    Exit: when score drops below entry threshold

    Drawdown scaling: sizes are multiplied by drawdown_scale() when
    current_drawdown_pct is provided to update().
    """

    def __init__(self) -> None:
        self.score: int = 0
        self.position: str | None = None  # 'LONG', 'SHORT', or None
        self.entry_threshold: int | None = None
        self.size: float = 0.0
        self.drawdown_mult: float = 1.0  # Latest drawdown multiplier

    def update(self, signals: dict, current_drawdown_pct: float = 0.0) -> dict:
        """
        Process today's signals and return a trading decision.

        Parameters
        ----------
        signals : dict
            {
              'DRY_20': {'action': 'ENTER_LONG'|'EXIT'|None},
              'DRY_12': {'action': 'ENTER_LONG'|'ENTER_SHORT'|'EXIT'|None},
              'DRY_16': {'action': 'ENTER_LONG'|'ENTER_SHORT'|'EXIT'|None},
            }
        current_drawdown_pct : float
            Current portfolio drawdown as positive percentage (e.g. 8.5 for -8.5%).
            Used for dynamic position scaling.

        Returns
        -------
        dict
            {
              'score': int,
              'prev_score': int,
              'action': 'ENTER_LONG'|'ENTER_SHORT'|'EXIT'|None,
              'size': float,
              'drawdown_mult': float,
              'reason': str,
            }
        """
        # Compute drawdown-based scaling
        self.drawdown_mult = drawdown_scale(current_drawdown_pct)
        prev_score = self.score

        # --- Compute fresh score for today ---
        new_score = 0
        reason_parts: list[str] = []

        dry20 = signals.get("DRY_20", {})
        dry12 = signals.get("DRY_12", {})
        dry16 = signals.get("DRY_16", {})

        dry20_action = dry20.get("action") if dry20 else None
        dry12_action = dry12.get("action") if dry12 else None
        dry16_action = dry16.get("action") if dry16 else None

        # DRY_20 EXIT resets score to 0 regardless of other signals
        if dry20_action == "EXIT":
            new_score = 0
            reason_parts.append("DRY_20(EXIT->0)")
            self.score = new_score
            action, size, reason_suffix = self._decide(prev_score, new_score, reason_parts)
            return self._build_result(prev_score, new_score, action, size, reason_suffix)

        # DRY_20 contribution
        if dry20_action == "ENTER_LONG":
            new_score += 2
            reason_parts.append("DRY_20(+2)")

        # DRY_12 contribution
        if dry12_action == "ENTER_LONG":
            new_score += 1
            reason_parts.append("DRY_12(+1)")
        elif dry12_action == "ENTER_SHORT":
            new_score -= 1
            reason_parts.append("DRY_12(-1)")

        # DRY_16 contribution
        if dry16_action == "ENTER_LONG":
            new_score += 1
            reason_parts.append("DRY_16(+1)")
        elif dry16_action == "ENTER_SHORT":
            new_score -= 1
            reason_parts.append("DRY_16(-1)")

        self.score = new_score
        action, size, reason_suffix = self._decide(prev_score, new_score, reason_parts)
        return self._build_result(prev_score, new_score, action, size, reason_suffix)

    def _decide(
        self, prev_score: int, new_score: int, reason_parts: list[str]
    ) -> tuple[str | None, float, str]:
        """Determine action, size, and build reason string."""
        score_expr = " ".join(reason_parts) if reason_parts else "no signals"
        action: str | None = None
        size: float = 0.0

        # --- Check for EXIT from existing position ---
        if self.position is not None and self.entry_threshold is not None:
            should_exit = False
            if self.position == "LONG" and new_score < self.entry_threshold:
                should_exit = True
            elif self.position == "SHORT" and new_score > -self.entry_threshold:
                # entry_threshold is stored as positive; SHORT entered at <= -threshold
                should_exit = True

            if should_exit:
                action = "EXIT"
                size = 0.0
                reason = f"{score_expr} = {new_score} -> EXIT (below threshold {self.entry_threshold})"
                self.position = None
                self.entry_threshold = None
                self.size = 0.0
                return action, size, reason

        # --- Check for new entries ---
        if new_score >= 3:
            desired_action = "ENTER_LONG"
            desired_size = 1.0
            desired_threshold = 3
            label = "LONG full"
        elif new_score >= 2:
            desired_action = "ENTER_LONG"
            desired_size = 0.5
            desired_threshold = 2
            label = "LONG half"
        elif new_score <= -2:
            desired_action = "ENTER_SHORT"
            desired_size = 0.5
            desired_threshold = 2  # stored as positive magnitude
            label = "SHORT half"
        else:
            # No action zone
            reason = f"{score_expr} = {new_score} -> no action"
            return None, 0.0, reason

        # Position conflict: no new LONG if already SHORT and vice versa
        if self.position == "LONG" and desired_action == "ENTER_SHORT":
            reason = f"{score_expr} = {new_score} -> no action (conflict: already LONG)"
            return None, 0.0, reason
        if self.position == "SHORT" and desired_action == "ENTER_LONG":
            reason = f"{score_expr} = {new_score} -> no action (conflict: already SHORT)"
            return None, 0.0, reason

        # If already in the same direction, keep position (no new entry)
        if self.position == "LONG" and desired_action == "ENTER_LONG":
            # Update size if threshold changed
            self.size = desired_size
            if desired_threshold > (self.entry_threshold or 0):
                self.entry_threshold = desired_threshold
            reason = f"{score_expr} = {new_score} -> hold {label}"
            return None, desired_size, reason
        if self.position == "SHORT" and desired_action == "ENTER_SHORT":
            self.size = desired_size
            reason = f"{score_expr} = {new_score} -> hold {label}"
            return None, desired_size, reason

        # New entry
        action = desired_action
        size = desired_size
        self.position = "LONG" if desired_action == "ENTER_LONG" else "SHORT"
        self.entry_threshold = desired_threshold
        self.size = size
        reason = f"{score_expr} = {new_score} -> {label}"
        return action, size, reason

    def _build_result(
        self,
        prev_score: int,
        new_score: int,
        action: str | None,
        size: float,
        reason: str,
    ) -> dict:
        # Apply drawdown scaling to the raw size
        scaled_size = round(size * self.drawdown_mult, 3)
        dd_note = ""
        if self.drawdown_mult < 1.0:
            dd_note = f" [DD scale: {self.drawdown_mult:.2f}x]"
        return {
            "score": new_score,
            "prev_score": prev_score,
            "action": action,
            "size": scaled_size,
            "raw_size": size,
            "drawdown_mult": self.drawdown_mult,
            "reason": reason + dd_note,
        }

    def get_daily_summary(self) -> str:
        """Return a one-line string suitable for Telegram notification."""
        pos_str = self.position if self.position else "FLAT"
        size_str = f"{self.size:.1f}x" if self.size > 0 else "-"
        return f"Score={self.score} | Pos={pos_str} | Size={size_str}"
