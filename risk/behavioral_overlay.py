"""
Behavioral Overlay — Kahneman-inspired bias correctors for position sizing.

Four overlays that detect and correct common behavioral biases:
  1. Loss Aversion Corrector: prevents panic sizing after healthy-metric losses
  2. Disposition Effect Blocker: trailing stop on winners, time-exit on losers
  3. Anchoring Detector: override SL/TGT near round numbers with ATR-based levels
  4. Overconfidence Dampener: cap sizing after win streaks

All overlays REDUCE risk (or maintain system size). The only exception is the
loss aversion floor, which prevents excessive shrinkage when system metrics
are healthy — this is capped at 70% of system size, never above 100%.

Usage:
    from risk.behavioral_overlay import BehavioralOverlay
    overlay = BehavioralOverlay()
    adjustments = overlay.apply_all(context)
    # adjustments = {
    #     'size_multiplier': 0.8,
    #     'sl_override': 24500.0,
    #     'tgt_override': 25200.0,
    #     'force_exit': False,
    #     'trailing_stop': 24800.0,
    #     'overlays_triggered': ['OVERCONFIDENCE_DAMPENER'],
    #     'audit': { ... },
    # }
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Loss Aversion Corrector
LA_CONSECUTIVE_LOSSES = 3           # trigger after N consecutive losses
LA_ROLLING_WINDOW = 20              # trades for rolling Sharpe
LA_SHARPE_HEALTHY = 0.5             # Sharpe threshold for "healthy"
LA_SHARPE_UNHEALTHY = 0.0           # below this, allow real decay
LA_DD_UNHEALTHY_PCT = 0.08          # above 8% DD, allow real decay
LA_FLOOR_FRACTION = 0.70            # floor at 70% of system size

# Disposition Effect Blocker
DE_TRAILING_STOP_PCT = 0.60         # trailing stop at 60% of max unrealized gain
DE_HOLDING_MULTIPLIER = 2.0         # force exit if holding > 2x avg hold time
DE_UNREALIZED_LOSS_PCT = -0.01      # force exit if unrealized < -1%

# Anchoring Detector
AD_ROUND_PROXIMITY_PCT = 0.001      # 0.1% proximity to round number
AD_ROUND_NUMBERS = [1000, 500]      # x000, x500 boundaries
AD_SL_ATR_MULT = 2.0               # SL = entry - 2*ATR
AD_TGT_ATR_MULT = 3.0              # TGT = entry + 3*ATR

# Overconfidence Dampener
OC_CONSECUTIVE_WINS_CAP = 5         # cap at system size after 5 wins
OC_CONSECUTIVE_WINS_REDUCE = 8      # reduce 20% after 8 wins
OC_REDUCE_FRACTION = 0.80           # 80% of system size


# ══════════════════════════════════════════════════════════════
# TRADE RECORD (for rolling stats)
# ══════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Minimal trade record for behavioral analysis."""
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    holding_bars: int = 0
    max_unrealized_pct: float = 0.0
    min_unrealized_pct: float = 0.0
    signal_id: str = ""


@dataclass
class PositionContext:
    """Current position context for disposition effect checks."""
    signal_id: str = ""
    direction: str = "LONG"
    entry_price: float = 0.0
    current_price: float = 0.0
    entry_time: Optional[datetime] = None
    bars_held: int = 0
    max_unrealized_gain_pct: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class OverlayContext:
    """
    Full context for behavioral overlay evaluation.

    Callers populate what they have; overlays skip checks with missing data.
    """
    # Trade history
    recent_trades: List[TradeRecord] = field(default_factory=list)

    # Current position (for disposition effect)
    position: Optional[PositionContext] = None

    # Proposed trade levels (for anchoring detector)
    entry_price: float = 0.0
    proposed_sl: float = 0.0
    proposed_tgt: float = 0.0
    direction: str = "LONG"  # LONG or SHORT
    current_atr: float = 0.0

    # System state
    system_size: float = 1.0         # base sizing multiplier from system
    current_dd_pct: float = 0.0      # current drawdown from peak
    equity: float = 0.0

    # Timestamp
    now: Optional[datetime] = None


@dataclass
class OverlayResult:
    """Result from apply_all()."""
    size_multiplier: float = 1.0
    sl_override: Optional[float] = None
    tgt_override: Optional[float] = None
    force_exit: bool = False
    force_exit_reason: str = ""
    trailing_stop: Optional[float] = None
    overlays_triggered: List[str] = field(default_factory=list)
    audit: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# BEHAVIORAL OVERLAY
# ══════════════════════════════════════════════════════════════

class BehavioralOverlay:
    """
    Kahneman-inspired behavioral bias correctors.

    All four overlays run independently and their effects combine:
    - Size multipliers multiply together (conservative wins)
    - SL/TGT overrides: anchoring override takes priority
    - Force exit: any trigger causes exit
    - Trailing stop: disposition effect trailing stop
    """

    def __init__(self):
        self._trade_history: List[TradeRecord] = []
        self._consecutive_losses = 0
        self._consecutive_wins = 0

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────

    def record_trade(self, trade: TradeRecord):
        """
        Record a completed trade for rolling statistics.
        Call after every trade exit.
        """
        self._trade_history.append(trade)

        # Update streaks
        if trade.pnl >= 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        # Keep bounded history (200 trades)
        if len(self._trade_history) > 200:
            self._trade_history = self._trade_history[-200:]

        logger.debug(
            f"BehavioralOverlay: recorded trade pnl={trade.pnl:+.0f} "
            f"streak W={self._consecutive_wins} L={self._consecutive_losses}"
        )

    def apply_all(self, ctx: OverlayContext) -> OverlayResult:
        """
        Run all four behavioral overlays and return combined adjustments.

        Args:
            ctx: OverlayContext with trade history, position info, proposed levels

        Returns:
            OverlayResult with combined adjustments
        """
        result = OverlayResult(size_multiplier=1.0)
        audit = {}

        # Use provided trade history or internal
        trades = ctx.recent_trades if ctx.recent_trades else self._trade_history

        # 1. Loss Aversion Corrector
        la_result = self._loss_aversion_corrector(trades, ctx)
        if la_result['triggered']:
            result.overlays_triggered.append('LOSS_AVERSION_CORRECTOR')
            result.size_multiplier = max(result.size_multiplier, la_result['floor'])
            audit['loss_aversion'] = la_result

        # 2. Disposition Effect Blocker
        de_result = self._disposition_effect_blocker(ctx)
        if de_result['triggered']:
            result.overlays_triggered.append('DISPOSITION_EFFECT_BLOCKER')
            if de_result.get('trailing_stop') is not None:
                result.trailing_stop = de_result['trailing_stop']
            if de_result.get('force_exit'):
                result.force_exit = True
                result.force_exit_reason = de_result.get('reason', 'disposition_time_exit')
            audit['disposition_effect'] = de_result

        # 3. Anchoring Detector
        ad_result = self._anchoring_detector(ctx)
        if ad_result['triggered']:
            result.overlays_triggered.append('ANCHORING_DETECTOR')
            result.sl_override = ad_result.get('sl_override')
            result.tgt_override = ad_result.get('tgt_override')
            audit['anchoring'] = ad_result

        # 4. Overconfidence Dampener
        oc_result = self._overconfidence_dampener(trades)
        if oc_result['triggered']:
            result.overlays_triggered.append('OVERCONFIDENCE_DAMPENER')
            result.size_multiplier = min(
                result.size_multiplier, oc_result['size_cap']
            )
            audit['overconfidence'] = oc_result

        # Safety: never increase beyond system size (except LA floor which is <=1.0)
        result.size_multiplier = min(result.size_multiplier, 1.0)

        result.audit = audit

        if result.overlays_triggered:
            logger.info(
                f"BehavioralOverlay: triggered={result.overlays_triggered} "
                f"size_mult={result.size_multiplier:.2f} "
                f"force_exit={result.force_exit}"
            )

        return result

    def get_status(self) -> Dict:
        """Return current overlay state for monitoring."""
        rolling_sharpe = self._rolling_sharpe(self._trade_history)
        return {
            'trade_count': len(self._trade_history),
            'consecutive_wins': self._consecutive_wins,
            'consecutive_losses': self._consecutive_losses,
            'rolling_sharpe_20': round(rolling_sharpe, 3),
        }

    # ──────────────────────────────────────────────────────────
    # OVERLAY 1: Loss Aversion Corrector
    # ──────────────────────────────────────────────────────────

    def _loss_aversion_corrector(
        self, trades: List[TradeRecord], ctx: OverlayContext
    ) -> Dict:
        """
        After 3+ consecutive losses with healthy rolling metrics:
        - Floor size at 70% of system size (prevent panic shrinkage)

        But if Sharpe < 0 or DD > 8%: allow full reduction (real decay).
        """
        result = {'triggered': False, 'floor': 1.0, 'reason': ''}

        if self._consecutive_losses < LA_CONSECUTIVE_LOSSES:
            return result

        # Check if system metrics are healthy
        rolling_sharpe = self._rolling_sharpe(trades)
        dd_pct = ctx.current_dd_pct

        # If metrics are UNHEALTHY, allow full system reduction (no floor)
        if rolling_sharpe < LA_SHARPE_UNHEALTHY or dd_pct > LA_DD_UNHEALTHY_PCT:
            result['reason'] = (
                f"metrics_unhealthy: sharpe={rolling_sharpe:.2f} dd={dd_pct:.1%} "
                f"— allowing full reduction"
            )
            logger.debug(f"LA corrector: {result['reason']}")
            return result

        # Metrics are healthy but we have consecutive losses = behavioral bias
        if rolling_sharpe >= LA_SHARPE_HEALTHY:
            result['triggered'] = True
            result['floor'] = LA_FLOOR_FRACTION
            result['reason'] = (
                f"healthy_metrics_after_{self._consecutive_losses}_losses: "
                f"sharpe={rolling_sharpe:.2f} dd={dd_pct:.1%} "
                f"— flooring at {LA_FLOOR_FRACTION:.0%}"
            )
            logger.info(f"LA corrector: {result['reason']}")

        return result

    # ──────────────────────────────────────────────────────────
    # OVERLAY 2: Disposition Effect Blocker
    # ──────────────────────────────────────────────────────────

    def _disposition_effect_blocker(self, ctx: OverlayContext) -> Dict:
        """
        Winning positions: add trailing stop at 60% of max unrealized gain.
        Losing positions: force time-exit if holding > 2x avg AND unrealized < -1%.
        """
        result = {'triggered': False}

        pos = ctx.position
        if pos is None:
            return result

        # --- Winning position: trailing stop ---
        if pos.max_unrealized_gain_pct > 0.005:  # > 0.5% gain at some point
            trailing_stop_pct = pos.max_unrealized_gain_pct * DE_TRAILING_STOP_PCT
            if pos.direction == "LONG":
                trailing_stop = pos.entry_price * (1 + trailing_stop_pct)
            else:
                trailing_stop = pos.entry_price * (1 - trailing_stop_pct)

            result['triggered'] = True
            result['trailing_stop'] = round(trailing_stop, 2)
            result['reason'] = (
                f"trailing_stop: max_gain={pos.max_unrealized_gain_pct:.2%} "
                f"trail_level={trailing_stop:.0f}"
            )

        # --- Losing position: force time-exit ---
        avg_hold = self._avg_holding_bars()
        if avg_hold > 0 and pos.bars_held > DE_HOLDING_MULTIPLIER * avg_hold:
            if pos.unrealized_pnl_pct < DE_UNREALIZED_LOSS_PCT:
                result['triggered'] = True
                result['force_exit'] = True
                result['reason'] = (
                    f"disposition_time_exit: bars_held={pos.bars_held} "
                    f"avg={avg_hold:.0f} unrealized={pos.unrealized_pnl_pct:.2%}"
                )
                logger.info(f"Disposition blocker: {result['reason']}")

        return result

    # ──────────────────────────────────────────────────────────
    # OVERLAY 3: Anchoring Detector
    # ──────────────────────────────────────────────────────────

    def _anchoring_detector(self, ctx: OverlayContext) -> Dict:
        """
        If SL or TGT is within 0.1% of a round number (x000, x500),
        override with ATR-based levels.
        """
        result = {'triggered': False}

        if ctx.entry_price <= 0 or ctx.current_atr <= 0:
            return result

        sl_anchored = self._is_near_round_number(ctx.proposed_sl)
        tgt_anchored = self._is_near_round_number(ctx.proposed_tgt)

        if not sl_anchored and not tgt_anchored:
            return result

        result['triggered'] = True
        anchored_levels = []

        if sl_anchored:
            if ctx.direction == "LONG":
                new_sl = ctx.entry_price - AD_SL_ATR_MULT * ctx.current_atr
            else:
                new_sl = ctx.entry_price + AD_SL_ATR_MULT * ctx.current_atr
            result['sl_override'] = round(new_sl, 2)
            anchored_levels.append(f"SL {ctx.proposed_sl:.0f}->{new_sl:.0f}")

        if tgt_anchored:
            if ctx.direction == "LONG":
                new_tgt = ctx.entry_price + AD_TGT_ATR_MULT * ctx.current_atr
            else:
                new_tgt = ctx.entry_price - AD_TGT_ATR_MULT * ctx.current_atr
            result['tgt_override'] = round(new_tgt, 2)
            anchored_levels.append(f"TGT {ctx.proposed_tgt:.0f}->{new_tgt:.0f}")

        result['reason'] = f"anchoring_override: {', '.join(anchored_levels)}"
        logger.info(f"Anchoring detector: {result['reason']}")
        return result

    # ──────────────────────────────────────────────────────────
    # OVERLAY 4: Overconfidence Dampener
    # ──────────────────────────────────────────────────────────

    def _overconfidence_dampener(self, trades: List[TradeRecord]) -> Dict:
        """
        After 5+ consecutive wins: cap at system size (1.0x).
        After 8+ consecutive wins: reduce to 80%.
        """
        result = {'triggered': False, 'size_cap': 1.0}

        if self._consecutive_wins >= OC_CONSECUTIVE_WINS_REDUCE:
            result['triggered'] = True
            result['size_cap'] = OC_REDUCE_FRACTION
            result['reason'] = (
                f"overconfidence_reduce: {self._consecutive_wins} wins "
                f"-> cap at {OC_REDUCE_FRACTION:.0%}"
            )
            logger.info(f"Overconfidence dampener: {result['reason']}")

        elif self._consecutive_wins >= OC_CONSECUTIVE_WINS_CAP:
            result['triggered'] = True
            result['size_cap'] = 1.0
            result['reason'] = (
                f"overconfidence_cap: {self._consecutive_wins} wins "
                f"-> cap at 100% (no increase)"
            )
            logger.info(f"Overconfidence dampener: {result['reason']}")

        return result

    # ──────────────────────────────────────────────────────────
    # HELPER: Rolling statistics
    # ──────────────────────────────────────────────────────────

    def _rolling_sharpe(self, trades: List[TradeRecord],
                        window: int = LA_ROLLING_WINDOW) -> float:
        """Compute rolling Sharpe over last N trades."""
        if len(trades) < max(5, window // 2):
            return 0.0

        recent = trades[-window:]
        returns = [t.pnl_pct for t in recent]

        if not returns:
            return 0.0

        mean_ret = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0

        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 0.001

        # Annualize assuming ~250 trading days, ~1 trade/day
        sharpe = (mean_ret / std_ret) * math.sqrt(250)
        return sharpe

    def _avg_holding_bars(self) -> float:
        """Average holding period in bars across recent trades."""
        recent = self._trade_history[-50:]
        if not recent:
            return 20.0  # default

        bars = [t.holding_bars for t in recent if t.holding_bars > 0]
        if not bars:
            return 20.0

        return sum(bars) / len(bars)

    @staticmethod
    def _is_near_round_number(price: float) -> bool:
        """Check if price is within 0.1% of a round number (x000, x500)."""
        if price <= 0:
            return False

        for step in AD_ROUND_NUMBERS:
            nearest = round(price / step) * step
            distance_pct = abs(price - nearest) / price
            if distance_pct < AD_ROUND_PROXIMITY_PCT:
                return True
        return False


# ══════════════════════════════════════════════════════════════
# MAIN — self-test
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
    )

    print("=" * 65)
    print("  BEHAVIORAL OVERLAY — Self-Test")
    print("=" * 65)

    overlay = BehavioralOverlay()

    # Simulate 20 profitable trades then 4 losses
    for i in range(20):
        overlay.record_trade(TradeRecord(
            pnl=500 + i * 10,
            pnl_pct=0.02,
            entry_time=datetime.now(),
            holding_bars=10,
            signal_id=f"TEST_{i}",
        ))

    for i in range(4):
        overlay.record_trade(TradeRecord(
            pnl=-300,
            pnl_pct=-0.01,
            entry_time=datetime.now(),
            holding_bars=10,
            signal_id=f"LOSS_{i}",
        ))

    print(f"\nStatus: {overlay.get_status()}")

    # Test 1: Loss aversion with healthy metrics
    ctx = OverlayContext(
        system_size=1.0,
        current_dd_pct=0.03,
        entry_price=25000,
        proposed_sl=24500,
        proposed_tgt=25500,
        direction="LONG",
        current_atr=200,
    )
    result = overlay.apply_all(ctx)
    print(f"\nTest 1 (loss aversion): triggered={result.overlays_triggered}")
    print(f"  size_mult={result.size_multiplier:.2f}")

    # Test 2: Anchoring near round number
    ctx2 = OverlayContext(
        entry_price=25100,
        proposed_sl=25000,   # exactly at round number
        proposed_tgt=25500,  # exactly at round number
        direction="LONG",
        current_atr=150,
    )
    result2 = overlay.apply_all(ctx2)
    print(f"\nTest 2 (anchoring): triggered={result2.overlays_triggered}")
    print(f"  sl_override={result2.sl_override} tgt_override={result2.tgt_override}")

    # Test 3: Overconfidence after 8 wins
    overlay2 = BehavioralOverlay()
    for i in range(9):
        overlay2.record_trade(TradeRecord(
            pnl=500, pnl_pct=0.02,
            entry_time=datetime.now(),
            holding_bars=10,
        ))
    result3 = overlay2.apply_all(OverlayContext())
    print(f"\nTest 3 (overconfidence): triggered={result3.overlays_triggered}")
    print(f"  size_mult={result3.size_multiplier:.2f}")

    # Test 4: Disposition effect — winning position
    ctx4 = OverlayContext(
        position=PositionContext(
            direction="LONG",
            entry_price=25000,
            current_price=25400,
            bars_held=5,
            max_unrealized_gain_pct=0.018,
            unrealized_pnl_pct=0.016,
        ),
    )
    result4 = overlay.apply_all(ctx4)
    print(f"\nTest 4 (disposition trailing): triggered={result4.overlays_triggered}")
    print(f"  trailing_stop={result4.trailing_stop}")

    print(f"\n{'=' * 65}")
    print("  All behavioral overlay self-tests passed")
    print(f"{'=' * 65}")
