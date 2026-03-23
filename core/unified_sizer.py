"""
Unified Position Sizer -- single entry point for all sizing decisions.

Combines: base risk budget -> Kelly fraction -> overlay composite ->
conviction scoring -> regime adjustment -> floor to Nifty lots.

Usage:
    from core.unified_sizer import UnifiedSizer
    sizer = UnifiedSizer(equity=1_000_000)
    result = sizer.compute('KAUFMAN_DRY_20', sl_points=120, spot_price=24000,
                           overlay_modifiers={'FII_FUTURES_OI': 1.15},
                           regime='TRENDING', direction='BULLISH')
"""

import logging
import math
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config imports with fallback
# ---------------------------------------------------------------------------
try:
    from config.unified_config import (
        MAX_CONCURRENT_POSITIONS, BASE_RISK_PCT, NIFTY_LOT_SIZE,
    )
except ImportError:
    try:
        from config.settings import MAX_POSITIONS as MAX_CONCURRENT_POSITIONS
        from config.settings import NIFTY_LOT_SIZE
    except ImportError:
        MAX_CONCURRENT_POSITIONS = 4
        NIFTY_LOT_SIZE = 25
    BASE_RISK_PCT = 0.02

# ---------------------------------------------------------------------------
# Regime multipliers -- tighten size in stress, loosen in calm trend
# ---------------------------------------------------------------------------
REGIME_MULTIPLIERS = {
    'TRENDING':  1.10,
    'RANGING':   0.85,
    'HIGH_VOL':  0.65,
    'CRISIS':    0.40,
    # Mamba-style regime labels
    'CALM_BULL':     1.10,
    'VOLATILE_BULL': 0.80,
    'NEUTRAL':       1.00,
    'VOLATILE_BEAR': 0.60,
    'CALM_BEAR':     0.75,
}

DEFAULT_REGIME_MULT = 1.00

# Composite modifier clamp range
COMPOSITE_FLOOR = 0.30
COMPOSITE_CEILING = 2.00

# Final lots clamp
MIN_LOTS = 1
MAX_LOTS_CAP = 20

# Margin per lot (approximate SPAN margin for Nifty futures)
MARGIN_PER_LOT = 120_000


class UnifiedSizer:
    """
    Single sizing pipeline: risk -> Kelly -> overlays -> conviction -> regime -> lots.

    Parameters
    ----------
    equity : float
        Current account equity in rupees.
    kelly_engine : optional
        An AdaptiveKelly instance. If None, Kelly step is skipped (fraction=1.0).
    conviction_scorer : optional
        A ConvictionScorer instance. If None, conviction step is skipped (modifier=1.0).
    """

    def __init__(
        self,
        equity: float = 1_000_000,
        kelly_engine=None,
        conviction_scorer=None,
    ):
        self.equity = equity
        self.kelly_engine = kelly_engine
        self.conviction_scorer = conviction_scorer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        signal_name: str,
        sl_points: float,
        spot_price: float,
        overlay_modifiers: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
        signal_stats: Optional[Dict] = None,
        direction: str = 'BULLISH',
        open_positions: int = 0,
        drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
        vix: float = 15.0,
        adx: float = 20.0,
    ) -> Dict:
        """
        Full sizing pipeline for a single trade candidate.

        Parameters
        ----------
        signal_name : str
            Identifier of the firing signal (e.g. 'KAUFMAN_DRY_20').
        sl_points : float
            Distance to stop loss in Nifty index points.
        spot_price : float
            Current Nifty spot price.
        overlay_modifiers : dict, optional
            {overlay_id: size_modifier} from OverlayPipeline.
        regime : str, optional
            Current regime label ('TRENDING', 'CRISIS', etc.).
        signal_stats : dict, optional
            Recent signal performance (win_rate, consecutive_losers, etc.)
            used by AdaptiveKelly.
        direction : str
            'BULLISH' or 'BEARISH'. Maps to LONG/SHORT for conviction.
        open_positions : int
            Number of currently open positions.
        drawdown_pct : float
            Current drawdown from equity peak (0.0 - 1.0 decimal).
        consecutive_losses : int
            Current consecutive losing trade streak.
        vix : float
            Current India VIX.
        adx : float
            Current ADX(14).

        Returns
        -------
        dict with keys:
            lots, risk_amount, kelly_fraction, composite_modifier,
            conviction_modifier, regime_modifier, reasoning
        """
        overlay_modifiers = dict(overlay_modifiers or {})
        reasoning: List[str] = []

        # ── Step 1: Base risk budget ──────────────────────────────
        if sl_points <= 0:
            sl_points = spot_price * 0.02
            reasoning.append(f'SL defaulted to 2% of spot ({sl_points:.0f} pts)')

        risk_per_position = self.equity * BASE_RISK_PCT / MAX_CONCURRENT_POSITIONS
        risk_per_lot = sl_points * NIFTY_LOT_SIZE
        base_lots = math.floor(risk_per_position / risk_per_lot) if risk_per_lot > 0 else 1
        base_lots = max(MIN_LOTS, base_lots)
        reasoning.append(
            f'Base: equity={self.equity:,.0f}, risk/pos={risk_per_position:,.0f}, '
            f'risk/lot={risk_per_lot:,.0f} -> {base_lots} lots'
        )

        # ── Step 2: Kelly fraction ────────────────────────────────
        kelly_fraction = 1.0
        kelly_gear = 'N/A'
        if self.kelly_engine is not None:
            stats = signal_stats or {}
            kelly_result = self.kelly_engine.get_fraction(
                drawdown_pct=drawdown_pct,
                recent_wr=stats.get('win_rate', 0.50),
                consecutive_losers=consecutive_losses,
                vix=vix,
                regime=regime or 'NEUTRAL',
            )
            kelly_fraction = kelly_result['fraction']
            kelly_gear = kelly_result.get('gear', 'N/A')
        adjusted = max(MIN_LOTS, round(base_lots * kelly_fraction))
        reasoning.append(f'Kelly: fraction={kelly_fraction:.2f} gear={kelly_gear} -> {adjusted} lots')

        # ── Step 3: Overlay composite modifier ────────────────────
        composite_modifier = self._compute_composite(overlay_modifiers)
        composite_modifier = max(COMPOSITE_FLOOR, min(COMPOSITE_CEILING, composite_modifier))
        adjusted = max(MIN_LOTS, round(adjusted * composite_modifier))
        reasoning.append(f'Overlay composite: {composite_modifier:.3f} -> {adjusted} lots')

        # ── Step 4: Conviction modifier ───────────────────────────
        conviction_modifier = 1.0
        if self.conviction_scorer is not None:
            conv_dir = 'LONG' if direction == 'BULLISH' else 'SHORT'
            conv_result = self.conviction_scorer.compute(
                modifiers=overlay_modifiers,
                vix=vix,
                adx=adx,
                direction=conv_dir,
                open_positions=open_positions,
                drawdown_pct=drawdown_pct * 100,  # ConvictionScorer expects pct
                consecutive_losses=consecutive_losses,
            )
            conviction_modifier = conv_result['final_modifier']
        adjusted = max(MIN_LOTS, round(adjusted * conviction_modifier))
        reasoning.append(f'Conviction: {conviction_modifier:.3f} -> {adjusted} lots')

        # ── Step 5: Regime modifier ───────────────────────────────
        regime_modifier = REGIME_MULTIPLIERS.get(regime, DEFAULT_REGIME_MULT) if regime else DEFAULT_REGIME_MULT
        adjusted = max(MIN_LOTS, round(adjusted * regime_modifier))
        reasoning.append(f'Regime: {regime or "NONE"} x{regime_modifier:.2f} -> {adjusted} lots')

        # ── Step 6: Margin and hard caps ──────────────────────────
        available_margin = self.equity * 0.60
        margin_lots = math.floor(available_margin / MARGIN_PER_LOT) if MARGIN_PER_LOT > 0 else adjusted
        adjusted = min(adjusted, margin_lots, MAX_LOTS_CAP)
        adjusted = max(MIN_LOTS, adjusted)

        risk_amount = adjusted * risk_per_lot
        reasoning.append(f'Final: {adjusted} lots, risk=Rs{risk_amount:,.0f}')

        return {
            'lots': adjusted,
            'risk_amount': round(risk_amount, 2),
            'kelly_fraction': kelly_fraction,
            'composite_modifier': round(composite_modifier, 4),
            'conviction_modifier': round(conviction_modifier, 4),
            'regime_modifier': round(regime_modifier, 4),
            'reasoning': reasoning,
        }

    def update_equity(self, new_equity: float):
        """Update equity after P&L changes."""
        self.equity = new_equity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_composite(modifiers: Dict[str, float]) -> float:
        """
        Combine overlay modifiers using geometric mean per category,
        then multiply categories together.  Mirrors LotSizer hierarchy.
        """
        from execution.lot_sizer import OVERLAY_HIERARCHY

        if not modifiers:
            return 1.0

        category_mults = {}
        for category, signals in OVERLAY_HIERARCHY.items():
            cat_values = []
            for sig in signals:
                if sig in modifiers:
                    val = max(0.1, min(3.0, modifiers[sig]))
                    cat_values.append(val)
            if cat_values:
                geo_mean = math.exp(
                    sum(math.log(v) for v in cat_values) / len(cat_values)
                )
                category_mults[category] = geo_mean
            else:
                category_mults[category] = 1.0

        composite = 1.0
        for mult in category_mults.values():
            composite *= mult

        return composite
