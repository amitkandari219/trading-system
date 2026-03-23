"""
Lot-Based Position Sizer — converts overlay modifiers into actual lot counts.

Replaces fixed-fraction sizing so that the 22 overlay signals
(VIX, FII, PCR, regime, ML, etc.) actually change position size in rupees.

Key formula:
  base_lots = floor(equity * risk_pct / (stop_loss_pts * lot_size))
  adjusted_lots = clamp(round(base_lots * composite_modifier), 1, max_lots)

Usage:
    from execution.lot_sizer import LotSizer
    sizer = LotSizer(equity=1_000_000)
    result = sizer.compute(stop_loss_pts=100, nifty_price=24000,
                           overlay_modifiers={'PCR_AUTOTRENDER': 1.2, ...})
"""

import logging
import math
from datetime import date
from typing import Dict, Optional

from config import settings

logger = logging.getLogger(__name__)

# Lot size history
LOT_SIZE_PRE_JUL2023 = 75   # before July 2023
LOT_SIZE_POST_JUL2023 = 25  # after July 2023
LOT_SIZE_CHANGE_DATE = date(2023, 7, 1)

# Margin per lot (approximate SPAN margin for Nifty futures)
MARGIN_PER_LOT = 120_000  # ~₹1.2L per lot

# Hierarchy weights for overlay combination
OVERLAY_HIERARCHY = {
    # Event filters: hard override, applied first
    'EVENT_FILTER': ['RBI_MACRO_FILTER'],
    # Regime: structural market state
    'REGIME': ['MAMBA_REGIME', 'GAMMA_EXPOSURE', 'VOL_TERM_STRUCTURE',
               'GUJRAL_DRY_7', 'CRISIS_SHORT'],
    # Flow: institutional positioning
    'FLOW': ['FII_FUTURES_OI', 'ROLLOVER_ANALYSIS', 'DELIVERY_PCT',
             'AMFI_MF_FLOW', 'CREDIT_CARD_SPENDING', 'ORDER_FLOW_IMBALANCE'],
    # Sentiment: market mood
    'SENTIMENT': ['SENTIMENT_COMPOSITE', 'NLP_SENTIMENT', 'BOND_YIELD_SPREAD',
                  'PCR_AUTOTRENDER'],
    # Calendar
    'CALENDAR': ['SSRN_JANUARY_EFFECT', 'GLOBAL_OVERNIGHT_COMPOSITE'],
    # Meta: final adjustment
    'META': ['XGBOOST_META_LEARNER', 'RL_POSITION_SIZER', 'TFT_FORECAST',
             'GNN_SECTOR_ROTATION'],
}

# Dampening: for each category, take geometric mean instead of product
# to prevent extreme compound effects


def get_lot_size(trade_date: date) -> int:
    """Return correct lot size for a given date."""
    if trade_date >= LOT_SIZE_CHANGE_DATE:
        return LOT_SIZE_POST_JUL2023
    return LOT_SIZE_PRE_JUL2023


class LotSizer:
    """Compute lot count from equity, risk, stop loss, and overlay modifiers."""

    def __init__(
        self,
        equity: float = 1_000_000,
        base_risk_pct: float = 0.02,
        max_exposure_pct: float = 0.60,
        max_lots_cap: int = 20,
        margin_per_lot: float = MARGIN_PER_LOT,
    ):
        self.equity = equity
        self.base_risk_pct = base_risk_pct
        self.max_exposure_pct = max_exposure_pct
        self.max_lots_cap = max_lots_cap
        self.margin_per_lot = margin_per_lot

    def compute(
        self,
        stop_loss_pts: float,
        nifty_price: float,
        trade_date: date = None,
        overlay_modifiers: Optional[Dict[str, float]] = None,
        direction: str = 'LONG',
        open_positions: int = 0,
    ) -> Dict:
        """
        Compute lot count with overlay modifiers.

        Args:
            stop_loss_pts: distance to stop loss in Nifty points
            nifty_price: current Nifty price
            trade_date: for lot size lookup (75 vs 25)
            overlay_modifiers: {signal_id: size_modifier} from all overlays
            direction: LONG or SHORT
            open_positions: number of currently open positions

        Returns:
            dict with lots, notional, margin, risk, modifier breakdown
        """
        trade_date = trade_date or date.today()
        lot_size = get_lot_size(trade_date)
        overlay_modifiers = overlay_modifiers or {}

        # Step 1: Base lots from risk budget, scaled by position capacity
        if stop_loss_pts <= 0:
            stop_loss_pts = nifty_price * 0.02  # default 2% SL

        # Divide risk budget by MAX_POSITIONS for consistent per-position allocation
        max_concurrent = settings.MAX_POSITIONS  # 4
        available_slots = max(1, max_concurrent - open_positions)
        adjusted_risk = self.equity * self.base_risk_pct / max_concurrent

        risk_per_lot = stop_loss_pts * lot_size
        base_lots = math.floor(adjusted_risk / risk_per_lot) if risk_per_lot > 0 else 1
        base_lots = max(1, base_lots)

        # Step 2: Apply Kelly fraction if available
        kelly_mult = overlay_modifiers.pop('_KELLY_FRACTION', 1.0)
        if kelly_mult != 1.0:
            base_lots = max(1, round(base_lots * kelly_mult))

        # Step 3: Composite overlay modifier (hierarchical)
        composite, breakdown = self._compute_composite_modifier(overlay_modifiers)

        # Step 4: Adjusted lots
        adjusted_lots = max(1, round(base_lots * composite))

        # Step 4: Margin constraint
        available_margin = self.equity * self.max_exposure_pct
        margin_lots = math.floor(available_margin / self.margin_per_lot) if self.margin_per_lot > 0 else adjusted_lots
        adjusted_lots = min(adjusted_lots, margin_lots)

        # Step 5: Hard cap
        adjusted_lots = min(adjusted_lots, self.max_lots_cap)
        adjusted_lots = max(1, adjusted_lots)

        # Compute derived values
        notional = adjusted_lots * lot_size * nifty_price
        margin_required = adjusted_lots * self.margin_per_lot
        risk_rupees = adjusted_lots * lot_size * stop_loss_pts

        return {
            'lots': adjusted_lots,
            'base_lots': base_lots,
            'lot_size': lot_size,
            'notional': round(notional),
            'margin_required': round(margin_required),
            'risk_rupees': round(risk_rupees),
            'risk_pct_of_equity': round(risk_rupees / self.equity * 100, 2),
            'composite_modifier': round(composite, 3),
            'modifier_breakdown': breakdown,
            'margin_utilization_pct': round(margin_required / self.equity * 100, 1),
        }

    def _compute_composite_modifier(
        self, modifiers: Dict[str, float]
    ) -> tuple:
        """
        Combine overlay modifiers using hierarchical dampening.

        Each category (EVENT, REGIME, FLOW, SENTIMENT, META) produces
        a geometric mean of its constituent modifiers. Categories are
        then multiplied together. Final result clamped to [0.3, 2.0].
        """
        if not modifiers:
            return 1.0, {}

        category_mults = {}
        breakdown = {}

        for category, signals in OVERLAY_HIERARCHY.items():
            cat_values = []
            for sig in signals:
                if sig in modifiers:
                    val = modifiers[sig]
                    val = max(0.1, min(3.0, val))  # per-signal clamp
                    cat_values.append(val)
                    breakdown[sig] = round(val, 3)

            if cat_values:
                # Geometric mean for the category
                geo_mean = math.exp(sum(math.log(v) for v in cat_values) / len(cat_values))
                category_mults[category] = geo_mean
            else:
                category_mults[category] = 1.0

        # Multiply category results
        composite = 1.0
        for cat, mult in category_mults.items():
            composite *= mult

        # Clamp final composite
        composite = max(0.3, min(2.0, composite))

        breakdown['_categories'] = {k: round(v, 3) for k, v in category_mults.items()}
        breakdown['_composite_raw'] = round(composite, 3)

        return composite, breakdown

    def update_equity(self, new_equity: float):
        self.equity = new_equity
