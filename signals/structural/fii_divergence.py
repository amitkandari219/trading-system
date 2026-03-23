"""
FII Divergence Signal — OVERLAY comparing FII futures vs options positioning.

FIIs (Foreign Institutional Investors) often hold offsetting positions in
index futures and index options.  The relationship between these positions
reveals their true directional view and hedging stance.

This is an OVERLAY signal — it does not generate standalone trades but
provides a regime classification and size multiplier that modifies other
scoring signals.

Regimes:
    HEDGED_BULLISH    : Long futures + long puts   → bullish with protection → 1.15×
    AGGRESSIVE_BULLISH: Long futures + short calls  → bullish, selling upside → 1.05×
    HEDGED_BEARISH    : Short futures + long calls  → bearish with protection → 0.85×
    CONFUSED          : Short futures + long puts   → bearish + buying puts   → 0.90×

When FII data is unavailable, a proxy regime is inferred from PCR + VIX.

Data source:
    - Historical: NSE FII/DII daily derivatives data.
    - Live: NSE bulk deal / participant-wise OI data.

Usage:
    from signals.structural.fii_divergence import FIIDivergenceSignal

    sig = FIIDivergenceSignal()
    result = sig.evaluate(market_data)
    # result['size_multiplier'] can be applied to other signal sizes.

Academic basis: Informed trader positioning in derivatives markets
(Easley, O'Hara & Srinivas 1998).  FII positioning is a leading indicator
of Nifty direction with ~55% weekly predictive accuracy.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = 'FII_DIVERGENCE'

# Regime definitions: (name, size_multiplier, description)
REGIME_HEDGED_BULLISH = 'HEDGED_BULLISH'
REGIME_AGGRESSIVE_BULLISH = 'AGGRESSIVE_BULLISH'
REGIME_HEDGED_BEARISH = 'HEDGED_BEARISH'
REGIME_CONFUSED = 'CONFUSED'
REGIME_NEUTRAL = 'NEUTRAL'

# Size multipliers per regime
REGIME_MULTIPLIERS = {
    REGIME_HEDGED_BULLISH: 1.15,
    REGIME_AGGRESSIVE_BULLISH: 1.05,
    REGIME_HEDGED_BEARISH: 0.85,
    REGIME_CONFUSED: 0.90,
    REGIME_NEUTRAL: 1.00,
}

# Confidence per regime (how reliable is this positioning signal)
REGIME_CONFIDENCE = {
    REGIME_HEDGED_BULLISH: 0.60,
    REGIME_AGGRESSIVE_BULLISH: 0.55,
    REGIME_HEDGED_BEARISH: 0.58,
    REGIME_CONFUSED: 0.45,
    REGIME_NEUTRAL: 0.40,
}

# Direction implied by regime
REGIME_DIRECTION = {
    REGIME_HEDGED_BULLISH: 'LONG',
    REGIME_AGGRESSIVE_BULLISH: 'LONG',
    REGIME_HEDGED_BEARISH: 'SHORT',
    REGIME_CONFUSED: 'SHORT',
    REGIME_NEUTRAL: 'NEUTRAL',
}

# Thresholds for FII net contracts (to avoid noise near zero)
FII_FUT_MIN_CONTRACTS = 1000    # Ignore if net < 1000 contracts
FII_OPT_MIN_OI = 500           # Ignore if net OI < 500

# Proxy thresholds (when FII data unavailable)
PCR_BULLISH = 1.2               # PCR > 1.2 → put writers confident → bullish
PCR_BEARISH = 0.8               # PCR < 0.8 → call writers confident → bearish
VIX_HIGH = 20.0                 # VIX > 20 → hedging demand = protection
VIX_LOW = 14.0                  # VIX < 14 → complacency

# FII futures change threshold for momentum
FII_FUT_CHANGE_LARGE = 5000     # >5000 contracts net change = strong flow


# ================================================================
# Helpers
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely cast to int."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _classify_fii_regime(
    fii_fut_net: int,
    fii_ce_net_oi: int,
    fii_pe_net_oi: int,
) -> str:
    """
    Classify FII regime from futures and options positioning.

    Parameters
    ----------
    fii_fut_net   : Net FII index futures contracts (positive = long)
    fii_ce_net_oi : Net FII index call OI (positive = long calls)
    fii_pe_net_oi : Net FII index put OI (positive = long puts)

    Returns
    -------
    Regime string.
    """
    fut_long = fii_fut_net > FII_FUT_MIN_CONTRACTS
    fut_short = fii_fut_net < -FII_FUT_MIN_CONTRACTS

    # For options: positive net OI = long (bought), negative = short (written)
    pe_long = fii_pe_net_oi > FII_OPT_MIN_OI
    ce_long = fii_ce_net_oi > FII_OPT_MIN_OI
    ce_short = fii_ce_net_oi < -FII_OPT_MIN_OI

    if fut_long and pe_long:
        return REGIME_HEDGED_BULLISH
    elif fut_long and ce_short:
        return REGIME_AGGRESSIVE_BULLISH
    elif fut_short and ce_long:
        return REGIME_HEDGED_BEARISH
    elif fut_short and pe_long:
        return REGIME_CONFUSED
    else:
        return REGIME_NEUTRAL


def _proxy_regime_from_pcr_vix(
    pcr: float,
    vix: float,
    nifty_change_pct: float,
) -> str:
    """
    Infer approximate FII regime from put-call ratio and VIX
    when direct FII positioning data is unavailable.

    Parameters
    ----------
    pcr               : Put-call ratio (OI-based)
    vix               : India VIX
    nifty_change_pct  : Nifty daily % change

    Returns
    -------
    Proxy regime string.
    """
    bullish_pcr = pcr > PCR_BULLISH
    bearish_pcr = pcr < PCR_BEARISH
    high_vix = vix > VIX_HIGH
    low_vix = vix < VIX_LOW

    if bullish_pcr and not high_vix:
        # High put writing + calm VIX → bullish confidence
        return REGIME_AGGRESSIVE_BULLISH
    elif bullish_pcr and high_vix:
        # High puts + high VIX → hedged bullish (protection in place)
        return REGIME_HEDGED_BULLISH
    elif bearish_pcr and high_vix:
        # Low puts + high VIX → bearish with hedging
        return REGIME_HEDGED_BEARISH
    elif bearish_pcr and low_vix:
        # Low puts + low VIX → confusing, mixed signals
        return REGIME_CONFUSED
    else:
        return REGIME_NEUTRAL


# ================================================================
# Signal Class
# ================================================================

class FIIDivergenceSignal:
    """
    OVERLAY signal — classifies FII positioning regime and provides
    a size multiplier for other scoring signals.

    Compares FII index futures net position vs FII index options
    positioning to determine the institutional directional view.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        self._last_regime: Optional[str] = None
        logger.info('FIIDivergenceSignal initialised')

    # ----------------------------------------------------------
    # No-signal helper
    # ----------------------------------------------------------
    @staticmethod
    def _no_signal() -> None:
        """Return None — no signal generated."""
        return None

    # ----------------------------------------------------------
    # Live evaluate
    # ----------------------------------------------------------
    def evaluate(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate FII divergence overlay.

        Parameters
        ----------
        market_data : dict
            Primary keys (FII data available):
                fii_index_fut_net_contracts : int   — net FII index futures (+ = long)
                fii_index_ce_net_oi         : int   — net FII call OI (+ = long)
                fii_index_pe_net_oi         : int   — net FII put OI (+ = long)
                fii_fut_net_change          : int   — daily change in net contracts

            Fallback keys (FII data unavailable):
                put_call_ratio              : float — OI-based PCR
                india_vix                   : float — India VIX
                nifty_daily_change_pct      : float — Nifty daily % change

            Optional:
                trade_date                  : date  — trading date

        Returns
        -------
        dict with regime classification and multiplier, or None if
        insufficient data for even proxy classification.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'FIIDivergenceSignal.evaluate error: %s', e, exc_info=True
            )
            return self._no_signal()

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date = market_data.get('trade_date')

        # ── Try primary FII data first ──────────────────────────
        fii_fut_net = market_data.get('fii_index_fut_net_contracts')
        fii_ce_net_oi = market_data.get('fii_index_ce_net_oi')
        fii_pe_net_oi = market_data.get('fii_index_pe_net_oi')
        fii_fut_change = market_data.get('fii_fut_net_change')

        using_proxy = False

        if (fii_fut_net is not None and
                fii_ce_net_oi is not None and
                fii_pe_net_oi is not None):
            # ── Direct FII classification ───────────────────────
            fii_fut_net = _safe_int(fii_fut_net)
            fii_ce_net_oi = _safe_int(fii_ce_net_oi)
            fii_pe_net_oi = _safe_int(fii_pe_net_oi)
            fii_fut_change = _safe_int(fii_fut_change, 0)

            regime = _classify_fii_regime(fii_fut_net, fii_ce_net_oi, fii_pe_net_oi)

            # ── Momentum adjustment ─────────────────────────────
            # If FII futures change is large, boost confidence
            momentum_boost = 0.0
            if abs(fii_fut_change) > FII_FUT_CHANGE_LARGE:
                momentum_boost = 0.05

        else:
            # ── Fallback to PCR + VIX proxy ─────────────────────
            pcr = _safe_float(market_data.get('put_call_ratio'))
            vix = _safe_float(market_data.get('india_vix'))
            nifty_change = _safe_float(market_data.get('nifty_daily_change_pct'), 0.0)

            if math.isnan(pcr) or math.isnan(vix):
                logger.debug(
                    'No FII data and insufficient proxy data (PCR=%s, VIX=%s)',
                    pcr, vix,
                )
                return self._no_signal()

            regime = _proxy_regime_from_pcr_vix(pcr, vix, nifty_change)
            using_proxy = True
            momentum_boost = 0.0
            fii_fut_net = None
            fii_ce_net_oi = None
            fii_pe_net_oi = None
            fii_fut_change = None

        # ── Look up regime properties ───────────────────────────
        size_multiplier = REGIME_MULTIPLIERS.get(regime, 1.0)
        base_confidence = REGIME_CONFIDENCE.get(regime, 0.40)
        direction = REGIME_DIRECTION.get(regime, 'NEUTRAL')

        # Proxy data is less reliable → reduce confidence
        if using_proxy:
            base_confidence *= 0.85

        confidence = min(0.90, base_confidence + momentum_boost)

        # ── Build reason string ─────────────────────────────────
        reason_parts = [
            f"FII_DIVERGENCE (OVERLAY)",
            f"Regime={regime}",
            f"Dir={direction}",
            f"Multiplier={size_multiplier}x",
        ]

        if not using_proxy:
            reason_parts.append(f"FII_Fut={fii_fut_net:+,}")
            reason_parts.append(f"FII_CE_OI={fii_ce_net_oi:+,}")
            reason_parts.append(f"FII_PE_OI={fii_pe_net_oi:+,}")
            if fii_fut_change:
                reason_parts.append(f"FutChange={fii_fut_change:+,}")
        else:
            pcr_val = _safe_float(market_data.get('put_call_ratio'))
            vix_val = _safe_float(market_data.get('india_vix'))
            reason_parts.append("PROXY_MODE")
            reason_parts.append(f"PCR={pcr_val:.2f}")
            reason_parts.append(f"VIX={vix_val:.1f}")

        # ── Track state ─────────────────────────────────────────
        regime_changed = (self._last_regime is not None and self._last_regime != regime)
        self._last_fire_date = trade_date
        self._last_regime = regime

        logger.info(
            '%s overlay: regime=%s dir=%s multiplier=%.2f conf=%.3f proxy=%s',
            self.SIGNAL_ID, regime, direction, size_multiplier,
            confidence, using_proxy,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'signal_type': 'OVERLAY',
            'regime': regime,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_multiplier': round(size_multiplier, 2),
            'using_proxy': using_proxy,
            'regime_changed': regime_changed,
            'fii_index_fut_net_contracts': fii_fut_net,
            'fii_index_ce_net_oi': fii_ce_net_oi,
            'fii_index_pe_net_oi': fii_pe_net_oi,
            'fii_fut_net_change': fii_fut_change,
            'put_call_ratio': _safe_float(market_data.get('put_call_ratio'), None),
            'india_vix': _safe_float(market_data.get('india_vix'), None),
            'nifty_daily_change_pct': _safe_float(
                market_data.get('nifty_daily_change_pct'), None
            ),
            'trade_date': trade_date,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None
        self._last_regime = None

    def __repr__(self) -> str:
        return f"FIIDivergenceSignal(signal_id='{self.SIGNAL_ID}')"
