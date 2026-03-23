"""
Macro / Cross-Asset signals for Nifty trading system.

These signals monitor global macro indicators and cross-asset relationships
that lead or influence NSE Nifty price action.
"""

from signals.macro.usdinr_momentum import UsdInrMomentumSignal
from signals.macro.us_yield_shock import UsYieldShockSignal
from signals.macro.china_decouple import ChinaDecoupleSignal
from signals.macro.gold_nifty_ratio import GoldNiftyRatioSignal
from signals.macro.crude_nifty_divergence import CrudeNiftyDivergenceSignal

__all__ = [
    'UsdInrMomentumSignal',
    'UsYieldShockSignal',
    'ChinaDecoupleSignal',
    'GoldNiftyRatioSignal',
    'CrudeNiftyDivergenceSignal',
]
