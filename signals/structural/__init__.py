"""
Structural signals for Nifty F&O.

These signals exploit structural features of the Indian derivatives market:

SCORING signals — fire actual trades:
    GiftConvergenceSignal  — GIFT Nifty → NSE open convergence (09:15-09:45)
    MaxOIBarrierSignal     — Max OI strike support/resistance barrier trades
    RolloverFlowSignal     — Monthly rollover OI flow (ratio + CoC + FII)
    IndexRebalanceSignal   — MSCI/FTSE index rebalancing flow
"""

from signals.structural.gift_convergence import GiftConvergenceSignal
from signals.structural.index_rebalance import IndexRebalanceSignal
from signals.structural.max_oi_barrier import MaxOIBarrierSignal
from signals.structural.rollover_flow import RolloverFlowSignal

__all__ = [
    'GiftConvergenceSignal',
    'IndexRebalanceSignal',
    'MaxOIBarrierSignal',
    'RolloverFlowSignal',
]
