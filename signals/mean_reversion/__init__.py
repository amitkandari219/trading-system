"""
Mean-reversion / statistical arbitrage signals for NSE Nifty.

Signals:
    - NiftyBankNiftySpreadSignal : Pairs z-score on Nifty/BankNifty ratio
    - RSI2ReversionSignal        : Connors RSI(2) oversold/overbought
    - BasisZScoreSignal          : Cash-futures basis mispricing
    - BollingerSqueezeSignal     : BB squeeze -> expansion breakout
"""

from signals.mean_reversion.nifty_banknifty_spread import NiftyBankNiftySpreadSignal
from signals.mean_reversion.rsi2_reversion import RSI2ReversionSignal
from signals.mean_reversion.basis_zscore import BasisZScoreSignal
from signals.mean_reversion.bollinger_squeeze import BollingerSqueezeSignal

__all__ = [
    'NiftyBankNiftySpreadSignal',
    'RSI2ReversionSignal',
    'BasisZScoreSignal',
    'BollingerSqueezeSignal',
]
