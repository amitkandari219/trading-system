"""
Intraday SHADOW signal configuration.

These signals run on 5-min bars during market hours. They log trades
without risking capital, collecting real performance data for promotion.

Used by intraday_runner.py (live) and intraday_forward_test.py (backtest).

Tuned on real Kite 5-min data (2025-03 to 2026-03):
  ORB_TUNED: OR/ATR < 0.6, VIX < 25, TP = 1.5x OR width
             Filters relaxed after expert audit (2026-03-22):
             - VIX 17→25 (was blocking 60% of post-2023 days)
             - ATR ratio 0.3→0.6 (was blocking 85% of days)
             - Volume filter disabled (Kite returns trade count, not volume)
"""

INTRADAY_SHADOW_SIGNALS = {
    'ORB_TUNED': {
        'signal_class': 'OpeningRangeBreakout',
        'module': 'signals.intraday_framework',
        'params': {
            'range_minutes': 15,
            'atr_max_ratio': 0.6,    # Relaxed: 0.3→0.6 (was filtering 85%+ of days)
            'atr_min_ratio': 0.02,   # Skip degenerate ranges
            'rr_ratio': 1.5,         # TP at 1.5x OR width
            'vix_max': 25.0,         # Relaxed: 17→25 (block CRISIS only, not ELEVATED)
            'long_only': True,       # SHORT side loses -₹25K, LONG profits +₹1.5K
        },
        'trade_type': 'SHADOW',
        'instrument': 'NIFTY_FUT',
        'timeframe': '5min',
        'status': 'SHADOW',
        'backtest_results': {
            'period': '2025-03-21 to 2026-03-20',
            'data_source': 'kite_real_5min',
            'trades': 116,
            'win_rate': 0.414,
            'profit_factor': 1.00,
            'sharpe': 0.03,
            'note': 'Break-even without volume. Needs futures volume confirmation.',
        },
    },
}
