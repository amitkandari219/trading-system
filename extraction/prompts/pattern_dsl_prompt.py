"""
Pattern DSL Translation Prompt.

Converts Bulkowski/Candlestick pattern descriptions to indicator-based
DSL conditions. Standard DSL prompt fails on pattern language because
it can't convert "higher highs and lower lows" to indicator comparisons.

This prompt includes:
1. Pattern-to-indicator conversion table (few-shot examples)
2. Bulkowski statistics lookup for take_profit and confidence
3. ATR-based stop loss instead of fixed percentage
"""

# Bulkowski published statistics for pattern lookup
BULKOWSKI_STATS = {
    'High Tight Flag':        {'success': 0.86, 'avg_rise': 0.69},
    'Ascending Triangle':     {'success': 0.75, 'avg_rise': 0.38},
    'Descending Triangle':    {'success': 0.72, 'avg_decline': 0.20},
    'Double Bottom':          {'success': 0.78, 'avg_rise': 0.40},
    'Double Top':             {'success': 0.75, 'avg_decline': 0.19},
    'Head and Shoulders':     {'success': 0.83, 'avg_decline': 0.22},
    'Inverse Head Shoulders': {'success': 0.79, 'avg_rise': 0.37},
    'Cup with Handle':        {'success': 0.68, 'avg_rise': 0.34},
    'Flag Bull':              {'success': 0.64, 'avg_rise': 0.23},
    'Flag Bear':              {'success': 0.67, 'avg_decline': 0.22},
    'Pennant Bull':           {'success': 0.65, 'avg_rise': 0.25},
    'Pennant Bear':           {'success': 0.66, 'avg_decline': 0.24},
    'Rectangle Bull':         {'success': 0.68, 'avg_rise': 0.38},
    'Rectangle Bear':         {'success': 0.65, 'avg_decline': 0.19},
    'Wedge Rising':           {'success': 0.69, 'avg_decline': 0.18},
    'Wedge Falling':          {'success': 0.68, 'avg_rise': 0.32},
    'Symmetrical Triangle':   {'success': 0.54, 'avg_rise': 0.31},
    'Broadening Top':         {'success': 0.60, 'avg_decline': 0.17},
    'Broadening Bottom':      {'success': 0.58, 'avg_rise': 0.27},
    'Diamond Top':            {'success': 0.69, 'avg_decline': 0.21},
    'Diamond Bottom':         {'success': 0.70, 'avg_rise': 0.33},
    'Pipe Bottom':            {'success': 0.74, 'avg_rise': 0.43},
    'Pipe Top':               {'success': 0.73, 'avg_decline': 0.20},
    'Three Rising Valleys':   {'success': 0.75, 'avg_rise': 0.41},
    'Rounding Bottom':        {'success': 0.74, 'avg_rise': 0.43},
    'Bump and Run Reversal':  {'success': 0.71, 'avg_decline': 0.25},
    'Hammer':                 {'success': 0.60, 'avg_rise': 0.11},
    'Inverted Hammer':        {'success': 0.58, 'avg_rise': 0.09},
    'Shooting Star':          {'success': 0.59, 'avg_decline': 0.08},
    'Hanging Man':            {'success': 0.57, 'avg_decline': 0.07},
    'Bullish Engulfing':      {'success': 0.63, 'avg_rise': 0.08},
    'Bearish Engulfing':      {'success': 0.62, 'avg_decline': 0.08},
    'Morning Star':           {'success': 0.61, 'avg_rise': 0.12},
    'Evening Star':           {'success': 0.60, 'avg_decline': 0.11},
    'Doji':                   {'success': 0.55, 'avg_rise': 0.05},
    'Three White Soldiers':   {'success': 0.65, 'avg_rise': 0.14},
    'Three Black Crows':      {'success': 0.64, 'avg_decline': 0.13},
}


PATTERN_DSL_PROMPT = """You are converting a chart pattern trading signal into indicator-based conditions for a backtest engine.

PATTERN TO CONVERT:
  Pattern name: {pattern_name}
  Rule: {rule_text}
  Entry conditions (text): {entry_conditions}
  Exit conditions (text): {exit_conditions}
  Direction: {direction}
  Statistics: success_rate={success_rate}, avg_rise={avg_rise}, avg_decline={avg_decline}

PATTERN-TO-INDICATOR CONVERSION TABLE:
  "Higher highs"           → high > prev_high
  "Higher lows"            → low > prev_low
  "Lower lows"             → low < prev_low
  "Lower highs"            → high < prev_high
  "Breakout above"         → close > dc_upper (Donchian 20-day high)
  "Breakout below"         → close < dc_lower (Donchian 20-day low)
  "Volume confirmation"    → vol_ratio_20 > 1.5
  "Volume declining"       → volume < prev_volume
  "Inside bar"             → high < prev_high AND low > prev_low
  "Outside bar/engulfing"  → high > prev_high AND low < prev_low
  "Gap up"                 → open > prev_close
  "Gap down"               → open < prev_close
  "Narrow range"           → bb_bandwidth < 0.03
  "Wide range"             → bb_bandwidth > 0.06
  "Price above MA"         → close > sma_50
  "Price below MA"         → close < sma_50
  "Strong trend"           → adx_14 > 25
  "Hammer/lower shadow"    → lower_wick > body * 2
  "Shooting star"          → upper_wick > body * 2
  "Doji"                   → body_pct < 0.001
  "Price near range high"  → price_pos_20 > 0.8
  "Price near range low"   → price_pos_20 < 0.2
  "Overbought"             → rsi_14 > 70
  "Oversold"               → rsi_14 < 30

AVAILABLE INDICATORS (use ONLY these):
  Price: open, high, low, close, volume
  MAs: sma_5, sma_10, sma_20, sma_40, sma_50, sma_80, sma_100, sma_200
  EMA: ema_5, ema_10, ema_20, ema_50, ema_100, ema_200
  RSI: rsi_7, rsi_14, rsi_21
  ATR: atr_7, atr_14, atr_20
  Bollinger: bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth
  Stochastic: stoch_k, stoch_d, stoch_k_5, stoch_d_5
  ADX: adx_14
  Donchian: dc_upper, dc_lower, dc_middle
  Pivots: pivot, r1, s1, r2, s2, r3, s3
  Volume: vol_ratio_20
  Volatility: hvol_6, hvol_20, hvol_100, india_vix
  Price Position: price_pos_20 (0=low, 1=high in 20-day range)
  Previous Bar: prev_close, prev_high, prev_low, prev_open, prev_volume
  Returns: returns, log_returns
  Bar Props: body, body_pct, upper_wick, lower_wick, range
  True Range: true_range

SCALE NOTES:
  0-100 scale: rsi_14, stoch_k, adx_14
  0-1 scale: bb_pct_b, price_pos_20, body_pct
  Decimal: returns, hvol_20, bb_bandwidth

OPERATORS: >, <, >=, <=, ==, crosses_above, crosses_below

RULES:
1. Convert EVERY pattern description to indicator conditions
2. Use the conversion table above as guide
3. For breakout patterns: use dc_upper/dc_lower
4. For reversal patterns: use price_pos_20 extremes
5. For candlestick patterns: use body, upper_wick, lower_wick
6. If the pattern truly cannot be expressed: set untranslatable=true
7. Stop loss: use atr_14-based formula (2× ATR from entry)
8. Take profit: use avg_rise/avg_decline if available

WORKED EXAMPLES:

Example 1 - Ascending Triangle (breakout):
  {{"entry_long": [{{"left":"close","operator":"crosses_above","right":"dc_upper"}},{{"left":"adx_14","operator":">","right":"20"}},{{"left":"vol_ratio_20","operator":">","right":"1.2"}}],
    "entry_short": [],
    "exit_long": [{{"left":"close","operator":"<","right":"sma_20"}}],
    "exit_short": [],
    "stop_loss_pct": 3.0, "take_profit_pct": 5.0, "hold_days_max": 20,
    "direction": "LONG", "target_regime": ["TRENDING_UP","RANGING"],
    "untranslatable": false}}

Example 2 - Hammer candlestick (reversal):
  {{"entry_long": [{{"left":"lower_wick","operator":">","right":"0.02"}},{{"left":"body_pct","operator":">","right":"0.001"}},{{"left":"price_pos_20","operator":"<","right":"0.3"}},{{"left":"vol_ratio_20","operator":">","right":"1.0"}}],
    "entry_short": [],
    "exit_long": [{{"left":"rsi_14","operator":">","right":"65"}}],
    "exit_short": [],
    "stop_loss_pct": 2.5, "take_profit_pct": 3.0, "hold_days_max": 15,
    "direction": "LONG", "target_regime": ["ANY"],
    "untranslatable": false}}

Example 3 - Head and Shoulders (reversal, bearish):
  {{"entry_long": [],
    "entry_short": [{{"left":"close","operator":"crosses_below","right":"sma_50"}},{{"left":"high","operator":"<","right":"prev_high"}},{{"left":"adx_14","operator":">","right":"20"}}],
    "exit_long": [],
    "exit_short": [{{"left":"close","operator":"crosses_above","right":"sma_20"}}],
    "stop_loss_pct": 3.0, "take_profit_pct": 4.0, "hold_days_max": 25,
    "direction": "SHORT", "target_regime": ["ANY"],
    "untranslatable": false}}

Return ONLY valid JSON. Do NOT invent indicator names not in the list above.
If the pattern is too complex for these indicators, set untranslatable=true with a reason."""


def get_pattern_stats(pattern_name: str) -> dict:
    """Look up Bulkowski statistics for a pattern name."""
    if not pattern_name:
        return {}

    # Try exact match
    if pattern_name in BULKOWSKI_STATS:
        return BULKOWSKI_STATS[pattern_name]

    # Try case-insensitive
    for key, stats in BULKOWSKI_STATS.items():
        if key.lower() == pattern_name.lower():
            return stats

    # Try partial match
    name_lower = pattern_name.lower()
    for key, stats in BULKOWSKI_STATS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return stats

    return {}


def format_pattern_prompt(signal: dict) -> str:
    """Format the pattern DSL prompt for a specific signal."""
    params = signal.get('parameters', {})
    pattern_name = params.get('pattern_name', params.get('_canonical_name', ''))

    # Get stats from signal or lookup table
    stats = get_pattern_stats(pattern_name)
    success = params.get('success_rate_bull', params.get('success_rate_bear', ''))
    if success in ('AUTHOR_SILENT', 'MISSING', None, ''):
        success = stats.get('success', 'UNKNOWN')
    avg_rise = params.get('average_rise_pct', '')
    if avg_rise in ('AUTHOR_SILENT', 'MISSING', None, ''):
        avg_rise = stats.get('avg_rise', 'UNKNOWN')
    avg_decline = params.get('average_decline_pct', '')
    if avg_decline in ('AUTHOR_SILENT', 'MISSING', None, ''):
        avg_decline = stats.get('avg_decline', 'UNKNOWN')

    return PATTERN_DSL_PROMPT.format(
        pattern_name=pattern_name or signal.get('rule_text', '')[:50],
        rule_text=signal.get('rule_text', '')[:300],
        entry_conditions=str(signal.get('entry_conditions', []))[:500],
        exit_conditions=str(signal.get('exit_conditions', []))[:300],
        direction=signal.get('direction', 'CONTEXT_DEPENDENT'),
        success_rate=success,
        avg_rise=avg_rise,
        avg_decline=avg_decline,
    )
