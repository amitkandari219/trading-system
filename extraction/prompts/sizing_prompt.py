"""
Position Sizing DSL Translation Prompt.

Extracts THARP R-multiple and VINCE optimal-f sizing rules.
These are NOT entry/exit signals — they describe HOW MUCH to trade.
"""

SIZING_DSL_PROMPT = """You are extracting a POSITION SIZING rule from a trading book excerpt. This is NOT an entry/exit signal. It describes how to size positions.

SIGNAL TO EXTRACT:
  Signal ID: {signal_id}
  Rule: {rule_text}
  Category: {signal_category}
  Entry conditions: {entry_conditions}
  Parameters: {parameters}

SIZING METHODS (choose the most appropriate):
  R_MULTIPLE:        Risk a fixed dollar/% amount per trade (Van Tharp 1R concept)
  OPTIMAL_F:         Mathematically optimal fraction of capital (Ralph Vince)
  FIXED_FRACTIONAL:  Risk a fixed percentage of equity per trade
  VOLATILITY_SCALED: Scale position size inversely to volatility
  KELLY:             Kelly criterion or fractional Kelly
  ANTI_MARTINGALE:   Increase position size after winning trades
  DRAWDOWN_SCALED:   Reduce position size during drawdowns

EXAMPLES:

"Never risk more than 1% on a single trade":
{{"sizing_method":"FIXED_FRACTIONAL","base_risk_pct":0.01,"r_multiple_stop":1.0,"confidence":"HIGH","untranslatable":false}}

"After 3 consecutive wins increase size by 25%":
{{"sizing_method":"ANTI_MARTINGALE","scale_up_multiplier":1.25,"trigger_condition":{{"metric":"consecutive_wins","threshold":3,"direction":"above"}},"confidence":"HIGH","untranslatable":false}}

"Reduce to half size when drawdown exceeds 10%":
{{"sizing_method":"DRAWDOWN_SCALED","scale_down_multiplier":0.50,"trigger_condition":{{"metric":"drawdown_pct","threshold":0.10,"direction":"above"}},"confidence":"HIGH","untranslatable":false}}

"The optimal f is 0.19 for 45% wins and 2:1 ratio":
{{"sizing_method":"OPTIMAL_F","base_risk_pct":0.19,"note":"Full Kelly — use 0.095 half Kelly for safety","confidence":"HIGH","untranslatable":false}}

"Use half the optimal f to reduce variance":
{{"sizing_method":"KELLY","scale_factor":0.50,"note":"Half Kelly","confidence":"HIGH","untranslatable":false}}

"Scale inversely to 20-day realized volatility":
{{"sizing_method":"VOLATILITY_SCALED","trigger_condition":{{"metric":"rolling_volatility_20","direction":"inverse"}},"confidence":"HIGH","untranslatable":false}}

SKIP and return {{"untranslatable":true,"untranslatable_reason":"Not a sizing rule"}} for:
  - Philosophy statements without specific numbers
  - Market commentary
  - Risk warnings without thresholds
  - Entry/exit conditions (use standard DSL instead)
  - Expectancy calculations (these are filters, not sizing)

Return ONLY valid JSON:
{{
  "signal_type": "POSITION_SIZING",
  "sizing_method": "one of the methods above",
  "base_risk_pct": float or null,
  "r_multiple_target": float or null,
  "r_multiple_stop": float or null,
  "scale_up_multiplier": float or null,
  "scale_down_multiplier": float or null,
  "scale_factor": float or null,
  "trigger_condition": {{"metric": str, "threshold": float, "direction": str}} or null,
  "max_risk_pct": float or null,
  "min_risk_pct": float or null,
  "applies_to": "ALL" or "TRENDING" or "VOLATILE",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "note": "brief explanation",
  "untranslatable": false,
  "untranslatable_reason": null
}}"""


def format_sizing_prompt(signal: dict) -> str:
    """Format the sizing DSL prompt for a specific signal."""
    import json
    return SIZING_DSL_PROMPT.format(
        signal_id=signal.get('signal_id', 'unknown'),
        rule_text=signal.get('rule_text', '')[:500],
        signal_category=signal.get('signal_category', ''),
        entry_conditions=json.dumps(signal.get('entry_conditions', []))[:500],
        parameters=json.dumps(signal.get('parameters', {}))[:400],
    )
