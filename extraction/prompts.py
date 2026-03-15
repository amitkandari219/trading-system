"""
Extraction prompts for Claude API.
These are the EXACT prompts from the spec. Do not paraphrase.

Models used:
- Extraction:         claude-haiku-4-5-20251001 (cheap, fast)
- Hallucination check: claude-sonnet-4-20250514 (more careful)
"""

CONCRETE_BOOK_EXTRACTION_PROMPT = """
You are extracting a structured trading rule from a book excerpt.
Extract ONLY what the author explicitly states.
Do NOT infer, extend, or add information not in the text.
If a parameter is not specified, use the string "AUTHOR_SILENT".

BOOK: {book_title} by {author}
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}
ABSTRACTION LEVEL: CONCRETE (this book states specific rules)
CHUNK TYPE: {chunk_type}

CHUNK TYPE GUIDANCE:
- RULE: Extract the specific trading rule with exact entry/exit conditions
- FORMULA: Extract the formula-based signal. Entry conditions MUST include the formula and threshold values
- EMPIRICAL: Extract the empirical finding as a signal. Entry conditions MUST include the statistical observation and sample period
- CODE: Extract the algorithmic signal. Entry conditions MUST include the logic/pseudocode
- SUMMARY: Extract the key takeaway as a rule
- PSYCHOLOGY: Extract any discipline-based rule (e.g. "stop trading after 3 losses")
- TABLE: Extract any pattern or threshold from the table data

EXCERPT:
{chunk_text}

Extract the trading rule as JSON. Return ONLY valid JSON, no other text.

{{
  "rule_found": true or false,
  "rule_text": "One sentence. What triggers what action.",
  "signal_category": "TREND|REVERSION|VOL|PATTERN|EVENT|SIZING|RISK|REGIME",
  "direction": "LONG|SHORT|NEUTRAL|CONTEXT_DEPENDENT",
  "entry_conditions": [
    "Exact condition 1 as stated by author",
    "Exact condition 2 as stated by author"
  ],
  "parameters": {{
    "parameter_name": "exact value as stated, or AUTHOR_SILENT"
  }},
  "exit_conditions": [
    "Exact exit rule as stated, or AUTHOR_SILENT"
  ],
  "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD|ANY|AUTHOR_SILENT",
  "timeframe": "INTRADAY|POSITIONAL|SWING|ANY|AUTHOR_SILENT",
  "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"] (array — pick all that apply, or ["ANY"] if AUTHOR_SILENT),
  "author_confidence": "HIGH|MEDIUM|LOW",
  "author_confidence_basis": "Why the author is or is not confident",
  "source_citation": "{book_title}, {author}, Ch.{chapter_number}, p.{page_start}",
  "conflicts_with": "If author mentions a conflicting approach, state it. Else: NONE",
  "completeness_warning": "Important context that may be in surrounding chapters, or NONE"
}}

If no clear trading rule exists in this excerpt: return {{"rule_found": false}}
"""

PRINCIPLE_BOOK_EXTRACTION_PROMPT = """
You are extracting a trading PRINCIPLE from a book excerpt.
This book operates at principle level, not specific rule level.
Your job is to identify the principle AND generate 3 parameter variants
that operationalize it. The backtest will determine which variant is best.

BOOK: {book_title} by {author}
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}
ABSTRACTION LEVEL: PRINCIPLE (generate 3 variants with parameter ranges)
CHUNK TYPE: {chunk_type}

CHUNK TYPE GUIDANCE:
- RULE: Extract the specific principle with actionable conditions
- FORMULA: The principle is formula-based. Variants MUST use different formula thresholds/parameters
- EMPIRICAL: The principle is based on a statistical finding. Variants MUST preserve the empirical basis
- CODE: The principle is algorithmic. Variants MUST include logic/pseudocode in entry conditions
- PSYCHOLOGY: Extract discipline or mindset rules (e.g. position sizing based on emotional state)

IMPORTANT — DO NOT extract a principle if the excerpt is primarily:
- A mathematical derivation, proof, or formula explanation
- A definition of a term (what vega IS, not what to DO with vega)
- Historical market context or background information
- A mechanical description of how an instrument works
- A comparison of two approaches with no stated preference
- A worked example that illustrates a concept without prescribing action

An actionable principle MUST contain an instruction — explicit or implicit.
Test: Can you complete the sentence "A trader should [action] when [condition]"
using only what this excerpt states?
If NO → return {{"principle_found": false}}
If YES → extract the principle and generate 3 variants.

EXCERPT:
{chunk_text}

Extract as JSON. Return ONLY valid JSON, no other text.

{{
  "principle_found": true or false,
  "principle_text": "The core principle in one sentence",
  "trader_should_statement": "A trader should [action] when [condition]",
  "signal_category": "TREND|REVERSION|VOL|PATTERN|EVENT|SIZING|RISK|REGIME",
  "variants": [
    {{
      "variant_id": "CONSERVATIVE",
      "description": "Conservative operationalization of this principle",
      "entry_conditions": ["condition 1", "condition 2"],
      "parameters": {{"param_name": "conservative_value"}},
      "exit_conditions": ["exit rule"],
      "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD",
      "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"],
      "rationale": "Why these are conservative parameters"
    }},
    {{
      "variant_id": "MODERATE",
      "description": "Moderate operationalization",
      "entry_conditions": ["condition 1", "condition 2"],
      "parameters": {{"param_name": "moderate_value"}},
      "exit_conditions": ["exit rule"],
      "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD",
      "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"],
      "rationale": "Why these are moderate parameters"
    }},
    {{
      "variant_id": "AGGRESSIVE",
      "description": "Aggressive operationalization",
      "entry_conditions": ["condition 1", "condition 2"],
      "parameters": {{"param_name": "aggressive_value"}},
      "exit_conditions": ["exit rule"],
      "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD",
      "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"],
      "rationale": "Why these are aggressive parameters"
    }}
  ],
  "source_citation": "{book_title}, {author}, Ch.{chapter_number}, p.{page_start}",
  "author_stated_conditions": "Any conditions the author DOES specify explicitly",
  "what_author_leaves_unspecified": "Parameters the author does not define"
}}

If no actionable principle exists: return {{"principle_found": false}}
"""

PSYCHOLOGY_BOOK_EXTRACTION_PROMPT = """
You are extracting a DISCIPLINE or MINDSET rule from a trading psychology book.
This book teaches mental frameworks, NOT specific indicators or price-based rules.

BOOK: {book_title} by {author}
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}
ABSTRACTION LEVEL: PSYCHOLOGY (discipline rules only)
CHUNK TYPE: {chunk_type}

CRITICAL CONSTRAINTS:
- Extract discipline/process rules the author states or strongly implies
- Do NOT invent numeric thresholds, scores, or percentages the author never mentions
- Do NOT add indicator-based conditions (RSI, ATR, MA, Bollinger, etc.)
- Entry conditions must be behavioral or process-based, not price-based
- If the author says "control your emotions", the entry condition is "Trader recognizes emotional state affecting decisions" — NOT "Emotion score >= 70%"
- A discipline rule can be: when to trade, when NOT to trade, how to think about risk, how to handle losses/wins, position sizing mindset
- If the excerpt is purely narrative/anecdotal with no extractable discipline rule, return {{"rule_found": false}}

EXCERPT:
{chunk_text}

Extract as JSON. Return ONLY valid JSON, no other text.

{{
  "rule_found": true or false,
  "rule_text": "The discipline rule in one sentence, as the author states it",
  "signal_category": "PSYCHOLOGY|RISK|SIZING",
  "direction": "NEUTRAL",
  "entry_conditions": [
    "Behavioral condition as author states it (no invented numbers)"
  ],
  "parameters": {{}},
  "exit_conditions": [
    "When to stop applying this rule, or AUTHOR_SILENT"
  ],
  "instrument": "ANY",
  "timeframe": "ANY",
  "target_regimes": ["ANY"],
  "author_confidence": "HIGH|MEDIUM|LOW",
  "source_citation": "{book_title}, {author}, Ch.{chapter_number}, p.{page_start}",
  "completeness_warning": "NONE"
}}

If no clear discipline rule exists: return {{"rule_found": false}}
"""

CONFLICT_DETECTION_PROMPT = """
You are comparing two extracted trading rules to determine if they conflict.

RULE A:
{rule_a}

RULE B:
{rule_b}

Analyze and return JSON only:

{{
  "conflict_type": "DIRECT_CONTRADICTION|PARAMETER_DISAGREEMENT|CONDITION_DISAGREEMENT|EXIT_DISAGREEMENT|NO_CONFLICT|COMPLEMENTARY",
  "conflict_description": "Exactly what disagrees between the two rules",
  "conflict_on_field": "direction|entry_conditions|parameters|exit_conditions|target_regimes",
  "resolution_method": "BACKTEST_BOTH|REGIME_SEPARATE|PARAMETER_TEST|HUMAN_REVIEW",
  "resolution_notes": "Specific guidance for how to resolve this conflict in backtesting",
  "can_coexist": true or false,
  "coexistence_condition": "If they can coexist, under what conditions"
}}
"""

HALLUCINATION_CHECK_PROMPT = """
A trading rule has been extracted from a book excerpt.
Verify whether the extracted rule is faithful to the source text.

ORIGINAL EXCERPT:
{original_chunk}

EXTRACTED RULE:
{extracted_rule}

Check each field and return JSON only:

{{
  "overall_verdict": "FAITHFUL|MINOR_ISSUES|SIGNIFICANT_ISSUES|HALLUCINATED",
  "field_checks": {{
    "rule_text": "FAITHFUL|INFERRED|HALLUCINATED",
    "entry_conditions": "FAITHFUL|INFERRED|HALLUCINATED",
    "parameters": "FAITHFUL|INFERRED|HALLUCINATED",
    "exit_conditions": "FAITHFUL|INFERRED|HALLUCINATED",
    "instrument": "FAITHFUL|INFERRED|HALLUCINATED",
    "target_regimes": "FAITHFUL|INFERRED|HALLUCINATED"
  }},
  "issues_found": ["Issue 1", "Issue 2"],
  "recommendation": "APPROVE|REVISE|REJECT",
  "revision_notes": "What needs to change if REVISE"
}}
"""


# ================================================================
# BOOK-SPECIFIC PROMPT OVERRIDES
# ================================================================

BULKOWSKI_EXTRACTION_PROMPT = """
You are extracting chart pattern trading rules from Bulkowski's Encyclopedia of Chart Patterns.

BOOK: Encyclopedia of Chart Patterns (2nd Ed.) by Thomas Bulkowski
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}

This book contains detailed STATISTICAL data for each pattern. Your job is to extract
BOTH the pattern identification rules AND the performance statistics.

EXCERPT:
{chunk_text}

Extract as JSON. Return ONLY valid JSON.

{{
  "rule_found": true or false,
  "rule_text": "Pattern name: brief description of entry conditions",
  "signal_category": "PATTERN",
  "direction": "LONG|SHORT|CONTEXT_DEPENDENT",
  "entry_conditions": [
    "Pattern identification criterion 1 (as stated by author)",
    "Pattern identification criterion 2",
    "Volume confirmation requirement (if stated)"
  ],
  "parameters": {{
    "pattern_name": "exact pattern name",
    "breakout_direction": "UP|DOWN|BOTH",
    "success_rate_bull": "exact percentage or AUTHOR_SILENT",
    "success_rate_bear": "exact percentage or AUTHOR_SILENT",
    "average_rise_pct": "exact percentage or AUTHOR_SILENT",
    "average_decline_pct": "exact percentage or AUTHOR_SILENT",
    "failure_rate_5pct": "percentage that fail to move 5% or AUTHOR_SILENT",
    "volume_trend": "description of volume behavior or AUTHOR_SILENT",
    "throwback_pullback_pct": "percentage of throwbacks/pullbacks or AUTHOR_SILENT"
  }},
  "exit_conditions": [
    "Stop placement rule (as stated by author, or AUTHOR_SILENT)",
    "Target price computation (if stated, or AUTHOR_SILENT)"
  ],
  "instrument": "ANY",
  "timeframe": "POSITIONAL|SWING|ANY",
  "target_regimes": ["ANY"],
  "author_confidence": "HIGH|MEDIUM|LOW",
  "author_confidence_basis": "Based on the statistics cited",
  "source_citation": "Encyclopedia of Chart Patterns, Bulkowski, Ch.{chapter_number}, p.{page_start}"
}}

IMPORTANT: Extract ALL statistical numbers exactly as stated. Do not round or approximate.
If a statistics table is present, extract every row of data.
If no pattern rule exists in this excerpt: return {{"rule_found": false}}
"""

CHAN_EXTRACTION_PROMPT = """
You are extracting trading strategies from Ernest Chan's quantitative/algorithmic trading books.

BOOK: {book_title} by Ernest Chan
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}

Chan often expresses strategies as code (Python/MATLAB) or mathematical formulas.
Translate code into natural language entry/exit conditions.

EXCERPT:
{chunk_text}

Extract as JSON. Return ONLY valid JSON.

{{
  "rule_found": true or false,
  "rule_text": "One sentence describing the strategy",
  "signal_category": "TREND|REVERSION|VOL|PATTERN|EVENT|REGIME",
  "direction": "LONG|SHORT|CONTEXT_DEPENDENT",
  "entry_conditions": [
    "Condition 1 (translated from code/formula to plain English)",
    "Condition 2"
  ],
  "parameters": {{
    "lookback_period": "exact value or AUTHOR_SILENT",
    "threshold": "exact value or AUTHOR_SILENT",
    "half_life": "if mean reversion, the half-life or AUTHOR_SILENT",
    "backtest_sharpe": "if Chan reports a Sharpe ratio, state it",
    "backtest_period": "date range of backtest if stated"
  }},
  "exit_conditions": [
    "Exit condition (from code or text, or AUTHOR_SILENT)"
  ],
  "instrument": "FUTURES|ANY|AUTHOR_SILENT",
  "timeframe": "INTRADAY|POSITIONAL|SWING|ANY|AUTHOR_SILENT",
  "target_regimes": ["ANY"],
  "author_confidence": "HIGH|MEDIUM|LOW",
  "source_citation": "{book_title}, Chan, Ch.{chapter_number}, p.{page_start}"
}}

When translating code:
  'returns.rolling(20).mean() > 0' → '20-day rolling average return is positive'
  'zscore > 2' → 'z-score of spread exceeds 2'
  'halflife < 30' → 'mean reversion half-life is less than 30 days'

If no trading strategy in this excerpt: return {{"rule_found": false}}
"""

VINCE_EXTRACTION_PROMPT = """
You are extracting position sizing and money management rules from Ralph Vince's Mathematics of Money Management.

BOOK: Mathematics of Money Management by Ralph Vince
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}

This book does NOT contain entry/exit signals. It contains POSITION SIZING rules.
Extract conditions for WHEN and HOW MUCH to trade.

EXCERPT:
{chunk_text}

Extract as JSON. Return ONLY valid JSON.

{{
  "rule_found": true or false,
  "rule_text": "One sentence describing the sizing rule",
  "signal_category": "SIZING",
  "direction": "NEUTRAL",
  "entry_conditions": [
    "Condition when this sizing rule applies",
    "Formula or threshold for position size"
  ],
  "parameters": {{
    "optimal_f": "value or formula or AUTHOR_SILENT",
    "kelly_fraction": "value or AUTHOR_SILENT",
    "max_drawdown_tolerance": "value or AUTHOR_SILENT",
    "formula": "exact mathematical formula if stated"
  }},
  "exit_conditions": ["AUTHOR_SILENT"],
  "instrument": "ANY",
  "timeframe": "ANY",
  "target_regimes": ["ANY"],
  "author_confidence": "HIGH|MEDIUM|LOW",
  "source_citation": "Mathematics of Money Management, Vince, Ch.{chapter_number}, p.{page_start}"
}}

If no sizing rule in this excerpt: return {{"rule_found": false}}
"""

# Map book_id to specific prompt (if None, use default based on abstraction level)
BOOK_SPECIFIC_PROMPTS = {
    'BULKOWSKI': BULKOWSKI_EXTRACTION_PROMPT,
    'CANDLESTICK': BULKOWSKI_EXTRACTION_PROMPT,  # same pattern structure
    'CHAN_QT': CHAN_EXTRACTION_PROMPT,
    'CHAN_AT': CHAN_EXTRACTION_PROMPT,
    'VINCE': VINCE_EXTRACTION_PROMPT,
    # All other books use default prompts based on abstraction level
}


# ================================================================
# PATTERN → INDICATOR DSL CONVERSION PROMPT
# Used AFTER extraction, BEFORE DSL translation
# Specifically for Bulkowski/Candlestick pattern signals
# ================================================================

PATTERN_TO_INDICATOR_PROMPT = """You are converting a chart pattern description into indicator-based entry/exit conditions for a backtest engine.

PATTERN SIGNAL:
  Pattern name: {pattern_name}
  Entry conditions (text): {entry_conditions}
  Exit conditions (text): {exit_conditions}
  Direction: {direction}
  Pattern type: {pattern_type}
  Average rise: {avg_rise}
  Success rate: {success_rate}

AVAILABLE INDICATORS for the backtest engine:
  Price: open, high, low, close, volume
  Moving Averages: sma_5, sma_10, sma_20, sma_50, sma_100, sma_200
  Previous Bar: prev_close, prev_high, prev_low, prev_open, prev_volume
  Bollinger: bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth
  Donchian: dc_upper, dc_lower, dc_middle
  RSI: rsi_7, rsi_14, rsi_21
  Stochastic: stoch_k, stoch_d, stoch_k_5, stoch_d_5
  ADX: adx_14
  ATR: atr_7, atr_14, atr_20
  Volume: vol_ratio_20
  Volatility: hvol_6, hvol_20, hvol_100
  Price Position: price_pos_20 (0=low, 1=high of 20-day range)
  Returns: returns, log_returns
  Bar: body, body_pct, upper_wick, lower_wick, range
  Pivots: pivot, r1, s1, r2, s2

CONVERSION RULES:
  "Higher highs" → high > prev_high
  "Higher lows" → low > prev_low
  "Lower lows" → low < prev_low
  "Lower highs" → high < prev_high
  "Breakout above resistance" → close > dc_upper
  "Breakout below support" → close < dc_lower
  "Volume confirmation" → vol_ratio_20 > 1.5
  "Volume declining" → volume < prev_volume
  "Inside bar" → high < prev_high AND low > prev_low
  "Outside bar" → high > prev_high AND low < prev_low
  "Gap up" → open > prev_close
  "Gap down" → open < prev_close
  "Price above MA" → close > sma_50
  "Price below MA" → close < sma_50
  "Narrow range" → bb_bandwidth < 0.03
  "Wide range" → bb_bandwidth > 0.06
  "Overbought" → rsi_14 > 70
  "Oversold" → rsi_14 < 30
  "Strong trend" → adx_14 > 25
  "Weak trend" → adx_14 < 20

Convert the pattern to JSON with indicator conditions.
If the pattern CANNOT be expressed with these indicators, set untranslatable=true.

Return ONLY valid JSON:
{{
  "entry_long": [{{"left": "indicator", "operator": "op", "right": "value_or_indicator"}}],
  "entry_short": [],
  "exit_long": [{{"left": "indicator", "operator": "op", "right": "value_or_indicator"}}],
  "exit_short": [],
  "stop_loss_pct": {stop_loss},
  "take_profit_pct": {take_profit},
  "hold_days_max": 20,
  "direction": "{direction}",
  "target_regime": ["ANY"],
  "untranslatable": false,
  "untranslatable_reason": null,
  "translation_notes": "How the pattern was converted to indicators"
}}"""

