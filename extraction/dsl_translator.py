"""
DSL-based signal translator.

Replaces the free-form Haiku translator with a strict JSON template.
Haiku can ONLY use indicators from ALLOWED_INDICATORS and operators
from ALLOWED_OPERATORS. If it cannot express the rule within these
constraints, it must set untranslatable=true.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from extraction.dsl_schema import (
    DSLSignalRule, DSLCondition, DSLTranslationError,
    ALLOWED_INDICATORS, ALLOWED_OPERATORS, ALLOWED_REGIMES,
    INDICATOR_SET, OPERATOR_SET,
)
from extraction.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Categories that are not standalone entry/exit signals
NON_TRADEABLE_CATEGORIES = {
    'RISK', 'SIZING', 'PSYCHOLOGY', 'LABELING', 'PRICING',
    'ARBITRAGE', 'SENTIMENT', 'TIMING',
}

TRANSLATE_PROMPT = """You are converting a trading signal into a structured JSON rule for a backtest engine.

SIGNAL TO TRANSLATE:
  Signal ID: {signal_id}
  Rule: {rule_text}
  Category: {signal_category}
  Direction: {direction}
  Entry conditions: {entry_conditions}
  Exit conditions: {exit_conditions}
  Parameters: {parameters}
  Target regimes: {target_regimes}

ORIGINAL SOURCE TEXT (from the book):
{source_chunk}

ALLOWED INDICATORS (you MUST only use these exact names):
{indicators_list}

ALLOWED OPERATORS: >, <, >=, <=, ==, crosses_above, crosses_below, is

ALLOWED REGIMES: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOL, ANY

RULES:
1. Each condition has exactly 3 fields: "left", "operator", "right"
2. "left" MUST be one of the ALLOWED INDICATORS listed above
3. "right" can be an ALLOWED INDICATOR name OR a number (as a string like "30" or "0.02")
4. For regime comparisons use operator "is" with right being a regime string
5. Do NOT invent indicator names. If the signal requires an indicator not in the list, set untranslatable=true
6. Do NOT use array indexing like close[1] — use prev_close, prev_high, prev_low, prev_open instead
7. Each entry/exit list should have 1-4 conditions
8. If exit conditions are unknown (AUTHOR_SILENT), use a reasonable reversal condition
9. If direction is LONG, leave entry_short and exit_short empty
10. If direction is SHORT, leave entry_long and exit_long empty

SCALE NOTES (use correct scale — do NOT confuse percentage vs decimal):
  0-100 scale: rsi_7, rsi_9, rsi_14, rsi_21, stoch_k, stoch_d, stoch_k_5, stoch_d_5, adx_14
  0-1 scale (NOT percentage): bb_pct_b, price_pos_20, body_pct
  Decimal (0.02 = 2%): returns, log_returns, hvol_6, hvol_20, hvol_100, bb_bandwidth, stop_loss_pct
  Z-score (centered ~0, typically -3 to +3): zscore_20, zscore_50 (positive = above mean, negative = below)

WORKED EXAMPLE:
  Signal: "Buy when RSI is oversold and price is above the 50-day moving average"
  Output:
  {{
    "entry_long": [{{"left": "rsi_14", "operator": "<", "right": "30"}}, {{"left": "close", "operator": ">", "right": "sma_50"}}],
    "entry_short": [],
    "exit_long": [{{"left": "rsi_14", "operator": ">", "right": "70"}}],
    "exit_short": [],
    "entry_logic": "AND",
    "exit_logic": "OR",
    "stop_loss_pct": 2.0,
    "hold_days_max": 10,
    "direction": "LONG",
    "target_regime": ["TRENDING_UP", "RANGING"],
    "untranslatable": false,
    "untranslatable_reason": null,
    "translation_notes": "Used rsi_14 as closest match for oversold RSI"
  }}

Return ONLY valid JSON matching this exact schema. If you cannot express a condition using only these indicators and operators, set untranslatable=true with a clear reason. Do NOT invent indicator names."""

RETRANSLATE_PROMPT = """You previously translated this trading signal but used invalid indicator names. Retranslate using ONLY the allowed indicators below.

SIGNAL ID: {signal_id}
ORIGINAL BOOK TEXT:
{source_chunk}

PREVIOUS (BROKEN) TRANSLATION:
{broken_rule}

ISSUES WITH PREVIOUS TRANSLATION:
{issues}

ALLOWED INDICATORS (you MUST only use these exact names):
{indicators_list}

ALLOWED OPERATORS: >, <, >=, <=, ==, crosses_above, crosses_below, is

RULES:
1. Each condition has exactly 3 fields: "left", "operator", "right"
2. "left" MUST be one of the ALLOWED INDICATORS
3. "right" can be an ALLOWED INDICATOR name OR a number (as a string)
4. Do NOT use array indexing like close[1] — use prev_close, prev_high, prev_low instead
5. Do NOT invent indicator names
6. If the concept truly cannot be expressed with these indicators, set untranslatable=true

Return ONLY valid JSON with the same schema as the example above."""


def _format_indicators_list() -> str:
    """Format the allowed indicators as a readable list."""
    lines = []
    categories = {
        'Price': ['open', 'high', 'low', 'close', 'volume'],
        'Moving Averages': [i for i in ALLOWED_INDICATORS if i.startswith(('sma_', 'ema_'))],
        'RSI': [i for i in ALLOWED_INDICATORS if i.startswith('rsi_')],
        'ATR': [i for i in ALLOWED_INDICATORS if i.startswith('atr_')] + ['true_range'],
        'Bollinger (20p)': [i for i in ALLOWED_INDICATORS if i.startswith('bb_') and '_30' not in i],
        'Bollinger (30p)': [i for i in ALLOWED_INDICATORS if i.startswith('bb_') and '_30' in i],
        'MACD': ['macd', 'macd_signal', 'macd_hist'],
        'ADX': ['adx_14'],
        'Stochastic': [i for i in ALLOWED_INDICATORS if i.startswith('stoch_')],
        'Donchian': [i for i in ALLOWED_INDICATORS if i.startswith('dc_')],
        'Pivots': ['pivot', 'r1', 's1', 'r2', 's2'],
        'Volume/Vol': ['vol_ratio_20', 'hvol_20', 'india_vix'],
        'Position': ['price_pos_20'],
        'Z-score': ['zscore_20', 'zscore_50'],
        'Previous Bar': ['prev_close', 'prev_high', 'prev_low', 'prev_open', 'prev_volume'],
        'Returns': ['returns', 'log_returns'],
        'Bar Props': ['body', 'body_pct', 'upper_wick', 'lower_wick', 'range'],
    }
    for cat, indicators in categories.items():
        lines.append(f"  {cat}: {', '.join(indicators)}")
    return '\n'.join(lines)


INDICATORS_LIST_STR = _format_indicators_list()


class DSLTranslator:
    """Translates signal candidates into DSLSignalRule using strict JSON template."""

    def __init__(self, num_workers: int = 6):
        self.llm = LLMClient()
        self.num_workers = num_workers

    def _is_tradeable(self, signal: dict) -> bool:
        """Pre-check: is this signal category tradeable?"""
        book_id = signal.get('book_id', '')

        # THARP and VINCE sizing rules are always "tradeable" (for sizing DSL)
        if book_id in ('THARP', 'VINCE'):
            return True

        cat = signal.get('signal_category', '')
        primary = cat.split('|')[0].strip()
        if primary in NON_TRADEABLE_CATEGORIES:
            return False
        if not signal.get('entry_conditions'):
            return False
        return True

    def translate(self, signal: dict, source_chunk: str = '') -> DSLSignalRule:
        """
        Translate a signal candidate dict into a DSLSignalRule.
        Uses pattern-specific prompt for BULKOWSKI/CANDLESTICK books.
        """
        signal_id = signal.get('signal_id', 'unknown')
        book_id = signal.get('book_id', '')

        if not self._is_tradeable(signal):
            return DSLSignalRule(
                signal_id=signal_id,
                untranslatable=True,
                untranslatable_reason=f"Category {signal.get('signal_category')} is not tradeable",
            )

        # Use book-specific prompts
        if book_id in ('BULKOWSKI', 'CANDLESTICK'):
            prompt = self._build_pattern_prompt(signal)
        elif book_id in ('THARP', 'VINCE'):
            prompt = self._build_sizing_prompt(signal)
        else:
            prompt = TRANSLATE_PROMPT.format(
                signal_id=signal_id,
                rule_text=signal.get('rule_text', '')[:500],
                signal_category=signal.get('signal_category', ''),
                direction=signal.get('direction', ''),
                entry_conditions=json.dumps(signal.get('entry_conditions', []))[:800],
                exit_conditions=json.dumps(signal.get('exit_conditions', []))[:400],
                parameters=json.dumps(signal.get('parameters', {}))[:400],
                target_regimes=json.dumps(signal.get('target_regimes', [])),
                source_chunk=(source_chunk or signal.get('raw_chunk_text', ''))[:1500],
                indicators_list=INDICATORS_LIST_STR,
            )

        return self._call_and_parse(signal_id, prompt)

    def _build_pattern_prompt(self, signal: dict) -> str:
        """Build pattern-specific DSL prompt for Bulkowski/Candlestick signals."""
        from extraction.prompts.pattern_dsl_prompt import format_pattern_prompt
        return format_pattern_prompt(signal)

    def _build_sizing_prompt(self, signal: dict) -> str:
        """Build sizing-specific DSL prompt for Tharp/Vince signals."""
        from extraction.prompts.sizing_prompt import format_sizing_prompt
        return format_sizing_prompt(signal)

    def retranslate_fixable(self, signal_id: str, source_chunk: str,
                            broken_rule: dict, issues: list) -> DSLSignalRule:
        """
        Retranslate a previously FIXABLE signal using the DSL constraints.

        Args:
            signal_id: the signal ID
            source_chunk: original book text
            broken_rule: the previous (broken) translation rules dict
            issues: list of issue strings from the validator

        Returns:
            DSLSignalRule
        """
        prompt = RETRANSLATE_PROMPT.format(
            signal_id=signal_id,
            source_chunk=source_chunk[:1500],
            broken_rule=json.dumps(broken_rule, indent=2)[:1500],
            issues='\n'.join(f"  - {i}" for i in issues[:10]),
            indicators_list=INDICATORS_LIST_STR,
        )

        return self._call_and_parse(signal_id, prompt)

    def _call_and_parse(self, signal_id: str, prompt: str) -> DSLSignalRule:
        """Call LLM and parse response into DSLSignalRule."""
        try:
            result = self.llm._call_anthropic(prompt, 'claude-haiku-4-5-20251001')
        except Exception as e:
            logger.warning(f"LLM call failed for {signal_id}: {e}")
            return DSLSignalRule(
                signal_id=signal_id,
                untranslatable=True,
                untranslatable_reason=f"LLM call failed: {str(e)[:100]}",
            )

        if not result or not isinstance(result, dict):
            return DSLSignalRule(
                signal_id=signal_id,
                untranslatable=True,
                untranslatable_reason="LLM returned no valid JSON",
            )

        # Check if Haiku flagged it as untranslatable
        if result.get('untranslatable'):
            return DSLSignalRule(
                signal_id=signal_id,
                untranslatable=True,
                untranslatable_reason=result.get('untranslatable_reason', 'Unknown'),
                translation_notes=result.get('translation_notes'),
            )

        # Handle POSITION_SIZING responses (different schema)
        if result.get('signal_type') == 'POSITION_SIZING':
            # Store sizing rule as a DSLSignalRule with translation_notes
            return DSLSignalRule(
                signal_id=signal_id,
                translation_notes=json.dumps(result),
                # Mark as "translatable" so it gets saved to PASS
                # The sizing data is in translation_notes as JSON
            )

        # Parse conditions (standard entry/exit DSL)
        try:
            entry_long = [DSLCondition.from_dict(c) for c in result.get('entry_long', [])]
            entry_short = [DSLCondition.from_dict(c) for c in result.get('entry_short', [])]
            exit_long = [DSLCondition.from_dict(c) for c in result.get('exit_long', [])]
            exit_short = [DSLCondition.from_dict(c) for c in result.get('exit_short', [])]
        except (KeyError, TypeError) as e:
            return DSLSignalRule(
                signal_id=signal_id,
                untranslatable=True,
                untranslatable_reason=f"Failed to parse conditions: {e}",
            )

        # Clamp values to valid ranges
        stop_loss = result.get('stop_loss_pct', 2.0)
        if not isinstance(stop_loss, (int, float)):
            stop_loss = 2.0
        stop_loss = max(0.5, min(10.0, float(stop_loss)))

        hold_days = result.get('hold_days_max', 10)
        if not isinstance(hold_days, (int, float)):
            hold_days = 10
        hold_days = max(1, min(30, int(hold_days)))

        direction = result.get('direction', 'BOTH')
        if direction not in ('LONG', 'SHORT', 'BOTH'):
            direction = 'BOTH'

        target_regime = result.get('target_regime', ['ANY'])
        if not isinstance(target_regime, list):
            target_regime = ['ANY']

        entry_logic = result.get('entry_logic', 'AND')
        if entry_logic not in ('AND', 'OR'):
            entry_logic = 'AND'

        exit_logic = result.get('exit_logic', 'AND')
        if exit_logic not in ('AND', 'OR'):
            exit_logic = 'AND'

        return DSLSignalRule(
            signal_id=signal_id,
            entry_long=entry_long,
            entry_short=entry_short,
            exit_long=exit_long,
            exit_short=exit_short,
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            stop_loss_pct=stop_loss,
            hold_days_max=hold_days,
            direction=direction,
            target_regime=target_regime,
            translation_notes=result.get('translation_notes'),
        )

    def translate_batch(self, signals: List[dict],
                        source_chunks: dict = None) -> List[DSLSignalRule]:
        """
        Translate a batch of signals in parallel.

        Args:
            signals: list of signal dicts
            source_chunks: optional dict mapping signal_id -> source text

        Returns:
            list of DSLSignalRule
        """
        source_chunks = source_chunks or {}
        total = len(signals)
        print(f"\n--- DSL Translation: {total} signals, "
              f"{self.num_workers} workers ---", flush=True)

        # Pre-filter non-tradeable
        tradeable = []
        results = []
        for s in signals:
            if self._is_tradeable(s):
                tradeable.append(s)
            else:
                results.append(DSLSignalRule(
                    signal_id=s.get('signal_id', 'unknown'),
                    untranslatable=True,
                    untranslatable_reason=f"Category {s.get('signal_category')} not tradeable",
                ))

        print(f"  Pre-filter: {len(tradeable)} tradeable, "
              f"{len(results)} skipped", flush=True)

        done = 0
        start = time.time()
        stats = {'pass': 0, 'untranslatable': 0, 'error': 0}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for s in tradeable:
                sid = s.get('signal_id', 'unknown')
                chunk = source_chunks.get(sid, s.get('raw_chunk_text', ''))
                futures[executor.submit(self.translate, s, chunk)] = s

            for future in as_completed(futures):
                try:
                    rule = future.result()
                    results.append(rule)
                    if rule.untranslatable:
                        stats['untranslatable'] += 1
                    else:
                        stats['pass'] += 1
                except Exception as e:
                    signal = futures[future]
                    stats['error'] += 1
                    results.append(DSLSignalRule(
                        signal_id=signal.get('signal_id', 'unknown'),
                        untranslatable=True,
                        untranslatable_reason=f"Error: {str(e)[:100]}",
                    ))

                done += 1
                if done % 50 == 0 or done == len(tradeable):
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{len(tradeable)}] ({rate:.1f}/s) "
                          f"PASS:{stats['pass']} UNTRANS:{stats['untranslatable']} "
                          f"ERR:{stats['error']}", flush=True)

        print(f"\n  DSL TRANSLATION COMPLETE", flush=True)
        print(f"  Total: {total}", flush=True)
        print(f"  Translated: {stats['pass']}", flush=True)
        print(f"  Untranslatable: {len(results) - stats['pass'] - stats['error']}", flush=True)
        print(f"  Errors: {stats['error']}", flush=True)

        return results
