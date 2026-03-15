"""
Signal Translator: converts approved signal text into structured backtest rules.

Uses Claude Haiku to parse natural language entry/exit conditions into
structured indicator-based rules that the generic backtest engine can execute.

Available indicators (column names in the DataFrame):
  sma_{5,10,20,50,100,200}, ema_{5,10,20,50,100,200}
  rsi_{7,14,21}, macd, macd_signal, macd_hist
  atr_{7,14,20}, true_range
  bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth
  stoch_k, stoch_d, adx_14
  dc_upper, dc_lower, dc_middle (Donchian 20)
  vol_ratio_20, hvol_20
  pivot, r1, s1, r2, s2
  price_pos_20 (0=low of range, 1=high)
  returns, log_returns, body, body_pct, upper_wick, lower_wick, range
  open, high, low, close, volume, india_vix (if available)

Operators: >, <, >=, <=, crosses_above, crosses_below

Usage:
    translator = SignalTranslator()
    rules = translator.translate_signal(signal_dict)
    # rules is a dict ready for generic_backtest.run_generic_backtest()
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from extraction.llm_client import LLMClient

logger = logging.getLogger(__name__)

TRANSLATE_PROMPT = """You are converting a trading signal's text-based entry/exit conditions into structured rules for a backtest engine.

SIGNAL:
  Rule: {rule_text}
  Category: {signal_category}
  Direction: {direction}
  Entry conditions: {entry_conditions}
  Exit conditions: {exit_conditions}
  Parameters: {parameters}
  Instrument: {instrument}
  Target regimes: {target_regimes}

AVAILABLE INDICATORS (exact column names you MUST use):
  Price: open, high, low, close, volume
  Moving Averages: sma_5, sma_10, sma_20, sma_50, sma_100, sma_200, ema_5, ema_10, ema_20, ema_50, ema_100, ema_200
  RSI: rsi_7, rsi_14, rsi_21
  MACD: macd, macd_signal, macd_hist
  ATR: atr_7, atr_14, atr_20, true_range
  Bollinger: bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth
  Stochastic: stoch_k, stoch_d
  ADX: adx_14
  Donchian: dc_upper, dc_lower, dc_middle
  Volume: vol_ratio_20 (current vol / 20-day avg)
  Volatility: hvol_20 (20-day historical vol), india_vix
  Pivots: pivot, r1, s1, r2, s2
  Price position: price_pos_20 (0=range low, 1=range high)
  Returns: returns, log_returns
  Bar: body, body_pct, upper_wick, lower_wick, range

OPERATORS: >, <, >=, <=, crosses_above, crosses_below

RULES:
1. Each condition is {{"indicator": "<column_name>", "op": "<operator>", "value": <number_or_column_name>}}
2. "value" can be a number (30, 0.02) or another column name ("sma_50")
3. Use the CLOSEST matching indicator. If the signal says "20-period MA", use "sma_20"
4. If direction is LONG-only, leave entry_short/exit_short empty
5. If direction is SHORT-only, leave entry_long/exit_long empty
6. If direction is CONTEXT_DEPENDENT or BOTH, provide both long and short rules
7. If exit conditions say AUTHOR_SILENT, use reasonable defaults (hold_days or a reversal condition)
8. stop_loss_pct should be 0.02 (2%) unless the signal specifies otherwise
9. For RISK/SIZING signals that aren't tradeable entries, set "backtestable": false

Return ONLY valid JSON:
{{
  "backtestable": true,
  "entry_long": [list of conditions],
  "entry_short": [list of conditions],
  "exit_long": [list of conditions],
  "exit_short": [list of conditions],
  "regime_filter": ["TRENDING_UP", ...] or [],
  "hold_days": 0,
  "stop_loss_pct": 0.02,
  "take_profit_pct": 0,
  "direction": "BOTH",
  "reason": "one sentence explaining the translation"
}}"""

# Categories that are not standalone entry/exit signals
NON_TRADEABLE_CATEGORIES = {
    'RISK', 'SIZING', 'PSYCHOLOGY', 'LABELING', 'PRICING',
    'ARBITRAGE', 'SENTIMENT', 'TIMING',
}


class SignalTranslator:
    """Translates approved text signals into structured backtest rules."""

    def __init__(self, num_workers: int = 6):
        self.llm = LLMClient()
        self.num_workers = num_workers

    def _is_backtestable(self, signal: dict) -> bool:
        """Quick pre-check: is this signal category backtestable?"""
        cat = signal.get('signal_category', '')
        # Check primary category (before any |)
        primary = cat.split('|')[0].strip()
        if primary in NON_TRADEABLE_CATEGORIES:
            return False
        # Check if entry conditions are too vague
        entries = signal.get('entry_conditions', [])
        if not entries:
            return False
        return True

    def _translate_one(self, signal: dict) -> dict:
        """Translate one signal using Haiku."""
        signal_id = signal.get('signal_id', 'unknown')

        # Quick filter for non-tradeable categories
        if not self._is_backtestable(signal):
            return {
                'signal_id': signal_id,
                'backtestable': False,
                'reason': f"Category {signal.get('signal_category')} is not a standalone entry signal",
                'rules': None,
            }

        prompt = TRANSLATE_PROMPT.format(
            rule_text=signal.get('rule_text', '')[:500],
            signal_category=signal.get('signal_category', ''),
            direction=signal.get('direction', ''),
            entry_conditions=json.dumps(signal.get('entry_conditions', []))[:800],
            exit_conditions=json.dumps(signal.get('exit_conditions', []))[:400],
            parameters=json.dumps(signal.get('parameters', {}))[:400],
            instrument=signal.get('instrument', ''),
            target_regimes=json.dumps(signal.get('target_regimes', [])),
        )

        try:
            result = self.llm._call_anthropic(prompt, 'claude-haiku-4-5-20251001')
            if result and isinstance(result, dict):
                return {
                    'signal_id': signal_id,
                    'backtestable': result.get('backtestable', True),
                    'reason': result.get('reason', ''),
                    'rules': result if result.get('backtestable', True) else None,
                }
            else:
                return {
                    'signal_id': signal_id,
                    'backtestable': False,
                    'reason': 'Translation failed (no valid response)',
                    'rules': None,
                }
        except Exception as e:
            logger.warning(f"Translation error for {signal_id}: {e}")
            return {
                'signal_id': signal_id,
                'backtestable': False,
                'reason': f'Translation error: {str(e)[:100]}',
                'rules': None,
            }

    def translate_batch(self, signals: List[dict],
                        output_path: Optional[str] = None) -> List[dict]:
        """Translate a batch of signals in parallel."""
        total = len(signals)
        print(f"\n--- Signal Translation: {total} signals, "
              f"{self.num_workers} workers ---", flush=True)

        # Pre-filter non-tradeable
        tradeable = []
        non_tradeable = []
        for s in signals:
            if self._is_backtestable(s):
                tradeable.append(s)
            else:
                non_tradeable.append({
                    'signal_id': s.get('signal_id', 'unknown'),
                    'backtestable': False,
                    'reason': f"Category {s.get('signal_category')} not backtestable",
                    'rules': None,
                })

        print(f"  Pre-filter: {len(tradeable)} tradeable, "
              f"{len(non_tradeable)} skipped (RISK/SIZING/etc)", flush=True)

        results = list(non_tradeable)
        done = 0
        start = time.time()
        stats = {'backtestable': 0, 'not_backtestable': 0, 'error': 0}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._translate_one, s): s
                       for s in tradeable}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if result.get('backtestable'):
                        stats['backtestable'] += 1
                    else:
                        stats['not_backtestable'] += 1
                except Exception as e:
                    stats['error'] += 1
                    signal = futures[future]
                    results.append({
                        'signal_id': signal.get('signal_id', 'unknown'),
                        'backtestable': False,
                        'reason': f'Error: {str(e)[:100]}',
                        'rules': None,
                    })

                done += 1
                if done % 100 == 0 or done == len(tradeable):
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  Translated: {done}/{len(tradeable)} ({rate:.1f}/s) | "
                          f"OK:{stats['backtestable']} Skip:{stats['not_backtestable']} "
                          f"Err:{stats['error']}", flush=True)

        print(f"\n  TRANSLATION COMPLETE", flush=True)
        print(f"  Total: {total}", flush=True)
        print(f"  Backtestable: {stats['backtestable']}", flush=True)
        print(f"  Not backtestable: {len(non_tradeable) + stats['not_backtestable']}",
              flush=True)
        print(f"  Errors: {stats['error']}", flush=True)

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved to: {output_path}", flush=True)

        return results

    def translate_signal(self, signal: dict) -> Optional[dict]:
        """Translate a single signal. Returns rules dict or None."""
        result = self._translate_one(signal)
        if result.get('backtestable') and result.get('rules'):
            return result['rules']
        return None
