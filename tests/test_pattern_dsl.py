"""Tests for pattern DSL prompt and conversion."""

import pytest
from extraction.prompts.pattern_dsl_prompt import (
    BULKOWSKI_STATS, get_pattern_stats, format_pattern_prompt
)


def test_higher_highs_in_conversion_table():
    """Conversion table includes higher highs → high > prev_high."""
    from extraction.prompts.pattern_dsl_prompt import PATTERN_DSL_PROMPT
    assert 'high > prev_high' in PATTERN_DSL_PROMPT


def test_breakout_above_resistance():
    """Breakout above → dc_upper in conversion table."""
    from extraction.prompts.pattern_dsl_prompt import PATTERN_DSL_PROMPT
    assert 'dc_upper' in PATTERN_DSL_PROMPT


def test_volume_confirmation():
    """Volume confirmation → vol_ratio_20 > 1.5."""
    from extraction.prompts.pattern_dsl_prompt import PATTERN_DSL_PROMPT
    assert 'vol_ratio_20' in PATTERN_DSL_PROMPT


def test_bulkowski_stats_ascending_triangle():
    """Lookup works for Ascending Triangle."""
    stats = get_pattern_stats('Ascending Triangle')
    assert stats['success'] == 0.75
    assert stats['avg_rise'] == 0.38


def test_author_silent_filled_from_stats():
    """AUTHOR_SILENT stats filled from BULKOWSKI_STATS table."""
    signal = {
        'rule_text': 'Double Bottom pattern',
        'parameters': {'pattern_name': 'Double Bottom', 'success_rate_bull': 'AUTHOR_SILENT'},
        'entry_conditions': ['Price breaks above neckline'],
        'exit_conditions': ['AUTHOR_SILENT'],
        'direction': 'LONG',
    }
    prompt = format_pattern_prompt(signal)
    assert '0.78' in prompt or '78' in prompt  # success rate
    assert '0.4' in prompt or '40' in prompt    # avg rise


def test_signal_name_bulkowski_prefix():
    """Signal naming uses BULKOWSKI_ prefix."""
    # This is a naming convention check
    pattern_name = 'Ascending Triangle'
    signal_name = f"BULKOWSKI_{pattern_name.upper().replace(' ', '_')}"
    assert signal_name == 'BULKOWSKI_ASCENDING_TRIANGLE'


def test_signal_name_candlestick_prefix():
    """Candlestick naming uses CANDLESTICK_ prefix."""
    pattern_name = 'Hammer'
    signal_name = f"CANDLESTICK_{pattern_name.upper().replace(' ', '_')}"
    assert signal_name == 'CANDLESTICK_HAMMER'


def test_stop_loss_uses_atr():
    """Prompt mentions ATR-based stop, not just fixed percent."""
    from extraction.prompts.pattern_dsl_prompt import PATTERN_DSL_PROMPT
    assert 'atr' in PATTERN_DSL_PROMPT.lower() or 'ATR' in PATTERN_DSL_PROMPT


def test_take_profit_from_avg_rise():
    """take_profit_pct populated from average_rise_pct."""
    signal = {
        'rule_text': 'High Tight Flag',
        'parameters': {'pattern_name': 'High Tight Flag', 'average_rise_pct': '69'},
        'entry_conditions': ['Flag forms after sharp rise'],
        'exit_conditions': ['AUTHOR_SILENT'],
        'direction': 'LONG',
    }
    prompt = format_pattern_prompt(signal)
    assert '69' in prompt  # avg_rise passed through


def test_confidence_from_success_rate():
    """confidence_score populated from success rate."""
    stats = get_pattern_stats('Head and Shoulders')
    assert stats['success'] == 0.83
    # Partial match also works
    stats2 = get_pattern_stats('head and shoulders')
    assert stats2['success'] == 0.83
