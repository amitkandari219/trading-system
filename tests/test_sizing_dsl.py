"""Tests for Position Sizing DSL schema."""

import pytest
from extraction.dsl_schema_sizing import PositionSizingRule, ScaleCondition


def test_r_multiple_base_risk():
    """R_MULTIPLE sizing extracts base_risk_pct."""
    rule = PositionSizingRule(
        signal_id='THARP_001', sizing_method='R_MULTIPLE',
        base_risk_pct=0.01)
    assert rule.base_risk_pct == 0.01


def test_r_multiple_target():
    """r_multiple_target correctly set."""
    rule = PositionSizingRule(
        signal_id='THARP_002', sizing_method='R_MULTIPLE',
        base_risk_pct=0.01, r_multiple_target=3.0)
    # take_profit = stop × r_multiple_target
    take_profit = rule.r_multiple_stop * rule.r_multiple_target
    assert take_profit == 3.0


def test_anti_martingale_scale_up():
    """ANTI_MARTINGALE has scale_up_condition."""
    rule = PositionSizingRule(
        signal_id='THARP_003', sizing_method='ANTI_MARTINGALE',
        scale_up_condition=ScaleCondition(
            trigger='consecutive_wins', threshold=3, multiplier=1.25))
    assert rule.scale_up_condition is not None
    assert rule.scale_up_condition.trigger == 'consecutive_wins'
    assert rule.scale_up_condition.multiplier == 1.25


def test_optimal_f_value():
    """OPTIMAL_F has optimal_f value."""
    rule = PositionSizingRule(
        signal_id='VINCE_001', sizing_method='OPTIMAL_F',
        optimal_f=0.25)
    assert rule.optimal_f == 0.25


def test_half_kelly():
    """Half Kelly = optimal_f × 0.5."""
    optimal_f = 0.30
    half_kelly = optimal_f * 0.5
    rule = PositionSizingRule(
        signal_id='VINCE_002', sizing_method='KELLY',
        kelly_fraction=half_kelly)
    assert rule.kelly_fraction == 0.15


def test_scale_down_drawdown():
    """scale_down_condition has drawdown trigger."""
    rule = PositionSizingRule(
        signal_id='VINCE_003',
        scale_down_condition=ScaleCondition(
            trigger='drawdown_pct', threshold=0.10, multiplier=0.50))
    assert rule.scale_down_condition.trigger == 'drawdown_pct'
    assert rule.scale_down_condition.threshold == 0.10


def test_expectancy_high():
    """Expectancy HIGH when win_rate 60%+ with 2R target."""
    rule = PositionSizingRule(signal_id='TEST')
    exp = rule.compute_expectancy(win_rate=0.60, avg_win_r=2.0, avg_loss_r=1.0)
    # (0.60 × 2.0) - (0.40 × 1.0) = 1.2 - 0.4 = 0.8
    assert exp == pytest.approx(0.8, abs=0.01)
    assert exp > 0.5  # HIGH confidence threshold


def test_applies_to_all():
    """Portfolio-level rules apply to ALL."""
    rule = PositionSizingRule(
        signal_id='VINCE_PORTFOLIO', applies_to='ALL')
    assert rule.applies_to == 'ALL'
    d = rule.to_dict()
    assert d['applies_to'] == 'ALL'
