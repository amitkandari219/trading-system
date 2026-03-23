"""
Trend Confirmation Filter — gates conviction-based amplification.

Before the ConvictionScorer is allowed to amplify position size above 1.0x,
the trend must be confirmed by multiple structural conditions. This prevents
sizing up into choppy, range-bound, or deteriorating markets.

Usage:
    from execution.trend_confirmer import TrendConfirmer
    confirmer = TrendConfirmer()

    trend = confirmer.is_trend_confirmed(
        close=24500, sma_20=24300, sma_50=24000,
        adx=25, high_52w=25000,
        recent_closes_5d=[(24100, 24050), (24200, 24150), ...]  # (close, open)
    )
    # trend['confirmed'] -> True

    gated = confirmer.gate_modifier(conviction_modifier=1.50, trend_confirmed=True)
    # gated -> 1.50  (amplification allowed)
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Trend confirmation thresholds
_ADX_THRESHOLD = 20
_GREEN_DAYS_REQUIRED = 3
_GREEN_DAYS_WINDOW = 5
_HIGH_52W_PCT = 0.05  # within 5% of 52-week high

# Gating cap when trend is NOT confirmed
_NO_TREND_CAP = 1.15


class TrendConfirmer:
    """
    Multi-condition trend confirmation filter.

    Five conditions must ALL be met for uptrend confirmation:
      1. close > sma_20
      2. sma_20 > sma_50
      3. adx > 20
      4. At least 3 of last 5 days were green (close > open)
      5. close within 5% of 52-week high
    """

    def __init__(self):
        # No persistent state needed — stateless filter
        pass

    def is_trend_confirmed(
        self,
        close: float,
        sma_20: float,
        sma_50: float,
        adx: float,
        high_52w: float,
        recent_closes_5d: List[Tuple[float, float]],
    ) -> Dict:
        """
        Check whether current market conditions confirm an uptrend.

        Args:
            close: current closing price
            sma_20: 20-period simple moving average
            sma_50: 50-period simple moving average
            adx: ADX(14) value
            high_52w: 52-week high price
            recent_closes_5d: list of (close, open) tuples for the last 5 days,
                              most recent first. Each tuple is (day_close, day_open).

        Returns:
            dict with keys:
              - confirmed (bool): True only if ALL 5 conditions met
              - conditions_met (int): count of conditions satisfied
              - conditions_total (int): always 5
              - details (list[str]): human-readable condition results
        """
        conditions: List[Tuple[bool, str]] = []

        # Condition 1: Price above short-term MA
        c1 = close > sma_20
        conditions.append((
            c1,
            f'close({close:.0f}) > sma_20({sma_20:.0f}): {"PASS" if c1 else "FAIL"}',
        ))

        # Condition 2: Short MA above long MA (golden cross structure)
        c2 = sma_20 > sma_50
        conditions.append((
            c2,
            f'sma_20({sma_20:.0f}) > sma_50({sma_50:.0f}): {"PASS" if c2 else "FAIL"}',
        ))

        # Condition 3: ADX shows trending market
        c3 = adx > _ADX_THRESHOLD
        conditions.append((
            c3,
            f'adx({adx:.1f}) > {_ADX_THRESHOLD}: {"PASS" if c3 else "FAIL"}',
        ))

        # Condition 4: Majority green days in last 5 sessions
        green_days = 0
        total_days = min(len(recent_closes_5d), _GREEN_DAYS_WINDOW)
        for day_close, day_open in recent_closes_5d[:_GREEN_DAYS_WINDOW]:
            if day_close > day_open:
                green_days += 1
        c4 = green_days >= _GREEN_DAYS_REQUIRED
        conditions.append((
            c4,
            f'green_days({green_days}/{total_days}) >= {_GREEN_DAYS_REQUIRED}: '
            f'{"PASS" if c4 else "FAIL"}',
        ))

        # Condition 5: Close within 5% of 52-week high
        if high_52w > 0:
            pct_from_high = (high_52w - close) / high_52w
            c5 = pct_from_high <= _HIGH_52W_PCT
        else:
            pct_from_high = 0.0
            c5 = False
        conditions.append((
            c5,
            f'close within {_HIGH_52W_PCT*100:.0f}% of 52w high({high_52w:.0f}): '
            f'{pct_from_high*100:.1f}% away — {"PASS" if c5 else "FAIL"}',
        ))

        # Aggregate
        met = sum(1 for passed, _ in conditions if passed)
        confirmed = met == len(conditions)

        details = [desc for _, desc in conditions]

        logger.debug(
            'TrendConfirmer: %d/%d conditions met, confirmed=%s',
            met, len(conditions), confirmed,
        )

        return {
            'confirmed': confirmed,
            'conditions_met': met,
            'conditions_total': len(conditions),
            'details': details,
        }

    def gate_modifier(
        self,
        conviction_modifier: float,
        trend_confirmed: bool,
    ) -> float:
        """
        Gate the conviction modifier based on trend confirmation.

        Rules:
          - If trend_confirmed AND modifier > 1.0: return modifier (allow amplification)
          - If NOT trend_confirmed: cap at 1.15 regardless of conviction
          - If modifier <= 1.0: return modifier (defensive sizing always allowed)

        Args:
            conviction_modifier: raw modifier from ConvictionScorer
            trend_confirmed: result from is_trend_confirmed()

        Returns:
            Gated modifier (float)
        """
        # Defensive sizing (reducing) is always allowed
        if conviction_modifier <= 1.0:
            return conviction_modifier

        # Amplification requires trend confirmation
        if trend_confirmed:
            return conviction_modifier

        # No trend confirmation: cap amplification
        capped = min(conviction_modifier, _NO_TREND_CAP)
        logger.debug(
            'TrendConfirmer gate: trend not confirmed, capping %.2f -> %.2f',
            conviction_modifier, capped,
        )
        return capped


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    confirmer = TrendConfirmer()

    print('=' * 70)
    print('TrendConfirmer Self-Test')
    print('=' * 70)

    # ── Test 1: All conditions met — strong uptrend ──
    recent_5d = [
        (24500, 24400),  # green
        (24400, 24350),  # green
        (24350, 24200),  # green
        (24200, 24250),  # red
        (24250, 24100),  # green
    ]
    result = confirmer.is_trend_confirmed(
        close=24500, sma_20=24300, sma_50=24000,
        adx=28, high_52w=25000,
        recent_closes_5d=recent_5d,
    )
    print('\nTest 1: Strong uptrend')
    print(f'  Confirmed:       {result["confirmed"]}')
    print(f'  Conditions met:  {result["conditions_met"]}/{result["conditions_total"]}')
    for d in result['details']:
        print(f'    {d}')
    assert result['confirmed'] is True
    assert result['conditions_met'] == 5

    # ── Test 2: Below SMA20 — trend broken ──
    result = confirmer.is_trend_confirmed(
        close=24100, sma_20=24300, sma_50=24000,
        adx=28, high_52w=25000,
        recent_closes_5d=recent_5d,
    )
    print('\nTest 2: Below SMA20')
    print(f'  Confirmed:       {result["confirmed"]}')
    print(f'  Conditions met:  {result["conditions_met"]}/{result["conditions_total"]}')
    assert result['confirmed'] is False

    # ── Test 3: Low ADX (ranging market) ──
    result = confirmer.is_trend_confirmed(
        close=24500, sma_20=24300, sma_50=24000,
        adx=15, high_52w=25000,
        recent_closes_5d=recent_5d,
    )
    print('\nTest 3: Low ADX (ranging)')
    print(f'  Confirmed:       {result["confirmed"]}')
    print(f'  Conditions met:  {result["conditions_met"]}/{result["conditions_total"]}')
    assert result['confirmed'] is False

    # ── Test 4: Too many red days ──
    red_5d = [
        (24100, 24200),  # red
        (24200, 24300),  # red
        (24300, 24400),  # red
        (24400, 24350),  # green
        (24350, 24300),  # green
    ]
    result = confirmer.is_trend_confirmed(
        close=24500, sma_20=24300, sma_50=24000,
        adx=28, high_52w=25000,
        recent_closes_5d=red_5d,
    )
    print('\nTest 4: Too many red days')
    print(f'  Confirmed:       {result["confirmed"]}')
    print(f'  Conditions met:  {result["conditions_met"]}/{result["conditions_total"]}')
    assert result['confirmed'] is False

    # ── Test 5: Far from 52-week high ──
    result = confirmer.is_trend_confirmed(
        close=22000, sma_20=21800, sma_50=21500,
        adx=28, high_52w=25000,
        recent_closes_5d=recent_5d,
    )
    print('\nTest 5: Far from 52w high (12% away)')
    print(f'  Confirmed:       {result["confirmed"]}')
    print(f'  Conditions met:  {result["conditions_met"]}/{result["conditions_total"]}')
    assert result['confirmed'] is False

    # ── Test 6: gate_modifier — trend confirmed, allow amplification ──
    gated = confirmer.gate_modifier(1.50, trend_confirmed=True)
    print(f'\nTest 6: gate_modifier(1.50, confirmed=True) -> {gated}')
    assert gated == 1.50

    # ── Test 7: gate_modifier — trend NOT confirmed, cap at 1.15 ──
    gated = confirmer.gate_modifier(1.50, trend_confirmed=False)
    print(f'Test 7: gate_modifier(1.50, confirmed=False) -> {gated}')
    assert gated == 1.15

    # ── Test 8: gate_modifier — defensive always allowed ──
    gated = confirmer.gate_modifier(0.70, trend_confirmed=False)
    print(f'Test 8: gate_modifier(0.70, confirmed=False) -> {gated}')
    assert gated == 0.70

    # ── Test 9: gate_modifier — modifier exactly 1.0 passes through ──
    gated = confirmer.gate_modifier(1.0, trend_confirmed=False)
    print(f'Test 9: gate_modifier(1.0, confirmed=False) -> {gated}')
    assert gated == 1.0

    # ── Test 10: gate_modifier — small amplification below cap ──
    gated = confirmer.gate_modifier(1.10, trend_confirmed=False)
    print(f'Test 10: gate_modifier(1.10, confirmed=False) -> {gated}')
    assert gated == 1.10

    # ── Test 11: Death cross (SMA20 < SMA50) ──
    result = confirmer.is_trend_confirmed(
        close=24500, sma_20=23900, sma_50=24000,
        adx=28, high_52w=25000,
        recent_closes_5d=recent_5d,
    )
    print(f'\nTest 11: Death cross (SMA20 < SMA50)')
    print(f'  Confirmed:       {result["confirmed"]}')
    assert result['confirmed'] is False

    print('\n' + '=' * 70)
    print('All TrendConfirmer tests passed.')
    print('=' * 70)
