"""
Cross-asset overlay signals for Nifty trading system.

8 signals that modify position sizing based on global macro conditions.
Each returns a multiplier (0.5-1.5x) applied to all SCORING signals.
Multiple overlays stack multiplicatively, capped 0.2x-2.0x.
"""

from typing import Dict, Optional


class CrossAssetOverlays:
    """Computes cross-asset overlay multipliers from daily data."""

    def compute(self, today_data: Dict, prev_data: Dict = None) -> Dict:
        """
        Compute all 8 overlay signals.

        Args:
            today_data: dict of {instrument: {close, daily_return, weekly_return, ...}}
            prev_data: previous day's data (for US overnight)

        Returns:
            dict with per-signal multipliers + composite
        """
        signals = {}

        # 1. USDINR Weakness (tightened: 0.8% daily, 2% weekly)
        usdinr = today_data.get('USDINR', {})
        dr = usdinr.get('daily_return', 0) or 0
        wr = usdinr.get('weekly_return', 0) or 0
        if wr > 0.02:
            signals['USDINR_WEAKNESS'] = {'mult': 0.60, 'reason': f'INR weekly -{wr:.1%}'}
        elif dr > 0.008:
            signals['USDINR_WEAKNESS'] = {'mult': 0.75, 'reason': f'INR daily -{dr:.1%}'}
        elif dr < -0.008:
            signals['USDINR_WEAKNESS'] = {'mult': 1.15, 'reason': f'INR strengthened {-dr:.1%}'}
        else:
            signals['USDINR_WEAKNESS'] = {'mult': 1.0, 'reason': 'neutral'}

        # 2. Crude Spike (tightened: 4% daily, 10% weekly)
        wti = today_data.get('CRUDE_WTI', {})
        brent = today_data.get('CRUDE_BRENT', {})
        crude_dr = ((wti.get('daily_return', 0) or 0) + (brent.get('daily_return', 0) or 0)) / 2
        crude_wr = ((wti.get('weekly_return', 0) or 0) + (brent.get('weekly_return', 0) or 0)) / 2
        if crude_wr > 0.10:
            signals['CRUDE_SPIKE'] = {'mult': 0.55, 'reason': f'crude weekly +{crude_wr:.1%}'}
        elif crude_dr > 0.04:
            signals['CRUDE_SPIKE'] = {'mult': 0.70, 'reason': f'crude daily +{crude_dr:.1%}'}
        elif crude_dr < -0.04:
            signals['CRUDE_SPIKE'] = {'mult': 1.10, 'reason': f'crude drop {crude_dr:.1%}'}
        else:
            signals['CRUDE_SPIKE'] = {'mult': 1.0, 'reason': 'neutral'}

        # 3. Gold Risk-Off (tightened: 2.5% weekly)
        gold = today_data.get('GOLD', {})
        gold_wr = gold.get('weekly_return', 0) or 0
        if gold_wr > 0.025:
            signals['GOLD_RISK_OFF'] = {'mult': 0.85, 'reason': f'gold weekly +{gold_wr:.1%}'}
        else:
            signals['GOLD_RISK_OFF'] = {'mult': 1.0, 'reason': 'neutral'}

        # 4. US Overnight Signal (tightened: 1.5% SPX, VIX>30)
        us_data = prev_data or today_data
        sp500 = us_data.get('SP500', {})
        vix_us = us_data.get('VIX_US', {})
        sp_dr = sp500.get('daily_return', 0) or 0
        us_vix = vix_us.get('close', 15) or 15

        if us_vix > 30:
            signals['US_OVERNIGHT'] = {'mult': 0.70, 'reason': f'US VIX={us_vix:.0f}'}
        elif sp_dr < -0.015:
            signals['US_OVERNIGHT'] = {'mult': 0.80, 'reason': f'SP500 {sp_dr:.1%}'}
        elif sp_dr > 0.015:
            signals['US_OVERNIGHT'] = {'mult': 1.15, 'reason': f'SP500 +{sp_dr:.1%}'}
        else:
            signals['US_OVERNIGHT'] = {'mult': 1.0, 'reason': 'neutral'}

        # 5. Asia Sentiment (tightened: 1.5% avg move)
        nikkei = today_data.get('NIKKEI', {})
        hangseng = today_data.get('HANGSENG', {})
        asia_dr = ((nikkei.get('daily_return', 0) or 0) + (hangseng.get('daily_return', 0) or 0)) / 2
        if asia_dr > 0.015:
            signals['ASIA_SENTIMENT'] = {'mult': 1.10, 'reason': f'Asia avg +{asia_dr:.1%}'}
        elif asia_dr < -0.015:
            signals['ASIA_SENTIMENT'] = {'mult': 0.85, 'reason': f'Asia avg {asia_dr:.1%}'}
        else:
            signals['ASIA_SENTIMENT'] = {'mult': 1.0, 'reason': 'neutral'}

        # 6. Yield Curve Stress (unchanged — already tight)
        us10y = today_data.get('US10Y', {})
        yield_close = us10y.get('close', 3.0) or 3.0
        yield_wr = us10y.get('weekly_return', 0) or 0
        if yield_close > 4.8 and yield_wr > 0.005:
            signals['YIELD_STRESS'] = {'mult': 0.80, 'reason': f'10Y={yield_close:.1f}% rising'}
        else:
            signals['YIELD_STRESS'] = {'mult': 1.0, 'reason': 'neutral'}

        # 7. Dollar Strength (tightened: z60 > 2.0)
        usdinr_z60 = usdinr.get('zscore_60', 0) or 0
        if usdinr_z60 > 2.0:
            signals['DOLLAR_STRENGTH'] = {'mult': 0.75, 'reason': f'USDINR z60={usdinr_z60:.1f}'}
        else:
            signals['DOLLAR_STRENGTH'] = {'mult': 1.0, 'reason': 'neutral'}

        # 8. Composite Global Stress (tightened thresholds)
        stress_count = sum([
            1 if (dr > 0.008) else 0,       # 0.8% INR weakness
            1 if (crude_dr > 0.03) else 0,   # 3% crude spike
            1 if (us_vix > 28) else 0,       # US VIX > 28
            1 if (sp_dr < -0.008) else 0,    # 0.8% SP500 drop
            1 if (gold_wr > 0.015) else 0,   # 1.5% gold weekly
        ])
        stress_mults = {0: 1.0, 1: 1.0, 2: 0.90, 3: 0.70, 4: 0.50, 5: 0.30}
        signals['COMPOSITE_STRESS'] = {
            'mult': stress_mults.get(stress_count, 0.30),
            'reason': f'stress={stress_count}/5',
            'stress_count': stress_count,
        }

        # Compute composite multiplier
        composite = 1.0
        for sig in signals.values():
            composite *= sig['mult']
        composite = max(0.2, min(2.0, composite))

        return {
            'signals': signals,
            'composite_multiplier': round(composite, 3),
            'stress_count': stress_count,
        }

    def get_long_multiplier(self, state):
        """Get multiplier for LONG signals."""
        return state['composite_multiplier']

    def get_short_multiplier(self, state):
        """Inverse for SHORT — stress helps shorts."""
        cm = state['composite_multiplier']
        if cm < 1.0:
            return min(2.0, 2.0 - cm)  # stress boosts shorts
        return max(0.5, 2.0 - cm)      # calm reduces shorts
