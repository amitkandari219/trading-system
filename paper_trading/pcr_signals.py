"""
PCR (Put/Call Ratio) contrarian signals.

High PCR = too many puts = fear = contrarian buy opportunity.
Low PCR = too many calls = greed = contrarian sell opportunity.

NSE publishes options OI data daily. PCR is computed from
Nifty options chain: sum(Put OI) / sum(Call OI).
"""

import logging
from typing import List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PCRSignals:
    """PCR-based contrarian signals for Nifty."""

    # Empirical Nifty PCR thresholds
    EXTREME_FEAR_PCR = 1.5
    ELEVATED_FEAR_PCR = 1.3
    EXTREME_GREED_PCR = 0.7
    ELEVATED_GREED_PCR = 0.8
    ZSCORE_EXTREME = 1.5
    ZSCORE_ELEVATED = 1.0

    def compute(self, df: pd.DataFrame) -> List[Dict]:
        """
        Compute PCR contrarian signals from DataFrame.

        df must have columns: pcr_oi, pcr_zscore (optional)
        If pcr_zscore not present, computes from pcr_oi rolling stats.
        """
        signals = []

        if 'pcr_oi' not in df.columns:
            return signals

        pcr = df['pcr_oi'].iloc[-1]
        if pd.isna(pcr):
            return signals

        # Compute z-score if not present
        if 'pcr_zscore' in df.columns:
            zscore = df['pcr_zscore'].iloc[-1]
        else:
            pcr_series = df['pcr_oi'].dropna()
            if len(pcr_series) >= 20:
                mean = pcr_series.rolling(20).mean().iloc[-1]
                std = pcr_series.rolling(20).std().iloc[-1]
                zscore = (pcr - mean) / std if std > 0 else 0
            else:
                zscore = 0

        if pd.isna(zscore):
            zscore = 0

        # Extreme fear — contrarian buy
        if pcr > self.EXTREME_FEAR_PCR and zscore > self.ZSCORE_EXTREME:
            signals.append({
                'signal_id': 'PCR_EXTREME_FEAR',
                'direction': 'LONG',
                'confidence': 0.65,
                'source': 'PCR_CONTRARIAN',
                'stop_pct': 0.02,
                'hold_days': 5,
                'note': f'PCR={pcr:.2f} z={zscore:.1f} extreme fear',
            })

        # Elevated fear — mild buy
        elif pcr > self.ELEVATED_FEAR_PCR and zscore > self.ZSCORE_ELEVATED:
            signals.append({
                'signal_id': 'PCR_ELEVATED_FEAR',
                'direction': 'LONG',
                'confidence': 0.58,
                'source': 'PCR_CONTRARIAN',
                'stop_pct': 0.02,
                'hold_days': 5,
                'note': f'PCR={pcr:.2f} z={zscore:.1f} elevated fear',
            })

        # Extreme greed — contrarian sell
        if pcr < self.EXTREME_GREED_PCR and zscore < -self.ZSCORE_EXTREME:
            signals.append({
                'signal_id': 'PCR_EXTREME_GREED',
                'direction': 'SHORT',
                'confidence': 0.63,
                'source': 'PCR_CONTRARIAN',
                'stop_pct': 0.02,
                'hold_days': 5,
                'note': f'PCR={pcr:.2f} z={zscore:.1f} extreme greed',
            })

        # Elevated greed — mild sell
        elif pcr < self.ELEVATED_GREED_PCR and zscore < -self.ZSCORE_ELEVATED:
            signals.append({
                'signal_id': 'PCR_ELEVATED_GREED',
                'direction': 'SHORT',
                'confidence': 0.57,
                'source': 'PCR_CONTRARIAN',
                'stop_pct': 0.02,
                'hold_days': 5,
                'note': f'PCR={pcr:.2f} z={zscore:.1f} elevated greed',
            })

        return signals

    def get_daily_summary(self, df: pd.DataFrame) -> str:
        """One-line summary for Telegram digest."""
        if 'pcr_oi' not in df.columns:
            return "PCR: not loaded"

        pcr = df['pcr_oi'].iloc[-1]
        if pd.isna(pcr):
            return "PCR: no data"

        if pcr > self.EXTREME_FEAR_PCR:
            zone = "EXTREME FEAR"
        elif pcr > self.ELEVATED_FEAR_PCR:
            zone = "ELEVATED FEAR"
        elif pcr < self.EXTREME_GREED_PCR:
            zone = "EXTREME GREED"
        elif pcr < self.ELEVATED_GREED_PCR:
            zone = "ELEVATED GREED"
        else:
            zone = "NEUTRAL"

        return f"PCR: {pcr:.2f} ({zone})"
