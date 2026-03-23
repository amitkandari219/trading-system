"""
PCR (Put-Call Ratio) validator — validates PCR regimes and backtests
predictive power of extreme PCR for 5-day forward returns.

PCR regime mapping:
    > 1.5:     BULLISH (smart money hedging, contrarian bullish)
    1.0 - 1.5: NEUTRAL_BULLISH (mild put bias)
    0.7 - 1.0: COMPLACENT (low put demand, market complacent)
    < 0.7:     CONTRARIAN_BEARISH (extreme complacency, danger)

Integration with FII overlay:
    When FII direction and PCR regime align → stronger modifier.
    Example: FII bullish + PCR > 1.5 (smart money puts) → 1.2x boost.
    Example: FII bearish + PCR < 0.7 (complacent) → 0.5x reduce.

Backtest: does extreme PCR (top/bottom 10%) predict 5-day returns?

Usage:
    from data.pcr_validator import PCRValidator
    validator = PCRValidator(db_conn=conn)
    regime = validator.get_pcr_regime(date.today())
    alignment = validator.check_fii_alignment(date.today(), 'BULLISH')
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ================================================================
# PCR REGIME THRESHOLDS
# ================================================================

PCR_REGIMES = {
    'BULLISH': {
        'min_pcr': 1.5,
        'max_pcr': float('inf'),
        'description': 'High PCR — smart money hedging, contrarian bullish',
        'fii_alignment': {'BULLISH': 1.15, 'BEARISH': 0.85, 'NEUTRAL': 1.05},
    },
    'NEUTRAL_BULLISH': {
        'min_pcr': 1.0,
        'max_pcr': 1.5,
        'description': 'Mild put bias — normal hedging',
        'fii_alignment': {'BULLISH': 1.10, 'BEARISH': 0.90, 'NEUTRAL': 1.0},
    },
    'COMPLACENT': {
        'min_pcr': 0.7,
        'max_pcr': 1.0,
        'description': 'Low put demand — market complacent',
        'fii_alignment': {'BULLISH': 0.95, 'BEARISH': 1.05, 'NEUTRAL': 0.95},
    },
    'CONTRARIAN_BEARISH': {
        'min_pcr': 0.0,
        'max_pcr': 0.7,
        'description': 'Extreme complacency — contrarian bearish signal',
        'fii_alignment': {'BULLISH': 0.80, 'BEARISH': 1.15, 'NEUTRAL': 0.85},
    },
}


class PCRValidator:
    """
    PCR regime classification, FII alignment checking, and backtest validation.
    """

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self._pcr_cache: Dict[str, Dict] = {}

    # ================================================================
    # PUBLIC: get_pcr_regime
    # ================================================================

    def get_pcr_regime(self, as_of: date) -> Dict:
        """
        Get PCR regime for a given date.

        Returns dict with:
            date: date
            pcr_oi: float (raw PCR)
            regime: str ('BULLISH', 'NEUTRAL_BULLISH', 'COMPLACENT', 'CONTRARIAN_BEARISH')
            description: str
            pcr_zscore: float (from pcr_daily)
            pcr_5d_avg: float
        """
        cache_key = str(as_of)
        if cache_key in self._pcr_cache:
            return self._pcr_cache[cache_key]

        result = {
            'date': as_of,
            'pcr_oi': None,
            'regime': 'UNKNOWN',
            'description': 'No PCR data available',
            'pcr_zscore': None,
            'pcr_5d_avg': None,
        }

        if self.conn is None:
            return result

        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT pcr_oi, pcr_5d_avg, pcr_20d_avg, pcr_zscore
                FROM pcr_daily
                WHERE date <= %s
                ORDER BY date DESC LIMIT 1
            """, (as_of,))
            row = cur.fetchone()

            if row is None:
                return result

            pcr_oi = float(row[0]) if row[0] is not None else None
            pcr_5d = float(row[1]) if row[1] is not None else None
            pcr_zscore = float(row[3]) if row[3] is not None else None

            if pcr_oi is None:
                return result

            # Classify regime
            regime_name = 'UNKNOWN'
            for name, config in PCR_REGIMES.items():
                if config['min_pcr'] <= pcr_oi < config['max_pcr']:
                    regime_name = name
                    break

            regime_info = PCR_REGIMES.get(regime_name, {})

            result = {
                'date': as_of,
                'pcr_oi': round(pcr_oi, 3),
                'regime': regime_name,
                'description': regime_info.get('description', ''),
                'pcr_zscore': round(pcr_zscore, 2) if pcr_zscore is not None else None,
                'pcr_5d_avg': round(pcr_5d, 3) if pcr_5d is not None else None,
            }

            self._pcr_cache[cache_key] = result
            return result

        except Exception as e:
            logger.debug(f"PCR regime lookup failed: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return result

    # ================================================================
    # PUBLIC: compute_pcr
    # ================================================================

    @staticmethod
    def compute_pcr(total_put_oi: int, total_call_oi: int) -> float:
        """
        Compute PCR from raw options chain OI.

        Args:
            total_put_oi: sum of all put open interest
            total_call_oi: sum of all call open interest

        Returns:
            float: PCR value (put_oi / call_oi), or 1.0 if call_oi is 0
        """
        if total_call_oi <= 0:
            return 1.0
        return total_put_oi / total_call_oi

    # ================================================================
    # PUBLIC: check_fii_alignment
    # ================================================================

    def check_fii_alignment(
        self, as_of: date, fii_direction: str,
    ) -> Dict:
        """
        Check alignment between PCR regime and FII directional bias.

        Args:
            as_of: date to check
            fii_direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'

        Returns dict with:
            aligned: bool (True if FII and PCR agree)
            pcr_regime: str
            fii_direction: str
            modifier: float (sizing modifier from alignment)
            description: str
        """
        pcr_info = self.get_pcr_regime(as_of)
        pcr_regime = pcr_info['regime']

        if pcr_regime == 'UNKNOWN':
            return {
                'aligned': False,
                'pcr_regime': pcr_regime,
                'fii_direction': fii_direction,
                'modifier': 1.0,
                'description': 'No PCR data — neutral modifier',
            }

        regime_config = PCR_REGIMES.get(pcr_regime, {})
        alignment_map = regime_config.get('fii_alignment', {})
        modifier = alignment_map.get(fii_direction, 1.0)

        # Determine alignment
        pcr_direction = 'BULLISH' if pcr_regime in ('BULLISH', 'NEUTRAL_BULLISH') else 'BEARISH'
        aligned = (
            (fii_direction == 'BULLISH' and pcr_direction == 'BULLISH')
            or (fii_direction == 'BEARISH' and pcr_direction == 'BEARISH')
        )

        desc_parts = [
            f"PCR={pcr_info['pcr_oi']:.2f} ({pcr_regime})",
            f"FII={fii_direction}",
            'ALIGNED' if aligned else 'DIVERGENT',
            f"modifier={modifier:.2f}x",
        ]

        return {
            'aligned': aligned,
            'pcr_regime': pcr_regime,
            'fii_direction': fii_direction,
            'modifier': modifier,
            'description': ' | '.join(desc_parts),
        }

    # ================================================================
    # PUBLIC: backtest_extreme_pcr
    # ================================================================

    def backtest_extreme_pcr(
        self,
        percentile_threshold: float = 10.0,
        forward_days: int = 5,
    ) -> Dict:
        """
        Backtest: does extreme PCR predict forward returns?

        Splits PCR into top/bottom percentiles and measures
        average 5-day forward returns for each bucket.

        Returns dict with results and statistical significance.
        """
        if self.conn is None:
            return {'error': 'No DB connection'}

        try:
            df = pd.read_sql("""
                SELECT p.date, p.pcr_oi, n.close
                FROM pcr_daily p
                JOIN nifty_daily n ON p.date = n.date
                WHERE p.pcr_oi IS NOT NULL
                ORDER BY p.date
            """, self.conn)
        except Exception as e:
            return {'error': f'Query failed: {e}'}

        if len(df) < 50:
            return {'error': f'Insufficient data ({len(df)} rows)'}

        # Forward returns
        df['fwd_return'] = df['close'].shift(-forward_days) / df['close'] - 1

        # Remove rows without forward return
        df = df.dropna(subset=['fwd_return'])

        # Percentile thresholds
        low_threshold = np.percentile(df['pcr_oi'], percentile_threshold)
        high_threshold = np.percentile(df['pcr_oi'], 100 - percentile_threshold)

        low_pcr = df[df['pcr_oi'] <= low_threshold]
        high_pcr = df[df['pcr_oi'] >= high_threshold]
        middle = df[(df['pcr_oi'] > low_threshold) & (df['pcr_oi'] < high_threshold)]

        results = {
            'forward_days': forward_days,
            'percentile_threshold': percentile_threshold,
            'total_observations': len(df),
            'low_pcr': {
                'threshold': round(low_threshold, 3),
                'count': len(low_pcr),
                'avg_fwd_return': round(float(low_pcr['fwd_return'].mean()) * 100, 2),
                'median_fwd_return': round(float(low_pcr['fwd_return'].median()) * 100, 2),
                'win_rate': round(float((low_pcr['fwd_return'] > 0).mean()) * 100, 1),
                'interpretation': 'Low PCR = complacent → expect weakness',
            },
            'high_pcr': {
                'threshold': round(high_threshold, 3),
                'count': len(high_pcr),
                'avg_fwd_return': round(float(high_pcr['fwd_return'].mean()) * 100, 2),
                'median_fwd_return': round(float(high_pcr['fwd_return'].median()) * 100, 2),
                'win_rate': round(float((high_pcr['fwd_return'] > 0).mean()) * 100, 1),
                'interpretation': 'High PCR = smart money hedging → expect strength',
            },
            'middle': {
                'count': len(middle),
                'avg_fwd_return': round(float(middle['fwd_return'].mean()) * 100, 2),
            },
        }

        # Statistical significance (simple t-test)
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(
                high_pcr['fwd_return'].values,
                low_pcr['fwd_return'].values,
                equal_var=False,
            )
            results['t_test'] = {
                't_statistic': round(float(t_stat), 3),
                'p_value': round(float(p_value), 4),
                'significant': p_value < 0.05,
            }
        except ImportError:
            results['t_test'] = {'note': 'scipy not available for significance test'}
        except Exception:
            results['t_test'] = {'note': 'Insufficient data for t-test'}

        return results


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    import argparse
    import psycopg2
    from config.settings import DATABASE_DSN

    parser = argparse.ArgumentParser(description='PCR Validator and Backtest')
    parser.add_argument('--backtest', action='store_true', help='Run PCR backtest')
    parser.add_argument('--regime', action='store_true', help='Show current PCR regime')
    parser.add_argument('--date', type=str, default=None, help='Date (YYYY-MM-DD)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    conn = psycopg2.connect(DATABASE_DSN)
    validator = PCRValidator(db_conn=conn)

    if args.backtest:
        print("\nPCR Extreme Backtest (5-day forward returns):")
        print("=" * 60)
        result = validator.backtest_extreme_pcr()
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            for bucket in ('low_pcr', 'high_pcr', 'middle'):
                info = result[bucket]
                print(f"\n  {bucket.upper()} (n={info['count']}):")
                for k, v in info.items():
                    if k != 'count':
                        print(f"    {k}: {v}")
            if 't_test' in result:
                print(f"\n  T-test: {result['t_test']}")

    if args.regime:
        as_of = date.fromisoformat(args.date) if args.date else date.today()
        info = validator.get_pcr_regime(as_of)
        print(f"\nPCR Regime for {as_of}:")
        for k, v in info.items():
            print(f"  {k}: {v}")

    conn.close()
