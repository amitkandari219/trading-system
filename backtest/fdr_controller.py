"""
Benjamini-Hochberg False Discovery Rate controller.

Controls the expected proportion of false discoveries
in the confirmed signal pool. Complements individual
signal validation (DSR/walk-forward) with pool-level
statistical control.
"""

import numpy as np
from scipy import stats


class FDRController:

    def compute_p_value(self, sharpe: float, n_trades: int,
                        years: float = 10.0) -> float:
        """
        Compute p-value from walk-forward aggregate Sharpe.

        Under null hypothesis (no skill), Sharpe ~ N(0, 1/sqrt(T))
        where T = number of years of data.

        p = 1 - Phi(sharpe * sqrt(T))
        """
        if sharpe <= 0:
            return 1.0
        t_stat = sharpe * np.sqrt(years)
        return 1.0 - stats.norm.cdf(t_stat)

    def apply_bh_correction(self, signals: list,
                             alpha: float = 0.05) -> list:
        """
        Benjamini-Hochberg FDR correction.

        signals: list of dicts with 'signal_id', 'sharpe', 'p_value'
                 (p_value computed if not present)
        alpha: desired FDR level

        Returns signals with 'bh_accepted', 'bh_rank', 'bh_threshold' added.
        """
        m = len(signals)
        if m == 0:
            return signals

        # Compute p-values if not present
        for sig in signals:
            if 'p_value' not in sig:
                sig['p_value'] = self.compute_p_value(
                    sig.get('sharpe', 0),
                    sig.get('trades', 100),
                    sig.get('years', 10.0),
                )

        # Sort by p-value ascending
        sorted_sigs = sorted(signals, key=lambda x: x['p_value'])

        # Find BH critical value
        bh_threshold_index = 0
        for k, sig in enumerate(sorted_sigs, 1):
            threshold = (k / m) * alpha
            if sig['p_value'] <= threshold:
                bh_threshold_index = k

        # Mark accepted/rejected
        for k, sig in enumerate(sorted_sigs, 1):
            sig['bh_accepted'] = (k <= bh_threshold_index)
            sig['bh_rank'] = k
            sig['bh_threshold'] = round((k / m) * alpha, 6)

        accepted = sum(1 for s in sorted_sigs if s['bh_accepted'])
        expected_false = alpha * accepted if accepted > 0 else 0

        print(f"BH-FDR Results (alpha={alpha}):")
        print(f"  Total tested:    {m}")
        print(f"  Accepted:        {accepted}")
        print(f"  Rejected:        {m - accepted}")
        print(f"  Expected false:  {expected_false:.1f} of {accepted}")

        return sorted_sigs

    def compute_dsr(self, sharpe: float, n_trades: int,
                     skewness: float = 0.0, kurtosis: float = 3.0,
                     n_trials: int = 1) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado).

        Adjusts Sharpe ratio for:
        - Number of trials (multiple testing)
        - Non-normal returns (skewness, kurtosis)

        Returns: probability that the Sharpe is genuine (0-1)
        """
        if sharpe <= 0 or n_trades < 10:
            return 0.0

        # Expected maximum Sharpe under null (from n_trials)
        if n_trials > 1:
            # E[max(Z_1,...,Z_n)] ≈ (1-gamma) * Phi^{-1}(1 - 1/n) + gamma * Phi^{-1}(1 - 1/(n*e))
            # Simplified approximation:
            e_max_sharpe = stats.norm.ppf(1 - 1 / n_trials) * (1 / np.sqrt(n_trades))
        else:
            e_max_sharpe = 0

        # Adjust standard error for non-normality
        se = np.sqrt((1 + 0.5 * sharpe**2 - skewness * sharpe +
                      (kurtosis - 3) / 4 * sharpe**2) / n_trades)

        if se <= 0:
            return 0.0

        # DSR = Prob(SR > E[max SR] | H0)
        dsr_stat = (sharpe - e_max_sharpe) / se
        dsr = stats.norm.cdf(dsr_stat)

        return round(dsr, 4)

    def combined_acceptance(self, signals: list,
                             dsr_threshold: float = 0.90,
                             bh_alpha: float = 0.05,
                             n_trials: int = None) -> list:
        """
        Apply BOTH DSR and BH-FDR for maximum rigor.

        Tier A: DSR > 0.95 AND BH accepted
        Tier B: DSR > 0.90 AND BH accepted
        Ghost:  fails either test
        """
        if n_trials is None:
            n_trials = len(signals)

        # Compute DSR for each signal
        for sig in signals:
            sig['dsr'] = self.compute_dsr(
                sig.get('sharpe', 0),
                sig.get('trades', 100),
                sig.get('skewness', 0),
                sig.get('kurtosis', 3),
                n_trials,
            )

        # Apply BH
        signals = self.apply_bh_correction(signals, bh_alpha)

        # Combined tier
        for sig in signals:
            dsr = sig.get('dsr', 0)
            bh = sig.get('bh_accepted', False)

            if dsr > 0.95 and bh:
                sig['combined_tier'] = 'A'
            elif dsr > 0.90 and bh:
                sig['combined_tier'] = 'B'
            elif dsr > 0.85 and bh:
                sig['combined_tier'] = 'C'
            else:
                sig['combined_tier'] = 'GHOST'

        tiers = {}
        for sig in signals:
            t = sig['combined_tier']
            tiers[t] = tiers.get(t, 0) + 1

        print(f"\nCombined DSR + BH-FDR:")
        for tier in ['A', 'B', 'C', 'GHOST']:
            if tier in tiers:
                print(f"  Tier {tier}: {tiers[tier]} signals")

        return signals
