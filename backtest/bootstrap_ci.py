"""
Bootstrap confidence interval computation for trade-level metrics.

Draws bootstrap samples from per-trade returns and computes percentile-based
confidence intervals for Sharpe, CAGR, MaxDD, Profit Factor, and Win Rate.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class BootstrapCI:
    """
    Computes bootstrap confidence intervals for trade-level performance metrics.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples to draw.
    ci_levels : list of float
        Confidence interval levels (e.g. 0.90 means 90% CI).
    seed : int
        Random seed for reproducibility.
    """

    METRICS = ["sharpe", "cagr", "max_dd", "profit_factor", "win_rate"]

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        ci_levels: Optional[List[float]] = None,
        seed: int = 42,
    ):
        self.n_bootstrap = n_bootstrap
        self.ci_levels = ci_levels if ci_levels is not None else [0.90, 0.95, 0.99]
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, trade_returns: np.ndarray) -> Dict:
        """
        Compute bootstrap confidence intervals for all metrics.

        Parameters
        ----------
        trade_returns : np.ndarray
            1-D array of per-trade return percentages (e.g. 2.5 means +2.5%).

        Returns
        -------
        dict
            Keyed by metric name. Each value is a dict with:
              - 'point': point estimate from original sample
              - '{level*100:.0f}_ci': (lower, upper) tuple for each CI level
        """
        trade_returns = np.asarray(trade_returns, dtype=np.float64)
        n = len(trade_returns)

        if n < 2:
            raise ValueError("Need at least 2 trades for bootstrap CI computation")

        # Vectorised bootstrap: (n_bootstrap, n) index matrix
        idx = self._rng.integers(0, n, size=(self.n_bootstrap, n))
        boot_samples = trade_returns[idx]  # (n_bootstrap, n)

        # Compute each metric across all bootstrap samples at once
        metric_fns = {
            "sharpe": self._sharpe,
            "cagr": self._cagr,
            "max_dd": self._max_dd,
            "profit_factor": self._pf,
            "win_rate": self._win_rate,
        }

        results: Dict = {}
        for name, fn in metric_fns.items():
            # Point estimate from the original sample
            point = fn(trade_returns)

            # Bootstrap distribution
            boot_values = np.array([fn(boot_samples[i]) for i in range(self.n_bootstrap)])

            entry: Dict = {"point": point}
            for level in self.ci_levels:
                alpha = (1.0 - level) / 2.0
                lo = np.nanpercentile(boot_values, 100 * alpha)
                hi = np.nanpercentile(boot_values, 100 * (1.0 - alpha))
                key = f"{level * 100:.0f}_ci"
                entry[key] = (lo, hi)

            results[name] = entry

        return results

    def summary_table(self, results: Dict) -> str:
        """
        Format bootstrap CI results into a readable table.

        Parameters
        ----------
        results : dict
            Output from :meth:`compute`.

        Returns
        -------
        str
            Multi-line formatted table with point estimates and CI bounds.
        """
        lines: List[str] = []

        # Header
        ci_headers = [f"{lvl * 100:.0f}% CI" for lvl in self.ci_levels]
        header = f"{'Metric':<18} {'Point':>10}" + "".join(
            f"  {h:>22}" for h in ci_headers
        )
        lines.append(header)
        lines.append("-" * len(header))

        display_names = {
            "sharpe": "Sharpe",
            "cagr": "CAGR (%)",
            "max_dd": "Max DD (%)",
            "profit_factor": "Profit Factor",
            "win_rate": "Win Rate (%)",
        }

        fmt_val = {
            "sharpe": lambda v: f"{v:>10.3f}",
            "cagr": lambda v: f"{v:>10.2f}",
            "max_dd": lambda v: f"{v:>10.2f}",
            "profit_factor": lambda v: f"{v:>10.3f}",
            "win_rate": lambda v: f"{v:>10.2f}",
        }

        for metric in self.METRICS:
            entry = results[metric]
            dname = display_names[metric]
            fv = fmt_val[metric]

            row = f"{dname:<18} {fv(entry['point'])}"
            for level in self.ci_levels:
                key = f"{level * 100:.0f}_ci"
                lo, hi = entry[key]
                row += f"  [{fv(lo).strip():>9}, {fv(hi).strip():<9}]"
            lines.append(row)

        lines.append("-" * len(lines[0]))

        # Fragile edge warning: check if 5th percentile of Sharpe < 1.0
        sharpe_95_ci = results["sharpe"].get("90_ci")
        if sharpe_95_ci is not None and sharpe_95_ci[0] < 1.0:
            lines.append(
                "WARNING: 5th percentile Sharpe = {:.3f} < 1.0 — fragile edge".format(
                    sharpe_95_ci[0]
                )
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Metric helpers (static-ish, operate on 1-D arrays)
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Annualised Sharpe ratio: mean/std * sqrt(252)."""
        std = np.std(returns, ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(252))

    @staticmethod
    def _cagr(returns: np.ndarray) -> float:
        """
        CAGR from compounding per-trade returns.

        Assumes ~252 trades per year for annualisation.
        Returns percentage (e.g. 12.5 means 12.5%).
        """
        # Convert percentage returns to multipliers
        multipliers = 1.0 + returns / 100.0
        cumulative = np.prod(multipliers)
        if cumulative <= 0:
            return -100.0
        n_trades = len(returns)
        years = max(n_trades / 252.0, 1e-9)
        cagr = (cumulative ** (1.0 / years) - 1.0) * 100.0
        return float(cagr)

    @staticmethod
    def _max_dd(returns: np.ndarray) -> float:
        """
        Maximum drawdown from the equity curve built from per-trade returns.

        Returns a negative percentage (e.g. -15.3 means -15.3% drawdown).
        """
        multipliers = 1.0 + returns / 100.0
        equity = np.cumprod(multipliers)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100.0
        return float(np.min(drawdowns))

    @staticmethod
    def _pf(returns: np.ndarray) -> float:
        """Profit factor: sum of wins / abs(sum of losses)."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        sum_loss = np.abs(np.sum(losses))
        if sum_loss == 0:
            return float("inf") if np.sum(wins) > 0 else 0.0
        return float(np.sum(wins) / sum_loss)

    @staticmethod
    def _win_rate(returns: np.ndarray) -> float:
        """Win rate as percentage of trades with return > 0."""
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns) * 100.0)


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("BootstrapCI — self-test")
    print("=" * 72)

    # Simulate 200 trades: slight positive edge, fat tails
    rng = np.random.default_rng(123)
    sample_returns = np.concatenate([
        rng.normal(loc=0.15, scale=1.2, size=160),   # typical trades
        rng.normal(loc=-0.5, scale=3.0, size=20),    # loss tail
        rng.normal(loc=1.5, scale=2.0, size=20),     # win tail
    ])

    print(f"\nSample: {len(sample_returns)} trades")
    print(f"  Mean return:  {np.mean(sample_returns):.4f}%")
    print(f"  Std return:   {np.std(sample_returns):.4f}%")
    print(f"  Min / Max:    {np.min(sample_returns):.2f}% / {np.max(sample_returns):.2f}%")
    print()

    bci = BootstrapCI(n_bootstrap=10_000, ci_levels=[0.90, 0.95, 0.99], seed=42)
    results = bci.compute(sample_returns)

    table = bci.summary_table(results)
    print(table)
    print()

    # Quick sanity checks
    assert results["win_rate"]["point"] > 0, "Win rate should be positive"
    assert results["max_dd"]["point"] < 0, "Max DD should be negative"
    for metric in BootstrapCI.METRICS:
        for level in [0.90, 0.95, 0.99]:
            key = f"{level * 100:.0f}_ci"
            lo, hi = results[metric][key]
            assert lo <= hi, f"CI bounds inverted for {metric} {key}"

    print("All sanity checks passed.")
