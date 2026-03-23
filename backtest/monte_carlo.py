"""
Monte Carlo permutation test for portfolio robustness validation.

Randomly shuffles per-trade returns 10,000 times, recomputes key metrics
each time, and reports the percentile rank of the real strategy vs the
permutation distribution.  A low p-value (< 0.05) means the observed
metric ordering is unlikely to arise by chance.

Usage:
    from backtest.monte_carlo import MonteCarloPermutationTest
    mc = MonteCarloPermutationTest(n_permutations=10_000, seed=42)
    results = mc.run(trade_returns)
"""

import sys
import time
from typing import Optional
import numpy as np


class MonteCarloPermutationTest:
    """
    Permutation test: shuffle the sequence of trade returns many times,
    rebuild the equity curve each time, and compare the real strategy's
    Sharpe / CAGR / MaxDD / PF against the permuted distributions.
    """

    def __init__(self, n_permutations: int = 10_000, seed: int = 42):
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(self, trade_returns: np.ndarray, equity_curve: Optional[np.ndarray] = None) -> dict:
        """
        Run the full permutation test.

        Parameters
        ----------
        trade_returns : 1-D numpy array of per-trade return percentages
                        (e.g. +2.3 means +2.3 %).
        equity_curve  : optional pre-computed equity curve (unused for
                        permutation logic but kept for API symmetry).

        Returns
        -------
        dict with keys:
            real_metrics        – dict of Sharpe, CAGR, MaxDD, PF for the
                                  actual trade sequence.
            perm_distributions  – dict mapping metric name -> 1-D array of
                                  permuted values (length n_permutations).
            percentiles         – dict mapping metric name -> percentile
                                  rank (0-100) of the real value inside the
                                  permutation distribution.
            p_values            – dict mapping metric name -> one-sided
                                  p-value (fraction of permuted values
                                  >= real for Sharpe/CAGR/PF, or <= real
                                  for MaxDD).
            n_permutations      – how many permutations were run.
            n_trades            – number of trades in the input array.
        """
        trade_returns = np.asarray(trade_returns, dtype=np.float64)
        n_trades = len(trade_returns)
        if n_trades < 2:
            raise ValueError(f"Need at least 2 trades, got {n_trades}")

        # ---- real metrics ------------------------------------------------
        real_metrics = self._compute_metrics(trade_returns)

        # ---- permutation distributions -----------------------------------
        perm_sharpe = np.empty(self.n_permutations, dtype=np.float64)
        perm_cagr = np.empty(self.n_permutations, dtype=np.float64)
        perm_maxdd = np.empty(self.n_permutations, dtype=np.float64)
        perm_pf = np.empty(self.n_permutations, dtype=np.float64)

        shuffled = trade_returns.copy()
        t0 = time.perf_counter()

        for i in range(self.n_permutations):
            self.rng.shuffle(shuffled)
            m = self._compute_metrics(shuffled)
            perm_sharpe[i] = m["sharpe"]
            perm_cagr[i] = m["cagr"]
            perm_maxdd[i] = m["max_drawdown"]
            perm_pf[i] = m["profit_factor"]

            if (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (self.n_permutations - i - 1) / rate
                print(
                    f"  permutation {i + 1:>6,} / {self.n_permutations:,}  "
                    f"({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)",
                    file=sys.stderr,
                )

        elapsed_total = time.perf_counter() - t0
        print(
            f"  completed {self.n_permutations:,} permutations in {elapsed_total:.1f}s",
            file=sys.stderr,
        )

        perm_distributions = {
            "sharpe": perm_sharpe,
            "cagr": perm_cagr,
            "max_drawdown": perm_maxdd,
            "profit_factor": perm_pf,
        }

        # ---- percentile ranks --------------------------------------------
        # For Sharpe/CAGR/PF: higher is better -> percentile = % of perms
        #   that are *less than* the real value.
        # For MaxDD: lower (less negative) is better -> percentile = % of
        #   perms that are *worse* (more negative) than real.
        percentiles = {
            "sharpe": float(np.mean(perm_sharpe < real_metrics["sharpe"]) * 100),
            "cagr": float(np.mean(perm_cagr < real_metrics["cagr"]) * 100),
            "max_drawdown": float(np.mean(perm_maxdd < real_metrics["max_drawdown"]) * 100),
            "profit_factor": float(np.mean(perm_pf < real_metrics["profit_factor"]) * 100),
        }

        # ---- p-values (one-sided) ----------------------------------------
        # p_value = P(random >= real) for "higher is better" metrics.
        # p_value = P(random <= real) for MaxDD (lower = less drawdown = better).
        p_values = {
            "sharpe": float(np.mean(perm_sharpe >= real_metrics["sharpe"])),
            "cagr": float(np.mean(perm_cagr >= real_metrics["cagr"])),
            "max_drawdown": float(np.mean(perm_maxdd <= real_metrics["max_drawdown"])),
            "profit_factor": float(np.mean(perm_pf >= real_metrics["profit_factor"])),
        }

        return {
            "real_metrics": real_metrics,
            "perm_distributions": perm_distributions,
            "percentiles": percentiles,
            "p_values": p_values,
            "n_permutations": self.n_permutations,
            "n_trades": n_trades,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(returns_array: np.ndarray) -> dict:
        """
        Compute strategy metrics from a 1-D array of per-trade return
        percentages.

        Parameters
        ----------
        returns_array : e.g. [+1.2, -0.5, +3.1, ...] where values are
                        percentage returns per trade.

        Returns
        -------
        dict with sharpe, cagr, max_drawdown, profit_factor.
        """
        frac_returns = returns_array / 100.0  # pct -> fractional

        # ---- Sharpe (annualised, assuming ~252 trading days) -------------
        mu = np.mean(frac_returns)
        sigma = np.std(frac_returns, ddof=1)
        if sigma == 0 or np.isnan(sigma):
            sharpe = 0.0
        else:
            sharpe = (mu / sigma) * np.sqrt(252)

        # ---- Equity curve & CAGR -----------------------------------------
        equity = np.cumprod(1.0 + frac_returns)
        final_equity = equity[-1]
        n_trades = len(frac_returns)

        # Approximate years: assume average holding ~1 trade per trading day
        # for the purpose of annualisation.  With ~252 trading days/year the
        # number of "years" spanned equals n_trades / 252.
        years = n_trades / 252.0
        if years <= 0 or final_equity <= 0:
            cagr = 0.0
        else:
            cagr = final_equity ** (1.0 / years) - 1.0

        # ---- Max drawdown (fraction, always <= 0) ------------------------
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max  # negative values
        max_drawdown = float(np.min(drawdowns))  # most negative = worst

        # ---- Profit factor -----------------------------------------------
        wins = frac_returns[frac_returns > 0]
        losses = frac_returns[frac_returns < 0]
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = float(gross_profit / gross_loss)

        return {
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(profit_factor),
        }

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    @staticmethod
    def format_results(results: dict) -> str:
        """Return a human-readable summary string."""
        lines = []
        lines.append("=" * 64)
        lines.append("  MONTE CARLO PERMUTATION TEST RESULTS")
        lines.append("=" * 64)
        lines.append(f"  Trades analysed : {results['n_trades']:,}")
        lines.append(f"  Permutations    : {results['n_permutations']:,}")
        lines.append("-" * 64)

        real = results["real_metrics"]
        pctl = results["percentiles"]
        pval = results["p_values"]
        dist = results["perm_distributions"]

        row_fmt = "  {metric:<16s}  {real:>10s}  {pctl:>8s}  {pval:>8s}  {mean:>10s}  {std:>10s}"
        lines.append(
            row_fmt.format(
                metric="Metric",
                real="Real",
                pctl="Pctile",
                pval="p-value",
                mean="Perm Mean",
                std="Perm Std",
            )
        )
        lines.append("-" * 64)

        for key, label, fmt in [
            ("sharpe", "Sharpe", ".3f"),
            ("cagr", "CAGR", ".4f"),
            ("max_drawdown", "Max Drawdown", ".4f"),
            ("profit_factor", "Profit Factor", ".3f"),
        ]:
            lines.append(
                row_fmt.format(
                    metric=label,
                    real=format(real[key], fmt),
                    pctl=f"{pctl[key]:.1f}%",
                    pval=format(pval[key], ".4f"),
                    mean=format(float(np.mean(dist[key])), fmt),
                    std=format(float(np.std(dist[key])), fmt),
                )
            )

        lines.append("-" * 64)

        # Overall verdict
        sharpe_p = pval["sharpe"]
        if sharpe_p < 0.01:
            verdict = "STRONG EVIDENCE of skill (p < 0.01)"
        elif sharpe_p < 0.05:
            verdict = "MODERATE EVIDENCE of skill (p < 0.05)"
        elif sharpe_p < 0.10:
            verdict = "WEAK EVIDENCE of skill (p < 0.10)"
        else:
            verdict = "NO EVIDENCE of skill (p >= 0.10) -- likely random"

        lines.append(f"  Verdict (Sharpe): {verdict}")
        lines.append("=" * 64)
        return "\n".join(lines)


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("Monte Carlo Permutation Test -- self-test\n")

    rng = np.random.default_rng(123)

    # Simulate 200 trades: slight positive edge
    # ~55% win rate, avg win +1.8%, avg loss -1.2%
    n = 200
    is_win = rng.random(n) < 0.55
    trade_returns = np.where(
        is_win,
        rng.normal(loc=1.8, scale=0.6, size=n),   # wins
        rng.normal(loc=-1.2, scale=0.5, size=n),   # losses
    )

    print(f"Sample trades: {n}")
    print(f"  wins  : {np.sum(trade_returns > 0)}")
    print(f"  losses: {np.sum(trade_returns <= 0)}")
    print(f"  mean  : {np.mean(trade_returns):+.3f}%")
    print(f"  std   : {np.std(trade_returns):.3f}%")
    print()

    mc = MonteCarloPermutationTest(n_permutations=10_000, seed=42)
    results = mc.run(trade_returns)

    print()
    print(MonteCarloPermutationTest.format_results(results))
