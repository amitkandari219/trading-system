"""
Slippage sensitivity analysis for trade-level returns.

Sweeps a range of round-trip slippage assumptions and recomputes key
performance metrics at each level.  Interpolates to find the breakeven
slippage where Sharpe crosses below 1.0.

Usage:
    from backtest.slippage_sensitivity import SlippageSensitivity
    ss = SlippageSensitivity()
    results = ss.run(trade_returns, avg_trade_return=0.35)
"""

import numpy as np
from typing import Dict, List, Optional


class SlippageSensitivity:
    """
    Sweep slippage levels and measure how portfolio metrics degrade.

    Parameters
    ----------
    slippage_levels : list of float
        Per-side slippage percentages.  0.05 means 0.05% per side,
        i.e. 0.10% round-trip cost subtracted from each trade.
    """

    SHARPE_THRESHOLD = 1.0  # breakeven target

    def __init__(
        self,
        slippage_levels: Optional[List[float]] = None,
    ):
        if slippage_levels is None:
            slippage_levels = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
        self.slippage_levels = sorted(slippage_levels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, trade_returns: np.ndarray, avg_trade_return: float) -> Dict:
        """
        Run slippage sensitivity sweep.

        Parameters
        ----------
        trade_returns : np.ndarray
            1-D array of per-trade return percentages (before additional
            slippage).  E.g. +2.3 means +2.3%.
        avg_trade_return : float
            Average per-trade return (%) for context / reporting.

        Returns
        -------
        dict with keys:
            results_by_slippage : list of dicts (one per slippage level)
            breakeven_slippage  : float or None (per-side %, where Sharpe
                                  drops below 1.0)
            safety_margin       : float or None (breakeven minus current
                                  assumed slippage of 0.05%)
        """
        trade_returns = np.asarray(trade_returns, dtype=np.float64)
        if len(trade_returns) < 2:
            raise ValueError("Need at least 2 trades for slippage analysis")

        results_by_slippage: List[Dict] = []

        for slip_pct in self.slippage_levels:
            # Round-trip cost: 2 * per-side slippage
            rt_cost = 2.0 * slip_pct
            adjusted = trade_returns - rt_cost
            metrics = self._compute_metrics(adjusted)
            metrics["slippage_per_side_pct"] = slip_pct
            metrics["round_trip_cost_pct"] = rt_cost
            results_by_slippage.append(metrics)

        # ---- Breakeven interpolation ------------------------------------
        sharpes = np.array([r["sharpe"] for r in results_by_slippage])
        slips = np.array(self.slippage_levels)
        breakeven = self._interpolate_breakeven(slips, sharpes, self.SHARPE_THRESHOLD)

        # Safety margin relative to a baseline of 0.05% per side
        baseline_slippage = 0.05
        safety_margin = None
        if breakeven is not None:
            safety_margin = breakeven - baseline_slippage

        return {
            "results_by_slippage": results_by_slippage,
            "breakeven_slippage": breakeven,
            "safety_margin": safety_margin,
            "avg_trade_return": avg_trade_return,
            "n_trades": len(trade_returns),
        }

    def format_results(self, results: Dict) -> str:
        """Return a human-readable summary table."""
        lines = []
        lines.append("=" * 72)
        lines.append("  SLIPPAGE SENSITIVITY ANALYSIS")
        lines.append("=" * 72)
        lines.append(f"  Trades: {results['n_trades']:,}  |  "
                      f"Avg trade return: {results['avg_trade_return']:.3f}%")
        lines.append("-" * 72)

        header = (f"  {'Slip/side':>10}  {'RT cost':>8}  {'Sharpe':>8}  "
                  f"{'CAGR':>8}  {'PF':>8}  {'MaxDD':>8}  {'WinRate':>8}")
        lines.append(header)
        lines.append("-" * 72)

        for r in results["results_by_slippage"]:
            lines.append(
                f"  {r['slippage_per_side_pct']:>9.2f}%  "
                f"{r['round_trip_cost_pct']:>7.2f}%  "
                f"{r['sharpe']:>8.3f}  "
                f"{r['cagr']:>7.2f}%  "
                f"{r['profit_factor']:>8.3f}  "
                f"{r['max_drawdown']:>7.2f}%  "
                f"{r['win_rate']:>7.1f}%"
            )

        lines.append("-" * 72)

        be = results["breakeven_slippage"]
        sm = results["safety_margin"]
        if be is not None:
            lines.append(f"  Breakeven slippage (Sharpe < 1.0): {be:.3f}% per side")
            lines.append(f"  Safety margin (vs 0.05% baseline): {sm:+.3f}%")
        else:
            lines.append("  Breakeven slippage: Sharpe remains above 1.0 at all levels tested")

        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(returns: np.ndarray) -> Dict:
        """Compute Sharpe, CAGR, PF, MaxDD, WinRate from trade returns (%)."""
        frac = returns / 100.0

        # Sharpe
        mu = np.mean(frac)
        sigma = np.std(frac, ddof=1)
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0.0

        # Equity curve & CAGR
        equity = np.cumprod(1.0 + frac)
        final = equity[-1]
        years = max(len(frac) / 252.0, 1e-9)
        cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0 else -100.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max * 100.0
        max_dd = float(np.min(dd))

        # Profit factor
        wins = frac[frac > 0]
        losses = frac[frac < 0]
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        if gross_loss == 0:
            pf = float("inf") if gross_profit > 0 else 0.0
        else:
            pf = float(gross_profit / gross_loss)

        # Win rate
        wr = float(np.sum(returns > 0) / len(returns) * 100.0)

        return {
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "max_drawdown": float(max_dd),
            "profit_factor": float(pf),
            "win_rate": float(wr),
        }

    @staticmethod
    def _interpolate_breakeven(
        slippage_levels: np.ndarray,
        sharpe_values: np.ndarray,
        threshold: float,
    ) -> Optional[float]:
        """
        Linear interpolation to find where Sharpe crosses below threshold.

        Returns per-side slippage (%) at the crossing, or None if Sharpe
        stays above threshold across all tested levels.
        """
        # Already below threshold at zero slippage
        if sharpe_values[0] < threshold:
            return 0.0

        for i in range(1, len(sharpe_values)):
            if sharpe_values[i] < threshold:
                # Linear interpolation between [i-1] and [i]
                s0, s1 = sharpe_values[i - 1], sharpe_values[i]
                x0, x1 = slippage_levels[i - 1], slippage_levels[i]
                # threshold = s0 + (s1 - s0) * (x - x0) / (x1 - x0)
                frac = (threshold - s0) / (s1 - s0) if s1 != s0 else 0.5
                return float(x0 + frac * (x1 - x0))

        return None  # never crossed


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("SlippageSensitivity -- self-test\n")

    rng = np.random.default_rng(123)

    # Simulate 300 trades: slight positive edge
    n = 300
    is_win = rng.random(n) < 0.55
    trade_returns = np.where(
        is_win,
        rng.normal(loc=1.5, scale=0.8, size=n),
        rng.normal(loc=-1.0, scale=0.6, size=n),
    )

    avg_ret = float(np.mean(trade_returns))
    print(f"Sample: {n} trades, avg return = {avg_ret:.3f}%\n")

    ss = SlippageSensitivity()
    results = ss.run(trade_returns, avg_trade_return=avg_ret)
    print(ss.format_results(results))

    # Sanity checks
    sharpes = [r["sharpe"] for r in results["results_by_slippage"]]
    assert sharpes[0] >= sharpes[-1], "Sharpe should decrease with more slippage"
    assert results["n_trades"] == n
    print("\nAll sanity checks passed.")
