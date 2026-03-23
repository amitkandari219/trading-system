"""
Out-of-sample holdout validation.

The system's 6 SCORING signals were validated via walk-forward on
nifty_daily using 36mo train / 12mo test / 3mo step.  This module
tests on the most recent 6 months that were NOT in any WF test window,
providing a true unseen holdout.

Verdicts:
    ROBUST          – Sharpe degradation < 30%
    MODERATE_DECAY  – Sharpe degradation 30–60%
    OVERFITTING     – Sharpe degradation > 60%

Usage:
    from backtest.oos_holdout import OOSHoldout
    oos = OOSHoldout(holdout_start='2025-10-01')
    results = oos.run(trade_returns_is, trade_returns_oos)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class OOSHoldout:
    """
    True out-of-sample holdout test.

    Parameters
    ----------
    holdout_start : str
        ISO date string for the start of the holdout period.
        Default '2025-10-01' — the last ~6 months not in any WF test window.
    """

    # Degradation thresholds for verdict
    ROBUST_THRESHOLD = 0.30       # < 30% degradation
    MODERATE_THRESHOLD = 0.60     # 30–60% degradation

    METRICS = ["sharpe", "cagr", "max_drawdown", "profit_factor", "win_rate"]

    def __init__(self, holdout_start: str = "2025-10-01"):
        self.holdout_start = pd.Timestamp(holdout_start)

    # ------------------------------------------------------------------
    # Public API — trade-level
    # ------------------------------------------------------------------

    def run(
        self,
        trade_returns_is: np.ndarray,
        trade_returns_oos: np.ndarray,
    ) -> Dict:
        """
        Compare in-sample and out-of-sample trade-level metrics.

        Parameters
        ----------
        trade_returns_is : np.ndarray
            Per-trade return percentages from WF training period.
        trade_returns_oos : np.ndarray
            Per-trade return percentages from the holdout period.

        Returns
        -------
        dict with keys:
            is_metrics      – metrics computed on in-sample trades
            oos_metrics     – metrics computed on OOS trades
            degradation_pct – percentage degradation for each metric
            verdict         – ROBUST / MODERATE_DECAY / OVERFITTING
            holdout_start   – start date of holdout period
        """
        trade_returns_is = np.asarray(trade_returns_is, dtype=np.float64)
        trade_returns_oos = np.asarray(trade_returns_oos, dtype=np.float64)

        if len(trade_returns_is) < 10:
            raise ValueError(f"Need at least 10 IS trades, got {len(trade_returns_is)}")
        if len(trade_returns_oos) < 5:
            raise ValueError(f"Need at least 5 OOS trades, got {len(trade_returns_oos)}")

        is_metrics = self._compute_metrics(trade_returns_is)
        oos_metrics = self._compute_metrics(trade_returns_oos)
        degradation = self._compute_degradation(is_metrics, oos_metrics)
        verdict = self._determine_verdict(degradation)

        return {
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "degradation_pct": degradation,
            "verdict": verdict,
            "holdout_start": str(self.holdout_start.date()),
            "n_trades_is": len(trade_returns_is),
            "n_trades_oos": len(trade_returns_oos),
        }

    # ------------------------------------------------------------------
    # Public API — daily-returns convenience
    # ------------------------------------------------------------------

    def run_from_daily_returns(
        self,
        daily_returns_series: pd.Series,
        holdout_start: Optional[str] = None,
    ) -> Dict:
        """
        Split a daily returns Series at holdout_start and compare periods.

        Parameters
        ----------
        daily_returns_series : pd.Series
            Daily portfolio return percentages indexed by date.
        holdout_start : str, optional
            Override the instance holdout_start if provided.

        Returns
        -------
        dict — same schema as :meth:`run`, plus daily_metrics variants.
        """
        if holdout_start is not None:
            split_date = pd.Timestamp(holdout_start)
        else:
            split_date = self.holdout_start

        daily_returns_series = daily_returns_series.sort_index()
        is_returns = daily_returns_series[daily_returns_series.index < split_date]
        oos_returns = daily_returns_series[daily_returns_series.index >= split_date]

        if len(is_returns) < 20:
            raise ValueError(
                f"Need at least 20 IS daily returns, got {len(is_returns)}"
            )
        if len(oos_returns) < 10:
            raise ValueError(
                f"Need at least 10 OOS daily returns, got {len(oos_returns)}"
            )

        is_arr = is_returns.values.astype(np.float64)
        oos_arr = oos_returns.values.astype(np.float64)

        is_metrics = self._compute_daily_metrics(is_arr)
        oos_metrics = self._compute_daily_metrics(oos_arr)
        degradation = self._compute_degradation(is_metrics, oos_metrics)
        verdict = self._determine_verdict(degradation)

        return {
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "degradation_pct": degradation,
            "verdict": verdict,
            "holdout_start": str(split_date.date()),
            "n_days_is": len(is_returns),
            "n_days_oos": len(oos_returns),
            "is_period": f"{is_returns.index.min().date()} to {is_returns.index.max().date()}",
            "oos_period": f"{oos_returns.index.min().date()} to {oos_returns.index.max().date()}",
        }

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_results(self, results: Dict) -> str:
        """Return a human-readable summary."""
        lines = []
        lines.append("=" * 72)
        lines.append("  OUT-OF-SAMPLE HOLDOUT VALIDATION")
        lines.append("=" * 72)
        lines.append(f"  Holdout start: {results['holdout_start']}")

        if "n_trades_is" in results:
            lines.append(
                f"  IS trades: {results['n_trades_is']:,}  |  "
                f"OOS trades: {results['n_trades_oos']:,}"
            )
        else:
            lines.append(
                f"  IS period: {results.get('is_period', '?')}  "
                f"({results.get('n_days_is', '?')} days)"
            )
            lines.append(
                f"  OOS period: {results.get('oos_period', '?')}  "
                f"({results.get('n_days_oos', '?')} days)"
            )

        lines.append("-" * 72)
        header = f"  {'Metric':<18} {'IS':>10} {'OOS':>10} {'Degradation':>12}"
        lines.append(header)
        lines.append("-" * 72)

        is_m = results["is_metrics"]
        oos_m = results["oos_metrics"]
        deg = results["degradation_pct"]

        display = {
            "sharpe": ("Sharpe", ".3f"),
            "cagr": ("CAGR (%)", ".2f"),
            "max_drawdown": ("Max DD (%)", ".2f"),
            "profit_factor": ("Profit Factor", ".3f"),
            "win_rate": ("Win Rate (%)", ".1f"),
        }

        for key, (label, fmt) in display.items():
            is_val = format(is_m[key], fmt)
            oos_val = format(oos_m[key], fmt)
            deg_val = deg.get(key)
            deg_str = f"{deg_val:+.1f}%" if deg_val is not None else "n/a"
            lines.append(f"  {label:<18} {is_val:>10} {oos_val:>10} {deg_str:>12}")

        lines.append("-" * 72)
        verdict = results["verdict"]
        lines.append(f"  Verdict: {verdict}")
        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: metrics from trade-level returns (%)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(returns: np.ndarray) -> Dict:
        """Compute metrics from per-trade return percentages."""
        frac = returns / 100.0

        mu = np.mean(frac)
        sigma = np.std(frac, ddof=1)
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0.0

        equity = np.cumprod(1.0 + frac)
        final = equity[-1]
        years = max(len(frac) / 252.0, 1e-9)
        cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0 else -100.0

        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max * 100.0
        max_dd = float(np.min(dd))

        wins = frac[frac > 0]
        losses = frac[frac < 0]
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        if gross_loss == 0:
            pf = float("inf") if gross_profit > 0 else 0.0
        else:
            pf = float(gross_profit / gross_loss)

        wr = float(np.sum(returns > 0) / len(returns) * 100.0)

        return {
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "max_drawdown": float(max_dd),
            "profit_factor": float(pf),
            "win_rate": float(wr),
        }

    # ------------------------------------------------------------------
    # Internal: metrics from daily returns (%)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_daily_metrics(daily_returns: np.ndarray) -> Dict:
        """Compute metrics from daily return percentages."""
        frac = daily_returns / 100.0

        mu = np.mean(frac)
        sigma = np.std(frac, ddof=1)
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0.0

        equity = np.cumprod(1.0 + frac)
        final = equity[-1]
        n_days = len(frac)
        years = max(n_days / 252.0, 1e-9)
        cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0 else -100.0

        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max * 100.0
        max_dd = float(np.min(dd))

        # Profit factor from daily returns
        wins = frac[frac > 0]
        losses = frac[frac < 0]
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        if gross_loss == 0:
            pf = float("inf") if gross_profit > 0 else 0.0
        else:
            pf = float(gross_profit / gross_loss)

        wr = float(np.sum(daily_returns > 0) / len(daily_returns) * 100.0)

        return {
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "max_drawdown": float(max_dd),
            "profit_factor": float(pf),
            "win_rate": float(wr),
        }

    # ------------------------------------------------------------------
    # Internal: degradation computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_degradation(is_metrics: Dict, oos_metrics: Dict) -> Dict:
        """
        Compute percentage degradation from IS to OOS for each metric.

        For Sharpe / CAGR / PF / WinRate: degradation = (IS - OOS) / IS * 100
            Positive means OOS is worse.
        For MaxDD: degradation = (OOS - IS) / abs(IS) * 100
            Positive means OOS drawdown is deeper (worse).
        """
        deg = {}

        for key in ["sharpe", "cagr", "profit_factor", "win_rate"]:
            is_val = is_metrics[key]
            oos_val = oos_metrics[key]
            if abs(is_val) < 1e-9:
                deg[key] = None
            else:
                deg[key] = float((is_val - oos_val) / abs(is_val) * 100.0)

        # MaxDD: more negative OOS = worse
        is_dd = is_metrics["max_drawdown"]
        oos_dd = oos_metrics["max_drawdown"]
        if abs(is_dd) < 1e-9:
            deg["max_drawdown"] = None
        else:
            # OOS deeper drawdown -> positive degradation
            deg["max_drawdown"] = float((oos_dd - is_dd) / abs(is_dd) * 100.0)

        return deg

    def _determine_verdict(self, degradation: Dict) -> str:
        """
        Determine verdict based on Sharpe degradation.

        ROBUST:         < 30% degradation
        MODERATE_DECAY: 30–60% degradation
        OVERFITTING:    > 60% degradation
        """
        sharpe_deg = degradation.get("sharpe")

        if sharpe_deg is None:
            return "INCONCLUSIVE"

        if sharpe_deg < self.ROBUST_THRESHOLD * 100:
            return "ROBUST"
        elif sharpe_deg < self.MODERATE_THRESHOLD * 100:
            return "MODERATE_DECAY"
        else:
            return "OVERFITTING"


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("OOSHoldout -- self-test\n")

    rng = np.random.default_rng(42)

    # ---- Test 1: Trade-level API ------------------------------------
    print("Test 1: Trade-level run()")
    print("-" * 40)

    # In-sample: good edge
    n_is = 500
    is_win = rng.random(n_is) < 0.56
    trade_returns_is = np.where(
        is_win,
        rng.normal(loc=1.5, scale=0.7, size=n_is),
        rng.normal(loc=-1.0, scale=0.5, size=n_is),
    )

    # OOS: slightly degraded edge (realistic)
    n_oos = 120
    oos_win = rng.random(n_oos) < 0.52
    trade_returns_oos = np.where(
        oos_win,
        rng.normal(loc=1.3, scale=0.8, size=n_oos),
        rng.normal(loc=-1.1, scale=0.6, size=n_oos),
    )

    oos = OOSHoldout(holdout_start="2025-10-01")
    results = oos.run(trade_returns_is, trade_returns_oos)
    print(oos.format_results(results))
    print()

    # ---- Test 2: Daily returns API ----------------------------------
    print("Test 2: Daily returns run_from_daily_returns()")
    print("-" * 40)

    # Generate synthetic daily returns
    dates = pd.bdate_range("2023-01-01", "2026-03-15")
    daily_rets = pd.Series(
        rng.normal(loc=0.04, scale=1.1, size=len(dates)),
        index=dates,
    )

    results2 = oos.run_from_daily_returns(daily_rets)
    print(oos.format_results(results2))
    print()

    # ---- Sanity checks ----------------------------------------------
    assert results["verdict"] in ("ROBUST", "MODERATE_DECAY", "OVERFITTING", "INCONCLUSIVE")
    assert results2["verdict"] in ("ROBUST", "MODERATE_DECAY", "OVERFITTING", "INCONCLUSIVE")
    assert results["n_trades_is"] == n_is
    assert results["n_trades_oos"] == n_oos

    print("All sanity checks passed.")
