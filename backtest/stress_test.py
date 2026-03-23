"""
Stress test suite for portfolio robustness.

Runs historical scenario replays, synthetic shock injections,
and worst-case statistical analysis against a backtest equity curve.

Usage:
    python -m backtest.stress_test
"""

import copy
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DD_HALT_PCT = 0.15          # CompoundSizer halts at 15% drawdown
TIER_1_LOSS_PCT = 0.03      # DailyLossLimiter tier 1
TIER_2_LOSS_PCT = 0.05      # DailyLossLimiter tier 2
WEEKLY_LOSS_PCT = 0.12      # DailyLossLimiter weekly halt

HISTORICAL_SCENARIOS = {
    "Demonetization": ("2016-11-08", "2016-11-30"),
    "COVID Crash": ("2020-02-20", "2020-03-23"),
    "COVID Recovery": ("2020-03-24", "2020-06-30"),
    "Russia-Ukraine": ("2022-02-24", "2022-03-15"),
    "Adani Crisis": ("2023-01-25", "2023-02-28"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_drawdown_series(equity: pd.Series) -> pd.Series:
    """Running drawdown from peak as a fraction (negative values)."""
    peak = equity.cummax()
    return (equity - peak) / peak


def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown as a positive fraction."""
    dd = _max_drawdown_series(equity)
    return float(-dd.min()) if len(dd) > 0 else 0.0


def _days_to_recovery(equity: pd.Series, start_idx: int) -> Optional[int]:
    """Trading days from *start_idx* until equity exceeds prior peak."""
    if start_idx >= len(equity):
        return None
    peak = equity.iloc[:start_idx + 1].max()
    rest = equity.iloc[start_idx + 1:]
    recovered = rest[rest >= peak]
    if recovered.empty:
        return None  # never recovered within data
    return int((recovered.index[0] - equity.index[start_idx]).days)


def _equity_from_returns(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Cumulative equity curve from a daily return series."""
    return start * (1 + returns).cumprod()


# ---------------------------------------------------------------------------
# StressTest
# ---------------------------------------------------------------------------

class StressTest:
    """Portfolio stress testing against historical and synthetic scenarios."""

    def __init__(
        self,
        daily_returns: pd.Series,
        equity_curve: Optional[pd.Series] = None,
        trade_log: Optional[List[dict]] = None,
    ):
        self.daily_returns = daily_returns.copy()
        self.daily_returns.index = pd.to_datetime(self.daily_returns.index)
        self.daily_returns = self.daily_returns.sort_index()

        if equity_curve is not None:
            self.equity_curve = equity_curve.copy()
            self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
            self.equity_curve = self.equity_curve.sort_index()
        else:
            self.equity_curve = _equity_from_returns(self.daily_returns)

        self.trade_log = trade_log or []

    # ------------------------------------------------------------------
    # 1. Historical scenarios
    # ------------------------------------------------------------------

    def historical_scenarios(self) -> List[dict]:
        """Replay known market crises against the return series."""
        results = []
        data_start = self.daily_returns.index.min()
        data_end = self.daily_returns.index.max()

        for name, (s, e) in HISTORICAL_SCENARIOS.items():
            start = pd.Timestamp(s)
            end = pd.Timestamp(e)

            if start > data_end or end < data_start:
                results.append({
                    "scenario": name,
                    "start": s,
                    "end": e,
                    "skipped": True,
                    "note": "Period outside data range",
                })
                continue

            mask = (self.daily_returns.index >= start) & (
                self.daily_returns.index <= end
            )
            period_ret = self.daily_returns.loc[mask]

            if period_ret.empty:
                results.append({
                    "scenario": name,
                    "start": s,
                    "end": e,
                    "skipped": True,
                    "note": "No data in period",
                })
                continue

            period_equity = _equity_from_returns(period_ret)
            cum_pnl = float((1 + period_ret).prod() - 1)
            max_dd = _max_drawdown(period_equity)
            worst_day = float(period_ret.min())
            worst_day_date = str(period_ret.idxmin().date())

            # Recovery: days from end of stress period to a new equity high
            end_pos = self.equity_curve.index.searchsorted(end, side="right")
            recovery_days = _days_to_recovery(self.equity_curve, end_pos - 1)

            # Check circuit breaker triggers
            triggers = []
            if worst_day <= -TIER_1_LOSS_PCT:
                triggers.append(f"Tier1 ({TIER_1_LOSS_PCT:.0%})")
            if worst_day <= -TIER_2_LOSS_PCT:
                triggers.append(f"Tier2 ({TIER_2_LOSS_PCT:.0%})")
            weekly_ret = float(period_ret.iloc[:5].sum()) if len(period_ret) >= 5 else float(period_ret.sum())
            if weekly_ret <= -WEEKLY_LOSS_PCT:
                triggers.append(f"Weekly halt ({WEEKLY_LOSS_PCT:.0%})")
            if max_dd >= DD_HALT_PCT:
                triggers.append(f"DD halt ({DD_HALT_PCT:.0%})")

            results.append({
                "scenario": name,
                "start": s,
                "end": e,
                "skipped": False,
                "cumulative_pnl_pct": round(cum_pnl * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "worst_day_pct": round(worst_day * 100, 2),
                "worst_day_date": worst_day_date,
                "recovery_days": recovery_days,
                "trading_days": len(period_ret),
                "circuit_breakers_triggered": triggers or ["None"],
            })

        return results

    # ------------------------------------------------------------------
    # 2. Synthetic stress
    # ------------------------------------------------------------------

    def synthetic_stress(self) -> List[dict]:
        """Inject synthetic shocks and measure impact."""
        results = []
        base_equity = self.equity_curve.copy()
        n = len(self.daily_returns)

        scenarios = [
            {
                "name": "Single-day -5% gap",
                "description": "Insert a -5% return at a random date",
                "builder": self._build_single_day_gap,
            },
            {
                "name": "5-day sustained -2%/day",
                "description": "5 consecutive days of -2% returns",
                "builder": self._build_sustained_drop,
            },
            {
                "name": "Flash crash (-8% / +5%)",
                "description": "-8% one day followed by +5% recovery",
                "builder": self._build_flash_crash,
            },
            {
                "name": "VIX 40 spike (5 days at -3%)",
                "description": "5 days of -3% returns simulating VIX spike",
                "builder": self._build_vix_spike,
            },
            {
                "name": "Liquidity crunch (10 days 0.7x)",
                "description": "Returns dampened to 70% for 10 days (slippage)",
                "builder": self._build_liquidity_crunch,
            },
        ]

        for scenario in scenarios:
            shocked_returns = scenario["builder"]()
            shocked_equity = _equity_from_returns(shocked_returns)
            max_dd = _max_drawdown(shocked_equity)
            halt_triggered = max_dd >= DD_HALT_PCT
            total_impact = float((1 + shocked_returns).prod() - 1) - float(
                (1 + self.daily_returns).prod() - 1
            )

            results.append({
                "scenario": scenario["name"],
                "description": scenario["description"],
                "max_drawdown_pct": round(max_dd * 100, 2),
                "halt_triggered_15pct": halt_triggered,
                "capital_impact_pct": round(total_impact * 100, 2),
            })

        return results

    def _shock_start_idx(self, buffer: int = 20) -> int:
        """Pick a random insertion point with enough room."""
        n = len(self.daily_returns)
        lo = max(n // 4, buffer)
        hi = max(lo + 1, n - buffer)
        return random.randint(lo, hi)

    def _build_single_day_gap(self) -> pd.Series:
        ret = self.daily_returns.copy()
        idx = self._shock_start_idx()
        ret.iloc[idx] = -0.05
        return ret

    def _build_sustained_drop(self) -> pd.Series:
        ret = self.daily_returns.copy()
        idx = self._shock_start_idx(buffer=25)
        for i in range(5):
            if idx + i < len(ret):
                ret.iloc[idx + i] = -0.02
        return ret

    def _build_flash_crash(self) -> pd.Series:
        ret = self.daily_returns.copy()
        idx = self._shock_start_idx(buffer=25)
        ret.iloc[idx] = -0.08
        if idx + 1 < len(ret):
            ret.iloc[idx + 1] = 0.05
        return ret

    def _build_vix_spike(self) -> pd.Series:
        ret = self.daily_returns.copy()
        idx = self._shock_start_idx(buffer=25)
        for i in range(5):
            if idx + i < len(ret):
                ret.iloc[idx + i] = -0.03
        return ret

    def _build_liquidity_crunch(self) -> pd.Series:
        ret = self.daily_returns.copy()
        idx = self._shock_start_idx(buffer=30)
        for i in range(10):
            if idx + i < len(ret):
                ret.iloc[idx + i] *= 0.7
        return ret

    # ------------------------------------------------------------------
    # 3. Worst-case analysis
    # ------------------------------------------------------------------

    def worst_case_analysis(self) -> dict:
        """Statistical worst-case metrics from the daily return series."""
        ret = self.daily_returns.dropna()
        n = len(ret)

        # Rolling windows
        roll_5 = ret.rolling(5).sum()
        roll_21 = ret.rolling(21).sum()

        # Time underwater
        equity = self.equity_curve
        peak = equity.cummax()
        underwater = equity < peak
        if underwater.any():
            groups = (~underwater).cumsum()
            underwater_spans = underwater.groupby(groups).sum()
            max_underwater = int(underwater_spans.max())
        else:
            max_underwater = 0

        # CVaR
        sorted_ret = np.sort(ret.values)
        n_5 = max(int(np.floor(n * 0.05)), 1)
        n_1 = max(int(np.floor(n * 0.01)), 1)
        cvar_95 = float(sorted_ret[:n_5].mean())
        cvar_99 = float(sorted_ret[:n_1].mean())

        return {
            "worst_single_day_pct": round(float(ret.min()) * 100, 4),
            "worst_single_day_date": str(ret.idxmin().date()),
            "worst_5day_return_pct": round(float(roll_5.min()) * 100, 4),
            "worst_5day_end_date": str(roll_5.idxmin().date()) if not roll_5.isna().all() else None,
            "worst_week_5d_pct": round(float(roll_5.min()) * 100, 4),
            "worst_month_21d_pct": round(float(roll_21.min()) * 100, 4) if not roll_21.isna().all() else None,
            "worst_month_end_date": str(roll_21.idxmin().date()) if not roll_21.isna().all() else None,
            "max_time_underwater_days": max_underwater,
            "cvar_95_pct": round(cvar_95 * 100, 4),
            "cvar_99_pct": round(cvar_99 * 100, 4),
            "total_trading_days": n,
        }

    # ------------------------------------------------------------------
    # 4. Run all
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """Execute all three analyses and return combined results."""
        return {
            "historical_scenarios": self.historical_scenarios(),
            "synthetic_stress": self.synthetic_stress(),
            "worst_case_analysis": self.worst_case_analysis(),
        }

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(results: dict) -> None:
        """Pretty-print the stress test results to stdout."""
        sep = "=" * 72

        # --- Historical ---
        print(f"\n{sep}")
        print("  HISTORICAL SCENARIO ANALYSIS")
        print(sep)
        for s in results.get("historical_scenarios", []):
            if s.get("skipped"):
                print(f"\n  {s['scenario']:25s}  [{s['start']} - {s['end']}]  SKIPPED: {s.get('note')}")
                continue
            print(f"\n  {s['scenario']:25s}  [{s['start']} - {s['end']}]")
            print(f"    Cumulative P&L:        {s['cumulative_pnl_pct']:+.2f}%")
            print(f"    Max Drawdown:          {s['max_drawdown_pct']:.2f}%")
            print(f"    Worst Single Day:      {s['worst_day_pct']:+.2f}%  ({s['worst_day_date']})")
            rec = s['recovery_days']
            print(f"    Recovery Days:         {rec if rec is not None else 'Never (within data)'}")
            print(f"    Trading Days:          {s['trading_days']}")
            print(f"    Circuit Breakers:      {', '.join(s['circuit_breakers_triggered'])}")

        # --- Synthetic ---
        print(f"\n{sep}")
        print("  SYNTHETIC STRESS TESTS")
        print(sep)
        header = f"  {'Scenario':35s} {'Max DD%':>8s} {'Halt?':>6s} {'Impact%':>9s}"
        print(header)
        print("  " + "-" * 60)
        for s in results.get("synthetic_stress", []):
            halt = "YES" if s["halt_triggered_15pct"] else "no"
            print(
                f"  {s['scenario']:35s} "
                f"{s['max_drawdown_pct']:7.2f}% "
                f"{halt:>5s}  "
                f"{s['capital_impact_pct']:+8.2f}%"
            )

        # --- Worst case ---
        print(f"\n{sep}")
        print("  WORST-CASE ANALYSIS")
        print(sep)
        wc = results.get("worst_case_analysis", {})
        print(f"  Worst Single Day:        {wc.get('worst_single_day_pct', 0):+.4f}%  ({wc.get('worst_single_day_date', '?')})")
        print(f"  Worst 5-Day Return:      {wc.get('worst_5day_return_pct', 0):+.4f}%  (ending {wc.get('worst_5day_end_date', '?')})")
        print(f"  Worst Month (21d):       {wc.get('worst_month_21d_pct', 0):+.4f}%  (ending {wc.get('worst_month_end_date', '?')})")
        print(f"  Max Time Underwater:     {wc.get('max_time_underwater_days', 0)} trading days")
        print(f"  CVaR 95%:                {wc.get('cvar_95_pct', 0):+.4f}%")
        print(f"  CVaR 99%:                {wc.get('cvar_99_pct', 0):+.4f}%")
        print(f"  Total Trading Days:      {wc.get('total_trading_days', 0)}")
        print(sep)


# ---------------------------------------------------------------------------
# __main__ — load from nifty_daily and run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import psycopg2

    sys.path.insert(0, ".")
    from config.settings import DATABASE_DSN

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Connecting to database ...")
    conn = psycopg2.connect(DATABASE_DSN)

    query = "SELECT date, close FROM nifty_daily ORDER BY date"
    df = pd.read_sql(query, conn, parse_dates=["date"])
    conn.close()

    df = df.set_index("date").sort_index()
    df["return"] = df["close"].pct_change()
    daily_returns = df["return"].dropna()

    print(f"Loaded {len(daily_returns)} daily returns "
          f"({daily_returns.index.min().date()} to {daily_returns.index.max().date()})")

    random.seed(42)
    np.random.seed(42)

    st = StressTest(daily_returns)
    results = st.run_all()
    st.print_report(results)
