"""
Shared dataclasses for the backtest engine.
All backtest modules import from here.
"""

from dataclasses import dataclass
from typing import Optional
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """
    Standardised return type for all backtest_fn calls.
    Signature: backtest_fn(params, history_df, regime_labels) -> BacktestResult
    """
    sharpe: float
    calmar_ratio: float
    max_drawdown: float           # fraction e.g. 0.18 = 18% drawdown
    win_rate: float               # fraction e.g. 0.45 = 45% winners
    profit_factor: float          # gross_profit / gross_loss
    avg_win_loss_ratio: float     # avg_win / avg_loss
    trade_count: int
    nifty_correlation: float      # Pearson corr of daily returns vs Nifty
    annual_return: float          # annualised net return

    # Required for OPTIONS_SELLING signals only.
    # Max drawdown during Mar-Apr 2020 sub-period.
    drawdown_2020: Optional[float] = None


def add_months(dt, months):
    """Add calendar months to a date. Handles month-end correctly."""
    return dt + relativedelta(months=months)


def add_trading_days(dt, n, calendar_df):
    """
    Add n trading days to dt using the market_calendar table.
    calendar_df: DataFrame with columns [date, is_trading_day].
    """
    # Normalize dt to comparable type
    dt = pd.Timestamp(dt)
    trading_days = calendar_df[
        (calendar_df['date'] > dt) &
        (calendar_df['is_trading_day'] == True)
    ]['date'].sort_values()

    if len(trading_days) < n:
        raise ValueError(f"Not enough future trading days in calendar for +{n} days")
    return trading_days.iloc[n - 1]


def subtract_trading_days(dt, n, calendar_df):
    """Subtract n trading days from dt using the market_calendar table."""
    dt = pd.Timestamp(dt)
    trading_days = calendar_df[
        (calendar_df['date'] < dt) &
        (calendar_df['is_trading_day'] == True)
    ]['date'].sort_values(ascending=False)

    if len(trading_days) < n:
        raise ValueError(f"Not enough prior trading days in calendar for -{n} days")
    return trading_days.iloc[n - 1]


def harmonic_mean_sharpe(window_results):
    """
    Harmonic mean of per-window Sharpe ratios.
    Preferred over arithmetic mean: penalises windows with very low Sharpe.
    """
    positive = [w['result'].sharpe for w in window_results
                if w['result'].sharpe > 0]
    if not positive:
        return 0.0
    return len(positive) / sum(1 / s for s in positive)
