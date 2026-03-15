"""
Generic backtest engine for structured signal rules.

Takes a structured signal definition (parsed from text by the translator)
and executes it against OHLCV data to produce a BacktestResult.

Signal rule format (JSON):
{
    "entry_long": [{"indicator": "rsi_14", "op": "<", "value": 30}],
    "entry_short": [{"indicator": "rsi_14", "op": ">", "value": 70}],
    "exit_long": [{"indicator": "rsi_14", "op": ">", "value": 50}],
    "exit_short": [{"indicator": "rsi_14", "op": "<", "value": 50}],
    "regime_filter": ["TRENDING_UP", "TRENDING_DOWN"],
    "hold_days": 5,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
    "direction": "BOTH"
}

Operators: ">", "<", ">=", "<=", "crosses_above", "crosses_below"
"""

import numpy as np
import pandas as pd
from backtest.types import BacktestResult
from backtest.indicators import add_all_indicators


def _empty_result():
    return BacktestResult(
        sharpe=0.0, calmar_ratio=0.0, max_drawdown=1.0,
        win_rate=0.0, profit_factor=0.0, avg_win_loss_ratio=0.0,
        trade_count=0, nifty_correlation=0.0,
        annual_return=0.0, drawdown_2020=0.0
    )


def _eval_condition(row, prev_row, cond: dict) -> bool:
    """Evaluate a single condition against a DataFrame row."""
    indicator = cond.get('indicator', '')
    op = cond.get('op', '>')
    value = cond.get('value')

    # Handle special indicator references
    if indicator == 'True':
        return True
    if indicator == 'False':
        return False

    # Get indicator value from row
    if indicator in row.index:
        ind_val = row[indicator]
    else:
        return False

    if pd.isna(ind_val):
        return False

    # Value can be a number or another indicator name
    if isinstance(value, str) and value in row.index:
        target = row[value]
        if pd.isna(target):
            return False
    elif value is None:
        return False
    else:
        target = float(value)

    # Comparison operators
    if op == '>':
        return ind_val > target
    elif op == '<':
        return ind_val < target
    elif op == '>=':
        return ind_val >= target
    elif op == '<=':
        return ind_val <= target
    elif op == '==':
        return abs(ind_val - target) < 1e-9
    elif op == 'crosses_above':
        if prev_row is None:
            return False
        prev_ind = prev_row.get(indicator, np.nan)
        if isinstance(value, str) and value in prev_row.index:
            prev_target = prev_row.get(value, np.nan)
        else:
            prev_target = target
        if pd.isna(prev_ind) or pd.isna(prev_target):
            return False
        return prev_ind <= prev_target and ind_val > target
    elif op == 'crosses_below':
        if prev_row is None:
            return False
        prev_ind = prev_row.get(indicator, np.nan)
        if isinstance(value, str) and value in prev_row.index:
            prev_target = prev_row.get(value, np.nan)
        else:
            prev_target = target
        if pd.isna(prev_ind) or pd.isna(prev_target):
            return False
        return prev_ind >= prev_target and ind_val < target
    return False


def _eval_conditions(row, prev_row, conditions: list) -> bool:
    """All conditions must be true (AND logic)."""
    if not conditions:
        return False
    return all(_eval_condition(row, prev_row, c) for c in conditions)


def run_generic_backtest(rules: dict, history_df: pd.DataFrame,
                         regime_labels: dict) -> BacktestResult:
    """
    Execute a structured signal rule against OHLCV data.

    Args:
        rules: Structured signal definition (see module docstring)
        history_df: OHLCV DataFrame with date, open, high, low, close, volume
        regime_labels: dict mapping date -> regime string

    Returns:
        BacktestResult with all metrics
    """
    if len(history_df) < 50:
        return _empty_result()

    # Add all indicators
    df = add_all_indicators(history_df)

    entry_long = rules.get('entry_long', [])
    entry_short = rules.get('entry_short', [])
    exit_long = rules.get('exit_long', [])
    exit_short = rules.get('exit_short', [])
    regime_filter = rules.get('regime_filter', [])
    hold_days = rules.get('hold_days', 0)
    stop_loss_pct = rules.get('stop_loss_pct', 0.02)
    take_profit_pct = rules.get('take_profit_pct', 0)
    direction = rules.get('direction', 'BOTH')
    cooldown_days = rules.get('cooldown_days', 1)

    # Handle validation signals
    entry_rule = rules.get('entry_rule', '')
    if entry_rule in ('RANDOM', 'BUY_IF_TOMORROW_HIGHER', 'BUY_MONDAY_OPEN'):
        from backtest.validator import make_reference_backtest_fn
        ref_fn = make_reference_backtest_fn()
        return ref_fn(rules, history_df, regime_labels)

    trades = []
    position = None  # None, 'LONG', or 'SHORT'
    entry_price = 0.0
    entry_idx = 0
    days_in_trade = 0
    last_exit_idx = -cooldown_days

    closes = df['close'].values
    dates = df['date'].values if 'date' in df.columns else df.index.values
    n = len(df)

    for i in range(1, n):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Regime filter
        if regime_filter:
            date_key = dates[i]
            # Try multiple date formats for regime lookup
            regime = None
            for fmt in [date_key, str(date_key)[:10],
                        pd.Timestamp(date_key)]:
                regime = regime_labels.get(fmt)
                if regime:
                    break
            if regime and regime not in regime_filter:
                # Still check if we need to exit
                if position is not None:
                    days_in_trade += 1
                continue

        if position is None:
            # Check for entries (with cooldown)
            if i - last_exit_idx < cooldown_days:
                continue

            # Long entry
            if direction in ('BOTH', 'LONG', 'CONTEXT_DEPENDENT'):
                if entry_long and _eval_conditions(row, prev_row, entry_long):
                    position = 'LONG'
                    entry_price = closes[i]
                    entry_idx = i
                    days_in_trade = 0
                    continue

            # Short entry
            if direction in ('BOTH', 'SHORT', 'CONTEXT_DEPENDENT'):
                if entry_short and _eval_conditions(row, prev_row, entry_short):
                    position = 'SHORT'
                    entry_price = closes[i]
                    entry_idx = i
                    days_in_trade = 0
                    continue

        else:
            days_in_trade += 1
            current_price = closes[i]

            # Stop loss
            if stop_loss_pct > 0:
                if position == 'LONG':
                    if (entry_price - current_price) / entry_price >= stop_loss_pct:
                        pnl = current_price - entry_price
                        trades.append({
                            'entry_date': dates[entry_idx],
                            'exit_date': dates[i],
                            'pnl': pnl,
                            'entry_price': entry_price,
                            'direction': position,
                            'exit_reason': 'stop_loss',
                        })
                        position = None
                        last_exit_idx = i
                        continue
                else:
                    if (current_price - entry_price) / entry_price >= stop_loss_pct:
                        pnl = entry_price - current_price
                        trades.append({
                            'entry_date': dates[entry_idx],
                            'exit_date': dates[i],
                            'pnl': pnl,
                            'entry_price': entry_price,
                            'direction': position,
                            'exit_reason': 'stop_loss',
                        })
                        position = None
                        last_exit_idx = i
                        continue

            # Take profit
            if take_profit_pct > 0:
                if position == 'LONG':
                    if (current_price - entry_price) / entry_price >= take_profit_pct:
                        pnl = current_price - entry_price
                        trades.append({
                            'entry_date': dates[entry_idx],
                            'exit_date': dates[i],
                            'pnl': pnl,
                            'entry_price': entry_price,
                            'direction': position,
                            'exit_reason': 'take_profit',
                        })
                        position = None
                        last_exit_idx = i
                        continue
                else:
                    if (entry_price - current_price) / entry_price >= take_profit_pct:
                        pnl = entry_price - current_price
                        trades.append({
                            'entry_date': dates[entry_idx],
                            'exit_date': dates[i],
                            'pnl': pnl,
                            'entry_price': entry_price,
                            'direction': position,
                            'exit_reason': 'take_profit',
                        })
                        position = None
                        last_exit_idx = i
                        continue

            # Hold days exit
            if hold_days > 0 and days_in_trade >= hold_days:
                if position == 'LONG':
                    pnl = current_price - entry_price
                else:
                    pnl = entry_price - current_price
                trades.append({
                    'entry_date': dates[entry_idx],
                    'exit_date': dates[i],
                    'pnl': pnl,
                    'entry_price': entry_price,
                    'direction': position,
                    'exit_reason': 'hold_days',
                })
                position = None
                last_exit_idx = i
                continue

            # Signal-based exit
            exit_conditions = exit_long if position == 'LONG' else exit_short
            if exit_conditions and _eval_conditions(row, prev_row, exit_conditions):
                if position == 'LONG':
                    pnl = current_price - entry_price
                else:
                    pnl = entry_price - current_price
                trades.append({
                    'entry_date': dates[entry_idx],
                    'exit_date': dates[i],
                    'pnl': pnl,
                    'entry_price': entry_price,
                    'direction': position,
                    'exit_reason': 'signal',
                })
                position = None
                last_exit_idx = i

    # Close any open position at end
    if position is not None:
        current_price = closes[-1]
        if position == 'LONG':
            pnl = current_price - entry_price
        else:
            pnl = entry_price - current_price
        trades.append({
            'entry_date': dates[entry_idx],
            'exit_date': dates[-1],
            'pnl': pnl,
            'entry_price': entry_price,
            'direction': position,
            'exit_reason': 'end_of_data',
        })

    return _compute_result(trades, history_df)


def _compute_result(trades: list, history_df: pd.DataFrame) -> BacktestResult:
    """Compute BacktestResult from trade list.

    PnL is in absolute Nifty points. We normalise to a notional capital
    equal to the first trade's entry price (i.e. 1 lot of Nifty futures).
    This makes drawdown, annual return, and Sharpe all percentage-based.
    """
    if len(trades) < 5:
        return _empty_result()

    # Notional capital = first entry price (roughly the Nifty level).
    # All PnL is expressed as a fraction of this notional.
    first_entry = trades[0].get('entry_price')
    if first_entry is None or first_entry == 0:
        # Fallback: use median close from history as notional
        first_entry = float(history_df['close'].median())
    notional = float(first_entry)

    pnls = [t['pnl'] for t in trades]
    # Convert absolute PnL to percentage returns
    pct_returns = [p / notional for p in pnls]

    wins_pct = [r for r in pct_returns if r > 0]
    losses_pct = [r for r in pct_returns if r < 0]

    win_rate = len(wins_pct) / len(pct_returns) if pct_returns else 0

    # Sharpe from per-trade percentage returns
    ret_series = pd.Series(pct_returns)
    std = ret_series.std()
    sharpe = (ret_series.mean() / std * np.sqrt(252)
              if std > 0 else 0.0)

    # Equity curve: start at 1.0, compound trade returns
    equity = [1.0]
    for r in pct_returns:
        equity.append(equity[-1] * (1.0 + r))
    equity_series = pd.Series(equity)
    rolling_peak = equity_series.cummax()
    drawdown_series = (equity_series - rolling_peak) / rolling_peak
    max_dd = abs(drawdown_series.min())

    # Profit factor (absolute PnL, ratio is scale-invariant)
    gross_wins = sum(w for w in pnls if w > 0)
    gross_losses = abs(sum(l for l in pnls if l < 0))
    profit_factor = (gross_wins / gross_losses
                     if gross_losses > 0 else float('inf'))

    avg_win = np.mean(wins_pct) if wins_pct else 0.0
    avg_loss = abs(np.mean(losses_pct)) if losses_pct else 1e-9
    win_loss_ratio = avg_win / avg_loss

    # Annual return from equity curve
    total_equity_return = equity[-1] / equity[0] - 1.0  # fractional
    date_range = (pd.Timestamp(history_df['date'].max())
                  - pd.Timestamp(history_df['date'].min())).days
    years = max(1, date_range / 365.25)
    # CAGR
    if equity[-1] > 0:
        annual_return = (equity[-1] ** (1.0 / years)) - 1.0
    else:
        annual_return = -1.0

    # Calmar = CAGR / max_drawdown
    calmar = annual_return / max_dd if max_dd > 0 else 0.0

    # Nifty correlation: daily strategy returns vs Nifty returns
    # Build daily return series from trades (pct of notional)
    trade_ret_by_date = {}
    for t in trades:
        d = t['exit_date']
        trade_ret_by_date[d] = trade_ret_by_date.get(d, 0) + t['pnl'] / notional

    nifty_returns = history_df.set_index('date')['close'].pct_change()
    strat_returns = pd.Series(trade_ret_by_date)

    # Align on common dates where strategy had trades
    common = strat_returns.index.intersection(nifty_returns.index)
    if len(common) > 10:
        s = strat_returns.reindex(common)
        n = nifty_returns.reindex(common)
        if s.std() > 0 and n.std() > 0:
            nifty_correlation = float(s.corr(n))
        else:
            nifty_correlation = 0.0
    else:
        nifty_correlation = 0.0

    if np.isnan(nifty_correlation):
        nifty_correlation = 0.0

    # 2020 drawdown: max drawdown of equity curve during Mar-Apr 2020
    drawdown_2020 = 0.0
    crash_trades = [t for t in trades
                    if pd.Timestamp('2020-03-01') <= pd.Timestamp(t['entry_date'])
                    <= pd.Timestamp('2020-04-30')]
    if crash_trades:
        crash_equity = [1.0]
        for t in crash_trades:
            r = t['pnl'] / notional
            crash_equity.append(crash_equity[-1] * (1.0 + r))
        crash_series = pd.Series(crash_equity)
        crash_peak = crash_series.cummax()
        crash_dd = (crash_series - crash_peak) / crash_peak
        drawdown_2020 = abs(crash_dd.min())

    return BacktestResult(
        sharpe=round(sharpe, 3),
        calmar_ratio=round(calmar, 3),
        max_drawdown=round(max_dd, 3),
        win_rate=round(win_rate, 3),
        profit_factor=round(min(profit_factor, 999), 3),
        avg_win_loss_ratio=round(win_loss_ratio, 3),
        trade_count=len(trades),
        nifty_correlation=round(nifty_correlation, 3),
        annual_return=round(annual_return, 4),
        drawdown_2020=round(drawdown_2020, 3),
    )


def make_generic_backtest_fn(rules: dict):
    """Factory: returns a backtest_fn(params, history_df, regime_labels) closure."""
    def backtest_fn(params, history_df, regime_labels):
        # Merge params into rules (params can override)
        merged = {**rules, **params}
        return run_generic_backtest(merged, history_df, regime_labels)
    return backtest_fn
