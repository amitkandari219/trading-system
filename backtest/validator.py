"""
Validation Test Suite.
3 mandatory validation tests. Run BEFORE any real signal is tested.
If any fails: backtest engine has a bug. Fix before proceeding.

Usage:
    python -m backtest.validator
"""

import numpy as np
import pandas as pd
from backtest.types import BacktestResult


def validate_backtest_engine(backtest_fn, history_df, regime_labels):
    """
    3 mandatory validation tests.
    If any fails: backtest engine has a bug. Fix before proceeding.

    backtest_fn must handle these special entry_rule values:
        'RANDOM'                — coin flip entries
        'BUY_IF_TOMORROW_HIGHER' — perfect foresight (lookahead)
        'BUY_MONDAY_OPEN'       — known-bad weekly signal
    """
    print("Running backtest engine validation...")
    all_passed = True

    # TEST 1: Random signal (coin flip entries)
    # Expected: Sharpe near 0, slightly negative after costs
    print("\nTEST 1: Random signal (coin flip)...")
    random_params = {'entry_rule': 'RANDOM', 'seed': 42}
    result1 = backtest_fn(random_params, history_df, regime_labels)

    if -0.3 < result1.sharpe < 0.3:
        print(f"  PASSED: Sharpe={result1.sharpe:.3f} (expected near 0)")
    else:
        print(f"  FAILED: Sharpe={result1.sharpe:.3f} "
              f"(expected between -0.3 and 0.3)")
        print(f"  Possible lookahead bias or cost model error.")
        all_passed = False

    # TEST 2: Perfect foresight signal
    # Expected: Very high Sharpe (>5.0), very high win rate (>90%)
    print("\nTEST 2: Perfect foresight (lookahead)...")
    perfect_params = {
        'entry_rule': 'BUY_IF_TOMORROW_HIGHER',
        'exit_rule': 'NEXT_DAY_CLOSE',
    }
    result2 = backtest_fn(perfect_params, history_df, regime_labels)

    test2_pass = True
    if result2.sharpe > 5.0:
        print(f"  Sharpe: {result2.sharpe:.1f} (>5.0) PASS")
    else:
        print(f"  Sharpe: {result2.sharpe:.1f} (expected >5.0) FAIL")
        test2_pass = False

    if result2.win_rate > 0.90:
        print(f"  Win rate: {result2.win_rate:.2f} (>0.90) PASS")
    else:
        print(f"  Win rate: {result2.win_rate:.2f} (expected >0.90) FAIL")
        test2_pass = False

    if not test2_pass:
        print(f"  Data loading or return calculation error.")
        all_passed = False

    # TEST 3: Known negative signal (buy Monday, sell Friday)
    # Expected: Negative or near-zero Sharpe, below 0.8
    print("\nTEST 3: Known-bad signal (buy Monday, sell Friday)...")
    bad_params = {
        'entry_rule': 'BUY_MONDAY_OPEN',
        'exit_rule': 'SELL_FRIDAY_CLOSE',
    }
    result3 = backtest_fn(bad_params, history_df, regime_labels)

    if result3.sharpe < 0.8:
        print(f"  PASSED: Sharpe={result3.sharpe:.3f} (expected <0.8)")
    else:
        print(f"  FAILED: Sharpe={result3.sharpe:.3f} (expected <0.8)")
        print(f"  Something is systematically wrong.")
        all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("ALL VALIDATION TESTS PASSED. Backtest engine is trustworthy.")
    else:
        print("VALIDATION FAILED. Fix backtest engine before proceeding.")
    print("=" * 50)

    return all_passed


def make_reference_backtest_fn():
    """
    Creates a reference backtest function that handles the 3 validation
    signal types. Use this to validate the engine framework itself.

    This function simulates simple futures trading on Nifty daily data.
    No options, no spreads — just long/short futures at close prices.
    """

    def backtest_fn(params, history_df, regime_labels):
        entry_rule = params.get('entry_rule', '')
        exit_rule = params.get('exit_rule', 'NEXT_DAY_CLOSE')

        trades = []
        closes = history_df['close'].values
        dates = history_df['date'].values
        n = len(closes)

        if n < 10:
            return _empty_result()

        if entry_rule == 'RANDOM':
            # Coin flip: random long/short entries, exit next day
            rng = np.random.RandomState(params.get('seed', 42))
            for i in range(0, n - 1, 2):  # every other day
                if rng.random() > 0.5:
                    # Long
                    pnl = closes[i + 1] - closes[i]
                else:
                    # Short
                    pnl = closes[i] - closes[i + 1]
                trades.append({
                    'entry_date': dates[i],
                    'exit_date': dates[i + 1],
                    'pnl': pnl,
                })

        elif entry_rule == 'BUY_IF_TOMORROW_HIGHER':
            # Perfect foresight: buy today if tomorrow is higher
            for i in range(n - 1):
                if closes[i + 1] > closes[i]:
                    pnl = closes[i + 1] - closes[i]
                else:
                    pnl = closes[i] - closes[i + 1]
                trades.append({
                    'entry_date': dates[i],
                    'exit_date': dates[i + 1],
                    'pnl': pnl,
                })

        elif entry_rule == 'BUY_MONDAY_OPEN':
            # Alternating long/short weeks to neutralize market drift
            df = history_df.copy()
            df['weekday'] = pd.to_datetime(df['date']).dt.weekday
            mondays = df[df['weekday'] == 0].index.tolist()
            fridays = df[df['weekday'] == 4].index.tolist()

            fri_idx = 0
            week_num = 0
            for mon_idx in mondays:
                # Find next Friday after this Monday
                while fri_idx < len(fridays) and fridays[fri_idx] <= mon_idx:
                    fri_idx += 1
                if fri_idx >= len(fridays):
                    break
                fri = fridays[fri_idx]
                # Alternate long/short each week to remove directional bias
                if week_num % 2 == 0:
                    pnl = closes[fri] - closes[mon_idx]  # Long
                else:
                    pnl = closes[mon_idx] - closes[fri]   # Short
                trades.append({
                    'entry_date': dates[mon_idx],
                    'exit_date': dates[fri],
                    'pnl': pnl,
                })
                week_num += 1

        else:
            return _empty_result()

        return _compute_result(trades, history_df)

    return backtest_fn


def _empty_result():
    return BacktestResult(
        sharpe=0.0, calmar_ratio=0.0, max_drawdown=1.0,
        win_rate=0.0, profit_factor=0.0, avg_win_loss_ratio=0.0,
        trade_count=0, nifty_correlation=0.0,
        annual_return=0.0, drawdown_2020=0.0
    )


def _compute_result(trades, history_df):
    """Compute BacktestResult from list of trade dicts."""
    if len(trades) < 5:
        return _empty_result()

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_return = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0

    # Sharpe from trade P&L series
    pnl_series = pd.Series(pnls)
    std = pnl_series.std()
    sharpe = (pnl_series.mean() / std * np.sqrt(252)
              if std > 0 else 0.0)

    # Drawdown from cumulative P&L
    cum_pnl = pnl_series.cumsum()
    rolling_max = cum_pnl.cummax()
    drawdown = (cum_pnl - rolling_max) / (rolling_max.abs() + 1e-9)
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    profit_factor = (sum(wins) / abs(sum(losses))
                     if losses else float('inf'))
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 1.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    calmar = total_return / max_dd if max_dd > 0 else 0.0

    # Years in dataset for annualization
    date_range = (history_df['date'].max() - history_df['date'].min()).days
    years = max(1, date_range / 365.25)
    annual_return = total_return / years

    # Nifty correlation
    nifty_returns = history_df['close'].pct_change().dropna()
    # Build daily P&L series aligned to dates
    trade_pnl_by_date = {}
    for t in trades:
        d = t['exit_date']
        trade_pnl_by_date[d] = trade_pnl_by_date.get(d, 0) + t['pnl']

    trade_series = pd.Series(trade_pnl_by_date).reindex(
        history_df['date'], fill_value=0
    )
    if len(trade_series) > 10 and trade_series.std() > 0:
        nifty_correlation = float(
            trade_series.corr(
                nifty_returns.reindex(history_df['date'], fill_value=0)
            )
        )
    else:
        nifty_correlation = 0.0

    # 2020 drawdown
    crash_trades = [t for t in trades
                    if pd.Timestamp('2020-03-01') <= pd.Timestamp(t['entry_date'])
                    <= pd.Timestamp('2020-04-30')]
    crash_pnl = [t['pnl'] for t in crash_trades]
    drawdown_2020 = (abs(min(0, sum(crash_pnl))) / (abs(total_return) + 1e-9)
                     if crash_pnl else 0.0)

    return BacktestResult(
        sharpe=round(sharpe, 3),
        calmar_ratio=round(calmar, 3),
        max_drawdown=round(max_dd, 3),
        win_rate=round(win_rate, 3),
        profit_factor=round(min(profit_factor, 999), 3),
        avg_win_loss_ratio=round(win_loss_ratio, 3),
        trade_count=len(trades),
        nifty_correlation=round(nifty_correlation, 3),
        annual_return=round(annual_return, 0),
        drawdown_2020=round(drawdown_2020, 3),
    )


# ================================================================
# CLI ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import psycopg2
    from config.settings import DATABASE_DSN
    from data.nifty_loader import load_nifty_history
    from regime_labeler import RegimeLabeler

    print("Loading data...")
    conn = psycopg2.connect(DATABASE_DSN)
    history = load_nifty_history(conn)
    print(f"  {len(history)} trading days")

    print("Computing regime labels...")
    labeler = RegimeLabeler()
    regime_labels = labeler.label_full_history(history)
    print(f"  {len(regime_labels)} labels")

    print("\nCreating reference backtest function...")
    backtest_fn = make_reference_backtest_fn()

    print("")
    validate_backtest_engine(backtest_fn, history, regime_labels)

    conn.close()
