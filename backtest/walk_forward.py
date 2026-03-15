"""
Walk-Forward Engine.
Exact parameters: 36-month train / 12-month test / 3-month step.
Purge: 21 trading days before each test window.
Embargo: 5 trading days after each test window.

Pass criteria: >=75% of all windows AND the most recent window.
"""

import pandas as pd
from backtest.types import (
    BacktestResult, add_months, add_trading_days,
    subtract_trading_days, harmonic_mean_sharpe
)
from config.settings import (
    WF_TRAIN_MONTHS, WF_TEST_MONTHS, WF_STEP_MONTHS,
    WF_PURGE_DAYS, WF_EMBARGO_DAYS, WF_MIN_PASS_RATE
)


class WalkForwardEngine:
    """
    Exact walk-forward implementation.
    Train: 36 months. Test: 12 months. Step: 3 months.
    Purge: 21 trading days before each test window.
    Embargo: 5 trading days after each test window.
    """

    TRAIN_MONTHS = WF_TRAIN_MONTHS
    TEST_MONTHS = WF_TEST_MONTHS
    STEP_MONTHS = WF_STEP_MONTHS
    PURGE_DAYS = WF_PURGE_DAYS
    EMBARGO_DAYS = WF_EMBARGO_DAYS

    MIN_PASS_RATE = WF_MIN_PASS_RATE
    MUST_PASS_LAST_WINDOW = True

    # Differentiated pass criteria by signal type
    CRITERIA = {
        'FUTURES': {
            'min_sharpe': 1.2,
            'min_calmar': 0.8,
            'min_profit_factor': 1.6,
            'min_trades': 50,
        },
        'OPTIONS_BUYING': {
            'min_sharpe': 1.5,
            'min_win_rate': 0.40,
            'min_win_loss_ratio': 2.5,
            'min_trades': 50,
        },
        'OPTIONS_SELLING': {
            'min_sharpe': 1.8,
            'max_drawdown': 0.20,
            'must_survive_2020': True,
            'min_trades': 50,
        },
        'COMBINED': {
            'min_sharpe': 1.5,
            'max_nifty_correlation': 0.4,
            'min_trades': 50,
        },
    }

    def __init__(self, calendar_df):
        """
        calendar_df: DataFrame with columns [date, is_trading_day].
        Load from market_calendar table before instantiating.
        """
        self.calendar_df = calendar_df.copy()
        # Ensure date column is Timestamp for comparisons
        self.calendar_df['date'] = pd.to_datetime(self.calendar_df['date'])

    def run(self, signal_id, backtest_fn, history_df,
            regime_labels, params, signal_type):
        """
        Runs complete walk-forward analysis.
        Returns dict with per-window and aggregate metrics.
        """
        windows = self._generate_windows(history_df)

        if len(windows) < 4:
            return {
                'signal_id': signal_id,
                'overall_pass': False,
                'windows_passed': 0,
                'total_windows': len(windows),
                'pass_rate': 0.0,
                'last_window_passed': False,
                'window_details': [],
                'aggregate_sharpe': 0.0,
                'worst_window_drawdown': 0.0,
                'recommendation': 'ARCHIVE',
                'fail_reason': f'INSUFFICIENT_WINDOWS_{len(windows)}_NEED_4',
            }

        window_results = []

        for i, window in enumerate(windows):
            # Training data (with purge buffer excluded)
            train_data = history_df[
                (history_df['date'] >= window['train_start']) &
                (history_df['date'] < window['purge_start'])
            ]

            # Test data (after embargo buffer)
            test_data = history_df[
                (history_df['date'] >= window['test_start']) &
                (history_df['date'] <= window['test_end'])
            ]

            # Regime labels for test period only
            test_regimes = {
                d: regime_labels[d]
                for d in test_data['date']
                if d in regime_labels
            }

            # Run backtest on TEST data only
            result = backtest_fn(params, test_data, test_regimes)

            # Apply pass criteria by signal type
            passed = self._evaluate_window(result, signal_type, window)

            window_results.append({
                'window_index': i,
                'window': window,
                'result': result,
                'passed': passed,
                'trade_count': result.trade_count,
            })

        return self._aggregate_results(signal_id, window_results)

    def _generate_windows(self, history_df):
        """
        Generates all valid walk-forward windows.
        Minimum: 4 complete windows to proceed.
        """
        windows = []
        start_date = history_df['date'].min()
        end_date = history_df['date'].max()

        current = start_date
        while True:
            train_start = current
            train_end = add_months(train_start, self.TRAIN_MONTHS)

            try:
                purge_start = subtract_trading_days(
                    train_end, self.PURGE_DAYS, self.calendar_df
                )
                test_start = add_trading_days(
                    train_end, self.EMBARGO_DAYS, self.calendar_df
                )
            except ValueError:
                break  # Not enough calendar data

            test_end = add_months(test_start, self.TEST_MONTHS)

            if test_end > end_date:
                break

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'purge_start': purge_start,
                'test_start': test_start,
                'test_end': test_end,
            })

            current = add_months(current, self.STEP_MONTHS)

        return windows

    def _evaluate_window(self, result, signal_type, window):
        """
        Apply differentiated pass criteria by signal type.
        """
        c = self.CRITERIA.get(signal_type, self.CRITERIA['FUTURES'])

        # Minimum trades
        if result.trade_count < c.get('min_trades', 50):
            return False

        # Sharpe ratio — all types
        if result.sharpe < c.get('min_sharpe', 1.0):
            return False

        # Profit factor — FUTURES
        if 'min_profit_factor' in c:
            if result.profit_factor < c['min_profit_factor']:
                return False

        # Calmar ratio — FUTURES
        if 'min_calmar' in c:
            if result.calmar_ratio < c['min_calmar']:
                return False

        # Max drawdown — OPTIONS_SELLING
        if 'max_drawdown' in c:
            if result.max_drawdown > c['max_drawdown']:
                return False

        # Win rate — OPTIONS_BUYING
        if 'min_win_rate' in c:
            if result.win_rate < c['min_win_rate']:
                return False

        # Win/loss ratio — OPTIONS_BUYING
        if 'min_win_loss_ratio' in c:
            if result.avg_win_loss_ratio < c['min_win_loss_ratio']:
                return False

        # Nifty correlation — COMBINED
        if 'max_nifty_correlation' in c:
            if result.nifty_correlation > c['max_nifty_correlation']:
                return False

        # 2020 survival — OPTIONS_SELLING
        if c.get('must_survive_2020'):
            crash_start = pd.Timestamp('2020-03-01')
            crash_end = pd.Timestamp('2020-04-30')
            window_start = pd.Timestamp(window['test_start'])
            window_end = pd.Timestamp(window['test_end'])

            window_overlaps_2020 = (
                window_start <= crash_end and window_end >= crash_start
            )
            if window_overlaps_2020:
                sub_dd = getattr(result, 'drawdown_2020', None)
                if sub_dd is not None:
                    survival_limit = c['max_drawdown'] * 1.5
                    if sub_dd > survival_limit:
                        return False

        return True

    def _aggregate_results(self, signal_id, window_results):
        """Aggregate per-window results into overall pass/fail."""
        windows_passed = sum(1 for w in window_results if w['passed'])
        total_windows = len(window_results)
        pass_rate = windows_passed / total_windows if total_windows else 0.0

        last_window_passed = (
            window_results[-1]['passed'] if window_results else False
        )
        rate_ok = pass_rate >= self.MIN_PASS_RATE
        overall_pass = rate_ok and (
            last_window_passed if self.MUST_PASS_LAST_WINDOW else True
        )

        return {
            'signal_id': signal_id,
            'overall_pass': overall_pass,
            'windows_passed': windows_passed,
            'total_windows': total_windows,
            'pass_rate': pass_rate,
            'last_window_passed': last_window_passed,
            'window_details': window_results,
            'aggregate_sharpe': harmonic_mean_sharpe(window_results),
            'worst_window_drawdown': max(
                (w['result'].max_drawdown for w in window_results),
                default=0.0
            ),
            'recommendation': 'PROMOTE_TO_ACTIVE' if overall_pass else 'ARCHIVE',
            'fail_reason': (
                None if overall_pass else
                'LAST_WINDOW_FAILED' if rate_ok and not last_window_passed else
                f'PASS_RATE_{pass_rate:.0%}_BELOW_75PCT'
            ),
        }
