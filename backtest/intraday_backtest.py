"""
Intraday backtest engine for L9 signals.

Key differences from daily generic_backtest:
- Session-scoped: positions open and close within same day
- Forced exit at 15:20 (SESSION_CLOSE_TIME)
- Slippage modeled explicitly (0.05% each way)
- Uses intraday indicators (VWAP, opening range, etc.)
- PnL computed per-session, equity curve from session returns
"""

import numpy as np
import pandas as pd

from backtest.types import BacktestResult
from data.intraday_indicators import add_intraday_indicators
from signals.l9_signals import IntradaySignalComputer

SLIPPAGE_PCT = 0.0005       # 0.05% each way
SESSION_CLOSE_HOUR = 15
SESSION_CLOSE_MIN = 20
NIFTY_LOT_SIZE = 25


def _empty_result():
    return BacktestResult(
        sharpe=0.0, calmar_ratio=0.0, max_drawdown=1.0,
        win_rate=0.0, profit_factor=0.0, avg_win_loss_ratio=0.0,
        trade_count=0, nifty_correlation=0.0,
        annual_return=0.0, drawdown_2020=0.0
    )


def _is_session_close(dt):
    """Check if bar is at or past session close time."""
    return (dt.hour > SESSION_CLOSE_HOUR or
            (dt.hour == SESSION_CLOSE_HOUR and dt.minute >= SESSION_CLOSE_MIN))


def run_intraday_backtest(signal_id, signal_config, df_5min,
                           capital=1_000_000):
    """
    Run intraday backtest for a single L9 signal.

    Args:
        signal_id: signal identifier
        signal_config: dict with stop_loss_pct, max_hold_bars, etc.
        df_5min: DataFrame of 5-min bars with datetime column
        capital: starting capital

    Returns:
        BacktestResult with all metrics + trades list
    """
    if len(df_5min) < 200:
        return _empty_result(), []

    # Add indicators
    df = add_intraday_indicators(df_5min)

    computer = IntradaySignalComputer()
    check_method = getattr(computer, signal_config['check'])
    stop_loss = signal_config.get('stop_loss_pct', 0.004)
    max_hold = signal_config.get('max_hold_bars', 30)

    trades = []
    position = None
    entry_price = 0.0
    entry_bar = 0
    bars_held = 0
    equity = capital
    daily_pnl = {}

    # Group by trading date for session context
    df['_date'] = df['datetime'].dt.date
    dates = df['_date'].unique()

    for trading_date in dates:
        session = df[df['_date'] == trading_date].copy()
        if len(session) < 5:
            continue

        session_bars_so_far = pd.DataFrame()

        for idx in range(len(session)):
            bar = session.iloc[idx]
            prev_bar = session.iloc[idx - 1] if idx > 0 else None
            bar_dt = bar['datetime']
            close = float(bar['close'])

            # Accumulate session context
            session_bars_so_far = pd.concat(
                [session_bars_so_far, bar.to_frame().T], ignore_index=True
            )

            # ── CHECK EXITS ──────────────────────────────
            if position is not None:
                bars_held += 1
                direction = position['direction']
                ep = position['entry_price']

                exit_reason = None

                # Forced session close
                if _is_session_close(bar_dt):
                    exit_reason = 'session_close'

                # Stop loss
                if not exit_reason:
                    if direction == 'LONG':
                        loss = (ep - close) / ep
                    else:
                        loss = (close - ep) / ep
                    if loss >= stop_loss:
                        exit_reason = 'stop_loss'

                # Max hold bars
                if not exit_reason and bars_held >= max_hold:
                    exit_reason = 'max_hold'

                if exit_reason:
                    # Apply slippage
                    if direction == 'LONG':
                        exit_price = close * (1 - SLIPPAGE_PCT)
                        pnl_pts = exit_price - ep
                    else:
                        exit_price = close * (1 + SLIPPAGE_PCT)
                        pnl_pts = ep - exit_price

                    pnl_rs = pnl_pts * NIFTY_LOT_SIZE

                    trades.append({
                        'signal_id': signal_id,
                        'direction': direction,
                        'entry_time': position['entry_time'],
                        'exit_time': bar_dt,
                        'entry_price': ep,
                        'exit_price': round(exit_price, 2),
                        'pnl_pts': round(pnl_pts, 2),
                        'pnl_rs': round(pnl_rs, 2),
                        'bars_held': bars_held,
                        'exit_reason': exit_reason,
                        'date': trading_date,
                    })

                    equity += pnl_rs
                    daily_pnl.setdefault(trading_date, 0)
                    daily_pnl[trading_date] += pnl_rs
                    position = None
                    bars_held = 0

            # ── CHECK ENTRIES ─────────────────────────────
            if position is None and not _is_session_close(bar_dt):
                result = check_method(
                    signal_id, signal_config, bar, prev_bar, session_bars_so_far
                )
                if result:
                    # Apply entry slippage
                    if result['direction'] == 'LONG':
                        adj_price = close * (1 + SLIPPAGE_PCT)
                    else:
                        adj_price = close * (1 - SLIPPAGE_PCT)

                    position = {
                        'direction': result['direction'],
                        'entry_price': round(adj_price, 2),
                        'entry_time': bar_dt,
                    }
                    bars_held = 0

        # Force close any open position at session end
        if position is not None:
            last_bar = session.iloc[-1]
            close = float(last_bar['close'])
            direction = position['direction']
            ep = position['entry_price']

            if direction == 'LONG':
                exit_price = close * (1 - SLIPPAGE_PCT)
                pnl_pts = exit_price - ep
            else:
                exit_price = close * (1 + SLIPPAGE_PCT)
                pnl_pts = ep - exit_price

            pnl_rs = pnl_pts * NIFTY_LOT_SIZE

            trades.append({
                'signal_id': signal_id,
                'direction': direction,
                'entry_time': position['entry_time'],
                'exit_time': last_bar['datetime'],
                'entry_price': ep,
                'exit_price': round(exit_price, 2),
                'pnl_pts': round(pnl_pts, 2),
                'pnl_rs': round(pnl_rs, 2),
                'bars_held': bars_held,
                'exit_reason': 'forced_session_end',
                'date': trading_date,
            })

            equity += pnl_rs
            daily_pnl.setdefault(trading_date, 0)
            daily_pnl[trading_date] += pnl_rs
            position = None

    # ── COMPUTE METRICS ──────────────────────────────
    if not trades:
        return _empty_result(), []

    pnls = [t['pnl_rs'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
    avg_wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Equity curve from daily P&L
    daily_returns = pd.Series(daily_pnl) / capital
    if len(daily_returns) > 1:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = (cum_pnl - peak)
    max_dd = abs(dd.min()) / capital if capital > 0 else 0

    years = len(dates) / 252
    ratio = equity / capital if capital > 0 else 1
    annual_return = (ratio ** (1 / years) - 1) if years > 0 and ratio > 0 else 0
    annual_return = float(annual_return.real) if hasattr(annual_return, 'real') else float(annual_return)
    calmar = annual_return / max_dd if max_dd > 0 else 0

    result = BacktestResult(
        sharpe=round(sharpe, 2),
        calmar_ratio=round(calmar, 2),
        max_drawdown=round(max_dd, 4),
        win_rate=round(win_rate, 3),
        profit_factor=round(profit_factor, 2),
        avg_win_loss_ratio=round(avg_wl_ratio, 2),
        trade_count=len(trades),
        nifty_correlation=0.0,
        annual_return=round(annual_return, 4),
        drawdown_2020=0.0,
    )

    return result, trades
