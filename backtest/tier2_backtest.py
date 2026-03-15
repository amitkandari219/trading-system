"""
Tier 2 backtest template for OPTIONS signals.
Reconstructs option prices from OHLCV + VIX using Black-Scholes.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from backtest.types import BacktestResult


def black_scholes_price(S, K, T, r, sigma, option_type='call') -> float:
    """
    Standard Black-Scholes option price.
    S:     Nifty spot price
    K:     strike price
    T:     time to expiry in years (calendar days / 365)
    r:     risk-free rate (repo rate, e.g. 0.065)
    sigma: implied volatility (India VIX / 100, adjusted by vol_multiplier)
    option_type: 'call' or 'put'
    """
    if T <= 0:
        # Expired — intrinsic value only
        if option_type == 'call':
            return max(0, S - K)
        return max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_vol_multiplier(db, dte: int) -> float:
    """
    Read the DTE-based vol multiplier from vol_adjustment_factors table.
    Returns 1.0 if table not yet populated (safe default).
    """
    # Map DTE to bucket
    if   dte <= 1:  bucket = 1
    elif dte <= 2:  bucket = 2
    elif dte <= 3:  bucket = 3
    elif dte <= 4:  bucket = 4
    elif dte <= 5:  bucket = 5
    elif dte <= 6:  bucket = 6
    elif dte <= 7:  bucket = 7
    elif dte <= 14: bucket = 8    # 8-14
    elif dte <= 21: bucket = 15   # 15-21
    elif dte <= 30: bucket = 22   # 22-30
    else:           bucket = 31   # 31+

    row = db.execute(
        "SELECT vol_multiplier FROM vol_adjustment_factors WHERE dte_bucket = %s",
        (bucket,)
    ).fetchone()
    return row['vol_multiplier'] if row else 1.0


def make_tier2_backtest_fn(signal_params: dict, db, risk_free_rate: float = 0.065):
    """
    Factory: returns a backtest_fn() for an OPTIONS signal.
    signal_params must include:
        instrument, entry_dte, exit_dte, strike_offset,
        direction, lot_size, stop_loss_pct
    """
    def backtest_fn(params, history_df, regime_labels):
        """
        Simulate the options strategy on history_df.
        """
        trades = []
        in_trade = False
        entry_price = None
        entry_date = None
        direction = params.get('direction', 'LONG')
        lot_size = params.get('lot_size', 25)
        entry_dte = params.get('entry_dte', 7)
        exit_dte = params.get('exit_dte', 1)
        strike_offset = params.get('strike_offset', 0.0)
        stop_loss_pct = params.get('stop_loss_pct', 0.50)

        for _, row in history_df.iterrows():
            bar_date = row['date']
            spot = row['close']
            vix = row['india_vix'] if pd.notna(row['india_vix']) else 15.0

            # Days to next Thursday expiry
            days_to_thu = (3 - bar_date.weekday()) % 7 or 7
            dte = days_to_thu
            if dte > 30:
                dte = dte % 7 or 7

            # IV: VIX / 100 × vol_multiplier for this DTE
            vol_mult = get_vol_multiplier(db, dte)
            sigma = (vix / 100.0) * vol_mult

            # Strike: ATM ± offset
            strike = round(spot * (1 + strike_offset) / 50) * 50

            T = dte / 365.0
            opt_type = 'call' if direction == 'LONG' and strike_offset >= 0 else 'put'
            opt_price = black_scholes_price(spot, strike, T, risk_free_rate, sigma, opt_type)

            if not in_trade:
                regime = regime_labels.get(bar_date.date(), 'RANGING')
                if _check_entry(params, row, regime, dte, entry_dte):
                    in_trade = True
                    entry_price = opt_price
                    entry_date = bar_date
                    entry_spot = spot

            else:
                exit_reason = None
                if dte <= exit_dte:
                    exit_reason = 'DTE_TARGET'
                elif direction == 'LONG' and opt_price < entry_price * (1 - stop_loss_pct):
                    exit_reason = 'STOP_LOSS'
                elif direction == 'SHORT' and opt_price > entry_price * (1 + stop_loss_pct):
                    exit_reason = 'STOP_LOSS'
                elif _check_exit(params, row, regime_labels.get(bar_date.date(), 'RANGING')):
                    exit_reason = 'SIGNAL_EXIT'

                if exit_reason:
                    pnl_per_lot = (opt_price - entry_price) * lot_size
                    if direction == 'SHORT':
                        pnl_per_lot = -pnl_per_lot
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date':  bar_date,
                        'pnl':        pnl_per_lot,
                        'entry_price': entry_price,
                        'exit_price':  opt_price,
                        'exit_reason': exit_reason,
                    })
                    in_trade = False
                    entry_price = None

        return _compute_backtest_result(trades, history_df)

    return backtest_fn


def _check_entry(params, row, regime, dte, entry_dte) -> bool:
    """
    Signal-specific entry check. Override per signal.
    Default: enter when DTE matches entry_dte target.
    """
    return dte == entry_dte


def _check_exit(params, row, regime) -> bool:
    """
    Signal-specific exit check. Override per signal.
    Default: no early exit beyond DTE and stop-loss.
    """
    return False


def _compute_backtest_result(trades: list, history_df) -> BacktestResult:
    """Compute BacktestResult from list of trade dicts."""
    if len(trades) < 5:
        return BacktestResult(
            sharpe=0.0, calmar_ratio=0.0, max_drawdown=1.0,
            win_rate=0.0, profit_factor=0.0, avg_win_loss_ratio=0.0,
            trade_count=len(trades), nifty_correlation=0.0,
            annual_return=0.0, drawdown_2020=1.0
        )

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_return = sum(pnls)
    win_rate = len(wins) / len(pnls)

    # Approximate Sharpe from trade P&L series
    pnl_series = pd.Series(pnls)
    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(52)
              if pnl_series.std() > 0 else 0.0)

    # Drawdown from cumulative P&L
    cum_pnl = pd.Series(pnls).cumsum()
    rolling_max = cum_pnl.cummax()
    drawdown = ((cum_pnl - rolling_max) / (rolling_max.abs() + 1e-9))
    max_dd = abs(drawdown.min())

    profit_factor = (sum(wins) / abs(sum(losses))
                     if losses else float('inf'))
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 1.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    calmar = (total_return / max_dd) if max_dd > 0 else 0.0

    # 2020 drawdown sub-period
    crash_trades = [t for t in trades
                    if pd.Timestamp('2020-03-01') <= t['entry_date']
                    <= pd.Timestamp('2020-04-30')]
    crash_pnl = [t['pnl'] for t in crash_trades]
    drawdown_2020 = (abs(min(0, sum(crash_pnl))) / (abs(total_return) + 1e-9)
                     if crash_pnl else 0.0)

    # Nifty correlation (simplified)
    nifty_correlation = 0.0

    return BacktestResult(
        sharpe            = round(sharpe, 3),
        calmar_ratio      = round(calmar, 3),
        max_drawdown      = round(max_dd, 3),
        win_rate          = round(win_rate, 3),
        profit_factor     = round(profit_factor, 3),
        avg_win_loss_ratio= round(win_loss_ratio, 3),
        trade_count       = len(trades),
        nifty_correlation = nifty_correlation,
        annual_return     = round(total_return, 0),
        drawdown_2020     = round(drawdown_2020, 3),
    )
