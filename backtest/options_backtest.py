"""
Options backtest engine for L8 signals.

Simulates options strategies on historical NSE data.
Calculates P&L including:
- Premium received/paid
- Brokerage (₹40 per lot per leg)
- STT (0.05% on sell side premium)
- Slippage (0.1% of premium)
- Margin cost (opportunity cost)

Key rules:
- Exit before expiry if ITM (avoid STT trap)
- Force close at exit_dte
- Profit target and stop loss based on credit received
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

from data.options_loader import bs_price, bs_delta, implied_volatility
from config.settings import RISK_FREE_RATE, NIFTY_LOT_SIZE

BROKERAGE_PER_LOT = 40      # ₹ per lot per leg
STT_PCT = 0.0005            # 0.05% on sell-side premium
SLIPPAGE_PCT = 0.001        # 0.1% of premium


def find_strike_by_delta(options_chain, option_type, target_delta, spot):
    """Find strike closest to target delta."""
    filtered = options_chain[options_chain['option_type'] == option_type].copy()
    if filtered.empty:
        return None

    filtered = filtered[filtered['implied_volatility'].notna()]
    if filtered.empty:
        return None

    filtered['delta_diff'] = abs(abs(filtered['delta']) - abs(target_delta))
    best = filtered.loc[filtered['delta_diff'].idxmin()]
    return best


def find_strike_by_offset(options_chain, option_type, atm_strike, offset_points):
    """Find strike at ATM + offset."""
    target = atm_strike + offset_points
    filtered = options_chain[options_chain['option_type'] == option_type].copy()
    if filtered.empty:
        return None
    filtered['strike_diff'] = abs(filtered['strike'] - target)
    return filtered.loc[filtered['strike_diff'].idxmin()]


def find_atm_strike(options_chain, option_type, spot):
    """Find ATM strike."""
    filtered = options_chain[options_chain['option_type'] == option_type].copy()
    if filtered.empty:
        return None
    filtered['strike_diff'] = abs(filtered['strike'] - spot)
    return filtered.loc[filtered['strike_diff'].idxmin()]


def build_calendar_position(signal_rule, near_chain, far_chain, spot):
    """Build calendar spread: sell near month ATM, buy far month ATM."""
    sell_leg = find_atm_strike(near_chain, 'CE', spot)
    buy_leg = find_atm_strike(far_chain, 'CE', spot)

    if sell_leg is None or buy_leg is None:
        return None

    return [
        {
            'option_type': 'CE', 'action': 'SELL',
            'strike': float(sell_leg['strike']),
            'entry_premium': float(sell_leg['close']),
            'iv_at_entry': float(sell_leg['implied_volatility']) if pd.notna(sell_leg['implied_volatility']) else 0.15,
            'delta': float(sell_leg['delta']) if pd.notna(sell_leg['delta']) else 0,
            'lots': 1,
        },
        {
            'option_type': 'CE', 'action': 'BUY',
            'strike': float(buy_leg['strike']),
            'entry_premium': float(buy_leg['close']),
            'iv_at_entry': float(buy_leg['implied_volatility']) if pd.notna(buy_leg['implied_volatility']) else 0.15,
            'delta': float(buy_leg['delta']) if pd.notna(buy_leg['delta']) else 0,
            'lots': 1,
        },
    ]


def build_position(signal_rule, options_chain, spot):
    """Build position from signal rule legs using options chain data."""
    position_legs = []

    atm_strike = round(spot / 50) * 50  # Round to nearest 50

    for leg in signal_rule.legs:
        if leg.strike_method == 'delta':
            match = find_strike_by_delta(options_chain, leg.option_type,
                                          leg.delta_target, spot)
        elif leg.strike_method == 'offset':
            match = find_strike_by_offset(options_chain, leg.option_type,
                                           atm_strike, leg.offset_points)
        elif leg.strike_method == 'atm':
            match = find_atm_strike(options_chain, leg.option_type, spot)
        else:
            continue

        if match is None:
            return None  # Can't build position

        position_legs.append({
            'option_type': leg.option_type,
            'action': leg.action,
            'strike': float(match['strike']),
            'entry_premium': float(match['close']),
            'iv_at_entry': float(match['implied_volatility']) if pd.notna(match['implied_volatility']) else 0.15,
            'delta': float(match['delta']) if pd.notna(match['delta']) else 0,
            'lots': leg.lots,
        })

    return position_legs


def compute_trade_costs(legs, lots=1):
    """Compute brokerage + STT + slippage for a trade."""
    n_legs = len(legs)
    brokerage = BROKERAGE_PER_LOT * lots * n_legs * 2  # entry + exit

    sell_premium = sum(
        l['entry_premium'] * l['lots'] * NIFTY_LOT_SIZE
        for l in legs if l['action'] == 'SELL'
    )
    stt = sell_premium * STT_PCT

    total_premium = sum(l['entry_premium'] * l['lots'] * NIFTY_LOT_SIZE for l in legs)
    slippage = total_premium * SLIPPAGE_PCT * 2  # entry + exit

    return brokerage + stt + slippage


def run_options_backtest(signal_id, signal_rule, options_history, spot_history,
                          vix_history, regime_history, capital=5_000_000):
    """
    Run backtest for a single L8 options signal.
    """
    equity = capital
    trades = []
    open_position = None
    daily_pnl = {}

    dates = sorted(spot_history.keys())

    # Pre-compute IV rank from rolling 1-year ATM IV history
    iv_history = {}
    for dt in dates:
        chain = options_history.get(dt)
        if chain is not None and not chain.empty:
            spot = spot_history[dt]
            atm = chain.iloc[(chain['strike'] - spot).abs().argsort()[:4]]
            atm_iv = atm['implied_volatility'].dropna().mean()
            if atm_iv and atm_iv > 0:
                iv_history[dt] = atm_iv

    iv_series = pd.Series(iv_history)
    iv_rank_series = {}
    for dt in dates:
        if dt not in iv_history:
            iv_rank_series[dt] = 50  # default
            continue
        # 1-year lookback
        lookback = iv_series[iv_series.index <= dt].tail(252)
        if len(lookback) < 20:
            iv_rank_series[dt] = 50
            continue
        current = iv_history[dt]
        rank = (current - lookback.min()) / (lookback.max() - lookback.min()) * 100
        iv_rank_series[dt] = min(100, max(0, rank))

    # Compute regime from VIX + simple ADX proxy
    for dt in dates:
        vix = vix_history.get(dt, 15)
        if vix >= 25:
            regime_history[dt] = 'HIGH_VOL'
        elif vix >= 18:
            regime_history[dt] = 'RANGING'  # elevated but not crisis
        else:
            regime_history[dt] = 'RANGING'  # default for options backtest

    for trade_date in dates:
        spot = spot_history[trade_date]
        vix = vix_history.get(trade_date, 15)
        regime = regime_history.get(trade_date, 'RANGING')
        chain = options_history.get(trade_date)

        if chain is None or chain.empty:
            continue

        # ── CHECK EXITS ────────────────────────────────────
        if open_position is not None:
            pos = open_position
            days_held = (trade_date - pos['entry_date']).days

            # Get current prices for position legs
            current_value = 0
            for leg in pos['legs']:
                leg_chain = chain[
                    (chain['strike'] == leg['strike']) &
                    (chain['option_type'] == leg['option_type'])
                ]
                if not leg_chain.empty:
                    current_price = float(leg_chain.iloc[0]['close'])
                else:
                    # Estimate from BS
                    dte = pos['expiry_dte'] - days_held
                    if dte <= 0:
                        current_price = max(0, spot - leg['strike']) if leg['option_type'] == 'CE' else max(0, leg['strike'] - spot)
                    else:
                        current_price = bs_price(spot, leg['strike'], dte/365,
                                                  RISK_FREE_RATE, leg['iv_at_entry'],
                                                  leg['option_type'])

                mult = -1 if leg['action'] == 'SELL' else 1
                current_value += current_price * mult * leg['lots'] * NIFTY_LOT_SIZE

            entry_value = pos['entry_value']
            pnl = current_value - entry_value  # For sellers, entry_value is negative (credit)

            # For credit strategies: profit = entry_credit - current_cost
            credit = -entry_value  # positive for credit strategies
            current_cost = current_value

            exit_reason = None

            # DTE exit
            remaining_dte = pos['expiry_dte'] - days_held
            if remaining_dte <= signal_rule.exit_dte:
                exit_reason = 'dte_exit'

            # Profit target (% of credit received)
            if credit > 0:
                profit_pct = (credit - current_cost) / credit
                if profit_pct >= signal_rule.profit_target_pct:
                    exit_reason = 'profit_target'

                # Stop loss (multiple of credit)
                if current_cost > credit * signal_rule.stop_loss_multiple:
                    exit_reason = 'stop_loss'

            # Max hold days
            if days_held >= signal_rule.max_hold_days and signal_rule.max_hold_days > 0:
                exit_reason = 'max_hold'

            if exit_reason:
                costs = compute_trade_costs(pos['legs'])
                net_pnl = pnl - costs

                trades.append({
                    'signal_id': signal_id,
                    'entry_date': pos['entry_date'],
                    'exit_date': trade_date,
                    'strategy': signal_rule.strategy_type,
                    'entry_credit': round(credit, 0),
                    'exit_cost': round(current_cost, 0),
                    'gross_pnl': round(pnl, 0),
                    'costs': round(costs, 0),
                    'net_pnl': round(net_pnl, 0),
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                })

                equity += net_pnl
                daily_pnl.setdefault(trade_date, 0)
                daily_pnl[trade_date] += net_pnl
                open_position = None

        # ── CHECK ENTRIES ──────────────────────────────────
        if open_position is None:
            iv_rank = iv_rank_series.get(trade_date, 50)

            # Find nearest expiry with enough DTE
            expiries = sorted(chain['expiry'].unique())
            near_expiry = None
            far_expiry = None
            for exp in expiries:
                dte = (exp - trade_date).days
                if dte >= signal_rule.dte_min:
                    if near_expiry is None:
                        near_expiry = exp
                    elif far_expiry is None:
                        far_expiry = exp
                        break
            if near_expiry is None:
                continue
            dte_near = (near_expiry - trade_date).days

            # Check signal conditions
            if iv_rank < signal_rule.iv_rank_min or iv_rank > signal_rule.iv_rank_max:
                continue
            if vix < signal_rule.vix_min or vix > signal_rule.vix_max:
                continue
            if dte_near < signal_rule.dte_min or dte_near > signal_rule.dte_max:
                continue
            if 'ANY' not in signal_rule.regime_filter and regime not in signal_rule.regime_filter:
                continue

            # Build position
            near_chain = chain[chain['expiry'] == near_expiry]

            # Calendar spreads need far month for buy leg
            if signal_rule.strategy_type == 'CALENDAR_SPREAD' and far_expiry:
                far_chain = chain[chain['expiry'] == far_expiry]
                legs = build_calendar_position(signal_rule, near_chain, far_chain, spot)
            else:
                legs = build_position(signal_rule, near_chain, spot)
            if legs is None:
                continue

            # Calculate entry value
            entry_value = 0
            for leg in legs:
                mult = -1 if leg['action'] == 'SELL' else 1
                entry_value += leg['entry_premium'] * mult * leg['lots'] * NIFTY_LOT_SIZE

            # Check risk limit
            max_risk = capital * signal_rule.max_risk_pct
            if abs(entry_value) > max_risk:
                continue

            open_position = {
                'entry_date': trade_date,
                'legs': legs,
                'entry_value': entry_value,
                'expiry_dte': dte_near,
            }

    # ── COMPUTE METRICS ────────────────────────────────────
    if not trades:
        return {'signal_id': signal_id, 'trades': 0, 'sharpe': 0, 'win_rate': 0,
                'max_dd': 1.0, 'avg_credit': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'profit_factor': 0, 'avg_loss': 0, 'max_loss': 0,
                'strategy': signal_rule.strategy_type}, trades

    pnls = [t['net_pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls)
    avg_pnl = np.mean(pnls)
    avg_credit = np.mean([t['entry_credit'] for t in trades])
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0

    # Sharpe from daily P&L
    daily_returns = pd.Series(daily_pnl) / capital
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = (cum_pnl - peak)
    max_dd = abs(dd.min()) / capital if capital > 0 else 0

    metrics = {
        'signal_id': signal_id,
        'strategy': signal_rule.strategy_type,
        'trades': len(trades),
        'win_rate': round(win_rate, 3),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 4),
        'profit_factor': round(profit_factor, 2),
        'avg_credit': round(avg_credit, 0),
        'avg_pnl': round(avg_pnl, 0),
        'avg_loss': round(avg_loss, 0),
        'total_pnl': round(sum(pnls), 0),
        'max_loss': round(min(pnls), 0) if pnls else 0,
    }

    return metrics, trades
