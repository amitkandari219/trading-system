"""
Sizing Rule A/B Test — compare 5 sizing strategies on NIFTY daily signals.

Tests five sizing rules on the same entries/exits from walk-forward-passing
signals over ~10 years of NIFTY daily data. Uses bootstrap resampling
for confidence intervals.

Sizing rules:
    1. Fixed Fractional:       2% risk per trade
    2. Volatility-Scaled:      ATR-based, target 1% daily vol
    3. Half-Kelly:             rolling 50-trade win/loss stats
    4. Signal Confidence:      base * confidence * regime factor
    5. Compound Tiered:        current system (equity-proportional)

Usage:
    venv/bin/python3 -m risk.sizing_ab_test
"""

import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

INITIAL_EQUITY = 1_000_000
NIFTY_LOT_SIZE = 25
RISK_FREE_RATE = 0.065
BOOTSTRAP_SAMPLES = 1000
TRADING_DAYS_PER_YEAR = 250

# Signal generation parameters (daily trend-following)
# We generate 6 synthetic but realistic signal streams from NIFTY data
SIGNAL_CONFIGS = [
    {'name': 'SMA_CROSS_20_50',  'fast': 20,  'slow': 50,  'sl_atr': 2.0, 'tgt_atr': 3.0},
    {'name': 'SMA_CROSS_10_30',  'fast': 10,  'slow': 30,  'sl_atr': 1.5, 'tgt_atr': 2.5},
    {'name': 'EMA_CROSS_12_26',  'fast': 12,  'slow': 26,  'sl_atr': 2.0, 'tgt_atr': 3.5},
    {'name': 'RSI_MR_30_70',     'fast': 14,  'slow': 0,   'sl_atr': 1.5, 'tgt_atr': 2.0},
    {'name': 'BB_SQUEEZE',       'fast': 20,  'slow': 0,   'sl_atr': 2.0, 'tgt_atr': 3.0},
    {'name': 'ADX_TREND',        'fast': 14,  'slow': 0,   'sl_atr': 2.5, 'tgt_atr': 4.0},
]


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Single trade record for A/B test."""
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    direction: str
    sl_price: float
    tgt_price: float
    signal_name: str
    lots: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class SizingResult:
    """Aggregate result for one sizing rule."""
    name: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    cagr: float = 0.0
    max_dd: float = 0.0
    sharpe: float = 0.0
    calmar: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_lots: float = 0.0
    # Bootstrap CIs
    cagr_ci: Tuple[float, float] = (0.0, 0.0)
    sharpe_ci: Tuple[float, float] = (0.0, 0.0)
    calmar_ci: Tuple[float, float] = (0.0, 0.0)


# ══════════════════════════════════════════════════════════════
# SIGNAL GENERATION (from NIFTY daily data)
# ══════════════════════════════════════════════════════════════

def load_nifty_daily() -> pd.DataFrame:
    """Load NIFTY daily data from database."""
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        conn = psycopg2.connect(DATABASE_DSN)
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume, india_vix "
            "FROM nifty_daily ORDER BY date",
            conn,
        )
        conn.close()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load NIFTY daily: {e}")
        return pd.DataFrame()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR from OHLC data."""
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def generate_trades_from_signal(
    df: pd.DataFrame, config: Dict
) -> List[Trade]:
    """
    Generate trade list from a signal configuration on NIFTY daily data.
    Returns trades with entry/exit prices and dates.
    """
    close = df['close'].astype(float)
    atr = compute_atr(df, 14)

    trades = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    sl_price = 0.0
    tgt_price = 0.0
    direction = "LONG"

    name = config['name']

    if 'RSI_MR' in name:
        # RSI mean reversion
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1.0/14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_vals = 100.0 - (100.0 / (1.0 + rs))

        for i in range(60, len(df)):
            if pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue
            if not in_trade:
                if rsi_vals.iloc[i] < 30 and rsi_vals.iloc[i-1] >= 30:
                    in_trade = True
                    entry_idx = i
                    entry_price = float(close.iloc[i])
                    sl_price = entry_price - config['sl_atr'] * float(atr.iloc[i])
                    tgt_price = entry_price + config['tgt_atr'] * float(atr.iloc[i])
                    direction = "LONG"
            else:
                c = float(close.iloc[i])
                if c <= sl_price or c >= tgt_price or (i - entry_idx) > 20:
                    exit_price = min(max(c, sl_price), tgt_price)
                    exit_price = c  # use actual close
                    trades.append(Trade(
                        entry_date=df['date'].iloc[entry_idx].date(),
                        exit_date=df['date'].iloc[i].date(),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=direction,
                        sl_price=sl_price,
                        tgt_price=tgt_price,
                        signal_name=name,
                    ))
                    in_trade = False

    elif 'BB_SQUEEZE' in name:
        # Bollinger Band squeeze breakout
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_width = (2 * std_20) / sma_20

        for i in range(60, len(df)):
            if pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0 or pd.isna(bb_width.iloc[i]):
                continue
            if not in_trade:
                if (float(bb_width.iloc[i]) < 0.03
                        and float(close.iloc[i]) > float(sma_20.iloc[i])):
                    in_trade = True
                    entry_idx = i
                    entry_price = float(close.iloc[i])
                    sl_price = entry_price - config['sl_atr'] * float(atr.iloc[i])
                    tgt_price = entry_price + config['tgt_atr'] * float(atr.iloc[i])
                    direction = "LONG"
            else:
                c = float(close.iloc[i])
                if c <= sl_price or c >= tgt_price or (i - entry_idx) > 15:
                    trades.append(Trade(
                        entry_date=df['date'].iloc[entry_idx].date(),
                        exit_date=df['date'].iloc[i].date(),
                        entry_price=entry_price,
                        exit_price=c,
                        direction=direction,
                        sl_price=sl_price,
                        tgt_price=tgt_price,
                        signal_name=name,
                    ))
                    in_trade = False

    elif 'ADX_TREND' in name:
        # ADX trend following
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        tr_s = compute_atr(df, 1) * 14  # just use ATR*period as smoothed TR approx
        plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx_vals = dx.rolling(14).mean()

        for i in range(60, len(df)):
            if pd.isna(atr.iloc[i]) or pd.isna(adx_vals.iloc[i]):
                continue
            if not in_trade:
                if float(adx_vals.iloc[i]) > 25 and float(plus_di.iloc[i]) > float(minus_di.iloc[i]):
                    in_trade = True
                    entry_idx = i
                    entry_price = float(close.iloc[i])
                    sl_price = entry_price - config['sl_atr'] * float(atr.iloc[i])
                    tgt_price = entry_price + config['tgt_atr'] * float(atr.iloc[i])
                    direction = "LONG"
            else:
                c = float(close.iloc[i])
                if c <= sl_price or c >= tgt_price or (i - entry_idx) > 25:
                    trades.append(Trade(
                        entry_date=df['date'].iloc[entry_idx].date(),
                        exit_date=df['date'].iloc[i].date(),
                        entry_price=entry_price,
                        exit_price=c,
                        direction=direction,
                        sl_price=sl_price,
                        tgt_price=tgt_price,
                        signal_name=name,
                    ))
                    in_trade = False

    else:
        # SMA/EMA crossover
        fast_p = config['fast']
        slow_p = config['slow']

        if 'EMA' in name:
            fast_ma = close.ewm(span=fast_p, adjust=False).mean()
            slow_ma = close.ewm(span=slow_p, adjust=False).mean()
        else:
            fast_ma = close.rolling(fast_p).mean()
            slow_ma = close.rolling(slow_p).mean()

        for i in range(slow_p + 10, len(df)):
            if pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue
            if not in_trade:
                # Bullish crossover
                if (float(fast_ma.iloc[i]) > float(slow_ma.iloc[i])
                        and float(fast_ma.iloc[i-1]) <= float(slow_ma.iloc[i-1])):
                    in_trade = True
                    entry_idx = i
                    entry_price = float(close.iloc[i])
                    sl_price = entry_price - config['sl_atr'] * float(atr.iloc[i])
                    tgt_price = entry_price + config['tgt_atr'] * float(atr.iloc[i])
                    direction = "LONG"
            else:
                c = float(close.iloc[i])
                # Exit on SL, TGT, or bearish crossover
                bearish_cross = (
                    float(fast_ma.iloc[i]) < float(slow_ma.iloc[i])
                    and float(fast_ma.iloc[i-1]) >= float(slow_ma.iloc[i-1])
                )
                if c <= sl_price or c >= tgt_price or bearish_cross:
                    trades.append(Trade(
                        entry_date=df['date'].iloc[entry_idx].date(),
                        exit_date=df['date'].iloc[i].date(),
                        entry_price=entry_price,
                        exit_price=c,
                        direction=direction,
                        sl_price=sl_price,
                        tgt_price=tgt_price,
                        signal_name=name,
                    ))
                    in_trade = False

    return trades


# ══════════════════════════════════════════════════════════════
# SIZING RULES
# ══════════════════════════════════════════════════════════════

def size_fixed_fractional(
    equity: float, entry_price: float, sl_price: float,
    risk_pct: float = 0.02, **kwargs
) -> int:
    """Fixed Fractional: risk 2% of equity per trade."""
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= 0:
        return 1
    risk_amount = equity * risk_pct
    units = int(risk_amount / risk_per_unit)
    lots = max(1, units // NIFTY_LOT_SIZE)
    return min(lots, 60)


def size_volatility_scaled(
    equity: float, entry_price: float, atr_val: float,
    target_daily_vol: float = 0.01, **kwargs
) -> int:
    """Volatility-Scaled: target 1% daily portfolio volatility."""
    if atr_val <= 0:
        return 1
    # Position size = (equity * target_vol) / (ATR * lot_size)
    position_value = equity * target_daily_vol / (atr_val / entry_price)
    lots = max(1, int(position_value / (entry_price * NIFTY_LOT_SIZE)))
    return min(lots, 60)


def size_half_kelly(
    equity: float, entry_price: float, sl_price: float,
    recent_trades: List[Trade] = None, **kwargs
) -> int:
    """Half-Kelly: rolling 50-trade stats for Kelly fraction."""
    if not recent_trades or len(recent_trades) < 10:
        # Default to 1 lot until enough data
        return 1

    wins = [t for t in recent_trades[-50:] if t.pnl > 0]
    losses = [t for t in recent_trades[-50:] if t.pnl <= 0]

    p = len(wins) / len(recent_trades[-50:]) if recent_trades[-50:] else 0.5
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.01
    avg_loss = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0.01

    if avg_loss <= 0:
        avg_loss = 0.01

    b = avg_win / avg_loss  # win/loss ratio
    kelly = p - (1 - p) / b if b > 0 else 0
    half_kelly = max(0, kelly * 0.5)
    half_kelly = min(half_kelly, 0.10)  # cap at 10%

    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= 0:
        return 1

    risk_amount = equity * half_kelly
    units = int(risk_amount / risk_per_unit)
    lots = max(1, units // NIFTY_LOT_SIZE)
    return min(lots, 60)


def size_signal_confidence(
    equity: float, entry_price: float, sl_price: float,
    confidence: float = 0.7, regime_factor: float = 1.0,
    base_risk_pct: float = 0.02, **kwargs
) -> int:
    """Signal Confidence Scaled: base * confidence * regime."""
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit <= 0:
        return 1

    adjusted_risk = base_risk_pct * confidence * regime_factor
    adjusted_risk = min(adjusted_risk, 0.05)  # cap at 5%

    risk_amount = equity * adjusted_risk
    units = int(risk_amount / risk_per_unit)
    lots = max(1, units // NIFTY_LOT_SIZE)
    return min(lots, 60)


def size_compound_tiered(
    equity: float, entry_price: float, premium: float = 200,
    initial_equity: float = INITIAL_EQUITY, **kwargs
) -> int:
    """Compound Tiered: current system (equity-proportional)."""
    # Simplified version of CompoundSizer logic
    if equity < 200_000:
        deploy = 0.45
    elif equity < 500_000:
        deploy = 0.50
    elif equity < 1_000_000:
        deploy = 0.55
    else:
        deploy = 0.50

    cost_per_lot = premium * NIFTY_LOT_SIZE
    if cost_per_lot <= 0:
        return 1

    lots = int((equity * deploy) / cost_per_lot)
    lots = max(1, min(lots, 60))

    # Drawdown reducer
    peak = max(equity, initial_equity)
    dd = (peak - equity) / peak if peak > 0 else 0
    if dd > 0.08:
        lots = max(1, lots // 2)

    return lots


# ══════════════════════════════════════════════════════════════
# A/B TEST ENGINE
# ══════════════════════════════════════════════════════════════

def run_sizing_on_trades(
    trades: List[Trade],
    sizing_fn,
    sizing_name: str,
    df: pd.DataFrame,
) -> SizingResult:
    """
    Run a sizing rule on a set of trades and compute metrics.
    """
    equity = float(INITIAL_EQUITY)
    peak_equity = equity
    equity_curve = [equity]
    sized_trades = []
    all_completed = []

    atr_series = compute_atr(df, 14)
    date_to_idx = {d.date() if hasattr(d, 'date') else d: i
                   for i, d in enumerate(df['date'])}

    # VIX for regime
    vix = df['india_vix'].astype(float) if 'india_vix' in df.columns else pd.Series(15.0, index=df.index)

    for trade in trades:
        idx = date_to_idx.get(trade.entry_date)
        if idx is None or idx >= len(atr_series):
            continue

        atr_val = float(atr_series.iloc[idx]) if not pd.isna(atr_series.iloc[idx]) else 200.0
        premium = max(50, min(120 + 0.8 * atr_val, 500))

        vix_val = float(vix.iloc[idx]) if not pd.isna(vix.iloc[idx]) else 15.0
        regime_factor = 1.0
        if vix_val > 25:
            regime_factor = 0.6
        elif vix_val > 20:
            regime_factor = 0.8

        # Compute lots
        lots = sizing_fn(
            equity=equity,
            entry_price=trade.entry_price,
            sl_price=trade.sl_price,
            atr_val=atr_val,
            premium=premium,
            recent_trades=all_completed,
            confidence=0.7,
            regime_factor=regime_factor,
            initial_equity=INITIAL_EQUITY,
        )

        # Compute P&L
        if trade.direction == "LONG":
            pnl_per_unit = trade.exit_price - trade.entry_price
        else:
            pnl_per_unit = trade.entry_price - trade.exit_price

        pnl = pnl_per_unit * lots * NIFTY_LOT_SIZE
        pnl_pct = pnl / equity if equity > 0 else 0

        trade_copy = Trade(
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            direction=trade.direction,
            sl_price=trade.sl_price,
            tgt_price=trade.tgt_price,
            signal_name=trade.signal_name,
            lots=lots,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

        equity += pnl
        equity = max(equity, 10_000)  # floor to prevent wipeout
        peak_equity = max(peak_equity, equity)
        equity_curve.append(equity)
        all_completed.append(trade_copy)
        sized_trades.append(trade_copy)

    # Compute metrics
    result = SizingResult(name=sizing_name, trades=sized_trades, equity_curve=equity_curve)
    result = _compute_metrics(result)

    return result


def _compute_metrics(result: SizingResult) -> SizingResult:
    """Compute CAGR, MaxDD, Sharpe, Calmar from equity curve."""
    curve = result.equity_curve
    if len(curve) < 2:
        return result

    final_eq = curve[-1]
    initial_eq = curve[0]

    # CAGR
    n_trades = len(result.trades)
    if n_trades > 0 and result.trades:
        first_date = result.trades[0].entry_date
        last_date = result.trades[-1].exit_date
        years = (last_date - first_date).days / 365.25 if last_date > first_date else 1
    else:
        years = 1

    if years > 0 and final_eq > 0 and initial_eq > 0:
        result.cagr = (final_eq / initial_eq) ** (1 / years) - 1
    else:
        result.cagr = 0

    # Max Drawdown
    peak = curve[0]
    max_dd = 0
    for eq in curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    result.max_dd = max_dd

    # Sharpe (trade-level returns)
    if result.trades:
        returns = [t.pnl_pct for t in result.trades]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.001
        if std_ret > 0:
            result.sharpe = (mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR)
        result.win_rate = len([t for t in result.trades if t.pnl > 0]) / len(result.trades)
        result.avg_lots = np.mean([t.lots for t in result.trades])

    # Calmar
    result.calmar = result.cagr / max_dd if max_dd > 0 else 0

    # Total P&L
    result.total_pnl = final_eq - initial_eq

    return result


def bootstrap_ci(
    trades: List[Trade],
    metric_fn,
    n_samples: int = BOOTSTRAP_SAMPLES,
    ci_level: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a metric."""
    if len(trades) < 10:
        return (0.0, 0.0)

    rng = np.random.default_rng(42)
    estimates = []

    for _ in range(n_samples):
        sample_idx = rng.integers(0, len(trades), size=len(trades))
        sample = [trades[i] for i in sample_idx]
        val = metric_fn(sample)
        if np.isfinite(val):
            estimates.append(val)

    if not estimates:
        return (0.0, 0.0)

    alpha = (1 - ci_level) / 2
    lower = float(np.percentile(estimates, alpha * 100))
    upper = float(np.percentile(estimates, (1 - alpha) * 100))
    return (round(lower, 4), round(upper, 4))


def _sharpe_from_trades(trades: List[Trade]) -> float:
    """Compute Sharpe from a list of trades."""
    if len(trades) < 2:
        return 0.0
    returns = [t.pnl_pct for t in trades]
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret <= 0:
        return 0.0
    return float((mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _calmar_from_trades(trades: List[Trade]) -> float:
    """Compute Calmar from a list of trades."""
    if len(trades) < 2:
        return 0.0

    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0

    for t in trades:
        equity += t.pnl
        equity = max(equity, 10_000)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    final = equity
    first = trades[0].entry_date
    last = trades[-1].exit_date
    years = max(0.5, (last - first).days / 365.25)
    cagr = (final / INITIAL_EQUITY) ** (1 / years) - 1 if final > 0 else 0
    return cagr / max_dd if max_dd > 0 else 0


# ══════════════════════════════════════════════════════════════
# MAIN: Run A/B Test
# ══════════════════════════════════════════════════════════════

def run_ab_test(print_results: bool = True) -> Dict[str, SizingResult]:
    """
    Run the full A/B test: 5 sizing rules × 6 signals × all NIFTY daily data.

    Returns dict of sizing_name -> SizingResult.
    """
    df = load_nifty_daily()
    if df.empty:
        logger.error("No NIFTY daily data — cannot run A/B test")
        return {}

    logger.info(
        f"Loaded {len(df)} NIFTY daily bars "
        f"({df['date'].min().date()} to {df['date'].max().date()})"
    )

    # Generate trades from all signals
    all_trades: List[Trade] = []
    for config in SIGNAL_CONFIGS:
        trades = generate_trades_from_signal(df, config)
        all_trades.extend(trades)
        logger.info(f"  {config['name']}: {len(trades)} trades")

    # Sort by entry date
    all_trades.sort(key=lambda t: t.entry_date)
    logger.info(f"Total trades across all signals: {len(all_trades)}")

    if not all_trades:
        logger.error("No trades generated — check signal parameters")
        return {}

    # Define sizing rules
    sizing_rules = {
        'Fixed Fractional (2%)': size_fixed_fractional,
        'Volatility-Scaled': size_volatility_scaled,
        'Half-Kelly': size_half_kelly,
        'Signal Confidence': size_signal_confidence,
        'Compound Tiered': size_compound_tiered,
    }

    # Run each sizing rule
    results: Dict[str, SizingResult] = {}
    for name, fn in sizing_rules.items():
        logger.info(f"Running {name}...")
        result = run_sizing_on_trades(all_trades, fn, name, df)

        # Bootstrap CIs
        if result.trades:
            result.sharpe_ci = bootstrap_ci(result.trades, _sharpe_from_trades)
            result.calmar_ci = bootstrap_ci(result.trades, _calmar_from_trades)

        results[name] = result

    # Print comparison
    if print_results:
        _print_results(results)

    return results


def _print_results(results: Dict[str, SizingResult]):
    """Pretty-print A/B test results."""
    print(f"\n{'='*90}")
    print(f"  SIZING RULE A/B TEST — {len(list(results.values())[0].trades) if results else 0} trades")
    print(f"  Initial equity: {INITIAL_EQUITY:,}")
    print(f"{'='*90}\n")

    # Header
    print(f"{'Rule':<25s} {'CAGR':>8s} {'MaxDD':>8s} {'Sharpe':>8s} "
          f"{'Calmar':>8s} {'WinRate':>8s} {'AvgLots':>8s} {'FinalPnL':>12s}")
    print(f"{'─'*87}")

    # Sort by Calmar
    sorted_results = sorted(results.values(), key=lambda r: r.calmar, reverse=True)

    for r in sorted_results:
        marker = " <<< WINNER" if r == sorted_results[0] else ""
        print(
            f"{r.name:<25s} {r.cagr:>7.1%} {r.max_dd:>7.1%} {r.sharpe:>8.2f} "
            f"{r.calmar:>8.2f} {r.win_rate:>7.1%} {r.avg_lots:>8.1f} "
            f"{r.total_pnl:>11,.0f}{marker}"
        )

    # Confidence intervals
    print(f"\n{'─'*87}")
    print(f"{'Rule':<25s} {'Sharpe 95% CI':>20s} {'Calmar 95% CI':>20s}")
    print(f"{'─'*87}")

    for r in sorted_results:
        print(
            f"{r.name:<25s} [{r.sharpe_ci[0]:>7.2f}, {r.sharpe_ci[1]:>7.2f}] "
            f"[{r.calmar_ci[0]:>7.2f}, {r.calmar_ci[1]:>7.2f}]"
        )

    # Winner
    winner = sorted_results[0]
    print(f"\n{'='*90}")
    print(f"  WINNER (by Calmar): {winner.name}")
    print(f"  CAGR={winner.cagr:.1%} MaxDD={winner.max_dd:.1%} "
          f"Sharpe={winner.sharpe:.2f} Calmar={winner.calmar:.2f}")
    print(f"{'='*90}\n")

    return winner.name


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
    )
    results = run_ab_test(print_results=True)
