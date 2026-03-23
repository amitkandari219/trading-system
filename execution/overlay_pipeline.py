"""
Overlay Modifier Pipeline — collects all 22 overlay signals for a given date.

For backtesting: computes modifiers from historical data in nifty_daily.
For live trading: calls each overlay module's evaluate() method.

Usage:
    from execution.overlay_pipeline import OverlayPipeline
    pipeline = OverlayPipeline(df)  # df = nifty_daily with indicators
    modifiers = pipeline.get_modifiers(trade_date, direction='LONG')
"""

import calendar
import math
import logging
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OverlayPipeline:
    """Compute all overlay modifiers from historical data for backtesting."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: nifty_daily DataFrame with indicators added.
                Expected columns: date, close, india_vix, pcr_oi, rsi_14,
                adx_14, sma_50, bb_bandwidth, hvol_20, etc.
        """
        self._df = df.copy()
        if 'date' in self._df.columns:
            self._df['_date'] = self._df['date'].dt.date if hasattr(self._df['date'].iloc[0], 'date') else self._df['date']
            self._df = self._df.set_index('_date')

        # Pre-compute VIX percentiles for IV rank
        vix = self._df['india_vix'].dropna()
        self._vix_pctile = vix.expanding(min_periods=50).rank(pct=True)

    def get_modifiers(self, trade_date, direction: str = 'LONG') -> Dict[str, float]:
        """
        Get all overlay modifiers for a given date.
        Returns dict of {signal_id: size_modifier}.
        """
        modifiers = {}

        try:
            row = self._df.loc[trade_date]
        except KeyError:
            return modifiers

        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        vix = self._safe(row, 'india_vix', 15)
        adx = self._safe(row, 'adx_14', 20)
        rsi = self._safe(row, 'rsi_14', 50)
        pcr = self._safe(row, 'pcr_oi', 0)
        close = self._safe(row, 'close', 20000)
        sma50 = self._safe(row, 'sma_50', close)
        bb_bw = self._safe(row, 'bb_bandwidth', 0.05)
        hvol = self._safe(row, 'hvol_20', 15)

        # ── EVENT FILTER (RBI/Budget/FOMC) ──
        modifiers['RBI_MACRO_FILTER'] = self._rbi_macro(trade_date)

        # ── REGIME overlays ──
        modifiers['MAMBA_REGIME'] = self._vix_regime(vix)
        modifiers['GAMMA_EXPOSURE'] = self._gamma_proxy(vix, bb_bw)
        modifiers['VOL_TERM_STRUCTURE'] = self._vol_term(vix, hvol)
        modifiers['GUJRAL_DRY_7'] = self._trend_regime(close, sma50, adx, direction)
        modifiers['CRISIS_SHORT'] = self._crisis_mode(vix, close, sma50)

        # ── FLOW overlays ──
        modifiers['FII_FUTURES_OI'] = self._fii_proxy(vix, direction)
        modifiers['ROLLOVER_ANALYSIS'] = self._rollover_analysis(trade_date, vix, row)
        modifiers['DELIVERY_PCT'] = self._delivery_pct_proxy(row)
        modifiers['AMFI_MF_FLOW'] = 1.0  # needs monthly MF data
        modifiers['CREDIT_CARD_SPENDING'] = 1.0  # needs RBI CC data
        modifiers['ORDER_FLOW_IMBALANCE'] = self._order_flow_imbalance(row)

        # ── SENTIMENT overlays ──
        modifiers['SENTIMENT_COMPOSITE'] = self._sentiment(vix, rsi)
        modifiers['NLP_SENTIMENT'] = 1.0  # needs news data
        modifiers['BOND_YIELD_SPREAD'] = self._bond_yield_spread_proxy(vix)
        modifiers['PCR_AUTOTRENDER'] = self._pcr_signal(pcr)

        # ── CALENDAR ──
        modifiers['SSRN_JANUARY_EFFECT'] = self._january(trade_date, direction)
        modifiers['GLOBAL_OVERNIGHT_COMPOSITE'] = self._global_overnight_composite(row, trade_date)

        # ── META ──
        modifiers['XGBOOST_META_LEARNER'] = 1.0  # trained model output
        modifiers['RL_POSITION_SIZER'] = 1.0  # RL agent output
        modifiers['TFT_FORECAST'] = 1.0  # forecast model
        modifiers['GNN_SECTOR_ROTATION'] = 1.0  # sector model

        return modifiers

    # ── Individual overlay computations ──

    def _rbi_macro(self, dt) -> float:
        """Reduce size around RBI MPC, Budget, FOMC."""
        if not hasattr(dt, 'month'):
            return 1.0
        m, d, dow = dt.month, dt.day, dt.weekday()
        # Budget: Feb 1
        if m == 2 and d == 1:
            return 0.3
        # RBI MPC: ~first Wed of Feb/Apr/Jun/Aug/Oct/Dec
        if m in (2, 4, 6, 8, 10, 12) and d <= 7 and dow == 2:
            return 0.5
        # Day before/after RBI
        if m in (2, 4, 6, 8, 10, 12) and d <= 8 and dow in (1, 3):
            return 0.7
        return 1.0

    def _vix_regime(self, vix) -> float:
        """VIX-based regime: low vol → full size, high vol → reduce."""
        if vix < 12:
            return 1.1  # calm, slight boost
        if vix < 16:
            return 1.0
        if vix < 20:
            return 0.85
        if vix < 25:
            return 0.65
        return 0.4  # crisis

    def _gamma_proxy(self, vix, bb_bw) -> float:
        """Gamma exposure proxy from VIX + Bollinger bandwidth."""
        # Tight bands + low VIX = positive gamma (mean-reverting)
        if bb_bw < 0.03 and vix < 15:
            return 1.15  # gamma pinning — good for MR signals
        if bb_bw > 0.08 and vix > 22:
            return 0.7  # negative gamma — trending, reduce MR
        return 1.0

    def _vol_term(self, vix, hvol) -> float:
        """Vol term structure: VIX vs realized vol."""
        if hvol <= 0:
            return 1.0
        ratio = vix / hvol
        if ratio > 1.3:
            return 0.85  # VIX elevated vs realized — expect vol crush, reduce
        if ratio < 0.8:
            return 1.1  # VIX cheap vs realized — potential spike, careful
        return 1.0

    def _trend_regime(self, close, sma50, adx, direction) -> float:
        """Gujral regime: trending vs ranging."""
        if adx > 30:
            # Strong trend
            if close > sma50 and direction == 'LONG':
                return 1.2
            if close < sma50 and direction == 'SHORT':
                return 1.2
            return 0.7  # counter-trend
        if adx < 18:
            return 0.85  # weak trend — reduce all
        return 1.0

    def _crisis_mode(self, vix, close, sma50) -> float:
        """Crisis detection: VIX spike + below 50MA."""
        if vix > 25 and close < sma50:
            return 0.4  # crisis — heavy reduction
        if vix > 20 and close < sma50:
            return 0.7
        return 1.0

    def _fii_proxy(self, vix, direction) -> float:
        """FII flow proxy from VIX (FII sell when VIX high)."""
        # This is a rough proxy — real FII data not in nifty_daily
        if vix > 20:
            return 0.8 if direction == 'LONG' else 1.1
        if vix < 13:
            return 1.15 if direction == 'LONG' else 0.85
        return 1.0

    def _sentiment(self, vix, rsi) -> float:
        """Sentiment from VIX + RSI combination."""
        if vix > 25 and rsi < 30:
            return 1.2  # extreme fear → contrarian bullish
        if vix < 12 and rsi > 70:
            return 0.7  # extreme greed → reduce
        return 1.0

    def _pcr_signal(self, pcr) -> float:
        """PCR contrarian overlay."""
        if pcr <= 0 or math.isnan(pcr):
            return 1.0
        if pcr > 1.3:
            return 1.2  # extreme put buying → bullish
        if pcr < 0.5:
            return 0.8  # extreme call buying → bearish
        return 1.0

    def _january(self, dt, direction) -> float:
        """January effect: boost LONG in January."""
        if hasattr(dt, 'month') and dt.month == 1 and direction == 'LONG':
            return 1.15
        return 1.0

    def _delivery_pct_proxy(self, row) -> float:
        """DELIVERY_PCT: Use volume + range as proxy for delivery/accumulation.

        High volume + small range = accumulation (institutional buying) -> 1.10x
        Low volume = weak participation -> 0.90x
        """
        volume = self._safe(row, 'volume', 0)
        high = self._safe(row, 'high', 0)
        low = self._safe(row, 'low', 0)
        close = self._safe(row, 'close', 1)

        if volume <= 0 or close <= 0:
            return 1.0

        # Compute range as percentage of close
        range_pct = (high - low) / close if close > 0 else 0

        # Volume relative to recent average (use expanding mean as proxy)
        # Check if we have a volume column to compare against
        try:
            idx = row.name if hasattr(row, 'name') else None
            if idx is not None and 'volume' in self._df.columns:
                loc = self._df.index.get_loc(idx)
                if isinstance(loc, slice):
                    loc = loc.stop - 1
                elif hasattr(loc, '__len__'):
                    loc = loc[-1] if len(loc) > 0 else 0
                start = max(0, loc - 20)
                vol_window = self._df['volume'].iloc[start:loc + 1]
                avg_vol = vol_window.mean()
                vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0
            else:
                vol_ratio = 1.0
        except Exception:
            vol_ratio = 1.0

        # High vol + small range = accumulation
        if vol_ratio > 1.3 and range_pct < 0.01:
            return 1.10
        # Low volume = weak conviction
        if vol_ratio < 0.7:
            return 0.90
        return 1.0

    def _order_flow_imbalance(self, row) -> float:
        """ORDER_FLOW_IMBALANCE: Use bar direction as proxy for order flow.

        close > open = buy-dominated bar -> 1.12x
        close < open = sell-dominated bar -> 0.88x
        """
        close = self._safe(row, 'close', 0)
        open_ = self._safe(row, 'open', 0)

        if close <= 0 or open_ <= 0:
            return 1.0

        # Magnitude of the move matters
        move_pct = abs(close - open_) / open_ if open_ > 0 else 0

        if close > open_:
            # Buy-dominated: stronger moves get full boost
            if move_pct > 0.005:  # >0.5% bullish bar
                return 1.12
            return 1.05
        elif close < open_:
            # Sell-dominated
            if move_pct > 0.005:
                return 0.88
            return 0.95
        return 1.0

    def _bond_yield_spread_proxy(self, vix) -> float:
        """BOND_YIELD_SPREAD: Use VIX as proxy for yield stress.

        VIX > 22 = tight money / credit stress -> 0.85x (reduce risk)
        VIX < 12 = easy money / low stress -> 1.05x (increase risk)
        """
        if vix > 22:
            return 0.85
        if vix < 12:
            return 1.05
        return 1.0

    def _global_overnight_composite(self, row, trade_date) -> float:
        """GLOBAL_OVERNIGHT_COMPOSITE: Compute from overnight gap.

        Gap up > 0.3% from prev close to today's open = global risk-on -> 1.10x
        Gap down > 0.3% = global risk-off -> 0.85x
        """
        open_ = self._safe(row, 'open', 0)
        if open_ <= 0:
            return 1.0

        # Get previous day's close
        try:
            idx = row.name if hasattr(row, 'name') else trade_date
            loc = self._df.index.get_loc(idx)
            if isinstance(loc, slice):
                loc = loc.stop - 1
            elif hasattr(loc, '__len__'):
                loc = loc[-1] if len(loc) > 0 else 0
            if loc > 0:
                prev_close = float(self._df['close'].iloc[loc - 1])
            else:
                return 1.0
        except Exception:
            return 1.0

        if prev_close <= 0:
            return 1.0

        gap_pct = (open_ - prev_close) / prev_close

        if gap_pct > 0.003:   # >0.3% gap up
            return 1.10
        if gap_pct < -0.003:  # >0.3% gap down
            return 0.85
        return 1.0

    def _rollover_analysis(self, trade_date, vix, row) -> float:
        """ROLLOVER_ANALYSIS: Last 5 days before monthly expiry.

        If VIX falling during rollover = bullish roll -> 1.10x
        If VIX rising during rollover = bearish roll -> 0.90x
        Outside rollover window = neutral 1.0x
        """
        if not hasattr(trade_date, 'month'):
            return 1.0

        # Monthly expiry is last Thursday of month
        # Check if within 5 trading days of month end
        year, month = trade_date.year, trade_date.month
        last_day = calendar.monthrange(year, month)[1]

        # Find last Thursday of month
        last_thu = None
        for d in range(last_day, last_day - 8, -1):
            if d < 1:
                break
            try:
                candidate = date(year, month, d)
                if candidate.weekday() == 3:  # Thursday
                    last_thu = candidate
                    break
            except ValueError:
                continue

        if last_thu is None:
            return 1.0

        # Are we within 5 calendar days before expiry?
        days_to_expiry = (last_thu - trade_date).days
        if days_to_expiry < 0 or days_to_expiry > 5:
            return 1.0

        # Compare current VIX vs VIX 5 days ago
        try:
            idx = row.name if hasattr(row, 'name') else trade_date
            loc = self._df.index.get_loc(idx)
            if isinstance(loc, slice):
                loc = loc.stop - 1
            elif hasattr(loc, '__len__'):
                loc = loc[-1] if len(loc) > 0 else 0
            if loc >= 5:
                vix_5d_ago = float(self._df['india_vix'].iloc[loc - 5])
                if pd.isna(vix_5d_ago) or vix_5d_ago <= 0:
                    return 1.0
                if vix < vix_5d_ago:  # VIX falling = bullish roll
                    return 1.10
                else:  # VIX rising = bearish roll
                    return 0.90
        except Exception:
            return 1.0

        return 1.0

    @staticmethod
    def _safe(row, col, default=0):
        try:
            v = row[col]
            if pd.isna(v):
                return default
            return float(v)
        except (KeyError, TypeError, ValueError):
            return default
