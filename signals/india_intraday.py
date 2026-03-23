"""
India-specific intraday signals based on NSE microstructure.

Five signals tuned for NSE 5-min bar data (Nifty & BankNifty):

1. EXPIRY_PIN_FADE      — Fade toward nearest strike on expiry-day pin
2. FII_FLOW_MOMENTUM    — Ride 3-day consecutive FII buying/selling
3. VIX_MEANREV_SPIKE    — Buy mean-reversion after VIX spikes
4. LAST_HOUR_MOMENTUM   — Capture 14:30-15:20 continuation moves
5. BUDGET_STRADDLE_BUY  — Buy ATM straddle before macro events

Data requirements:
- intraday_bars  : 5-min OHLCV (NIFTY, BANKNIFTY) via Kite
- nifty_daily    : date, open, high, low, close, volume, india_vix
- fii_daily_metrics : trade_date, fii_net_buy (crores)

Integrates with:
- signals.calendar_overlay.CalendarOverlay.get_daily_context(date)
- signals.fii_overlay.FIIOverlay.get_multiplier(date)
- signals.expiry_day_detector.ExpiryDayDetector.get_expiry_info(date)

Usage:
    from signals.india_intraday import IndiaIntradaySignals
    engine = IndiaIntradaySignals()
    fired = engine.check_all(bar, prev_bar, session_bars, context)
"""

import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════

# NSE session boundaries (IST)
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
FORCE_EXIT_TIME = time(15, 20)

# Strike intervals
NIFTY_STRIKE_INTERVAL = 50
BANKNIFTY_STRIKE_INTERVAL = 100

# Round-strike multiples for pin detection (Signal 1 uses 100-pt)
NIFTY_PIN_INTERVAL = 100


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Convert to float, returning *default* on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _bar_time(bar: Dict) -> Optional[time]:
    """Extract time-of-day from a bar dict (expects 'timestamp')."""
    ts = bar.get('timestamp')
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.time()
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts).time()
        except ValueError:
            return None
    return None


def _bar_datetime(bar: Dict) -> Optional[datetime]:
    ts = bar.get('timestamp')
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None


def _nearest_strike(price: float, interval: int) -> float:
    """Round *price* to the nearest strike at *interval*."""
    return round(price / interval) * interval


def _bars_in_window(
    session_bars: Sequence[Dict],
    start: time,
    end: time,
) -> List[Dict]:
    """Return bars whose timestamp falls in [start, end)."""
    out: List[Dict] = []
    for b in session_bars:
        t = _bar_time(b)
        if t is not None and start <= t < end:
            out.append(b)
    return out


def _compute_atr_from_bars(bars: Sequence[Dict]) -> float:
    """ATR proxy from a short list of bar dicts (high-low range avg)."""
    if not bars:
        return float('nan')
    ranges = []
    for b in bars:
        h = _safe_float(b.get('high'))
        lo = _safe_float(b.get('low'))
        if not (math.isnan(h) or math.isnan(lo)):
            ranges.append(h - lo)
    return float(np.mean(ranges)) if ranges else float('nan')


def _compute_vwap(session_bars: Sequence[Dict]) -> float:
    """Cumulative VWAP from bar dicts (needs close, volume)."""
    cum_pv = 0.0
    cum_vol = 0
    for b in session_bars:
        # Typical price = (H+L+C)/3
        h = _safe_float(b.get('high'), 0)
        lo = _safe_float(b.get('low'), 0)
        c = _safe_float(b.get('close'), 0)
        v = int(_safe_float(b.get('volume'), 0))
        if v > 0 and c > 0:
            tp = (h + lo + c) / 3.0
            cum_pv += tp * v
            cum_vol += v
    if cum_vol == 0:
        return float('nan')
    return cum_pv / cum_vol


def _rsi_from_series(closes: Sequence[float], period: int = 14) -> float:
    """Compute latest RSI from a list of close prices."""
    if len(closes) < period + 1:
        return float('nan')
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _slope_pct_per_week(values: Sequence[float], days: int = 10) -> float:
    """
    Approximate slope of *values* (daily closes) as %/week.
    Uses simple linear regression over the last *days* values.
    """
    arr = list(values)
    if len(arr) < max(days, 3):
        return 0.0
    seg = arr[-days:]
    x = np.arange(len(seg), dtype=float)
    y = np.array(seg, dtype=float)
    if y[0] == 0:
        return 0.0
    # polyfit degree 1
    coeffs = np.polyfit(x, y, 1)
    daily_slope = coeffs[0]  # pts/day
    weekly_slope_pct = (daily_slope * 5.0 / y.mean()) * 100.0
    return weekly_slope_pct


# ════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ════════════════════════════════════════════════════════════════════

class IndiaIntradaySignals:
    """
    Five India-specific intraday signals for NSE 5-min bars.

    Each ``check_X`` method receives:
        bar          – current 5-min bar dict
        prev_bar     – previous 5-min bar dict (or None)
        session_bars – all bars so far today (list of dicts)
        context      – daily context dict:
            {
                'today':          date,
                'vix':            float,
                'daily_atr':      float,
                'expiry_info':    dict from ExpiryDayDetector,
                'fii_data':       list[dict] with recent fii_daily_metrics rows,
                'calendar_ctx':   dict from CalendarOverlay,
                'iv_rank':        float (0-100),
                'daily_bars_df':  pd.DataFrame (nifty_daily, last ~60 rows),
            }

    Returns: dict (signal result) or None if no signal.
    """

    # ────────────────────────────────────────────────────────────────
    # SIGNAL IDS
    # ────────────────────────────────────────────────────────────────
    SIG_EXPIRY_PIN = 'EXPIRY_PIN_FADE'
    SIG_FII_FLOW = 'FII_FLOW_MOMENTUM'
    SIG_VIX_MR = 'VIX_MEANREV_SPIKE'
    SIG_LAST_HOUR = 'LAST_HOUR_MOMENTUM'
    SIG_BUDGET_STRADDLE = 'BUDGET_STRADDLE_BUY'

    def __init__(self) -> None:
        logger.info('IndiaIntradaySignals initialised')

    # ================================================================
    # PUBLIC: check_all
    # ================================================================
    def check_all(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> List[Dict]:
        """Run all five signal checks; return list of fired signals."""
        fired: List[Dict] = []
        for method in (
            self.check_expiry_pin_fade,
            self.check_fii_flow_momentum,
            self.check_vix_meanrev_spike,
            self.check_last_hour_momentum,
            self.check_budget_straddle_buy,
            self.check_bn_orr_reversion,
            self.check_bn_expiry_pin_fade,
        ):
            try:
                result = method(bar, prev_bar, session_bars, context)
                if result is not None:
                    fired.append(result)
            except Exception:
                logger.exception('Signal check %s failed', method.__name__)
        return fired

    # ================================================================
    # SIGNAL 1: EXPIRY_PIN_FADE
    # ================================================================
    def check_expiry_pin_fade(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> Optional[Dict]:
        """
        Fade toward nearest round strike on expiry days.

        Entry window : 14:00 – 15:00
        Conditions   : expiry day, price within 0.15% of round 100-pt
                       strike, ATR of last 6 bars declining, VIX < 20
        Direction    : above strike → SHORT, below strike → LONG
        SL / TGT     : 0.3% / nearest strike
        Size         : 0.5x
        """
        # --- gate: expiry day ---
        expiry_info = context.get('expiry_info') or {}
        if not expiry_info.get('is_any_expiry'):
            return None

        # --- gate: time window ---
        t = _bar_time(bar)
        if t is None or not (time(14, 0) <= t < time(15, 0)):
            return None

        # --- gate: VIX < 20 ---
        vix = _safe_float(context.get('vix'))
        if math.isnan(vix) or vix >= 20.0:
            return None

        price = _safe_float(bar.get('close'))
        if math.isnan(price) or price <= 0:
            return None

        # --- nearest 100-pt strike ---
        nearest = _nearest_strike(price, NIFTY_PIN_INTERVAL)
        dist_pct = abs(price - nearest) / price * 100.0

        if dist_pct > 0.15:
            return None

        # --- ATR declining over last 6 bars ---
        if len(session_bars) < 6:
            return None
        last6 = session_bars[-6:]
        first3_atr = _compute_atr_from_bars(last6[:3])
        last3_atr = _compute_atr_from_bars(last6[3:])
        if math.isnan(first3_atr) or math.isnan(last3_atr):
            return None
        if last3_atr >= first3_atr:
            return None  # ATR not declining

        # --- direction ---
        if price > nearest:
            direction = 'SHORT'
        elif price < nearest:
            direction = 'LONG'
        else:
            return None  # exactly at strike — no edge

        # Determine instrument
        instrument = 'NIFTY'
        if expiry_info.get('is_banknifty_expiry') and not expiry_info.get('is_nifty_expiry'):
            instrument = 'BANKNIFTY'
            nearest = _nearest_strike(price, BANKNIFTY_STRIKE_INTERVAL)
            dist_pct = abs(price - nearest) / price * 100.0
            if dist_pct > 0.15:
                return None

        sl_pct = 0.3
        tgt_price = nearest
        tgt_pct = abs(tgt_price - price) / price * 100.0

        return {
            'signal_id': self.SIG_EXPIRY_PIN,
            'direction': direction,
            'price': price,
            'reason': (
                f'Expiry pin fade: price {price:.1f} near strike {nearest:.0f} '
                f'(dist {dist_pct:.3f}%), ATR declining, VIX {vix:.1f}'
            ),
            'instrument': instrument,
            'sl_pct': sl_pct,
            'tgt_pct': round(tgt_pct, 4),
            'tgt_price': tgt_price,
            'size_mult': 0.5,
            'max_hold_bars': None,  # time-based exit at 15:20
            'exit_time': FORCE_EXIT_TIME.isoformat(),
        }

    # ================================================================
    # SIGNAL 2: FII_FLOW_MOMENTUM
    # ================================================================
    def check_fii_flow_momentum(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> Optional[Dict]:
        """
        Follow 3-day consecutive FII net buying/selling.

        Entry   : 9:20
        Filters : no event days, VIX < 25, 3 consecutive days
                  net buy > 1500 cr (or net sell < -1500 cr)
        SL/TGT  : 0.7% / 1.5%, trailing stop at 0.4% after 0.8% gain
        """
        # --- gate: exactly 9:20 bar ---
        t = _bar_time(bar)
        if t is None or t != time(9, 20):
            return None

        # --- gate: no event days ---
        cal = context.get('calendar_ctx') or {}
        if cal.get('block_new_entries'):
            return None
        if cal.get('events_active'):
            return None

        # --- gate: VIX < 25 ---
        vix = _safe_float(context.get('vix'))
        if math.isnan(vix) or vix >= 25.0:
            return None

        # --- FII data: need 3 consecutive days ---
        fii_data = context.get('fii_data')
        if not fii_data or not isinstance(fii_data, (list, pd.DataFrame)):
            return None

        # Normalise to list of dicts
        if isinstance(fii_data, pd.DataFrame):
            if len(fii_data) < 3:
                return None
            rows = fii_data.sort_values('trade_date', ascending=False).head(3).to_dict('records')
        else:
            rows = list(fii_data)
            if len(rows) < 3:
                return None
            # Take the 3 most recent
            rows = sorted(rows, key=lambda r: r.get('trade_date', ''), reverse=True)[:3]

        # Extract net_buy values
        net_buys: List[float] = []
        for row in rows:
            nb = _safe_float(
                row.get('fii_net_buy') or row.get('net_buy') or row.get('fii_net')
            )
            if math.isnan(nb):
                return None  # missing data → skip gracefully
            net_buys.append(nb)

        # Check 3-day consecutive direction
        THRESHOLD_CR = 1500.0
        all_buying = all(nb > THRESHOLD_CR for nb in net_buys)
        all_selling = all(nb < -THRESHOLD_CR for nb in net_buys)

        if not (all_buying or all_selling):
            return None

        direction = 'LONG' if all_buying else 'SHORT'
        price = _safe_float(bar.get('close'))
        if math.isnan(price) or price <= 0:
            return None

        avg_flow = sum(net_buys) / len(net_buys)

        return {
            'signal_id': self.SIG_FII_FLOW,
            'direction': direction,
            'price': price,
            'reason': (
                f'FII 3-day {"buying" if all_buying else "selling"}: '
                f'avg {avg_flow:+.0f} cr/day, VIX {vix:.1f}'
            ),
            'instrument': 'NIFTY',
            'sl_pct': 0.7,
            'tgt_pct': 1.5,
            'trailing_stop_pct': 0.4,
            'trailing_activate_pct': 0.8,
            'size_mult': 1.0,
            'max_hold_bars': 75,  # ~6 hours
            'exit_time': FORCE_EXIT_TIME.isoformat(),
        }

    # ================================================================
    # SIGNAL 3: VIX_MEANREV_SPIKE
    # ================================================================
    def check_vix_meanrev_spike(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> Optional[Dict]:
        """
        Buy after a VIX spike starts reversing.

        Daily check at 9:30:
        - VIX was > 20 in last 3 days, now falling for 2+ days
        - RSI(14) on daily closes < 40
        - 50-day MA slope >= -0.5%/week (no sustained bear)

        LONG only, SL = lowest low of last 3 days.
        TGT = pre-spike level, max hold 5 days.
        4-6 trades/year, 1.5x size.
        """
        # --- gate: 9:30 bar ---
        t = _bar_time(bar)
        if t is None or t != time(9, 30):
            return None

        # --- need daily bars dataframe ---
        daily_df = context.get('daily_bars_df')
        if daily_df is None or not isinstance(daily_df, pd.DataFrame) or len(daily_df) < 20:
            return None

        # Sort by date ascending
        df = daily_df.sort_values('date').reset_index(drop=True)

        # --- VIX history ---
        vix_col = None
        for col_name in ('india_vix', 'vix'):
            if col_name in df.columns:
                vix_col = col_name
                break
        if vix_col is None:
            return None

        vix_vals = df[vix_col].dropna()
        if len(vix_vals) < 5:
            return None

        last5_vix = vix_vals.iloc[-5:].values

        # VIX was > 20 in last 3 days (at least once)
        last3_vix = last5_vix[-3:]
        if not any(v > 20.0 for v in last3_vix):
            return None

        # VIX falling for 2+ consecutive days
        recent_vix = vix_vals.iloc[-3:].values
        if len(recent_vix) < 3:
            return None
        falling_days = 0
        for i in range(1, len(recent_vix)):
            if recent_vix[i] < recent_vix[i - 1]:
                falling_days += 1
            else:
                falling_days = 0
        if falling_days < 2:
            return None

        # Current VIX (today's context or latest daily)
        current_vix = _safe_float(context.get('vix'))
        if math.isnan(current_vix):
            current_vix = float(vix_vals.iloc[-1])

        # --- RSI(14) on daily closes < 40 ---
        close_col = 'close'
        if close_col not in df.columns:
            return None
        closes = df[close_col].dropna().values
        if len(closes) < 16:
            return None
        rsi_val = _rsi_from_series(closes, period=14)
        if math.isnan(rsi_val) or rsi_val >= 40.0:
            return None

        # --- 50-MA slope check: slope >= -0.5%/week ---
        if len(closes) < 50:
            return None
        ma50 = closes[-50:]
        slope = _slope_pct_per_week(ma50, days=10)
        if slope < -0.5:
            return None  # sustained bear

        # --- SL: lowest low of last 3 days ---
        low_col = 'low'
        if low_col in df.columns:
            last3_lows = df[low_col].iloc[-3:].values
            sl_price = float(np.nanmin(last3_lows))
        else:
            sl_price = float('nan')

        price = _safe_float(bar.get('close'))
        if math.isnan(price) or price <= 0:
            return None

        if math.isnan(sl_price) or sl_price <= 0:
            sl_pct = 2.0  # fallback
        else:
            sl_pct = round((price - sl_price) / price * 100.0, 2)
            if sl_pct <= 0:
                return None  # price below recent low — no long entry

        # --- TGT: pre-spike level (VIX went > 20 from what level?) ---
        # Approximate: find the close when VIX first exceeded 20 in the
        # recent spike, then use the close from the day before as target.
        pre_spike_close = float('nan')
        for i in range(len(vix_vals) - 4, -1, -1):
            if i < 0:
                break
            if float(vix_vals.iloc[i]) <= 20.0:
                idx = vix_vals.index[i]
                if idx in df.index and close_col in df.columns:
                    pre_spike_close = float(df.loc[idx, close_col])
                break

        if math.isnan(pre_spike_close) or pre_spike_close <= price:
            tgt_pct = 2.5  # fallback: generous target
        else:
            tgt_pct = round((pre_spike_close - price) / price * 100.0, 2)

        # Max hold: 5 trading days ≈ 75 bars/day × 5 = 375 bars
        MAX_HOLD_DAYS = 5
        BARS_PER_DAY = 75  # (15:30-9:15)/5min = 75
        max_hold = MAX_HOLD_DAYS * BARS_PER_DAY

        return {
            'signal_id': self.SIG_VIX_MR,
            'direction': 'LONG',
            'price': price,
            'reason': (
                f'VIX mean-reversion: VIX fell {recent_vix[-2]:.1f}→{recent_vix[-1]:.1f}, '
                f'RSI(14) {rsi_val:.1f}, 50MA slope {slope:+.2f}%/wk'
            ),
            'instrument': 'NIFTY',
            'sl_pct': sl_pct,
            'sl_price': sl_price,
            'tgt_pct': tgt_pct,
            'size_mult': 1.5,
            'max_hold_bars': max_hold,
            'max_hold_days': MAX_HOLD_DAYS,
        }

    # ================================================================
    # SIGNAL 4: LAST_HOUR_MOMENTUM
    # ================================================================
    def check_last_hour_momentum(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> Optional[Dict]:
        """
        Capture last-hour continuation when intraday trend is strong.

        Entry   : 14:30 (non-expiry days)
        Trend   : close_14:30 vs close_12:00 must differ by > 0.3%
        VWAP    : price on same side as the trend
        SL/TGT  : 0.3% / 0.5%, exit at 15:20
        ~10-12 trades/month.
        """
        # --- gate: 14:30 bar ---
        t = _bar_time(bar)
        if t is None or t != time(14, 30):
            return None

        # --- gate: non-expiry day ---
        expiry_info = context.get('expiry_info') or {}
        if expiry_info.get('is_any_expiry'):
            return None

        price = _safe_float(bar.get('close'))
        if math.isnan(price) or price <= 0:
            return None

        # --- find 12:00 close ---
        close_1200 = None
        for b in session_bars:
            bt = _bar_time(b)
            if bt is not None and bt == time(12, 0):
                close_1200 = _safe_float(b.get('close'))
                break

        if close_1200 is None or math.isnan(close_1200) or close_1200 <= 0:
            # Try nearest bar to 12:00 (within 10 min)
            close_1200 = self._find_close_near_time(
                session_bars, time(12, 0), tolerance_min=10,
            )
            if close_1200 is None:
                return None

        # --- trend magnitude ---
        move_pct = (price - close_1200) / close_1200 * 100.0
        if abs(move_pct) < 0.3:
            return None

        direction = 'LONG' if move_pct > 0 else 'SHORT'

        # --- VWAP confirmation ---
        vwap = _compute_vwap(session_bars)
        if math.isnan(vwap):
            return None
        if direction == 'LONG' and price < vwap:
            return None
        if direction == 'SHORT' and price > vwap:
            return None

        return {
            'signal_id': self.SIG_LAST_HOUR,
            'direction': direction,
            'price': price,
            'reason': (
                f'Last-hour momentum: {move_pct:+.2f}% since 12:00, '
                f'VWAP {vwap:.1f}, price {"above" if price >= vwap else "below"} VWAP'
            ),
            'instrument': 'NIFTY',
            'sl_pct': 0.3,
            'tgt_pct': 0.5,
            'size_mult': 1.0,
            'max_hold_bars': 10,  # ~50 min to 15:20
            'exit_time': FORCE_EXIT_TIME.isoformat(),
        }

    # ================================================================
    # SIGNAL 5: BUDGET_STRADDLE_BUY
    # ================================================================
    def check_budget_straddle_buy(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> Optional[Dict]:
        """
        Buy ATM straddle before major macro events.

        Detected via calendar overlay (Budget, RBI MPC, Election).
        Entry      : 9:30 on day before event (T-1)
        IV filter  : IV rank < 80
        SL         : -40% of premium
        TGT        : +100% of premium
        7-8 trades/year.
        """
        # --- gate: 9:30 bar ---
        t = _bar_time(bar)
        if t is None or t != time(9, 30):
            return None

        # --- calendar context: event upcoming ---
        cal = context.get('calendar_ctx') or {}

        # We need an event upcoming tomorrow (this is T-1).
        # CalendarOverlay's get_daily_context flags events_active when within
        # proximity of an event. We look for a specific flag indicating T-1.
        is_event_tminus1 = cal.get('is_event_tminus1', False)

        # Fallback: check if events_active but not block_new_entries
        # (block_new_entries is typically the event day itself)
        if not is_event_tminus1:
            events_active = cal.get('events_active', False)
            block_tomorrow = cal.get('block_new_entries', False)
            # If events are flagged but entries aren't blocked, treat as T-1
            if not events_active:
                return None
            if block_tomorrow:
                return None  # this is the event day itself, too late

        # --- IV rank filter: < 80 ---
        iv_rank = _safe_float(context.get('iv_rank'))
        if math.isnan(iv_rank):
            # If IV rank unavailable, skip conservatively
            return None
        if iv_rank >= 80.0:
            return None  # IV already elevated → straddle too expensive

        price = _safe_float(bar.get('close'))
        if math.isnan(price) or price <= 0:
            return None

        atm_strike = _nearest_strike(price, NIFTY_STRIKE_INTERVAL)

        # Determine event type from calendar context
        event_type = cal.get('event_type', 'MACRO_EVENT')

        return {
            'signal_id': self.SIG_BUDGET_STRADDLE,
            'direction': 'STRADDLE',  # non-directional
            'price': price,
            'reason': (
                f'Event straddle: {event_type} upcoming, '
                f'ATM strike {atm_strike:.0f}, IV rank {iv_rank:.0f}'
            ),
            'instrument': 'NIFTY',
            'atm_strike': atm_strike,
            'sl_pct': 40.0,   # -40% of premium paid
            'tgt_pct': 100.0, # +100% of premium
            'size_mult': 1.0,
            'max_hold_bars': None,  # hold through event
            'max_hold_days': 3,
            'strategy_type': 'STRADDLE_BUY',
        }

    # ════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_close_near_time(
        session_bars: Sequence[Dict],
        target: time,
        tolerance_min: int = 10,
    ) -> Optional[float]:
        """Find bar close nearest to *target* time within tolerance."""
        best_bar = None
        best_delta = timedelta(minutes=tolerance_min + 1)
        for b in session_bars:
            bt = _bar_time(b)
            if bt is None:
                continue
            # Compute delta (same day, so direct subtraction OK)
            delta = abs(
                timedelta(hours=bt.hour, minutes=bt.minute)
                - timedelta(hours=target.hour, minutes=target.minute)
            )
            if delta < best_delta:
                best_delta = delta
                best_bar = b
        if best_bar is None:
            return None
        val = _safe_float(best_bar.get('close'))
        return val if not math.isnan(val) else None

    # ================================================================
    # SIGNAL 6: BN_ORR_REVERSION (BankNifty gap fade)
    # ================================================================
    SIG_BN_ORR_REVERSION = 'BN_ORR_REVERSION'

    def check_bn_orr_reversion(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> Optional[Dict]:
        """
        BankNifty overnight gap fade.

        Same thesis as ORR_REVERSION but tuned for BankNifty:
        - Gap threshold: > 0.6% (BN has higher daily range)
        - Entry: 9:25 IST
        - Exit: 11:00 or 60% gap retraced or SL at gap extension
        - Wider SL (2x ATR vs 1.5x for Nifty)
        - Block on BN monthly expiry day (last Thursday)
        """
        t = _bar_time(bar)
        if t is None or not (time(9, 25) <= t <= time(10, 30)):
            return None

        # Need previous day close
        prev_day_close = _safe_float(context.get('daily_close'))
        if math.isnan(prev_day_close) or prev_day_close <= 0:
            return None

        # First bar open = today's open
        if not session_bars:
            return None
        first_bar = session_bars[0]
        day_open = _safe_float(first_bar.get('open'))
        if math.isnan(day_open) or day_open <= 0:
            return None

        gap_pct = (day_open - prev_day_close) / prev_day_close
        if abs(gap_pct) < 0.006:  # < 0.6% gap → skip
            return None

        price = _safe_float(bar.get('close'))
        if math.isnan(price) or price <= 0:
            return None

        # Block on BN monthly expiry (last Thursday)
        today = context.get('today')
        if today and hasattr(today, 'weekday'):
            if today.weekday() == 3 and today.day > 21:
                return None  # monthly expiry Thursday

        # Check if gap still open (price hasn't already reverted)
        if gap_pct > 0 and price < prev_day_close:
            return None  # gap already filled
        if gap_pct < 0 and price > prev_day_close:
            return None

        # Current bar must show reversal sign
        bar_close = _safe_float(bar.get('close'))
        bar_open = _safe_float(bar.get('open'))
        if math.isnan(bar_close) or math.isnan(bar_open):
            return None

        # Gap up → need bearish bar (close < open)
        if gap_pct > 0.006 and bar_close < bar_open:
            return {
                'signal_id': self.SIG_BN_ORR_REVERSION,
                'direction': 'SHORT',
                'price': price,
                'reason': f'BN gap fade short: gap={gap_pct:+.2%} target={prev_day_close:.0f}',
                'instrument': 'BANKNIFTY',
                'sl_pct': 0.006,
                'tgt_pct': abs(gap_pct) * 0.6,
                'size_mult': 0.8,
                'max_hold_bars': 20,
                'exit_time': time(11, 0).isoformat(),
            }

        # Gap down → need bullish bar (close > open)
        if gap_pct < -0.006 and bar_close > bar_open:
            return {
                'signal_id': self.SIG_BN_ORR_REVERSION,
                'direction': 'LONG',
                'price': price,
                'reason': f'BN gap fade long: gap={gap_pct:+.2%} target={prev_day_close:.0f}',
                'instrument': 'BANKNIFTY',
                'sl_pct': 0.006,
                'tgt_pct': abs(gap_pct) * 0.6,
                'size_mult': 0.8,
                'max_hold_bars': 20,
                'exit_time': time(11, 0).isoformat(),
            }

        return None

    # ================================================================
    # SIGNAL 7: BN_EXPIRY_PIN_FADE (BankNifty monthly expiry pinning)
    # ================================================================
    SIG_BN_EXPIRY_PIN_FADE = 'BN_EXPIRY_PIN_FADE'

    BN_STRIKE_INTERVAL = 100  # BankNifty strikes at 100-pt intervals
    BN_PIN_ZONE_PCT = 0.002   # 0.2% of a round 500-pt strike

    def check_bn_expiry_pin_fade(
        self,
        bar: Dict,
        prev_bar: Optional[Dict],
        session_bars: List[Dict],
        context: Dict,
    ) -> Optional[Dict]:
        """
        BankNifty monthly expiry pinning near round 500-pt strikes.

        Only fires on BN monthly expiry (last Thursday of month).
        Entry: 14:00-15:00 IST
        Pin zone: within 0.2% of a round 500-pt strike
        ~12 trades/year (monthly only).
        """
        t = _bar_time(bar)
        if t is None or not (time(14, 0) <= t <= time(15, 0)):
            return None

        # Must be monthly expiry day
        today = context.get('today')
        if not today or not hasattr(today, 'weekday'):
            return None
        if today.weekday() != 3 or today.day <= 21:
            return None  # not last Thursday

        price = _safe_float(bar.get('close'))
        if math.isnan(price) or price <= 0:
            return None

        vix = _safe_float(context.get('vix', 15))
        if vix > 22:
            return None  # high vol breaks pinning

        # Check proximity to round 500-pt strike
        nearest_500 = round(price / 500) * 500
        distance_pct = abs(price - nearest_500) / price

        if distance_pct > self.BN_PIN_ZONE_PCT:
            return None  # too far from pin

        # ATR check: last 6 bars should show declining range (compression)
        if len(session_bars) >= 6:
            recent_6 = session_bars[-6:]
            ranges = [_safe_float(b.get('high', 0)) - _safe_float(b.get('low', 0))
                      for b in recent_6]
            ranges = [r for r in ranges if not math.isnan(r) and r > 0]
            if len(ranges) >= 4:
                first_half = sum(ranges[:3]) / 3
                second_half = sum(ranges[3:]) / max(len(ranges[3:]), 1)
                if second_half > first_half * 1.2:
                    return None  # range expanding, not compressing

        # Direction: fade away from pin strike
        if price > nearest_500:
            direction = 'SHORT'
        else:
            direction = 'LONG'

        return {
            'signal_id': self.SIG_BN_EXPIRY_PIN_FADE,
            'direction': direction,
            'price': price,
            'reason': (
                f'BN pin fade {direction}: price={price:.0f} '
                f'pin_strike={nearest_500:.0f} dist={distance_pct:.3%} VIX={vix:.1f}'
            ),
            'instrument': 'BANKNIFTY',
            'sl_pct': 0.004,
            'tgt_price': nearest_500,
            'size_mult': 0.5,  # conservative — monthly only, fewer trades
            'max_hold_bars': 12,
            'exit_time': FORCE_EXIT_TIME.isoformat(),
        }


# ════════════════════════════════════════════════════════════════════
# SELF-TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    from pprint import pprint

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print('=' * 72)
    print('IndiaIntradaySignals — self-test with synthetic data')
    print('=' * 72)

    engine = IndiaIntradaySignals()
    passed = 0
    failed = 0
    total = 0

    # ── Helper to build a bar dict ──
    def make_bar(
        dt: datetime,
        o: float, h: float, lo: float, c: float,
        vol: int = 100000,
    ) -> Dict:
        return {
            'timestamp': dt,
            'open': o, 'high': h, 'low': lo, 'close': c,
            'volume': vol,
        }

    # ── Build synthetic daily bars for VIX signal ──
    def make_daily_df(n: int = 60) -> pd.DataFrame:
        dates = [date(2026, 1, 5) + timedelta(days=i) for i in range(n)]
        closes = [22000 + i * 10 + np.random.randn() * 20 for i in range(n)]
        highs = [c + 80 for c in closes]
        lows = [c - 80 for c in closes]
        # VIX: starts ~16, spikes to 22 around day 50, then falls
        vix_vals = []
        for i in range(n):
            if i < 45:
                vix_vals.append(16 + np.random.rand())
            elif i < 52:
                vix_vals.append(22 + np.random.rand() * 2)
            else:
                # Falling: 22 → 18
                vix_vals.append(max(17, 22 - (i - 52) * 0.8 + np.random.rand() * 0.3))
        return pd.DataFrame({
            'date': dates,
            'open': [c - 30 for c in closes],
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': [5000000] * n,
            'india_vix': vix_vals,
        })

    daily_df = make_daily_df(60)

    # ────────────────────────────────────────────────────────────────
    # TEST 1: EXPIRY_PIN_FADE — should fire
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 1: EXPIRY_PIN_FADE (expect FIRE) ---')
    # Price ~22050 → nearest 100-pt strike = 22100, dist ~0.22%
    # Too far. Use price ~22090 → dist 0.045%
    pin_price = 22090.0
    session = []
    base_dt = datetime(2026, 3, 19, 13, 30)
    for i in range(12):
        t = base_dt + timedelta(minutes=5 * i)
        atr_mult = 1.2 - (i * 0.05)  # declining ATR
        session.append(make_bar(
            t,
            pin_price - 5, pin_price + 10 * atr_mult,
            pin_price - 10 * atr_mult, pin_price,
        ))
    bar_1430 = make_bar(
        datetime(2026, 3, 19, 14, 30),
        pin_price - 2, pin_price + 3, pin_price - 3, pin_price,
    )
    session.append(bar_1430)
    ctx1 = {
        'today': date(2026, 3, 19),
        'vix': 17.5,
        'daily_atr': 180.0,
        'expiry_info': {
            'is_any_expiry': True,
            'is_nifty_expiry': True,
            'is_banknifty_expiry': False,
        },
        'fii_data': [],
        'calendar_ctx': {},
        'iv_rank': 45.0,
        'daily_bars_df': daily_df,
    }
    res = engine.check_expiry_pin_fade(bar_1430, session[-2], session, ctx1)
    if res is not None:
        print(f'  PASS: fired {res["signal_id"]} {res["direction"]}')
        pprint(res)
        passed += 1
    else:
        print('  FAIL: expected signal, got None')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 2: EXPIRY_PIN_FADE — should NOT fire (non-expiry)
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 2: EXPIRY_PIN_FADE (expect NO fire — non-expiry) ---')
    ctx2 = dict(ctx1)
    ctx2['expiry_info'] = {'is_any_expiry': False}
    res = engine.check_expiry_pin_fade(bar_1430, session[-2], session, ctx2)
    if res is None:
        print('  PASS: correctly returned None')
        passed += 1
    else:
        print('  FAIL: should not have fired')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 3: FII_FLOW_MOMENTUM — should fire (3-day buying)
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 3: FII_FLOW_MOMENTUM (expect FIRE — buying) ---')
    bar_920 = make_bar(
        datetime(2026, 3, 19, 9, 20), 22000, 22020, 21980, 22010,
    )
    fii_rows = [
        {'trade_date': date(2026, 3, 18), 'fii_net_buy': 2100},
        {'trade_date': date(2026, 3, 17), 'fii_net_buy': 1800},
        {'trade_date': date(2026, 3, 16), 'fii_net_buy': 1600},
    ]
    ctx3 = {
        'today': date(2026, 3, 19),
        'vix': 18.0,
        'daily_atr': 180.0,
        'expiry_info': {'is_any_expiry': False},
        'fii_data': fii_rows,
        'calendar_ctx': {},
        'iv_rank': 50.0,
        'daily_bars_df': daily_df,
    }
    res = engine.check_fii_flow_momentum(bar_920, None, [bar_920], ctx3)
    if res is not None and res['direction'] == 'LONG':
        print(f'  PASS: fired {res["signal_id"]} {res["direction"]}')
        pprint(res)
        passed += 1
    else:
        print('  FAIL: expected LONG signal')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 4: FII_FLOW_MOMENTUM — missing data → None
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 4: FII_FLOW_MOMENTUM (expect NO fire — missing data) ---')
    ctx4 = dict(ctx3)
    ctx4['fii_data'] = None
    res = engine.check_fii_flow_momentum(bar_920, None, [bar_920], ctx4)
    if res is None:
        print('  PASS: correctly returned None')
        passed += 1
    else:
        print('  FAIL: should not have fired')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 5: LAST_HOUR_MOMENTUM — should fire
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 5: LAST_HOUR_MOMENTUM (expect FIRE) ---')
    session5: List[Dict] = []
    # 9:15 → 14:30 = 63 bars at 5-min
    base = datetime(2026, 3, 19, 9, 15)
    drift_per_bar = 3.0  # upward drift
    p = 22000.0
    for i in range(64):
        t = base + timedelta(minutes=5 * i)
        p += drift_per_bar + np.random.randn() * 2
        session5.append(make_bar(t, p - 5, p + 8, p - 8, p, vol=200000))
    # The bar at 14:30
    bar_1430_5 = session5[-1]
    # Overwrite to exactly 14:30
    bar_1430_5['timestamp'] = datetime(2026, 3, 19, 14, 30)
    # Set 12:00 bar
    idx_1200 = 33  # (12:00 - 9:15)/5 = 33
    session5[idx_1200]['timestamp'] = datetime(2026, 3, 19, 12, 0)
    session5[idx_1200]['close'] = 22050.0
    bar_1430_5['close'] = 22200.0  # 0.68% above 12:00

    ctx5 = {
        'today': date(2026, 3, 19),
        'vix': 16.0,
        'daily_atr': 180.0,
        'expiry_info': {'is_any_expiry': False},
        'fii_data': [],
        'calendar_ctx': {},
        'iv_rank': 40.0,
        'daily_bars_df': daily_df,
    }
    res = engine.check_last_hour_momentum(
        bar_1430_5, session5[-2], session5, ctx5,
    )
    if res is not None and res['direction'] == 'LONG':
        print(f'  PASS: fired {res["signal_id"]} {res["direction"]}')
        pprint(res)
        passed += 1
    else:
        print('  FAIL: expected LONG signal')
        if res:
            pprint(res)
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 6: BUDGET_STRADDLE_BUY — should fire
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 6: BUDGET_STRADDLE_BUY (expect FIRE) ---')
    bar_930 = make_bar(
        datetime(2026, 1, 30, 9, 30), 22000, 22020, 21980, 22010,
    )
    ctx6 = {
        'today': date(2026, 1, 30),
        'vix': 16.0,
        'daily_atr': 180.0,
        'expiry_info': {'is_any_expiry': False},
        'fii_data': [],
        'calendar_ctx': {
            'events_active': True,
            'block_new_entries': False,
            'event_type': 'BUDGET',
        },
        'iv_rank': 55.0,
        'daily_bars_df': daily_df,
    }
    res = engine.check_budget_straddle_buy(bar_930, None, [bar_930], ctx6)
    if res is not None and res['direction'] == 'STRADDLE':
        print(f'  PASS: fired {res["signal_id"]} {res["direction"]}')
        pprint(res)
        passed += 1
    else:
        print('  FAIL: expected STRADDLE signal')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 7: BUDGET_STRADDLE_BUY — IV rank too high → None
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 7: BUDGET_STRADDLE_BUY (expect NO fire — IV rank 85) ---')
    ctx7 = dict(ctx6)
    ctx7['iv_rank'] = 85.0
    res = engine.check_budget_straddle_buy(bar_930, None, [bar_930], ctx7)
    if res is None:
        print('  PASS: correctly returned None')
        passed += 1
    else:
        print('  FAIL: should not have fired')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 8: check_all — should not crash with empty context
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 8: check_all with empty context (expect no crash) ---')
    try:
        bar_empty = make_bar(
            datetime(2026, 3, 19, 10, 0), 22000, 22020, 21980, 22010,
        )
        result = engine.check_all(bar_empty, None, [bar_empty], {})
        print(f'  PASS: returned {len(result)} signals, no crash')
        passed += 1
    except Exception as exc:
        print(f'  FAIL: crashed with {exc}')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 9: check_all — accumulates multiple signals
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 9: check_all with FII + other triggers ---')
    # 9:20 bar with FII data → should fire FII signal
    result = engine.check_all(bar_920, None, [bar_920], ctx3)
    fii_signals = [s for s in result if s['signal_id'] == 'FII_FLOW_MOMENTUM']
    if len(fii_signals) == 1:
        print(f'  PASS: check_all collected {len(result)} signal(s), '
              f'including FII_FLOW_MOMENTUM')
        passed += 1
    else:
        print(f'  FAIL: expected 1 FII signal, got {len(fii_signals)}')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # TEST 10: VIX_MEANREV_SPIKE — gate: wrong time → None
    # ────────────────────────────────────────────────────────────────
    total += 1
    print('\n--- Test 10: VIX_MEANREV_SPIKE (wrong time → None) ---')
    bar_1000 = make_bar(
        datetime(2026, 3, 19, 10, 0), 22000, 22020, 21980, 22010,
    )
    res = engine.check_vix_meanrev_spike(bar_1000, None, [bar_1000], ctx1)
    if res is None:
        print('  PASS: correctly returned None (time gate)')
        passed += 1
    else:
        print('  FAIL: should not fire at 10:00')
        failed += 1

    # ────────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print(f'Results: {passed}/{total} passed, {failed} failed')
    if failed == 0:
        print('All tests passed.')
    else:
        print(f'WARNING: {failed} test(s) failed.')
        sys.exit(1)
