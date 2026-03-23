"""
Systematic Weekly Strangle Seller for Nifty Options.

Sells OTM call + OTM put strangles on Nifty weekly expiry with IV rank
gating, delta-based strike selection, and rule-based position management.

Relies on:
    - signals.iv_rank_filter.IVRankFilter  (IV regime checks)
    - execution.margin_calculator.MarginCalculator  (SPAN margin estimates)
    - data.options_loader.bs_price  (Black-Scholes pricing fallback)
    - config.settings  (DATABASE_DSN, RISK_FREE_RATE, NIFTY_LOT_SIZE)

Usage:
    from signals.strangle_seller import WeeklyStrangleSeller
    seller = WeeklyStrangleSeller()
    entry = seller.check_entry(today, spot=24500, vix=18.2)
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

import psycopg2

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE, RISK_FREE_RATE
from data.options_loader import bs_price
from execution.margin_calculator import MarginCalculator
from signals.iv_rank_filter import IVRankFilter

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

STRIKE_INTERVAL = 50
MAX_CONCURRENT_STRANGLES = 3
MIN_COMBINED_PREMIUM = 100          # per lot in rupees
MIN_PREMIUM_TO_MARGIN_PCT = 0.03    # 3 %
MIN_VIX = 16                        # premiums too thin below this
CRISIS_VIX = 25                     # VIX above this = crisis regime
IV_RANK_MIN = 60                    # minimum IV rank for entry
IV_RANK_WED_MIN = 80                # stricter on Wednesdays
PROFIT_TARGET_PCT = 0.50            # close at 50 % of premium captured
SL_LEG_MULTIPLIER = 2.0             # close leg if premium doubles
DELTA_ALERT_THRESHOLD = 0.30        # alert if |net delta| exceeds this
FORCE_CLOSE_HOUR = 14               # Thursday 14:00 IST force-close
ATR_MULTIPLIER = 1.5                # strike distance = 1.5 * ATR approx

# Blocked event dates (RBI policy, Union Budget, elections).
# Maintained manually; check_entry() blocks 2 days before each.
BLOCKED_EVENT_DATES: List[date] = [
    # -- placeholder: add actual dates each quarter --
    # date(2026, 4, 8),   # e.g., RBI policy
    # date(2026, 2, 1),   # e.g., Budget
]


# ================================================================
# WeeklyStrangleSeller
# ================================================================

class WeeklyStrangleSeller:
    """Rule-based weekly Nifty strangle seller with IV rank gating."""

    def __init__(
        self,
        kite=None,
        db_conn=None,
        iv_rank_filter: Optional[IVRankFilter] = None,
        margin_calc: Optional[MarginCalculator] = None,
    ):
        self.kite = kite
        self.db_conn = db_conn
        self.iv_rank_filter = iv_rank_filter or IVRankFilter(db_conn=db_conn)
        self.margin_calc = margin_calc or MarginCalculator()
        self._vix_recent: List[float] = []  # last few days' VIX for trend check
        self._active_positions: List[Dict[str, Any]] = []

    # ================================================================
    # 1.  Entry Check
    # ================================================================

    def check_entry(
        self,
        today: date,
        spot: float,
        vix: float,
        option_chain: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate whether to enter a new strangle today.

        Returns a signal dict or None if no entry.
        """
        reasons: List[str] = []

        # --- Day-of-week gate ---
        dow = today.weekday()  # 0=Mon .. 6=Sun
        if dow == 3:  # Thursday = expiry day
            logger.info("Thursday: never open new strangles on expiry day")
            return None
        if dow >= 4:  # Fri/Sat/Sun
            logger.info("Weekend: no entry")
            return None

        # --- IV rank ---
        iv_rank = self.iv_rank_filter.compute_iv_rank(vix)
        if dow == 2:  # Wednesday: stricter threshold
            if iv_rank < IV_RANK_WED_MIN:
                logger.info("Wednesday with IV rank %.1f < %d: skip", iv_rank, IV_RANK_WED_MIN)
                return None
            reasons.append(f"Wed entry, IV rank {iv_rank:.1f} >= {IV_RANK_WED_MIN}")
        else:
            if iv_rank < IV_RANK_MIN:
                logger.info("IV rank %.1f < %d: skip", iv_rank, IV_RANK_MIN)
                return None
            reasons.append(f"IV rank {iv_rank:.1f} >= {IV_RANK_MIN}")

        # --- VIX floor ---
        if vix < MIN_VIX:
            logger.info("VIX %.1f < %d: premiums too thin", vix, MIN_VIX)
            return None

        # --- Crisis check ---
        if vix > CRISIS_VIX:
            if not self._vix_falling():
                logger.info("CRISIS VIX %.1f without falling trend: skip", vix)
                return None
            reasons.append("Crisis VIX but falling (vol-crush opportunity)")

        # --- Event proximity ---
        if self._near_blocked_event(today):
            logger.info("Within 2 days of blocked event: skip")
            return None

        # --- Concurrent position limit ---
        if len(self._active_positions) >= MAX_CONCURRENT_STRANGLES:
            logger.info("Max %d concurrent strangles reached", MAX_CONCURRENT_STRANGLES)
            return None

        # --- Select strikes ---
        strikes = self.select_strikes(spot, option_chain, vix)
        if strikes is None:
            return None

        # --- IV regime size boost ---
        regime_check = self.iv_rank_filter.check_strategy('SHORT_STRANGLE', vix)
        size_mult = regime_check.get('size_boost', 1.0)
        if regime_check['action'] == 'BLOCK':
            logger.info("IV regime BLOCK for SHORT_STRANGLE")
            return None
        reasons.append(f"regime={regime_check['regime']}, action={regime_check['action']}")

        signal = {
            'signal_id': 'WEEKLY_STRANGLE_SELL',
            'date': today.isoformat(),
            'spot': spot,
            'vix': vix,
            'iv_rank': round(iv_rank, 2),
            'call_strike': strikes['call_strike'],
            'put_strike': strikes['put_strike'],
            'call_premium': strikes['call_premium'],
            'put_premium': strikes['put_premium'],
            'combined_premium': strikes['combined_premium'],
            'margin_required': strikes['margin_required'],
            'premium_to_margin_pct': strikes['premium_to_margin_pct'],
            'size_mult': size_mult,
            'reason': '; '.join(reasons),
        }
        logger.info("ENTRY signal: %s", signal)
        return signal

    # ================================================================
    # 2.  Strike Selection
    # ================================================================

    def select_strikes(
        self,
        spot: float,
        option_chain: Optional[List[Dict[str, Any]]],
        vix: float,
    ) -> Optional[Dict[str, Any]]:
        """Pick OTM call and put strikes targeting ~0.20-0.25 delta.

        Uses ATR_MULTIPLIER * implied ATR as distance from spot, rounded
        to the nearest STRIKE_INTERVAL.  Filters on OI and premium checks.
        """
        # Approximate ATR from VIX: daily_move ~ spot * (vix/100) / sqrt(252)
        daily_vol = spot * (vix / 100.0) / math.sqrt(252)
        atr_approx = daily_vol * 1.5  # rough 1.5x daily range
        distance = ATR_MULTIPLIER * atr_approx

        call_strike = self._round_strike(spot + distance, up=True)
        put_strike = self._round_strike(spot - distance, up=False)

        # Ensure minimum separation
        if call_strike <= spot:
            call_strike = self._round_strike(spot + STRIKE_INTERVAL, up=True)
        if put_strike >= spot:
            put_strike = self._round_strike(spot - STRIKE_INTERVAL, up=False)

        # --- OI filter: avoid strikes with unusually high OI (resistance) ---
        if option_chain:
            call_strike = self._adjust_for_oi(call_strike, option_chain, 'CE', spot)
            put_strike = self._adjust_for_oi(put_strike, option_chain, 'PE', spot)

        # --- Estimate premiums ---
        # Weekly expiry is Thursday; assume ~5 DTE for Monday entry
        dte = 5
        call_premium = self._get_chain_premium(
            option_chain, call_strike, 'CE'
        ) or self._estimate_premium(spot, call_strike, vix, dte, 'CE')
        put_premium = self._get_chain_premium(
            option_chain, put_strike, 'PE'
        ) or self._estimate_premium(spot, put_strike, vix, dte, 'PE')

        combined = call_premium + put_premium

        if combined < MIN_COMBINED_PREMIUM:
            logger.info(
                "Combined premium %.1f < %d: strikes too far OTM",
                combined, MIN_COMBINED_PREMIUM,
            )
            return None

        # --- Margin estimate ---
        margin = self.margin_calc.short_straddle_margin(
            spot, call_strike, call_premium, put_premium,
            NIFTY_LOT_SIZE, lots=1,
        )
        premium_value = combined * NIFTY_LOT_SIZE
        ptm = premium_value / margin if margin > 0 else 0.0

        if ptm < MIN_PREMIUM_TO_MARGIN_PCT:
            logger.info(
                "Premium-to-margin %.2f%% < %.2f%%: not worth the margin",
                ptm * 100, MIN_PREMIUM_TO_MARGIN_PCT * 100,
            )
            return None

        return {
            'call_strike': call_strike,
            'put_strike': put_strike,
            'call_premium': round(call_premium, 2),
            'put_premium': round(put_premium, 2),
            'combined_premium': round(combined, 2),
            'margin_required': round(margin, 2),
            'premium_to_margin_pct': round(ptm * 100, 2),
        }

    # ================================================================
    # 3.  Position Management
    # ================================================================

    def manage_position(
        self,
        position: Dict[str, Any],
        current_spot: float,
        current_premiums: Dict[str, float],
        dte: int,
    ) -> Dict[str, Any]:
        """Decide hold / close / adjust for an open strangle.

        Parameters
        ----------
        position : dict with keys entry_call_premium, entry_put_premium,
                   call_strike, put_strike, entry_combined_premium
        current_premiums : {'call': float, 'put': float}
        dte : days to expiry

        Returns
        -------
        dict with 'action' and context.
        """
        entry_combined = position['entry_combined_premium']
        current_combined = current_premiums['call'] + current_premiums['put']
        pnl_pct = (entry_combined - current_combined) / entry_combined if entry_combined > 0 else 0.0

        call_strike = position['call_strike']
        put_strike = position['put_strike']
        entry_call = position['entry_call_premium']
        entry_put = position['entry_put_premium']
        cur_call = current_premiums['call']
        cur_put = current_premiums['put']

        # --- Thursday force-close ---
        now = datetime.now()
        if now.weekday() == 3 and now.hour >= FORCE_CLOSE_HOUR:
            return {
                'action': 'CLOSE_TIME',
                'reason': f'Thursday {FORCE_CLOSE_HOUR}:00 force-close',
                'pnl_pct': round(pnl_pct * 100, 2),
                'current_combined': round(current_combined, 2),
            }

        # --- 50 % profit target ---
        if pnl_pct >= PROFIT_TARGET_PCT:
            return {
                'action': 'CLOSE_PROFIT',
                'reason': f'Profit target hit: {pnl_pct*100:.1f}% captured',
                'pnl_pct': round(pnl_pct * 100, 2),
                'current_combined': round(current_combined, 2),
            }

        # --- Stop loss: either leg doubles ---
        if cur_call >= entry_call * SL_LEG_MULTIPLIER:
            return {
                'action': 'CLOSE_SL',
                'reason': f'Call leg SL: premium {cur_call:.1f} >= 2x entry {entry_call:.1f}',
                'leg': 'CE',
                'pnl_pct': round(pnl_pct * 100, 2),
            }
        if cur_put >= entry_put * SL_LEG_MULTIPLIER:
            return {
                'action': 'CLOSE_SL',
                'reason': f'Put leg SL: premium {cur_put:.1f} >= 2x entry {entry_put:.1f}',
                'leg': 'PE',
                'pnl_pct': round(pnl_pct * 100, 2),
            }

        # --- Spot breaches a strike ---
        if current_spot >= call_strike:
            return {
                'action': 'ADJUST',
                'reason': f'Spot {current_spot:.0f} breached call strike {call_strike}',
                'close_leg': 'CE',
                'suggestion': 'Close CE leg; consider rolling up',
            }
        if current_spot <= put_strike:
            return {
                'action': 'ADJUST',
                'reason': f'Spot {current_spot:.0f} breached put strike {put_strike}',
                'close_leg': 'PE',
                'suggestion': 'Close PE leg; consider rolling down',
            }

        # --- DTE = 0 and still holding (shouldn't happen if Thursday gate works) ---
        if dte <= 0:
            return {
                'action': 'CLOSE_TIME',
                'reason': 'DTE exhausted',
                'pnl_pct': round(pnl_pct * 100, 2),
            }

        return {
            'action': 'HOLD',
            'reason': 'No exit trigger',
            'pnl_pct': round(pnl_pct * 100, 2),
            'current_combined': round(current_combined, 2),
            'dte': dte,
        }

    # ================================================================
    # 4.  Position Greeks
    # ================================================================

    def calculate_position_greeks(
        self,
        position: Dict[str, Any],
        spot: float,
        vix: float,
        dte: int,
    ) -> Dict[str, Any]:
        """Compute net Greeks for the short strangle position.

        Uses BS analytical Greeks.  Short positions have negated signs.
        """
        sigma = vix / 100.0
        T = max(dte / 365.0, 1e-6)
        r = RISK_FREE_RATE

        call_strike = position['call_strike']
        put_strike = position['put_strike']

        call_greeks = self._bs_greeks(spot, call_strike, T, r, sigma, 'CE')
        put_greeks = self._bs_greeks(spot, put_strike, T, r, sigma, 'PE')

        # Short both legs: negate
        net_delta = -(call_greeks['delta'] + put_greeks['delta'])
        net_theta = -(call_greeks['theta'] + put_greeks['theta'])  # negative theta -> positive for seller
        net_gamma = -(call_greeks['gamma'] + put_greeks['gamma'])
        net_vega = -(call_greeks['vega'] + put_greeks['vega'])

        # Alert flags
        entry_combined = position.get('entry_combined_premium', 0)
        gamma_risk = abs(net_gamma) * spot * 0.01  # 1% move impact
        alerts = []
        if abs(net_delta) > DELTA_ALERT_THRESHOLD:
            alerts.append(f'|delta| {abs(net_delta):.3f} > {DELTA_ALERT_THRESHOLD}')
        if entry_combined > 0 and gamma_risk > 2.0 * entry_combined:
            alerts.append(f'Gamma risk {gamma_risk:.1f} > 2x premium {entry_combined:.1f}')

        return {
            'net_delta': round(net_delta, 4),
            'net_theta': round(net_theta, 4),
            'net_gamma': round(net_gamma, 6),
            'net_vega': round(net_vega, 4),
            'call_delta': round(call_greeks['delta'], 4),
            'put_delta': round(put_greeks['delta'], 4),
            'gamma_risk_1pct': round(gamma_risk, 2),
            'alerts': alerts,
        }

    # ================================================================
    # 5.  Premium Estimation (BS fallback)
    # ================================================================

    def _estimate_premium(
        self,
        spot: float,
        strike: float,
        vix: float,
        dte: int,
        option_type: str,
    ) -> float:
        """BS price when no live data available."""
        sigma = vix / 100.0
        T = max(dte / 365.0, 1e-6)
        return bs_price(spot, strike, T, RISK_FREE_RATE, sigma, option_type)

    # ================================================================
    # Private helpers
    # ================================================================

    @staticmethod
    def _round_strike(value: float, up: bool = True) -> float:
        """Round to nearest STRIKE_INTERVAL."""
        if up:
            return math.ceil(value / STRIKE_INTERVAL) * STRIKE_INTERVAL
        return math.floor(value / STRIKE_INTERVAL) * STRIKE_INTERVAL

    @staticmethod
    def _bs_greeks(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    ) -> Dict[str, float]:
        """Analytical BS Greeks for a single leg."""
        from scipy.stats import norm

        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Gamma and Vega are the same for calls and puts
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
        vega = S * norm.pdf(d1) * sqrt_T / 100.0  # per 1-point vol move

        if option_type == 'CE':
            delta = norm.cdf(d1)
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                - r * K * math.exp(-r * T) * norm.cdf(d2)
            ) / 365.0
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                + r * K * math.exp(-r * T) * norm.cdf(-d2)
            ) / 365.0

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
        }

    def _vix_falling(self) -> bool:
        """True if VIX has been declining for 2+ consecutive days."""
        if len(self._vix_recent) < 3:
            return False
        return self._vix_recent[-1] < self._vix_recent[-2] < self._vix_recent[-3]

    def _near_blocked_event(self, today: date) -> bool:
        """True if *today* is within 2 calendar days of a blocked event."""
        for evt in BLOCKED_EVENT_DATES:
            diff = (evt - today).days
            if 0 <= diff <= 2:
                return True
        return False

    @staticmethod
    def _get_chain_premium(
        option_chain: Optional[List[Dict[str, Any]]],
        strike: float,
        option_type: str,
    ) -> Optional[float]:
        """Look up LTP from option chain data, if available."""
        if not option_chain:
            return None
        for row in option_chain:
            if row.get('strike') == strike and row.get('option_type') == option_type:
                ltp = row.get('ltp') or row.get('close') or row.get('last_price')
                if ltp and ltp > 0:
                    return float(ltp)
        return None

    @staticmethod
    def _adjust_for_oi(
        strike: float,
        option_chain: List[Dict[str, Any]],
        option_type: str,
        spot: float,
    ) -> float:
        """Shift strike away from abnormally high OI (acts as magnet/resistance).

        If the chosen strike has OI > 2x median for nearby strikes, move one
        interval further OTM.
        """
        nearby = [
            row for row in option_chain
            if row.get('option_type') == option_type
            and abs(row.get('strike', 0) - strike) <= 3 * STRIKE_INTERVAL
        ]
        if len(nearby) < 3:
            return strike

        oi_values = [row.get('oi', 0) or 0 for row in nearby]
        oi_values_sorted = sorted(oi_values)
        median_oi = oi_values_sorted[len(oi_values_sorted) // 2]

        target_oi = next(
            (row.get('oi', 0) or 0 for row in nearby if row.get('strike') == strike),
            0,
        )

        if median_oi > 0 and target_oi > 2 * median_oi:
            if option_type == 'CE':
                strike += STRIKE_INTERVAL  # move further OTM for calls
            else:
                strike -= STRIKE_INTERVAL  # move further OTM for puts
            logger.info("OI filter: shifted %s strike to %.0f (high OI avoidance)", option_type, strike)

        return strike

    def update_vix_history(self, vix: float) -> None:
        """Append today's VIX to recent-trend tracker (last 5 days)."""
        self._vix_recent.append(vix)
        if len(self._vix_recent) > 5:
            self._vix_recent = self._vix_recent[-5:]

    def register_position(self, position: Dict[str, Any]) -> None:
        """Track an active strangle for concurrency limit."""
        self._active_positions.append(position)

    def remove_position(self, signal_id: str) -> None:
        """Remove a closed position from the active list."""
        self._active_positions = [
            p for p in self._active_positions
            if p.get('signal_id') != signal_id
        ]


# ================================================================
# Self-test
# ================================================================

if __name__ == '__main__':
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    # --- Connect to DB ---
    try:
        conn = psycopg2.connect(DATABASE_DSN)
    except Exception as exc:
        logger.error("DB connection failed (%s): %s", DATABASE_DSN, exc)
        sys.exit(1)

    # --- Set up components ---
    iv_filt = IVRankFilter(db_conn=conn)
    n = iv_filt.load_vix_history(conn, as_of=date.today())
    if n == 0:
        logger.error("No VIX data found")
        sys.exit(1)

    seller = WeeklyStrangleSeller(
        db_conn=conn,
        iv_rank_filter=iv_filt,
        margin_calc=MarginCalculator(),
    )

    # --- Fetch latest spot and VIX ---
    with conn.cursor() as cur:
        cur.execute("""
            SELECT date, close, india_vix
            FROM   nifty_daily
            WHERE  close IS NOT NULL AND india_vix IS NOT NULL
            ORDER  BY date DESC
            LIMIT  5
        """)
        rows = cur.fetchall()

    if not rows:
        logger.error("No nifty_daily data")
        sys.exit(1)

    # Feed recent VIX for trend detection
    for row_date, row_close, row_vix in reversed(rows):
        seller.update_vix_history(float(row_vix))

    latest_date, spot, vix = rows[0]
    spot = float(spot)
    vix = float(vix)

    print(f"\n== Weekly Strangle Seller — Self-Test ==")
    print(f"   Date : {latest_date}")
    print(f"   Spot : {spot:,.0f}")
    print(f"   VIX  : {vix:.1f}")
    print(f"   IV rank : {iv_filt.compute_iv_rank(vix):.1f}")

    # --- Entry check ---
    today = date.today()
    entry = seller.check_entry(today, spot, vix)
    print(f"\n-- Entry check for {today} (weekday={today.strftime('%A')}) --")
    if entry:
        print(json.dumps(entry, indent=2, default=str))
    else:
        print("   No entry signal.")

    # --- Simulated position management ---
    if entry:
        pos = {
            'signal_id': 'WEEKLY_STRANGLE_SELL',
            'call_strike': entry['call_strike'],
            'put_strike': entry['put_strike'],
            'entry_call_premium': entry['call_premium'],
            'entry_put_premium': entry['put_premium'],
            'entry_combined_premium': entry['combined_premium'],
        }
        # Simulate current premiums at 60% of entry (some decay)
        cur_prem = {
            'call': entry['call_premium'] * 0.6,
            'put': entry['put_premium'] * 0.6,
        }
        mgmt = seller.manage_position(pos, spot, cur_prem, dte=3)
        print(f"\n-- Position management (simulated 40% decay) --")
        print(json.dumps(mgmt, indent=2))

        greeks = seller.calculate_position_greeks(pos, spot, vix, dte=3)
        print(f"\n-- Position Greeks --")
        print(json.dumps(greeks, indent=2))

    # --- Strike selection standalone ---
    print(f"\n-- Strike selection (spot={spot:.0f}, vix={vix:.1f}) --")
    strikes = seller.select_strikes(spot, None, vix)
    if strikes:
        print(json.dumps(strikes, indent=2))
    else:
        print("   No viable strikes found.")

    conn.close()
    print("\nDone.")
