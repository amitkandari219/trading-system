"""
Expiry Day Detector — identifies weekly expiry days and generates gamma scalp signals.

NSE expiry schedule:
- Nifty weekly: Thursday (shifted to Wednesday if Thursday is holiday)
- Bank Nifty weekly: Wednesday (shifted to Tuesday if Wednesday is holiday)
- Monthly: last Thursday of the month

Gamma scalp strategy on expiry days:
- ATM options lose time value rapidly (theta decay accelerates)
- Gamma is highest ATM on expiry day → small underlying move = large option premium move
- Strategy: buy ATM straddle/strangle at 9:30, scalp directional moves
- Key edge: high gamma + low theta (theta already priced out by morning)

Usage:
    from signals.expiry_day_detector import ExpiryDayDetector
    detector = ExpiryDayDetector()

    # Check if today is an expiry day
    info = detector.get_expiry_info(date(2026, 3, 19))

    # Get gamma scalp signals
    signals = detector.check_gamma_signals(bar, prev_bar, session_bars, info)
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# NSE holidays 2026 (known at time of writing — update annually)
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 10),   # Holi
    date(2026, 3, 30),   # Id-Ul-Fitr
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr Ambedkar Jayanti
    date(2026, 5, 1),    # May Day
    date(2026, 6, 5),    # Id-Ul-Adha
    date(2026, 7, 6),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 19),   # Janmashtami
    date(2026, 9, 4),    # Milad-Un-Nabi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali
    date(2026, 11, 10),  # Diwali (Balipratipada)
    date(2026, 11, 27),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}

# Generic approach: if year not in known holidays, just check weekday
# For production, load holidays from NSE website or database


class ExpiryDayDetector:
    """Detect expiry days and generate gamma scalp signals."""

    def __init__(self, holidays: Optional[set] = None):
        """
        Args:
            holidays: set of date objects for NSE holidays.
                      If None, uses 2026 holidays + empty for other years.
        """
        self.holidays = holidays or NSE_HOLIDAYS_2026

    def is_trading_day(self, d: date) -> bool:
        """Check if date is a trading day (not weekend, not holiday)."""
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        if d in self.holidays:
            return False
        return True

    def _prev_trading_day(self, d: date) -> date:
        """Find the previous trading day before d."""
        d = d - timedelta(days=1)
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def get_nifty_weekly_expiry(self, d: date) -> date:
        """
        Get the Nifty weekly expiry for the week containing date d.
        Nifty weekly = Thursday. If Thursday is a holiday, shifts to Wednesday.
        """
        # Find Thursday of this week
        days_to_thu = (3 - d.weekday()) % 7
        thu = d + timedelta(days=days_to_thu)
        if d.weekday() > 3:  # If we're past Thursday, get next week's
            thu += timedelta(days=7)

        if self.is_trading_day(thu):
            return thu
        # Thursday is holiday → Wednesday
        wed = thu - timedelta(days=1)
        if self.is_trading_day(wed):
            return wed
        # Wednesday also holiday → Tuesday
        return self._prev_trading_day(wed)

    def get_banknifty_weekly_expiry(self, d: date) -> date:
        """
        Get Bank Nifty weekly expiry for the week containing date d.
        Bank Nifty weekly = Wednesday. If Wednesday is holiday, shifts to Tuesday.
        """
        days_to_wed = (2 - d.weekday()) % 7
        wed = d + timedelta(days=days_to_wed)
        if d.weekday() > 2:
            wed += timedelta(days=7)

        if self.is_trading_day(wed):
            return wed
        tue = wed - timedelta(days=1)
        if self.is_trading_day(tue):
            return tue
        return self._prev_trading_day(tue)

    def get_monthly_expiry(self, year: int, month: int) -> date:
        """
        Get monthly expiry (last Thursday of the month).
        If last Thursday is holiday, shifts to previous trading day.
        """
        # Find last day of month
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        # Find last Thursday
        days_back = (last_day.weekday() - 3) % 7
        last_thu = last_day - timedelta(days=days_back)

        if self.is_trading_day(last_thu):
            return last_thu
        return self._prev_trading_day(last_thu)

    def get_expiry_info(self, d: date) -> Dict:
        """
        Get comprehensive expiry information for a given date.

        Returns dict with:
            is_nifty_expiry: bool
            is_banknifty_expiry: bool
            is_monthly_expiry: bool
            is_any_expiry: bool
            nifty_expiry_date: date
            banknifty_expiry_date: date
            days_to_nifty_expiry: int
            days_to_banknifty_expiry: int
            expiry_instruments: list of str
        """
        nifty_exp = self.get_nifty_weekly_expiry(d)
        bn_exp = self.get_banknifty_weekly_expiry(d)
        monthly_exp = self.get_monthly_expiry(d.year, d.month)

        is_nifty = (d == nifty_exp)
        is_bn = (d == bn_exp)
        is_monthly = (d == monthly_exp)

        instruments = []
        if is_nifty:
            instruments.append('NIFTY')
        if is_bn:
            instruments.append('BANKNIFTY')

        # Days to expiry (trading days)
        dte_nifty = self._trading_days_between(d, nifty_exp)
        dte_bn = self._trading_days_between(d, bn_exp)

        return {
            'date': d,
            'is_nifty_expiry': is_nifty,
            'is_banknifty_expiry': is_bn,
            'is_monthly_expiry': is_monthly,
            'is_any_expiry': is_nifty or is_bn,
            'nifty_expiry_date': nifty_exp,
            'banknifty_expiry_date': bn_exp,
            'days_to_nifty_expiry': dte_nifty,
            'days_to_banknifty_expiry': dte_bn,
            'expiry_instruments': instruments,
        }

    def _trading_days_between(self, d1: date, d2: date) -> int:
        """Count trading days between d1 and d2 (inclusive of d2)."""
        if d1 >= d2:
            return 0
        count = 0
        d = d1 + timedelta(days=1)
        while d <= d2:
            if self.is_trading_day(d):
                count += 1
            d += timedelta(days=1)
        return count

    # ================================================================
    # GAMMA SCALP SIGNALS
    # ================================================================

    def check_gamma_signals(self, bar, prev_bar, session_bars,
                            expiry_info: Dict) -> List[Dict]:
        """
        Generate gamma scalp signals on expiry days.

        Strategy:
        1. GAMMA_BREAKOUT: Strong directional move in first hour (ORB + volume)
           → Buy ATM CE/PE in direction of breakout
        2. GAMMA_REVERSAL: Failed move + reversal after 11:00
           → Fade the move (options premium catches up fast)
        3. GAMMA_SQUEEZE: Tight range compression then expansion after 13:00
           → Buy straddle components individually

        Args:
            bar: current 5-min bar
            prev_bar: previous bar
            session_bars: all bars today
            expiry_info: from get_expiry_info()

        Returns:
            list of signal dicts
        """
        if not expiry_info.get('is_any_expiry', False):
            return []

        fired = []
        bar_time = bar['datetime']

        # Don't enter after 15:00 on expiry (theta burns fast)
        if bar_time.hour >= 15:
            return []

        instruments = expiry_info.get('expiry_instruments', [])

        for instrument in instruments:
            # Signal 1: GAMMA_BREAKOUT (9:30-11:00)
            if 930 <= bar_time.hour * 100 + bar_time.minute <= 1100:
                sig = self._check_gamma_breakout(bar, prev_bar, session_bars, instrument)
                if sig:
                    fired.append(sig)

            # Signal 2: GAMMA_REVERSAL (11:00-14:00)
            if 1100 <= bar_time.hour * 100 + bar_time.minute <= 1400:
                sig = self._check_gamma_reversal(bar, prev_bar, session_bars, instrument)
                if sig:
                    fired.append(sig)

            # Signal 3: GAMMA_SQUEEZE (13:00-14:30)
            if 1300 <= bar_time.hour * 100 + bar_time.minute <= 1430:
                sig = self._check_gamma_squeeze(bar, prev_bar, session_bars, instrument)
                if sig:
                    fired.append(sig)

        return fired

    def _check_gamma_breakout(self, bar, prev_bar, session_bars,
                              instrument: str) -> Optional[Dict]:
        """
        Expiry day ORB breakout — gamma amplifies the move.
        Tighter stop (0.2%) since gamma works both ways.
        """
        if pd.isna(bar.get('opening_range_high')) or pd.isna(bar.get('vol_ratio_20')):
            return None

        close = float(bar['close'])
        or_high = float(bar['opening_range_high'])
        or_low = float(bar['opening_range_low'])
        vol_ratio = float(bar['vol_ratio_20'])
        prev_close = float(prev_bar['close']) if prev_bar is not None else close

        # Higher volume threshold on expiry (lots of noise)
        if vol_ratio < 1.5:
            return None

        if close > or_high and prev_close <= or_high:
            return {
                'signal_id': f'GAMMA_BREAKOUT_{instrument}',
                'direction': 'LONG',
                'price': close,
                'instrument': instrument,
                'stop_loss_pct': 0.002,   # tight 0.2% SL
                'max_hold_bars': 20,      # max ~100 min hold
                'reason': f'Expiry gamma BO long: {close:.0f} > OR {or_high:.0f} vol={vol_ratio:.1f}',
            }

        if close < or_low and prev_close >= or_low:
            return {
                'signal_id': f'GAMMA_BREAKOUT_{instrument}',
                'direction': 'SHORT',
                'price': close,
                'instrument': instrument,
                'stop_loss_pct': 0.002,
                'max_hold_bars': 20,
                'reason': f'Expiry gamma BO short: {close:.0f} < OR {or_low:.0f} vol={vol_ratio:.1f}',
            }

        return None

    def _check_gamma_reversal(self, bar, prev_bar, session_bars,
                              instrument: str) -> Optional[Dict]:
        """
        Expiry day reversal — afternoon mean reversion when morning move fades.
        Premium collapses on reversal = high R:R.
        """
        if len(session_bars) < 20:
            return None

        close = float(bar['close'])
        open_ = float(bar['open'])
        session_high = float(session_bars['high'].max())
        session_low = float(session_bars['low'].min())
        session_range = session_high - session_low

        if session_range <= 0:
            return None

        # Position in session range
        position = (close - session_low) / session_range

        # Reversal from high: was near top, now falling
        if position < 0.4 and float(session_bars.iloc[-5:]['high'].max()) > session_high * 0.99:
            vwap = float(bar['vwap']) if pd.notna(bar.get('vwap')) else close
            if close < vwap:  # Below VWAP confirms weakness
                return {
                    'signal_id': f'GAMMA_REVERSAL_{instrument}',
                    'direction': 'SHORT',
                    'price': close,
                    'instrument': instrument,
                    'stop_loss_pct': 0.003,
                    'max_hold_bars': 25,
                    'reason': f'Expiry reversal short: pos={position:.0%} below vwap',
                }

        # Reversal from low: was near bottom, now rising
        if position > 0.6 and float(session_bars.iloc[-5:]['low'].min()) < session_low * 1.01:
            vwap = float(bar['vwap']) if pd.notna(bar.get('vwap')) else close
            if close > vwap:
                return {
                    'signal_id': f'GAMMA_REVERSAL_{instrument}',
                    'direction': 'LONG',
                    'price': close,
                    'instrument': instrument,
                    'stop_loss_pct': 0.003,
                    'max_hold_bars': 25,
                    'reason': f'Expiry reversal long: pos={position:.0%} above vwap',
                }

        return None

    def _check_gamma_squeeze(self, bar, prev_bar, session_bars,
                             instrument: str) -> Optional[Dict]:
        """
        Afternoon squeeze breakout — range compression then expansion.
        On expiry, this often triggers stop-loss cascades in options.
        """
        if len(session_bars) < 30:
            return None

        close = float(bar['close'])
        body_pct = float(bar['body_pct']) if pd.notna(bar.get('body_pct')) else 0

        # Need strong current bar
        if body_pct < 0.65:
            return None

        # Check if last 6 bars were compressed (narrow range)
        recent = session_bars.tail(7).head(6)  # 6 bars before current
        if len(recent) < 6:
            return None

        ranges = (recent['high'] - recent['low']).values
        atr_val = float(bar['atr_14']) if pd.notna(bar.get('atr_14')) else np.mean(ranges)

        if atr_val <= 0:
            return None

        # Squeeze: recent ranges < 50% of ATR
        narrow_bars = sum(1 for r in ranges if r < atr_val * 0.5)
        if narrow_bars < 4:
            return None

        # Current bar breaks out of squeeze
        body = float(bar['close']) - float(bar['open'])
        direction = 'LONG' if body > 0 else 'SHORT'

        return {
            'signal_id': f'GAMMA_SQUEEZE_{instrument}',
            'direction': direction,
            'price': close,
            'instrument': instrument,
            'stop_loss_pct': 0.002,
            'max_hold_bars': 15,
            'reason': f'Expiry squeeze {direction.lower()}: {narrow_bars}/6 narrow bars, body={body_pct:.0%}',
        }


# ================================================================
# MAIN — self-test
# ================================================================

if __name__ == '__main__':
    from datetime import date

    detector = ExpiryDayDetector()

    # Test dates
    test_dates = [
        date(2026, 3, 18),  # Wednesday (BN expiry)
        date(2026, 3, 19),  # Thursday (Nifty expiry)
        date(2026, 3, 20),  # Friday (no expiry)
        date(2026, 3, 25),  # Wednesday (BN)
        date(2026, 3, 26),  # Thursday (Nifty + monthly)
    ]

    print("=" * 65)
    print("  EXPIRY DAY DETECTOR — Self-Test")
    print("=" * 65)

    for d in test_dates:
        info = detector.get_expiry_info(d)
        expiry_tags = []
        if info['is_nifty_expiry']:
            expiry_tags.append('NIFTY')
        if info['is_banknifty_expiry']:
            expiry_tags.append('BANKNIFTY')
        if info['is_monthly_expiry']:
            expiry_tags.append('MONTHLY')

        tag_str = ', '.join(expiry_tags) if expiry_tags else 'none'
        print(f"\n  {d} ({d.strftime('%A')})")
        print(f"    Expiry: {tag_str}")
        print(f"    Nifty DTE: {info['days_to_nifty_expiry']}  BN DTE: {info['days_to_banknifty_expiry']}")
        print(f"    Instruments: {info['expiry_instruments']}")
