"""
Intraday signal framework for Nifty F&O.

Implements three well-documented intraday patterns:
1. Opening Range Breakout (ORB) — first 15-minute range
2. Institutional Flow Window (2:15-2:45 PM) — mutual fund/FII closing trades
3. Expiry Day Gamma Squeeze — Thursday weekly expiry effects

These are UNCORRELATED to the daily signals (DRY_20/16/12) which
trade on end-of-day data. Adding even one profitable intraday signal
improves portfolio Sharpe through timeframe diversification.

Data requirements:
- 1-minute or 5-minute OHLCV bars (from Kite API or TrueData)
- Pre-computed daily indicators (ATR, VIX) for position sizing

Note: This is the framework/scaffold. Actual signal logic requires
intraday data which isn't available in the current daily-only DB.
The framework is designed to plug into the existing execution engine
once live data is connected.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntradayBar:
    """A single intraday price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int = 0  # Open interest (for derivatives)


@dataclass
class IntradaySignalResult:
    """Result from an intraday signal check."""
    signal_id: str
    action: Optional[str] = None  # 'ENTER_LONG', 'ENTER_SHORT', 'EXIT', None
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    size_multiplier: float = 1.0
    reason: str = ''
    expiry_time: Optional[time] = None  # When to force-exit
    metadata: Dict = field(default_factory=dict)


class IntradaySignal(ABC):
    """Base class for intraday signals."""

    def __init__(self, signal_id: str):
        self.signal_id = signal_id
        self.position = None  # 'LONG', 'SHORT', or None
        self.entry_price = 0.0
        self.entry_time = None

    @abstractmethod
    def on_bar(self, bar: IntradayBar, context: Dict) -> IntradaySignalResult:
        """
        Process a new intraday bar.

        Args:
            bar: Current price bar
            context: {
                'daily_atr': float,
                'daily_vix': float,
                'daily_regime': str,
                'bars_today': list of IntradayBar,
                'is_expiry': bool,
                'day_of_week': int (0=Mon, 4=Fri),
            }

        Returns:
            IntradaySignalResult with action (or None)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset state at start of new trading day."""
        pass


class OpeningRangeBreakout(IntradaySignal):
    """
    Opening Range Breakout (ORB) — classic intraday pattern.

    Rules:
    1. Define opening range: high/low of first N minutes (default 15)
    2. Entry: price breaks above range high (LONG) or below range low (SHORT)
    3. Stop loss: opposite end of the range
    4. Take profit: 2x the range width
    5. Time exit: force-close at 3:15 PM (no overnight)
    6. VIX filter: skip if VIX > 25 (range too wide, stops get hit)

    ATR-based filtering:
    - Skip if opening range > 1.5x daily ATR (abnormal day)
    - Skip if opening range < 0.3x daily ATR (too tight, false breakouts)
    """

    def __init__(self, range_minutes: int = 15, atr_max_ratio: float = 0.3,
                 atr_min_ratio: float = 0.02, rr_ratio: float = 1.5,
                 vix_max: float = 17.0, long_only: bool = True):
        """
        Tuned params from real Kite data analysis (2025-2026):
          - atr_max_ratio=0.3: Only trade tight opening ranges (<30% of ATR)
          - rr_ratio=1.5: TP at 1.5x OR width
          - vix_max=17: Skip when VIX>17
          - long_only=True: SHORT side loses -₹25K, LONG side profits +₹1.5K
        """
        super().__init__('ORB_15MIN')
        self.range_minutes = range_minutes
        self.atr_max_ratio = atr_max_ratio
        self.atr_min_ratio = atr_min_ratio
        self.rr_ratio = rr_ratio
        self.vix_max = vix_max
        self.long_only = long_only

        # State
        self.range_high = None
        self.range_low = None
        self.range_defined = False
        self.traded_today = False

    def reset(self):
        self.range_high = None
        self.range_low = None
        self.range_defined = False
        self.traded_today = False
        self.position = None
        self.entry_price = 0.0
        self.entry_time = None

    def on_bar(self, bar: IntradayBar, context: Dict) -> IntradaySignalResult:
        result = IntradaySignalResult(signal_id=self.signal_id)
        bars_today = context.get('bars_today', [])
        daily_atr = context.get('daily_atr', 0)
        daily_vix = context.get('daily_vix', 15)

        market_open = time(9, 15)
        range_end = time(9, 15 + self.range_minutes)
        force_exit = time(15, 15)

        current_time = bar.timestamp.time()

        # Phase 1: Build opening range (9:15 - 9:30)
        if not self.range_defined and current_time <= range_end:
            if self.range_high is None:
                self.range_high = bar.high
                self.range_low = bar.low
            else:
                self.range_high = max(self.range_high, bar.high)
                self.range_low = min(self.range_low, bar.low)

            if current_time >= range_end:
                self.range_defined = True
                range_width = self.range_high - self.range_low

                # ATR filter
                if daily_atr > 0:
                    ratio = range_width / daily_atr
                    if ratio > self.atr_max_ratio or ratio < self.atr_min_ratio:
                        self.traded_today = True  # Skip — abnormal range
                        result.metadata = {
                            'skip_reason': f'range/ATR ratio {ratio:.2f} outside [{self.atr_min_ratio}, {self.atr_max_ratio}]'
                        }
                        logger.info(f"ORB skip: range/ATR={ratio:.2f}")

                # VIX filter (tuned: 17 instead of 25)
                if daily_vix > self.vix_max:
                    self.traded_today = True
                    result.metadata = {'skip_reason': f'VIX {daily_vix:.1f} > 25'}

                logger.info(f"ORB range defined: {self.range_low:.0f} - {self.range_high:.0f} "
                           f"(width={range_width:.0f})")

            return result

        # Phase 2: Check for breakout (after range defined)
        if self.range_defined and self.position is None and not self.traded_today:
            if current_time > force_exit:
                return result  # Too late for new entries

            range_width = self.range_high - self.range_low

            if bar.close > self.range_high:
                # Long breakout
                self.position = 'LONG'
                self.entry_price = bar.close
                self.entry_time = bar.timestamp
                self.traded_today = True

                result.action = 'ENTER_LONG'
                result.price = bar.close
                result.stop_loss = self.range_low
                result.take_profit = bar.close + range_width * self.rr_ratio
                result.expiry_time = force_exit
                result.reason = f'ORB long breakout above {self.range_high:.0f}'
                logger.info(f"ORB LONG entry @ {bar.close:.0f}, SL={self.range_low:.0f}")

            elif bar.close < self.range_low and not self.long_only:
                # Short breakout (disabled by default — SHORT side loses on Nifty)
                self.position = 'SHORT'
                self.entry_price = bar.close
                self.entry_time = bar.timestamp
                self.traded_today = True

                result.action = 'ENTER_SHORT'
                result.price = bar.close
                result.stop_loss = self.range_high
                result.take_profit = bar.close - range_width * self.rr_ratio
                result.expiry_time = force_exit
                result.reason = f'ORB short breakout below {self.range_low:.0f}'
                logger.info(f"ORB SHORT entry @ {bar.close:.0f}, SL={self.range_high:.0f}")

        # Phase 3: Manage open position
        elif self.position is not None:
            range_width = self.range_high - self.range_low

            # Force exit at 3:15 PM
            if current_time >= force_exit:
                pnl = (bar.close - self.entry_price) if self.position == 'LONG' else (self.entry_price - bar.close)
                result.action = 'EXIT'
                result.price = bar.close
                result.reason = f'time_exit (3:15 PM), pnl={pnl:.0f}'
                self.position = None
                return result

            # Stop loss
            if self.position == 'LONG' and bar.low <= self.range_low:
                result.action = 'EXIT'
                result.price = self.range_low
                result.reason = f'stop_loss @ {self.range_low:.0f}'
                self.position = None
            elif self.position == 'SHORT' and bar.high >= self.range_high:
                result.action = 'EXIT'
                result.price = self.range_high
                result.reason = f'stop_loss @ {self.range_high:.0f}'
                self.position = None

            # Take profit
            elif self.position == 'LONG':
                tp = self.entry_price + range_width * self.rr_ratio
                if bar.high >= tp:
                    result.action = 'EXIT'
                    result.price = tp
                    result.reason = f'take_profit @ {tp:.0f}'
                    self.position = None
            elif self.position == 'SHORT':
                tp = self.entry_price - range_width * self.rr_ratio
                if bar.low <= tp:
                    result.action = 'EXIT'
                    result.price = tp
                    result.reason = f'take_profit @ {tp:.0f}'
                    self.position = None

        return result


class InstitutionalFlowWindow(IntradaySignal):
    """
    Institutional flow window signal (2:15-2:45 PM IST).

    Mutual funds and FIIs execute their closing trades in this window,
    creating predictable directional flow. If Nifty is trending into
    this window, institutions amplify the move.

    Rules:
    1. Compute price change from 2:00 PM to 2:15 PM (pre-window momentum)
    2. If momentum > +0.3%: LONG at 2:15 PM, exit at 3:15 PM
    3. If momentum < -0.3%: SHORT at 2:15 PM, exit at 3:15 PM
    4. Stop loss: 0.5% from entry
    5. Only trade on days where ADX > 20 (need some trend)
    """

    def __init__(self, momentum_threshold: float = 0.003,
                 stop_loss_pct: float = 0.005):
        super().__init__('INST_FLOW_WINDOW')
        self.momentum_threshold = momentum_threshold
        self.stop_loss_pct = stop_loss_pct
        self.price_at_2pm = None
        self.traded_today = False

    def reset(self):
        self.price_at_2pm = None
        self.traded_today = False
        self.position = None
        self.entry_price = 0.0

    def on_bar(self, bar: IntradayBar, context: Dict) -> IntradaySignalResult:
        result = IntradaySignalResult(signal_id=self.signal_id)
        current_time = bar.timestamp.time()

        # Record 2:00 PM price
        if time(14, 0) <= current_time <= time(14, 5) and self.price_at_2pm is None:
            self.price_at_2pm = bar.close
            return result

        # Check momentum at 2:15 PM
        if (time(14, 14) <= current_time <= time(14, 16) and
                not self.traded_today and self.price_at_2pm):

            adx = context.get('daily_adx', 20)
            if adx < 20:
                self.traded_today = True
                return result

            momentum = (bar.close - self.price_at_2pm) / self.price_at_2pm
            sl_points = bar.close * self.stop_loss_pct

            if momentum > self.momentum_threshold:
                self.position = 'LONG'
                self.entry_price = bar.close
                self.traded_today = True
                result.action = 'ENTER_LONG'
                result.price = bar.close
                result.stop_loss = bar.close - sl_points
                result.expiry_time = time(15, 15)
                result.reason = f'inst_flow long (2PM momentum={momentum:.2%})'
                logger.info(f"INST_FLOW LONG @ {bar.close:.0f}, momentum={momentum:.2%}")

            elif momentum < -self.momentum_threshold:
                self.position = 'SHORT'
                self.entry_price = bar.close
                self.traded_today = True
                result.action = 'ENTER_SHORT'
                result.price = bar.close
                result.stop_loss = bar.close + sl_points
                result.expiry_time = time(15, 15)
                result.reason = f'inst_flow short (2PM momentum={momentum:.2%})'
                logger.info(f"INST_FLOW SHORT @ {bar.close:.0f}, momentum={momentum:.2%}")

            else:
                self.traded_today = True  # No signal today

        # Manage position
        if self.position and current_time >= time(15, 15):
            pnl = (bar.close - self.entry_price) if self.position == 'LONG' else (self.entry_price - bar.close)
            result.action = 'EXIT'
            result.price = bar.close
            result.reason = f'time_exit 3:15PM, pnl={pnl:.0f}'
            self.position = None

        return result


class ExpiryDayGamma(IntradaySignal):
    """
    Expiry day gamma squeeze signal.

    On weekly Nifty expiry days (Thursday), options market makers
    are short gamma near ATM strikes. This creates:
    - Pin risk: price gravitates to max pain strike
    - Gamma squeeze: if price moves away from pin, acceleration follows

    Rules:
    1. Only active on expiry days (Thursday, or Wednesday if Thursday holiday)
    2. Identify max pain strike from options chain
    3. If price deviates > 0.5% from max pain after 1 PM: trade in direction of deviation
    4. If price stays within 0.3% of max pain: trade the pin (mean reversion)
    5. Exit: 3:15 PM (before expiry settlement at 3:30)
    """

    def __init__(self):
        super().__init__('EXPIRY_GAMMA')
        self.max_pain_strike = None
        self.traded_today = False

    def reset(self):
        self.max_pain_strike = None
        self.traded_today = False
        self.position = None
        self.entry_price = 0.0

    def on_bar(self, bar: IntradayBar, context: Dict) -> IntradaySignalResult:
        result = IntradaySignalResult(signal_id=self.signal_id)

        # Only on expiry days
        if not context.get('is_expiry', False):
            return result

        current_time = bar.timestamp.time()

        # Need max pain strike (from options chain analysis)
        if self.max_pain_strike is None:
            mp = context.get('max_pain_strike')
            if mp:
                self.max_pain_strike = mp
                logger.info(f"Expiry gamma: max pain = {mp:.0f}")
            else:
                return result

        # Entry after 1 PM
        if (time(13, 0) <= current_time <= time(13, 5) and
                not self.traded_today and self.max_pain_strike):

            deviation = (bar.close - self.max_pain_strike) / self.max_pain_strike

            if abs(deviation) > 0.005:
                # Gamma squeeze: trade in direction of deviation
                if deviation > 0:
                    self.position = 'LONG'
                    result.action = 'ENTER_LONG'
                    result.reason = f'gamma squeeze long (dev={deviation:.2%} from MP={self.max_pain_strike:.0f})'
                else:
                    self.position = 'SHORT'
                    result.action = 'ENTER_SHORT'
                    result.reason = f'gamma squeeze short (dev={deviation:.2%} from MP={self.max_pain_strike:.0f})'

                self.entry_price = bar.close
                result.price = bar.close
                result.stop_loss = self.max_pain_strike  # Stop at max pain
                result.expiry_time = time(15, 15)
                self.traded_today = True
                logger.info(f"EXPIRY_GAMMA {self.position} @ {bar.close:.0f}")

            elif abs(deviation) < 0.003:
                # Pin trade: mean reversion toward max pain
                # If slightly above MP, short; slightly below, long
                if deviation > 0:
                    self.position = 'SHORT'
                    result.action = 'ENTER_SHORT'
                else:
                    self.position = 'LONG'
                    result.action = 'ENTER_LONG'

                self.entry_price = bar.close
                result.price = bar.close
                result.take_profit = self.max_pain_strike
                result.stop_loss = bar.close * (1 + 0.003 * (1 if self.position == 'SHORT' else -1))
                result.expiry_time = time(15, 15)
                result.reason = f'pin_trade toward MP={self.max_pain_strike:.0f}'
                self.traded_today = True

        # Force exit at 3:15 PM (before 3:30 settlement)
        if self.position and current_time >= time(15, 15):
            pnl = (bar.close - self.entry_price) if self.position == 'LONG' else (self.entry_price - bar.close)
            result.action = 'EXIT'
            result.price = bar.close
            result.reason = f'expiry_exit 3:15PM, pnl={pnl:.0f}'
            self.position = None

        return result


class IntradayEngine:
    """
    Engine that manages multiple intraday signals.

    Coordinates signal execution, position tracking, and daily resets.
    Designed to be called by the execution engine on each new bar.
    """

    def __init__(self):
        self.signals: List[IntradaySignal] = [
            OpeningRangeBreakout(),
            InstitutionalFlowWindow(),
            ExpiryDayGamma(),
        ]
        self.current_date = None

    def on_new_day(self, trading_date: date):
        """Reset all signals for new trading day."""
        self.current_date = trading_date
        for sig in self.signals:
            sig.reset()
        logger.info(f"Intraday engine reset for {trading_date}")

    def process_bar(self, bar: IntradayBar, context: Dict) -> List[IntradaySignalResult]:
        """
        Process a new bar through all signals.

        Returns list of non-empty results (signals that fired or exited).
        """
        results = []
        for sig in self.signals:
            try:
                result = sig.on_bar(bar, context)
                if result.action:
                    results.append(result)
            except Exception as e:
                logger.error(f"Intraday signal {sig.signal_id} error: {e}")
        return results

    def get_open_positions(self) -> Dict[str, str]:
        """Return dict of signal_id -> position direction for open positions."""
        return {
            sig.signal_id: sig.position
            for sig in self.signals
            if sig.position is not None
        }
