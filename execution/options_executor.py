"""
Options Executor — converts directional signals into option orders.

Maps LONG signals → buy ATM CE, SHORT signals → buy ATM PE.
Handles lot sizing based on equity, premium stop-loss, and session-scoped exits.

Designed for the ₹2L→₹10K/day roadmap:
- 50% capital deployed per trade (rest is buffer)
- ATM options: delta ~0.50, premium ₹150–250 per unit
- Stop loss: 30% of premium paid
- Target: 50% of premium (2:1 risk-reward on notional)
- Forced exit at 15:20 IST (no overnight option holds)

Usage:
    from execution.options_executor import OptionsExecutor
    executor = OptionsExecutor(equity=200_000)
    orders = executor.signal_to_orders(signal, nifty_ltp=23500)
"""

import logging
import math
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple

from execution.spread_builder import SpreadBuilder, SpreadOrder, GAMMA_SIGNAL_IDS

logger = logging.getLogger(__name__)


# ================================================================
# CONSTANTS
# ================================================================

NIFTY_LOT_SIZE = 25
BANKNIFTY_LOT_SIZE = 15

# NSE strike intervals
NIFTY_STRIKE_INTERVAL = 50
BANKNIFTY_STRIKE_INTERVAL = 100

# Session boundaries (IST)
SESSION_OPEN = dt_time(9, 15)
SESSION_CLOSE = dt_time(15, 30)
FORCE_EXIT_TIME = dt_time(15, 20)
NO_ENTRY_AFTER = dt_time(15, 10)

# Default premium estimates when live chain unavailable
DEFAULT_ATM_PREMIUM = {
    'NIFTY': 200,       # ₹200 per unit for ATM weekly
    'BANKNIFTY': 300,   # ₹300 per unit for ATM weekly
}

# Risk parameters
DEPLOY_FRACTION = 0.50       # deploy 50% of equity per trade
PREMIUM_STOP_LOSS = 0.30     # exit if premium drops 30%
PREMIUM_TARGET = 0.50        # target 50% premium gain
MAX_LOTS_CAP = 40            # absolute lot cap regardless of equity
MIN_LOTS = 1                 # minimum 1 lot


class OptionsExecutor:
    """
    Converts directional signals into option buy orders.

    Flow:
        1. Signal fires (LONG/SHORT) with entry price
        2. Determine instrument (NIFTY/BANKNIFTY)
        3. Pick ATM strike from underlying LTP
        4. Size position: lots = (equity × deploy%) / (premium × lot_size)
        5. Set stop loss at premium × (1 - 30%) and target at premium × (1 + 50%)
        6. Return order dict ready for ExecutionEngine
    """

    def __init__(self, equity: float, deploy_fraction: float = DEPLOY_FRACTION,
                 premium_sl_pct: float = PREMIUM_STOP_LOSS,
                 premium_target_pct: float = PREMIUM_TARGET,
                 max_lots: int = MAX_LOTS_CAP,
                 kite=None, paper_mode: bool = True):
        self.equity = equity
        self.deploy_fraction = deploy_fraction
        self.premium_sl_pct = premium_sl_pct
        self.premium_target_pct = premium_target_pct
        self.max_lots = max_lots

        # Spread builder for multi-leg strategies
        self._spread_builder = SpreadBuilder(kite=kite, paper_mode=paper_mode)

        # Track daily P&L for loss limit
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date = None

    # ================================================================
    # PUBLIC API
    # ================================================================

    def signal_to_spread_orders(
        self,
        signal: Dict,
        underlying_ltp: float,
        instrument: str = 'NIFTY',
        atr: float = 300.0,
        vix: float = 15.0,
        regime: str = 'NORMAL',
        expiry_date: Optional[str] = None,
        lots: int = 0,
    ) -> Optional[SpreadOrder]:
        """
        Convert a fired signal into a SpreadOrder (multi-leg strategy).

        Gamma signals bypass spreads and return None (caller should use
        signal_to_orders with use_spreads=False).

        Args:
            signal:         dict with {signal_id, direction, price, reason}
            underlying_ltp: current underlying price
            instrument:     'NIFTY' or 'BANKNIFTY'
            atr:            daily ATR
            vix:            India VIX
            regime:         regime string
            expiry_date:    option expiry 'YYMMDD'
            lots:           override lot count (0 = auto from sizer)

        Returns:
            SpreadOrder or None.
        """
        now = datetime.now()
        self._maybe_reset_daily(now)

        if not self._is_entry_allowed(now):
            logger.info(f"Entry blocked: outside trading window at {now.time()}")
            return None

        signal_id = signal.get('signal_id', 'UNKNOWN')
        direction = signal.get('direction', '').upper()

        # Gamma signals should NOT use spreads
        if signal_id in GAMMA_SIGNAL_IDS or signal_id.startswith('GAMMA_'):
            logger.info(f"Gamma signal {signal_id} — bypassing spreads")
            return None

        if lots <= 0:
            lot_size = NIFTY_LOT_SIZE if instrument == 'NIFTY' else BANKNIFTY_LOT_SIZE
            premium_est = atm_premium if (atm_premium := signal.get('_premium')) else 200
            lots = self._compute_lots(premium_est, lot_size)

        if lots < MIN_LOTS:
            return None

        spread = self._spread_builder.select_strategy(
            signal_id=signal_id,
            direction=direction,
            instrument=instrument,
            price=underlying_ltp,
            atr=atr,
            vix=vix,
            regime=regime,
            equity=self.equity,
            expiry_date=expiry_date,
            lots=lots,
        )

        if spread is not None:
            self._daily_trades += 1

        return spread

    def signal_to_orders(self, signal: Dict, underlying_ltp: float,
                         instrument: str = 'NIFTY',
                         atm_premium: Optional[float] = None,
                         expiry_date: Optional[str] = None,
                         use_spreads: bool = False,
                         atr: float = 300.0,
                         vix: float = 15.0,
                         regime: str = 'NORMAL') -> Optional[Dict]:
        """
        Convert a fired signal into an option order.

        Args:
            signal: dict with keys {signal_id, direction, price, reason}
            underlying_ltp: current price of underlying (Nifty/BankNifty)
            instrument: 'NIFTY' or 'BANKNIFTY'
            atm_premium: live ATM option premium (if None, uses estimate)
            expiry_date: option expiry in 'YYMMDD' format (for symbol construction)
            use_spreads: if True, delegate to signal_to_spread_orders for
                         non-gamma signals (backward compatible — default False)
            atr: daily ATR (used when use_spreads=True)
            vix: India VIX (used when use_spreads=True)
            regime: regime string (used when use_spreads=True)

        Returns:
            Order dict ready for execution engine, or None if blocked.
            When use_spreads=True and a spread is constructed, returns the
            SpreadOrder object instead of a plain dict.
        """
        # Delegate to spread builder for non-gamma signals when requested
        if use_spreads:
            signal_id = signal.get('signal_id', 'UNKNOWN')
            is_gamma = (
                signal_id in GAMMA_SIGNAL_IDS
                or signal_id.startswith('GAMMA_')
            )
            if not is_gamma:
                return self.signal_to_spread_orders(
                    signal=signal,
                    underlying_ltp=underlying_ltp,
                    instrument=instrument,
                    atr=atr,
                    vix=vix,
                    regime=regime,
                    expiry_date=expiry_date,
                )

        now = datetime.now()

        # Reset daily counters
        self._maybe_reset_daily(now)

        # Time gate
        if not self._is_entry_allowed(now):
            logger.info(f"Entry blocked: outside trading window at {now.time()}")
            return None

        # Daily loss limit check (5% of equity)
        daily_loss_limit = self.equity * 0.05
        if self._daily_pnl < -daily_loss_limit:
            logger.warning(f"Daily loss limit hit: ₹{self._daily_pnl:,.0f} "
                           f"(limit: -₹{daily_loss_limit:,.0f})")
            return None

        direction = signal.get('direction', '').upper()
        if direction not in ('LONG', 'SHORT'):
            logger.error(f"Invalid direction: {direction}")
            return None

        # Option type: LONG → CE (call), SHORT → PE (put)
        option_type = 'CE' if direction == 'LONG' else 'PE'

        # ATM strike
        strike = self._atm_strike(underlying_ltp, instrument)

        # Premium
        premium = atm_premium or DEFAULT_ATM_PREMIUM.get(instrument, 200)

        # Lot sizing
        lot_size = NIFTY_LOT_SIZE if instrument == 'NIFTY' else BANKNIFTY_LOT_SIZE
        lots = self._compute_lots(premium, lot_size)

        if lots < MIN_LOTS:
            logger.info(f"Computed lots={lots} below minimum — skipping")
            return None

        # Stop loss and target (on premium)
        sl_premium = round(premium * (1 - self.premium_sl_pct), 2)
        target_premium = round(premium * (1 + self.premium_target_pct), 2)

        # Total risk for this trade
        quantity = lots * lot_size
        max_loss = (premium - sl_premium) * quantity
        max_gain = (target_premium - premium) * quantity

        # Build trading symbol
        # Format: NIFTY{YYMMDD}{STRIKE}{CE/PE}
        expiry_str = expiry_date or self._next_weekly_expiry(now)
        tradingsymbol = f"{instrument}{expiry_str}{strike}{option_type}"

        order = {
            'signal_id': signal['signal_id'],
            'signal_direction': direction,
            'signal_reason': signal.get('reason', ''),

            # Order fields (compatible with ExecutionEngine)
            'tradingsymbol': tradingsymbol,
            'exchange': 'NFO',
            'transaction_type': 'BUY',
            'quantity': quantity,
            'product': 'NRML',
            'order_type': 'LIMIT',
            'limit_price': premium,

            # Options metadata
            'instrument': instrument,
            'option_type': option_type,
            'strike': strike,
            'lots': lots,
            'lot_size': lot_size,
            'entry_premium': premium,
            'underlying_ltp': underlying_ltp,

            # Risk management
            'sl_premium': sl_premium,
            'target_premium': target_premium,
            'max_loss': max_loss,
            'max_gain': max_gain,
            'risk_reward': round(max_gain / max_loss, 2) if max_loss > 0 else 0,

            # Session management
            'force_exit_time': FORCE_EXIT_TIME.strftime('%H:%M'),
            'entry_time': now.strftime('%H:%M:%S'),

            # Execution priority (from execution_engine.py)
            'execution_priority': 4,  # OPTIONS_ATM_ENTRY
        }

        logger.info(
            f"Options order: {tradingsymbol} × {lots} lots "
            f"@ ₹{premium} | SL ₹{sl_premium} | TGT ₹{target_premium} "
            f"| Risk ₹{max_loss:,.0f} | Reward ₹{max_gain:,.0f}"
        )

        self._daily_trades += 1
        return order

    def process_exit(self, position: Dict, exit_premium: float,
                     exit_reason: str = 'SIGNAL') -> Dict:
        """
        Process an exit and update daily P&L.

        Args:
            position: the original order dict (from signal_to_orders)
            exit_premium: premium at exit
            exit_reason: 'SIGNAL', 'STOP_LOSS', 'TARGET', 'FORCE_EXIT', 'EOD'

        Returns:
            Trade result dict with P&L.
        """
        entry_premium = position['entry_premium']
        quantity = position['quantity']

        pnl = (exit_premium - entry_premium) * quantity
        pnl_pct = (exit_premium - entry_premium) / entry_premium if entry_premium > 0 else 0

        # Update daily tracking
        self._daily_pnl += pnl

        # Update equity (compound)
        self.equity += pnl

        result = {
            'signal_id': position['signal_id'],
            'tradingsymbol': position['tradingsymbol'],
            'direction': position['signal_direction'],
            'entry_premium': entry_premium,
            'exit_premium': exit_premium,
            'quantity': quantity,
            'lots': position['lots'],
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 4),
            'exit_reason': exit_reason,
            'equity_after': round(self.equity, 2),
            'daily_pnl': round(self._daily_pnl, 2),
        }

        emoji = "+" if pnl >= 0 else ""
        logger.info(
            f"EXIT {position['tradingsymbol']}: {exit_reason} "
            f"| {emoji}₹{pnl:,.0f} ({pnl_pct:+.1%}) "
            f"| Day P&L: ₹{self._daily_pnl:,.0f} | Equity: ₹{self.equity:,.0f}"
        )

        return result

    def check_position_exits(self, position: Dict,
                             current_premium: float) -> Optional[str]:
        """
        Check if a position should be exited.

        Returns exit reason string or None.
        """
        now = datetime.now().time()

        # Force exit at 15:20
        if now >= FORCE_EXIT_TIME:
            return 'FORCE_EXIT'

        entry_premium = position['entry_premium']

        # Stop loss hit
        if current_premium <= position['sl_premium']:
            return 'STOP_LOSS'

        # Target hit
        if current_premium >= position['target_premium']:
            return 'TARGET'

        # Trailing stop: if premium gained > 30%, trail SL to breakeven
        if current_premium >= entry_premium * 1.30:
            if current_premium < entry_premium * 1.10:
                return 'TRAILING_STOP'

        return None

    def update_equity(self, new_equity: float):
        """Update equity (e.g., after deposit/withdrawal)."""
        old = self.equity
        self.equity = new_equity
        logger.info(f"Equity updated: ₹{old:,.0f} → ₹{new_equity:,.0f}")

    def get_status(self) -> Dict:
        """Return current executor state."""
        return {
            'equity': self.equity,
            'deploy_fraction': self.deploy_fraction,
            'deployable': self.equity * self.deploy_fraction,
            'daily_pnl': self._daily_pnl,
            'daily_trades': self._daily_trades,
            'daily_loss_limit': -self.equity * 0.05,
            'max_lots': self.max_lots,
        }

    # ================================================================
    # PRIVATE HELPERS
    # ================================================================

    def _compute_lots(self, premium: float, lot_size: int) -> int:
        """
        Compute number of lots based on equity and deployment fraction.

        lots = (equity × deploy_fraction) / (premium × lot_size)
        """
        deployable = self.equity * self.deploy_fraction
        cost_per_lot = premium * lot_size

        if cost_per_lot <= 0:
            return 0

        lots = int(deployable / cost_per_lot)
        lots = min(lots, self.max_lots)
        lots = max(lots, 0)

        logger.debug(
            f"Sizing: ₹{deployable:,.0f} deployable / "
            f"₹{cost_per_lot:,.0f} per lot = {lots} lots"
        )
        return lots

    def _atm_strike(self, ltp: float, instrument: str) -> int:
        """Round LTP to nearest strike interval."""
        interval = (NIFTY_STRIKE_INTERVAL if instrument == 'NIFTY'
                    else BANKNIFTY_STRIKE_INTERVAL)
        return int(round(ltp / interval) * interval)

    def _is_entry_allowed(self, now: datetime) -> bool:
        """Check if current time is within entry window."""
        t = now.time()
        return SESSION_OPEN <= t <= NO_ENTRY_AFTER

    def _maybe_reset_daily(self, now: datetime):
        """Reset daily counters at start of new session."""
        today = now.date()
        if self._last_reset_date != today:
            if self._last_reset_date is not None:
                logger.info(
                    f"New session {today}: prior day P&L ₹{self._daily_pnl:,.0f}, "
                    f"{self._daily_trades} trades"
                )
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset_date = today

    def _next_weekly_expiry(self, now: datetime) -> str:
        """
        Estimate next weekly expiry in YYMMDD format.
        Nifty weekly expiry = Thursday. If today is Thu post-market, use next Thu.
        """
        from datetime import timedelta
        today = now.date()
        # Thursday = 3
        days_ahead = (3 - today.weekday()) % 7
        if days_ahead == 0 and now.time() > dt_time(15, 30):
            days_ahead = 7
        expiry = today + timedelta(days=days_ahead)
        return expiry.strftime('%y%m%d')


# ================================================================
# CONVENIENCE: batch processing for multiple signals
# ================================================================

def process_signal_batch(signals: List[Dict], executor: OptionsExecutor,
                         underlying_ltp: float, instrument: str = 'NIFTY',
                         atm_premium: Optional[float] = None) -> List[Dict]:
    """
    Process a batch of fired signals through the options executor.

    Returns list of order dicts (skipped signals produce no entry).
    """
    orders = []
    for sig in signals:
        order = executor.signal_to_orders(
            signal=sig,
            underlying_ltp=underlying_ltp,
            instrument=instrument,
            atm_premium=atm_premium,
        )
        if order:
            orders.append(order)
    return orders


# ================================================================
# MAIN — demo / self-test
# ================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    print("=" * 70)
    print("  OPTIONS EXECUTOR — Self-Test")
    print("=" * 70)

    executor = OptionsExecutor(equity=200_000)

    # Simulate signals from the two proven intraday strategies
    test_signals = [
        {'signal_id': 'ID_KAUFMAN_BB_MR', 'direction': 'LONG',
         'price': 23520, 'reason': 'BB MR long: close near lower band, ADX<25'},
        {'signal_id': 'ID_GUJRAL_RANGE', 'direction': 'SHORT',
         'price': 23680, 'reason': 'Range short: close near upper boundary'},
        {'signal_id': 'L9_ORB_BREAKOUT', 'direction': 'LONG',
         'price': 23550, 'reason': 'ORB long: close > OR_high with volume'},
    ]

    print(f"\nEquity: ₹{executor.equity:,.0f}")
    print(f"Deployable: ₹{executor.equity * executor.deploy_fraction:,.0f}")
    print()

    for sig in test_signals:
        print(f"\n{'─' * 60}")
        print(f"Signal: {sig['signal_id']} {sig['direction']} @ {sig['price']}")

        order = executor.signal_to_orders(
            signal=sig,
            underlying_ltp=sig['price'],
            instrument='NIFTY',
            atm_premium=200,
        )

        if order:
            print(f"  Symbol:    {order['tradingsymbol']}")
            print(f"  Lots:      {order['lots']} × {order['lot_size']} = {order['quantity']} units")
            print(f"  Premium:   ₹{order['entry_premium']}")
            print(f"  Cost:      ₹{order['entry_premium'] * order['quantity']:,.0f}")
            print(f"  SL:        ₹{order['sl_premium']} (-{executor.premium_sl_pct:.0%})")
            print(f"  Target:    ₹{order['target_premium']} (+{executor.premium_target_pct:.0%})")
            print(f"  Max Loss:  ₹{order['max_loss']:,.0f}")
            print(f"  Max Gain:  ₹{order['max_gain']:,.0f}")
            print(f"  R:R:       1:{order['risk_reward']}")

            # Simulate exit
            print(f"\n  Simulating target hit at ₹{order['target_premium']}...")
            result = executor.process_exit(order, order['target_premium'], 'TARGET')
            print(f"  P&L: ₹{result['pnl']:,.0f} ({result['pnl_pct']:+.1%})")
            print(f"  Equity: ₹{result['equity_after']:,.0f}")

    print(f"\n{'=' * 70}")
    status = executor.get_status()
    print(f"Final Status:")
    print(f"  Equity:      ₹{status['equity']:,.0f}")
    print(f"  Day P&L:     ₹{status['daily_pnl']:,.0f}")
    print(f"  Day Trades:  {status['daily_trades']}")
    print(f"  Loss Limit:  ₹{status['daily_loss_limit']:,.0f}")
