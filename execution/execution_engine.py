"""
Execution engine — processes signals from the Redis ORDER_QUEUE.

Wires together the full execution pipeline:
    daily_loss_limiter.can_trade()
    -> greek_pre_check()
    -> compound_sizer.get_lots()
    -> kite_bridge.place_order()
    -> fill_monitor.monitor_fill()
    -> record trade

Handles priority ordering, time guards, position limits, fill windows,
failure scenarios, emergency exits, and EOD reconciliation.
"""

import json
import os
import time
import logging
import threading
from datetime import datetime, time as dt_time, date

from config import settings
from execution.kite_bridge import KiteBridge
from execution.fill_monitor import FillMonitor
from execution.position_reconciler import PositionReconciler
from execution.greek_pre_check import greek_pre_check
from risk.daily_loss_limiter import DailyLossLimiter
from risk.compound_sizer import CompoundSizer
from risk.behavioral_overlay import BehavioralOverlay, OverlayContext


EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

EXECUTION_PRIORITY = {
    'EXIT_SIGNAL': 1,
    'STOP_LOSS_TRIGGER': 1,
    'GREEK_HEDGE': 2,
    'FUTURES_ENTRY': 3,
    'OPTIONS_ATM_ENTRY': 4,
    'OPTIONS_OTM_ENTRY': 5,
    'SPREAD_ENTRY': 6,
    'CONDOR_ENTRY': 7,
}

# Fill window by strategy type (seconds)
FILL_WINDOWS = {
    'SINGLE_LEG': 90,
    'TWO_LEG': 60,
    'FOUR_LEG_PER_LEG': 45,
}

# Trading time guard (IST)
MARKET_OPEN = dt_time(9, 16)
MARKET_CLOSE = dt_time(15, 19)

# Position limits
MAX_POSITIONS = settings.MAX_POSITIONS          # 4
MAX_SAME_DIRECTION = settings.MAX_SAME_DIRECTION  # 2


class ExecutionEngine:
    """
    Processes signals from Redis ORDER_QUEUE one at a time.
    Waits for current signal to resolve before processing the next.

    In PAPER mode, all logic runs but no Kite API calls are made.
    """

    def __init__(self, db, redis_client, kite, alerter,
                 loss_limiter: DailyLossLimiter = None,
                 sizer: CompoundSizer = None,
                 logger=None):
        """
        Args:
            db:           database connection
            redis_client: redis.Redis instance
            kite:         KiteConnect instance (None OK in PAPER mode)
            alerter:      TelegramAlerter instance
            loss_limiter: DailyLossLimiter instance
            sizer:        CompoundSizer instance
            logger:       Python Logger
        """
        self.db = db
        self.redis = redis_client
        self.kite = kite
        self.alerter = alerter
        self.logger = logger or logging.getLogger(__name__)

        # Risk components
        self.loss_limiter = loss_limiter or DailyLossLimiter(
            equity=settings.TOTAL_CAPITAL
        )
        self.sizer = sizer or CompoundSizer(
            initial_equity=settings.TOTAL_CAPITAL
        )

        # Behavioral overlay (Layer 7)
        self.behavioral_overlay = BehavioralOverlay()
        self.sizer.set_behavioral_overlay(self.behavioral_overlay)

        # Execution components
        self.bridge = KiteBridge(kite, db, alerter, logger_override=self.logger)
        self.fill_monitor = FillMonitor(kite, alerter, db, logger_override=self.logger)
        self.fill_monitor.start()  # Start background fill polling
        self.reconciler = PositionReconciler(kite, db, alerter, logger_override=self.logger)

        self.daily_loss_limit_hit = False

        # Threading lock to prevent race conditions between position-limit
        # check and order placement (FOR UPDATE may not work on aggregates
        # in all PostgreSQL versions)
        self._position_lock = threading.Lock()

    # ==================================================================
    # PUBLIC: Queue processing
    # ==================================================================

    def process_queue(self):
        """
        Main loop: read from ORDER_QUEUE, sort by priority, process sequentially.

        Priority: EXIT=1, HEDGE=2, ENTRY=3+
        Within same priority, sort by Sharpe descending.
        """
        # Time guard
        if not self._is_trading_time():
            return

        messages = self.redis.xrange('ORDER_QUEUE', '-', '+')
        if not messages:
            return

        # Decode Redis bytes to strings if needed
        decoded = []
        for msg_id, fields in messages:
            clean = {}
            for k, v in fields.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                clean[key] = val
            decoded.append((msg_id, clean))

        # Sort: exit signals first, then by priority, then by Sharpe
        sorted_msgs = sorted(
            decoded,
            key=lambda m: (
                int(m[1].get('execution_priority', 99)),
                -float(m[1].get('sharpe_ratio', 0)),
            )
        )

        for msg_id, fields in sorted_msgs:
            if self.daily_loss_limit_hit:
                self.logger.info("Daily loss limit hit — skipping remaining orders")
                break
            self._process_signal(msg_id, fields)
            # Acknowledge processed message
            self.redis.xdel('ORDER_QUEUE', msg_id)

    def emergency_exit_all(self):
        """
        Close ALL open positions immediately at market price.
        Called on Tier 2 loss limit, system crash, or manual trigger.
        """
        self.logger.critical("EMERGENCY EXIT ALL — closing all open positions")
        self.alerter.send(
            'EMERGENCY',
            'EMERGENCY EXIT ALL triggered — closing all positions at market'
        )

        # Get open positions from DB
        try:
            result = self.db.execute("""
                SELECT signal_id, tradingsymbol, transaction_type,
                       quantity, lots, lot_size
                FROM trades
                WHERE exit_date IS NULL
                  AND DATE(created_at) = CURRENT_DATE
            """)
            open_positions = result.fetchall() if result else []
        except Exception as e:
            self.logger.error(f"Failed to fetch open positions: {e}")
            open_positions = []

        if not open_positions:
            self.logger.info("No open positions to close")
            return

        for pos in open_positions:
            if isinstance(pos, dict):
                symbol = pos['tradingsymbol']
                qty = pos['quantity']
                tx_type = pos['transaction_type']
                signal_id = pos['signal_id']
            else:
                signal_id, symbol, tx_type, qty = pos[0], pos[1], pos[2], pos[3]

            # Reverse the transaction
            close_tx = 'SELL' if tx_type == 'BUY' else 'BUY'
            close_order = {
                'signal_id': f"EMERG_{signal_id}",
                'tradingsymbol': symbol,
                'transaction_type': close_tx,
                'lots': 1,  # quantity is already in units
                'lot_size': abs(qty),  # hack: lots*lot_size = qty
                'limit_price': 0,
                'instrument': 'NIFTY',
            }

            order_id = self.bridge.place_order(close_order)
            if order_id:
                self.logger.info(f"Emergency close placed: {symbol} order_id={order_id}")
            else:
                self.logger.error(f"FAILED emergency close: {symbol}")
                self.alerter.send(
                    'EMERGENCY',
                    f"FAILED to close {symbol} — MANUAL INTERVENTION REQUIRED"
                )

    def daily_reconciliation(self):
        """
        Run EOD reconciliation at 15:35 IST.
        Delegates to PositionReconciler.
        """
        return self.reconciler.reconcile()

    # ==================================================================
    # PRIVATE: Signal processing pipeline
    # ==================================================================

    def _process_signal(self, msg_id, fields):
        """
        Process a single signal through the full execution pipeline:

        1. Time guard
        2. Daily loss limiter check
        3. Position limit check
        4. Greek pre-check
        5. Lot sizing via compound sizer
        6. Price freshness check
        7. Place order via kite_bridge
        8. Monitor fill
        9. Record trade result
        """
        signal_id = fields.get('signal_id', 'UNKNOWN')
        signal_type = fields.get('signal_type', 'ENTRY')
        instrument = fields.get('instrument', 'NIFTY')

        self.logger.info(f"Processing signal: {signal_id} ({signal_type})")

        # --- STEP 1: Time guard ---
        if not self._is_trading_time():
            self.logger.info(f"Outside trading hours — skipping {signal_id}")
            self._log_trade(signal_id, 'TIME_BLOCKED', fields)
            return

        # --- STEP 2: Daily loss limiter ---
        if not self.loss_limiter.can_trade():
            self.logger.warning(f"Daily loss limit hit — skipping {signal_id}")
            self._log_trade(signal_id, 'LOSS_LIMIT_BLOCKED', fields)
            self.daily_loss_limit_hit = True

            # Trigger emergency exit if Tier 2
            if self.loss_limiter.tier >= 2:
                self.emergency_exit_all()
            return

        # --- STEP 3: Position limit check (skip for exits) ---
        is_exit = signal_type in ('EXIT_SIGNAL', 'STOP_LOSS_TRIGGER', 'EXIT')
        if not is_exit:
            if not self._check_position_limits(fields):
                self.logger.info(f"Position limit reached — skipping {signal_id}")
                self._log_trade(signal_id, 'POSITION_LIMIT', fields)
                return

        # --- STEP 4: Greek pre-check (skip for exits) ---
        if not is_exit:
            try:
                approved, reason = self._run_greek_check(fields)
                if not approved:
                    self.logger.info(f"Greek pre-check rejected {signal_id}: {reason}")
                    self._log_trade(signal_id, 'GREEK_REJECTED', fields,
                                    {'reason': reason})
                    return
            except Exception as e:
                self.logger.warning(
                    f"Greek pre-check error for {signal_id}: {e} — rejecting"
                )
                self._log_trade(signal_id, 'GREEK_ERROR', fields)
                return

        # --- STEP 5: Lot sizing (full chain: sizer -> behavioral -> limiter) ---
        if not is_exit:
            premium = float(fields.get('entry_premium', 0) or
                            fields.get('limit_price', 200))
            base_lots = self.sizer.get_lots(instrument=instrument, premium=premium)

            if base_lots <= 0:
                self.logger.info(f"Sizer returned 0 lots — skipping {signal_id}")
                self._log_trade(signal_id, 'ZERO_LOTS', fields)
                return

            # Record system_lots BEFORE any adjustments (for 2x safety cap)
            system_lots = base_lots
            lots = base_lots

            # Apply behavioral overlay
            overlay_ctx = OverlayContext(
                system_size=1.0,
                current_dd_pct=self.sizer.drawdown_pct,
                entry_price=float(fields.get('underlying_ltp', 0) or
                                  fields.get('limit_price', 0)),
                current_atr=float(fields.get('atr', 0)),
                proposed_sl=float(fields.get('sl_price', 0)),
                proposed_tgt=float(fields.get('tgt_price', 0)),
                direction="LONG" if fields.get('transaction_type') == 'BUY' else "SHORT",
            )
            lots, overlay_result = self.sizer.apply_behavioral_overlay(lots, overlay_ctx)

            # Apply loss limiter size factor
            size_factor = self.loss_limiter.size_factor
            if size_factor < 1.0:
                lots = max(1, int(lots * size_factor))

            # Log full sizing chain
            self.logger.info(
                f"Sizing chain {signal_id}: base={base_lots} "
                f"-> behavioral={lots if overlay_result else base_lots} "
                f"-> limiter(×{size_factor})={lots} | system_lots={system_lots}"
            )

            fields['lots'] = str(lots)
            fields['system_lots'] = str(system_lots)
            lot_size = int(fields.get('lot_size', 25))
            fields['quantity'] = str(lots * lot_size)

        # --- STEP 6: Price freshness check ---
        if not self._check_price_freshness(fields):
            self.logger.info(f"Price escape for {signal_id} — cancelled")
            self._log_trade(signal_id, 'PRICE_ESCAPE', fields)
            return

        # --- STEP 7: Place order via KiteBridge ---
        order_id = self.bridge.place_order(fields)
        if order_id is None:
            self._log_trade(signal_id, 'ORDER_FAILED', fields)
            return

        # --- STEP 8: Monitor fill ---
        # NOTE: Currently uses blocking monitor_fill() for sequential safety.
        # TODO: Switch to async submit() API with callback for parallelism
        #       when multi-leg execution is implemented. The blocking API
        #       internally delegates to the async monitor (started in __init__)
        #       so the background polling thread is already active.
        fill_window = self._get_fill_window(
            fields.get('instrument_type', instrument)
        )
        expected_price = float(fields.get('limit_price', 0) or
                               fields.get('entry_premium', 0))

        fill_result = self.fill_monitor.monitor_fill(
            order_id=order_id,
            timeout_seconds=fill_window,
            expected_price=expected_price,
            tradingsymbol=fields.get('tradingsymbol', ''),
            signal_id=signal_id,
        )

        # --- STEP 9: Handle result ---
        self._handle_fill_result(signal_id, order_id, fill_result, fields)

    # ==================================================================
    # PRIVATE: Checks and guards
    # ==================================================================

    def _is_trading_time(self) -> bool:
        """Check if current time is within 9:16 - 15:19 IST."""
        now = datetime.now().time()
        return MARKET_OPEN <= now <= MARKET_CLOSE

    def _check_position_limits(self, fields) -> bool:
        """
        Check position limits:
        - MAX_POSITIONS = 4 total open positions
        - MAX_SAME_DIRECTION = 2 in the same direction (LONG/SHORT)

        Uses a threading lock to prevent TOCTOU races between checking
        open positions and placing the order. The SQL also uses FOR UPDATE
        to lock the rows at the DB level where supported.
        """
        with self._position_lock:
            return self._check_position_limits_inner(fields)

    def _check_position_limits_inner(self, fields) -> bool:
        """Inner position limit check (called under lock)."""
        try:
            result = self.db.execute("""
                SELECT transaction_type, COUNT(*) as cnt
                FROM trades
                WHERE exit_date IS NULL
                  AND DATE(created_at) = CURRENT_DATE
                GROUP BY transaction_type
                FOR UPDATE
            """)
            rows = result.fetchall() if result else []

            total_open = 0
            direction_counts = {}
            for row in rows:
                if isinstance(row, dict):
                    tx = row['transaction_type']
                    cnt = row['cnt']
                else:
                    tx, cnt = row[0], row[1]
                direction_counts[tx] = cnt
                total_open += cnt

            # Total position check
            if total_open >= MAX_POSITIONS:
                self.logger.info(
                    f"Position limit: {total_open}/{MAX_POSITIONS} open"
                )
                return False

            # Same direction check
            signal_direction = fields.get('transaction_type', 'BUY')
            same_dir = direction_counts.get(signal_direction, 0)
            if same_dir >= MAX_SAME_DIRECTION:
                self.logger.info(
                    f"Same-direction limit: {same_dir}/{MAX_SAME_DIRECTION} "
                    f"{signal_direction} positions"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Position limit check failed: {e}")
            return True  # Proceed cautiously on DB error

    def _run_greek_check(self, fields) -> tuple:
        """
        Run greek_pre_check against the proposed signal.
        Returns (approved: bool, reason: str).
        """
        # Build proposed signal and portfolio from fields / DB
        proposed_signal = {
            'tradingsymbol': fields.get('tradingsymbol', ''),
            'instrument': fields.get('instrument', 'NIFTY'),
            'option_type': fields.get('option_type', 'CE'),
            'strike': float(fields.get('strike', 0)),
            'lots': int(fields.get('lots', 1)),
            'transaction_type': fields.get('transaction_type', 'BUY'),
        }

        # Fetch current portfolio from DB
        try:
            result = self.db.execute("""
                SELECT tradingsymbol, quantity, entry_price, transaction_type
                FROM trades
                WHERE exit_date IS NULL
                  AND DATE(created_at) = CURRENT_DATE
            """)
            portfolio = result.fetchall() if result else []
        except Exception:
            portfolio = []

        # Fetch market data (LTP of underlying)
        market_data = {
            'underlying_ltp': float(fields.get('underlying_ltp', 0)),
            'iv': float(fields.get('iv', 15)),
        }

        return greek_pre_check(proposed_signal, portfolio, market_data)

    def _check_price_freshness(self, fields) -> bool:
        """Check if live price is within 1% of price at signal fire time."""
        fire_price = float(fields.get('fire_price', 0))
        if fire_price <= 0:
            return True  # No reference price, proceed

        if EXECUTION_MODE == "PAPER":
            return True  # Skip in paper mode

        try:
            ltp = self._get_ltp(fields.get('tradingsymbol', ''))
        except Exception:
            return True  # Can't check, proceed cautiously

        deviation = abs(ltp - fire_price) / fire_price
        if deviation > 0.01:  # 1% price escape
            return False
        return True

    # ==================================================================
    # PRIVATE: Fill result handling
    # ==================================================================

    def _handle_fill_result(self, signal_id, order_id, fill_result, fields):
        """Handle the outcome of order monitoring."""

        if fill_result is None:
            self.logger.info(f"Order {order_id} not filled for {signal_id}")
            self._log_trade(signal_id, 'UNFILLED', fields)
            return

        status = fill_result.get('status', 'UNFILLED')

        if status == 'FILLED':
            fill_price = fill_result.get('fill_price', 0)
            fill_qty = fill_result.get('fill_quantity', 0)
            slippage = fill_result.get('slippage', 0)

            self.logger.info(
                f"FILLED: {signal_id} | order_id={order_id} "
                f"@ {fill_price:.2f} (slippage={slippage:+.2%})"
            )

            # Record trade in DB
            self._record_trade(signal_id, fields, fill_result)

            # Update portfolio state in Redis
            self._update_portfolio_state(fields, fill_result)

            # Log to trade_log
            self._log_trade(signal_id, 'FILLED', fields, fill_result)

        elif status == 'PARTIAL':
            fill_qty = fill_result.get('fill_quantity', 0)
            total_qty = int(fields.get('quantity', 0))

            if total_qty > 0 and fill_qty >= total_qty * 0.5:
                self.logger.info(
                    f"Partial fill >=50% for {signal_id}: "
                    f"{fill_qty}/{total_qty} — keeping"
                )
                self._record_trade(signal_id, fields, fill_result)
                self._update_portfolio_state(fields, fill_result)
                self._log_trade(signal_id, 'PARTIAL_FILL', fields, fill_result)
            else:
                self.logger.warning(
                    f"Partial fill <50% for {signal_id}: "
                    f"{fill_qty}/{total_qty} — closing at market"
                )
                self._close_at_market(fields, fill_qty)
                self._log_trade(signal_id, 'ABANDONED_PARTIAL', fields, fill_result)

    def _record_trade(self, signal_id, fields, fill_result):
        """Record a completed trade in the trades table."""
        try:
            self.db.execute(
                """
                INSERT INTO trades
                    (signal_id, tradingsymbol, transaction_type, quantity,
                     lots, lot_size, entry_price, instrument, option_type,
                     strike, mode, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    signal_id,
                    fields.get('tradingsymbol', ''),
                    fields.get('transaction_type', 'BUY'),
                    fill_result.get('fill_quantity', int(fields.get('quantity', 0))),
                    int(fields.get('lots', 1)),
                    int(fields.get('lot_size', 25)),
                    fill_result.get('fill_price', 0),
                    fields.get('instrument', 'NIFTY'),
                    fields.get('option_type', ''),
                    float(fields.get('strike', 0)),
                    EXECUTION_MODE,
                ),
            )

            # Record in loss limiter (use negative for now — actual P&L at exit)
            # At entry, no P&L yet; limiter tracks at exit time
            self.logger.info(f"Trade recorded: {signal_id}")

        except Exception as e:
            self.logger.error(f"Failed to record trade: {e}")

    def _update_portfolio_state(self, fields, result):
        """Update portfolio state in Redis after a fill."""
        try:
            self.redis.hset('PORTFOLIO_STATE', mapping={
                'last_fill_signal': fields.get('signal_id', ''),
                'last_fill_price': str(result.get('fill_price',
                                                   result.get('average_price', 0))),
                'last_fill_time': datetime.now().isoformat(),
            })
        except Exception as e:
            self.logger.error(f"Failed to update portfolio state: {e}")

    def _close_at_market(self, fields, quantity):
        """Close a partial position at market price — routed through KiteBridge for audit trail."""
        tx_type = 'SELL' if fields.get('transaction_type') == 'BUY' else 'BUY'
        lot_size = int(fields.get('lot_size', 25))
        lots = max(1, quantity // lot_size)

        close_order = {
            'signal_id': f"CLOSE_PARTIAL_{fields.get('signal_id', 'UNKNOWN')}",
            'tradingsymbol': fields.get('tradingsymbol', ''),
            'transaction_type': tx_type,
            'lots': lots,
            'lot_size': lot_size,
            'instrument': fields.get('instrument', 'NIFTY'),
            'limit_price': 0,  # MARKET order — KiteBridge handles pricing
        }

        order_id = self.bridge.place_order(close_order)
        if order_id:
            self.logger.info(
                f"Partial close placed via KiteBridge: "
                f"{fields.get('tradingsymbol')} x{quantity} order_id={order_id}"
            )
        else:
            self.logger.error(
                f"Failed to close partial: {fields.get('tradingsymbol')} x{quantity}"
            )
            self.alerter.send(
                'CRITICAL',
                f"Failed to close partial: {fields.get('tradingsymbol')} — "
                f"MANUAL INTERVENTION REQUIRED",
                priority='critical',
            )

    def _get_ltp(self, tradingsymbol: str) -> float:
        """Get last traded price from Kite."""
        if EXECUTION_MODE == "PAPER" or self.kite is None:
            return 0.0
        quote = self.kite.ltp(f"NFO:{tradingsymbol}")
        return quote[f"NFO:{tradingsymbol}"]['last_price']

    def _get_fill_window(self, instrument: str) -> int:
        """Return fill window in seconds based on instrument type."""
        if instrument in ('FUTURES', 'OPTIONS_BUYING', 'OPTIONS_SELLING',
                          'NIFTY', 'BANKNIFTY'):
            return FILL_WINDOWS['SINGLE_LEG']
        elif instrument == 'SPREAD':
            return FILL_WINDOWS['TWO_LEG']
        else:
            return FILL_WINDOWS['SINGLE_LEG']

    def _log_trade(self, signal_id, status, fields, result=None):
        """Log trade event to database."""
        try:
            self.db.execute("""
                INSERT INTO trade_log
                    (signal_id, status, fields, result, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (
                signal_id, status,
                json.dumps({k: str(v) for k, v in fields.items()}),
                json.dumps(result) if result else None,
            ))
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")
