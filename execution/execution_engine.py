"""
Execution engine — processes signals from the order queue.
Handles priority ordering, fill windows, failure scenarios,
and daily reconciliation.
"""

import json
import time
import logging
from datetime import datetime

from config import settings


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


class ExecutionEngine:
    """
    Processes signals from Redis ORDER_QUEUE one at a time.
    Waits for current signal to resolve before processing the next.
    """

    def __init__(self, db, redis_client, kite, alerter, logger=None):
        """
        db:           database connection
        redis_client: redis.Redis instance
        kite:         KiteConnect instance for order placement
        alerter:      TelegramAlerter instance
        logger:       Python Logger
        """
        self.db      = db
        self.redis   = redis_client
        self.kite    = kite
        self.alerter = alerter
        self.logger  = logger or logging.getLogger(__name__)
        self.daily_loss_limit_hit = False

    def process_queue(self):
        """
        Main loop: read from ORDER_QUEUE, process one signal at a time.
        Signals are ordered by execution_priority then Sharpe.
        """
        messages = self.redis.xrange('ORDER_QUEUE', '-', '+')
        if not messages:
            return

        # Sort by priority then Sharpe
        sorted_msgs = sorted(
            messages,
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

    def _process_signal(self, msg_id, fields):
        """Process a single signal through the execution flow."""
        signal_id = fields.get('signal_id', 'UNKNOWN')

        # STEP 5: Price freshness check
        if not self._check_price_freshness(fields):
            self.logger.info(f"Price escape for {signal_id} — cancelled")
            self._log_trade(signal_id, 'PRICE_ESCAPE', fields)
            return

        # STEP 6: Place order
        order_id = self._place_order(fields)
        if order_id is None:
            return

        # STEP 7: Monitor fill window
        instrument = fields.get('instrument', 'FUTURES')
        fill_window = self._get_fill_window(instrument)
        fill_result = self._monitor_fill(order_id, fill_window)

        # Handle fill result
        self._handle_fill_result(signal_id, order_id, fill_result, fields)

    def _check_price_freshness(self, fields) -> bool:
        """Check if live price is within 0.3% of price at signal fire time."""
        fire_price = float(fields.get('fire_price', 0))
        if fire_price <= 0:
            return True  # No reference price, proceed

        try:
            ltp = self._get_ltp(fields.get('tradingsymbol', ''))
        except Exception:
            return True  # Can't check, proceed cautiously

        deviation = abs(ltp - fire_price) / fire_price
        if deviation > 0.003:  # 0.3%
            # Check if direction changed
            if deviation > 0.01:  # 1% — price escape
                return False
        return True

    def _place_order(self, fields) -> str:
        """Place limit order via Kite API. Returns order_id or None."""
        signal_id = fields.get('signal_id', 'UNKNOWN')

        for attempt in range(2):  # Max 2 attempts (original + 1 retry)
            try:
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NFO,
                    tradingsymbol=fields.get('tradingsymbol', ''),
                    transaction_type=fields.get('transaction_type', 'BUY'),
                    quantity=int(fields.get('quantity', 0)),
                    product=self.kite.PRODUCT_NRML,
                    order_type=self.kite.ORDER_TYPE_LIMIT,
                    price=float(fields.get('limit_price', 0)),
                )
                return order_id

            except Exception as e:
                error_msg = str(e)

                # Scenario 2: Insufficient margin
                if 'margin' in error_msg.lower() or 'insufficient' in error_msg.lower():
                    self.logger.warning(f"Insufficient margin for {signal_id}: {e}")
                    self._log_trade(signal_id, 'CAPITAL_BLOCKED', fields)
                    return None

                # Scenario 1: API timeout — retry once
                if attempt == 0:
                    self.logger.warning(f"Kite API error for {signal_id}, retrying: {e}")
                    time.sleep(3)
                    continue

                # Second attempt failed
                self.logger.error(f"Kite API failed for {signal_id}: {e}")
                self.alerter.send('CRITICAL', f"Kite API failed: {e}",
                                  signal_id=signal_id)
                self._log_trade(signal_id, 'API_TIMEOUT', fields)
                return None

        return None

    def _get_fill_window(self, instrument: str) -> int:
        """Return fill window in seconds based on instrument type."""
        if instrument in ('FUTURES', 'OPTIONS_BUYING', 'OPTIONS_SELLING'):
            return FILL_WINDOWS['SINGLE_LEG']
        elif instrument == 'SPREAD':
            return FILL_WINDOWS['TWO_LEG']
        else:
            return FILL_WINDOWS['SINGLE_LEG']

    def _monitor_fill(self, order_id, fill_window_secs) -> dict:
        """
        Monitor order for fill within the window.
        Returns dict with status and details.
        """
        start = time.time()
        while time.time() - start < fill_window_secs:
            try:
                order_history = self.kite.order_history(order_id)
                if not order_history:
                    time.sleep(1)
                    continue
                latest = order_history[-1]
                status = latest.get('status', '')

                if status == 'COMPLETE':
                    return {
                        'status': 'FILLED',
                        'filled_quantity': latest.get('filled_quantity', 0),
                        'average_price': latest.get('average_price', 0),
                    }
                elif status in ('CANCELLED', 'REJECTED'):
                    return {'status': 'CANCELLED', 'reason': latest.get('status_message', '')}

            except Exception:
                pass
            time.sleep(1)

        # Fill window expired — check partial fill
        try:
            order_history = self.kite.order_history(order_id)
            latest = order_history[-1] if order_history else {}
            filled = latest.get('filled_quantity', 0)
            intended = latest.get('quantity', 0)

            if filled > 0:
                self.kite.cancel_order(self.kite.VARIETY_REGULAR, order_id)
                return {
                    'status': 'PARTIAL',
                    'filled_quantity': filled,
                    'intended_quantity': intended,
                    'average_price': latest.get('average_price', 0),
                }

            # Not filled at all
            self.kite.cancel_order(self.kite.VARIETY_REGULAR, order_id)
            return {'status': 'UNFILLED'}

        except Exception:
            return {'status': 'UNFILLED'}

    def _handle_fill_result(self, signal_id, order_id, result, fields):
        """Handle the outcome of order monitoring."""
        status = result.get('status', 'UNFILLED')

        if status == 'FILLED':
            self.logger.info(f"Order {order_id} filled for {signal_id}")
            self._update_portfolio_state(fields, result)
            self._log_trade(signal_id, 'FILLED', fields, result)

        elif status == 'PARTIAL':
            filled = result.get('filled_quantity', 0)
            intended = result.get('intended_quantity', 0)

            if intended > 0 and filled >= intended * 0.5:
                # Scenario 4: Keep partial fill >= 50%
                self.logger.info(
                    f"Partial fill for {signal_id}: {filled}/{intended} lots — keeping"
                )
                self._update_portfolio_state(fields, result)
                self._log_trade(signal_id, 'PARTIAL_FILL', fields, result)
            else:
                # Partial < 50% — close at market
                self.logger.warning(
                    f"Partial fill for {signal_id}: {filled}/{intended} lots — abandoning"
                )
                self._close_at_market(fields, filled)
                self._log_trade(signal_id, 'ABANDONED_PARTIAL', fields, result)

        elif status == 'CANCELLED':
            self.logger.info(f"Order {order_id} cancelled for {signal_id}: {result.get('reason', '')}")
            self._log_trade(signal_id, 'CANCELLED', fields, result)

        else:  # UNFILLED
            self.logger.info(f"Order {order_id} unfilled for {signal_id}")
            self._log_trade(signal_id, 'UNFILLED', fields)

    def _update_portfolio_state(self, fields, result):
        """Update portfolio state in Redis after a fill."""
        self.redis.hset('PORTFOLIO_STATE', mapping={
            'last_fill_signal': fields.get('signal_id', ''),
            'last_fill_price': str(result.get('average_price', 0)),
            'last_fill_time': datetime.now().isoformat(),
        })

    def _close_at_market(self, fields, quantity):
        """Close a partial position at market price."""
        try:
            tx_type = 'SELL' if fields.get('transaction_type') == 'BUY' else 'BUY'
            self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NFO,
                tradingsymbol=fields.get('tradingsymbol', ''),
                transaction_type=tx_type,
                quantity=quantity,
                product=self.kite.PRODUCT_NRML,
                order_type=self.kite.ORDER_TYPE_MARKET,
            )
        except Exception as e:
            self.logger.error(f"Failed to close partial position: {e}")

    def _get_ltp(self, tradingsymbol: str) -> float:
        """Get last traded price from Kite."""
        quote = self.kite.ltp(f"NFO:{tradingsymbol}")
        return quote[f"NFO:{tradingsymbol}"]['last_price']

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

    def daily_reconciliation(self):
        """
        Runs at 3:35 PM every trading day (5 min after market close).
        Compares system state vs broker state.
        """
        # Get current positions from Kite API
        kite_positions = self.kite.positions()
        kite_net = {p['tradingsymbol']: p for p in kite_positions.get('net', [])
                    if p.get('quantity', 0) != 0}

        # Get system's open positions from trades table
        system_open = self.db.execute("""
            SELECT trade_id, signal_id, instrument, direction, lots, entry_price
            FROM trades
            WHERE exit_date IS NULL
              AND trade_type = 'LIVE'
        """).fetchall()
        system_net = {row['instrument']: row for row in system_open}

        mismatches = []

        # Check: positions in Kite but not in system
        for symbol, kite_pos in kite_net.items():
            if symbol not in system_net:
                mismatches.append(('UNKNOWN_POSITION', symbol, kite_pos))
                self.alerter.send('EMERGENCY',
                                  f"Kite has {symbol} but system does not. "
                                  f"Halting all trading.")
            else:
                sys_lots = system_net[symbol]['lots']
                kite_lots = abs(kite_pos.get('quantity', 0))
                if sys_lots != kite_lots:
                    mismatches.append(('QUANTITY_MISMATCH', symbol,
                                       f"system={sys_lots} kite={kite_lots}"))
                    self.db.execute(
                        "UPDATE trades SET lots = %s WHERE instrument = %s "
                        "AND exit_date IS NULL AND trade_type = 'LIVE'",
                        (kite_lots, symbol)
                    )

        # Check: positions in system but not in Kite
        for symbol, sys_pos in system_net.items():
            if symbol not in kite_net:
                mismatches.append(('GHOST_POSITION', symbol, sys_pos))
                self.db.execute(
                    "UPDATE trades SET exit_date = CURRENT_DATE, "
                    "exit_reason = 'NOT_FOUND_IN_BROKER_RECONCILIATION' "
                    "WHERE instrument = %s AND exit_date IS NULL "
                    "AND trade_type = 'LIVE'",
                    (symbol,)
                )

        # Store reconciliation snapshot
        self.db.execute("""
            INSERT INTO portfolio_state
                (snapshot_time, snapshot_type, total_capital, deployed_capital,
                 cash_reserve, open_positions)
            VALUES (NOW(), 'RECONCILIATION', %s, %s, %s, %s)
        """, (
            settings.TOTAL_CAPITAL,
            sum(r['lots'] * r['entry_price'] for r in system_open) if system_open else 0,
            settings.CAPITAL_RESERVE,
            json.dumps([dict(r) for r in system_open]) if system_open else '[]',
        ))

        if mismatches:
            self.alerter.send('EMERGENCY',
                              f"{len(mismatches)} reconciliation mismatch(es). "
                              f"Review before next session.")
        else:
            self.logger.info("Reconciliation clean.")

        return mismatches
