"""
Order Verifier — cross-checks today's orders between Kite and order_audit.

Detects:
  - Orders in Kite but not in order_audit (rogue orders)
  - Orders in order_audit but not in Kite (failed placements)
  - Fill discrepancies / slippage analysis

Usage:
    from execution.order_verifier import OrderVerifier
    verifier = OrderVerifier(kite, db)
    issues = verifier.verify_todays_orders()
    fill_issues = verifier.verify_fills()
"""

import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# Slippage thresholds
SLIPPAGE_WARNING_PCT = 0.005   # 0.5% — log a warning
SLIPPAGE_CRITICAL_PCT = 0.02   # 2.0% — flag as issue


class OrderVerifier:
    """
    Cross-checks orders between broker (Kite) and internal audit log.
    Runs as part of EOD reconciliation to catch order-level discrepancies.
    """

    def __init__(self, kite, db, alerter=None, logger_override=None):
        """
        Args:
            kite:    authenticated KiteConnect instance (None OK in PAPER mode)
            db:      psycopg2 connection
            alerter: optional TelegramAlerter instance
        """
        self.kite = kite
        self.db = db
        self.alerter = alerter
        self.log = logger_override or logger

    def verify_todays_orders(self, as_of_date: Optional[date] = None) -> List[Dict]:
        """
        Compare today's Kite orders vs order_audit entries.

        Returns:
            List of issue dicts. Empty list = all orders reconciled.
        """
        as_of_date = as_of_date or date.today()
        issues = []

        self.log.info(f"Verifying orders for {as_of_date} (mode={EXECUTION_MODE})")

        # Fetch internal orders from order_audit
        internal_orders = self._fetch_internal_orders(as_of_date)

        # Fetch broker orders
        broker_orders = self._fetch_broker_orders(as_of_date)

        if not internal_orders and not broker_orders:
            self.log.info("No orders to verify today")
            return issues

        # Build lookup by broker_order_id
        internal_by_broker_id = {}
        for order in internal_orders:
            bid = order.get("broker_order_id")
            if bid:
                internal_by_broker_id[bid] = order

        broker_by_id = {}
        for order in broker_orders:
            oid = order.get("order_id")
            if oid:
                broker_by_id[str(oid)] = order

        # Check 1: Orders in broker but not in order_audit
        for oid, border in broker_by_id.items():
            if oid not in internal_by_broker_id:
                # Could be a legitimate order placed outside the system
                tag = border.get("tag", "")
                if tag and any(
                    t in tag for t in ["RECONCILE", "EMERG", "CLOSE"]
                ):
                    # System-generated, not necessarily in order_audit
                    continue

                issue = {
                    "type": "ROGUE_ORDER",
                    "severity": "CRITICAL",
                    "broker_order_id": oid,
                    "tradingsymbol": border.get("tradingsymbol", "?"),
                    "transaction_type": border.get("transaction_type", "?"),
                    "quantity": border.get("quantity", 0),
                    "status": border.get("status", "?"),
                    "detail": "Order exists in broker but not in order_audit",
                }
                issues.append(issue)
                self.log.warning(
                    f"ROGUE ORDER: {oid} {border.get('tradingsymbol')} "
                    f"not in order_audit"
                )

        # Check 2: Orders in order_audit but not in broker (LIVE mode only)
        if EXECUTION_MODE == "LIVE":
            for order in internal_orders:
                bid = order.get("broker_order_id")
                if bid and bid not in broker_by_id:
                    # Order was logged but never appeared in broker
                    if order.get("status") in ("PENDING", "PLACED"):
                        issue = {
                            "type": "GHOST_ORDER",
                            "severity": "WARNING",
                            "audit_id": order.get("id"),
                            "signal_id": order.get("signal_id"),
                            "broker_order_id": bid,
                            "tradingsymbol": order.get("tradingsymbol", "?"),
                            "status": order.get("status"),
                            "detail": "Order in audit log but not found in broker",
                        }
                        issues.append(issue)
                        self.log.warning(
                            f"GHOST ORDER: audit_id={order.get('id')} "
                            f"broker_id={bid} not found in Kite"
                        )

        # Check 3: Status mismatches
        for bid, internal in internal_by_broker_id.items():
            if bid in broker_by_id:
                broker = broker_by_id[bid]
                broker_status = broker.get("status", "").upper()
                internal_status = internal.get("status", "").upper()

                # Map internal statuses to expected broker statuses
                status_ok = False
                if internal_status in ("PLACED", "PLACED_MARKET"):
                    status_ok = broker_status in ("COMPLETE", "CANCELLED", "REJECTED", "OPEN")
                elif internal_status == "PAPER_PLACED":
                    status_ok = True  # Paper orders are always OK
                elif internal_status == "FILLED":
                    status_ok = broker_status == "COMPLETE"
                else:
                    status_ok = True

                if not status_ok:
                    issue = {
                        "type": "STATUS_MISMATCH",
                        "severity": "WARNING",
                        "broker_order_id": bid,
                        "tradingsymbol": internal.get("tradingsymbol", "?"),
                        "internal_status": internal_status,
                        "broker_status": broker_status,
                        "detail": f"Internal={internal_status} vs Broker={broker_status}",
                    }
                    issues.append(issue)

        self.log.info(f"Order verification complete: {len(issues)} issues found")
        return issues

    def verify_fills(self, as_of_date: Optional[date] = None) -> List[Dict]:
        """
        Analyze fill quality: slippage between expected and actual prices.

        Returns:
            List of fill discrepancy dicts.
        """
        as_of_date = as_of_date or date.today()
        issues = []

        self.log.info(f"Verifying fills for {as_of_date}")

        # Fetch order_audit entries with prices
        try:
            cur = self.db.cursor()
            cur.execute(
                """
                SELECT id, signal_id, tradingsymbol, transaction_type,
                       quantity, price, status, broker_order_id
                FROM order_audit
                WHERE DATE(created_at) = %s
                  AND status IN ('PLACED', 'PLACED_MARKET', 'PAPER_PLACED',
                                 'FILLED', 'RECONCILE_CLOSED')
                  AND price > 0
                ORDER BY created_at
                """,
                (as_of_date,),
            )
            audit_rows = cur.fetchall()
        except Exception as e:
            self.log.error(f"Failed to fetch order_audit for fill verification: {e}")
            return issues

        if not audit_rows:
            self.log.info("No fills to verify")
            return issues

        # For LIVE mode, check actual fill prices from Kite
        broker_fills = {}
        if EXECUTION_MODE == "LIVE" and self.kite:
            broker_fills = self._fetch_broker_fills(as_of_date)

        # Analyze each order
        for row in audit_rows:
            audit_id, signal_id, symbol, tx_type, qty, expected_price, status, broker_id = row

            if not expected_price or expected_price <= 0:
                continue

            actual_price = expected_price  # Default for paper mode
            if broker_id and broker_id in broker_fills:
                actual_price = broker_fills[broker_id].get("average_price", expected_price)

            # Compute slippage
            if expected_price > 0:
                slippage_pct = (actual_price - expected_price) / expected_price
                # For SELL orders, positive slippage is good (got more)
                # For BUY orders, negative slippage is good (paid less)
                adverse_slippage = slippage_pct if tx_type == "BUY" else -slippage_pct
            else:
                slippage_pct = 0
                adverse_slippage = 0

            fill_info = {
                "audit_id": audit_id,
                "signal_id": signal_id,
                "tradingsymbol": symbol,
                "transaction_type": tx_type,
                "quantity": qty,
                "expected_price": round(expected_price, 2),
                "actual_price": round(actual_price, 2),
                "slippage_pct": round(slippage_pct * 100, 4),
                "adverse_slippage_pct": round(adverse_slippage * 100, 4),
                "slippage_rupees": round(
                    (actual_price - expected_price) * (qty or 0), 2
                ),
            }

            # Flag excessive slippage
            if abs(adverse_slippage) >= SLIPPAGE_CRITICAL_PCT:
                fill_info["severity"] = "CRITICAL"
                fill_info["type"] = "EXCESSIVE_SLIPPAGE"
                fill_info["detail"] = (
                    f"Adverse slippage {adverse_slippage:+.2%} exceeds "
                    f"{SLIPPAGE_CRITICAL_PCT:.1%} threshold"
                )
                issues.append(fill_info)
                self.log.warning(
                    f"EXCESSIVE SLIPPAGE: {symbol} {tx_type} "
                    f"expected={expected_price:.2f} actual={actual_price:.2f} "
                    f"slippage={adverse_slippage:+.2%}"
                )
            elif abs(adverse_slippage) >= SLIPPAGE_WARNING_PCT:
                fill_info["severity"] = "WARNING"
                fill_info["type"] = "HIGH_SLIPPAGE"
                fill_info["detail"] = (
                    f"Adverse slippage {adverse_slippage:+.2%} above "
                    f"{SLIPPAGE_WARNING_PCT:.1%} warning"
                )
                issues.append(fill_info)

        # Summary stats
        if audit_rows:
            total_slippage_rs = sum(
                i.get("slippage_rupees", 0) for i in issues
            )
            self.log.info(
                f"Fill verification: {len(audit_rows)} orders, "
                f"{len(issues)} issues, "
                f"total adverse slippage: Rs {total_slippage_rs:,.0f}"
            )

        return issues

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_internal_orders(self, as_of_date: date) -> List[Dict]:
        """Fetch today's orders from order_audit."""
        try:
            cur = self.db.cursor()
            cur.execute(
                """
                SELECT id, signal_id, tradingsymbol, transaction_type,
                       quantity, product, order_type, price, status,
                       broker_order_id, error_message, created_at
                FROM order_audit
                WHERE DATE(created_at) = %s
                ORDER BY created_at
                """,
                (as_of_date,),
            )
            rows = cur.fetchall()
            columns = [
                "id", "signal_id", "tradingsymbol", "transaction_type",
                "quantity", "product", "order_type", "price", "status",
                "broker_order_id", "error_message", "created_at",
            ]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.log.error(f"Failed to fetch internal orders: {e}")
            return []

    def _fetch_broker_orders(self, as_of_date: date) -> List[Dict]:
        """Fetch today's orders from Kite."""
        if EXECUTION_MODE == "PAPER" or self.kite is None:
            return []

        try:
            orders = self.kite.orders()
            # Filter to today's orders
            today_orders = []
            for order in orders:
                order_date = order.get("order_timestamp")
                if order_date:
                    if isinstance(order_date, datetime):
                        if order_date.date() == as_of_date:
                            today_orders.append(order)
                    elif isinstance(order_date, str):
                        try:
                            dt = datetime.fromisoformat(order_date)
                            if dt.date() == as_of_date:
                                today_orders.append(order)
                        except ValueError:
                            pass

            self.log.info(f"Fetched {len(today_orders)} broker orders for {as_of_date}")
            return today_orders

        except Exception as e:
            self.log.error(f"Failed to fetch Kite orders: {e}")
            return []

    def _fetch_broker_fills(self, as_of_date: date) -> Dict[str, Dict]:
        """Fetch fill details from Kite orders, keyed by order_id."""
        result = {}
        broker_orders = self._fetch_broker_orders(as_of_date)
        for order in broker_orders:
            oid = str(order.get("order_id", ""))
            if oid and order.get("status", "").upper() == "COMPLETE":
                result[oid] = {
                    "order_id": oid,
                    "average_price": order.get("average_price", 0),
                    "filled_quantity": order.get("filled_quantity", 0),
                    "exchange_timestamp": order.get("exchange_timestamp"),
                }
        return result
