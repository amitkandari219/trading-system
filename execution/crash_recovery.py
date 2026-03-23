"""
Crash Recovery — detect and reconcile orphaned orders on startup.

When the process crashes between placing an order and recording its fill,
order_intents rows remain in PENDING or PLACED status. On next startup,
CrashRecovery scans these rows, queries the broker for actual status,
and either recovers the position or cancels the stale order.

Usage:
    from execution.crash_recovery import CrashRecovery
    recovery = CrashRecovery(kite, db_conn, alerter)
    result = recovery.recover_on_startup()

Helpers (used by kite_bridge before/after placing):
    from execution.crash_recovery import log_intent, update_intent
    intent_id = log_intent(db, 'RSI_OVERSOLD', 'BUY', 'NIFTY25MAR22000CE', 50)
    update_intent(db, intent_id, kite_order_id='240322000123456', status='PLACED')
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# Kite order statuses that mean the order executed
_FILLED_STATUSES = {"COMPLETE"}
# Kite statuses that mean the order is still live
_OPEN_STATUSES = {"OPEN", "TRIGGER PENDING", "OPEN PENDING"}
# Kite statuses that mean the order is dead
_DEAD_STATUSES = {"REJECTED", "CANCELLED"}


class CrashRecovery:
    """
    Scans order_intents for orphaned rows (PENDING / PLACED) and reconciles
    them against the broker on startup.
    """

    def __init__(self, kite, db_conn, alerter=None):
        """
        Args:
            kite:      authenticated KiteConnect instance (None in PAPER mode)
            db_conn:   psycopg2 connection
            alerter:   TelegramAlerter instance (optional)
        """
        self.kite = kite
        self.db = db_conn
        self.alerter = alerter
        self.is_paper = kite is None or EXECUTION_MODE == "PAPER"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recover_on_startup(self) -> Dict[str, Any]:
        """
        Main entry point. Scans order_intents for orphaned rows and
        reconciles each one against the broker.

        Returns:
            dict with keys: recovered, cancelled, errors, details
        """
        result = {
            "recovered": 0,
            "cancelled": 0,
            "crashed_before_place": 0,
            "errors": 0,
            "details": [],
        }

        orphans = self._fetch_orphaned_intents()
        if not orphans:
            logger.info("Crash recovery: no orphaned intents found")
            return result

        logger.warning(
            "Crash recovery: found %d orphaned intent(s)", len(orphans)
        )

        for intent in orphans:
            try:
                self._recover_single(intent, result)
            except Exception as e:
                result["errors"] += 1
                detail = {
                    "intent_id": str(intent["intent_id"]),
                    "signal_name": intent["signal_name"],
                    "action": "ERROR",
                    "error": str(e),
                }
                result["details"].append(detail)
                logger.exception(
                    "Crash recovery error for intent %s: %s",
                    intent["intent_id"], e,
                )

        self._send_recovery_alert(result)
        return result

    # ------------------------------------------------------------------
    # Internal — per-intent recovery
    # ------------------------------------------------------------------

    def _recover_single(self, intent: Dict, result: Dict) -> None:
        """Recover a single orphaned intent."""
        intent_id = intent["intent_id"]
        kite_order_id = intent.get("kite_order_id")

        # --- PAPER mode: mark everything as paper-recovered ---
        if self.is_paper:
            update_intent(
                self.db, intent_id, status="PAPER_RECOVERED"
            )
            result["recovered"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": "PAPER_RECOVERED",
            })
            logger.info(
                "Paper-recovered intent %s (%s)",
                intent_id, intent["signal_name"],
            )
            return

        # --- PENDING: crash happened before place_order returned ---
        if kite_order_id is None:
            update_intent(
                self.db, intent_id, status="CRASHED_BEFORE_PLACE"
            )
            result["crashed_before_place"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": "CRASHED_BEFORE_PLACE",
            })
            logger.info(
                "Intent %s marked CRASHED_BEFORE_PLACE (no kite_order_id)",
                intent_id,
            )
            return

        # --- PLACED: order was sent to Kite, need to check broker ---
        try:
            history = self.kite.order_history(kite_order_id)
        except Exception as e:
            # If order_history fails, mark error and move on
            update_intent(
                self.db, intent_id, status="RECOVERY_ERROR",
                error_message=f"order_history failed: {e}",
            )
            result["errors"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": "ERROR",
                "error": f"order_history failed: {e}",
            })
            return

        if not history:
            update_intent(
                self.db, intent_id, status="RECOVERY_ERROR",
                error_message="Empty order history from Kite",
            )
            result["errors"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": "ERROR",
                "error": "Empty order history",
            })
            return

        # Last entry in history is the most recent status
        latest = history[-1]
        broker_status = (latest.get("status") or "").upper()

        if broker_status in _FILLED_STATUSES:
            # Order filled — recover the position
            fill_price = latest.get("average_price", 0)
            update_intent(
                self.db, intent_id,
                status="FILLED",
                fill_price=fill_price,
            )
            self._record_recovered_position(intent, latest)
            result["recovered"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": "RECOVERED_FILL",
                "fill_price": float(fill_price),
                "kite_order_id": kite_order_id,
            })
            logger.info(
                "Recovered filled order %s at %.2f for intent %s",
                kite_order_id, fill_price, intent_id,
            )

        elif broker_status in _OPEN_STATUSES:
            # Order still open — stale from crashed session, cancel it
            self._cancel_stale_order(kite_order_id)
            update_intent(
                self.db, intent_id, status="CANCELLED",
                error_message="Stale order cancelled on crash recovery",
            )
            result["cancelled"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": "CANCELLED_STALE",
                "kite_order_id": kite_order_id,
            })
            logger.info(
                "Cancelled stale order %s for intent %s",
                kite_order_id, intent_id,
            )

        elif broker_status in _DEAD_STATUSES:
            # Already cancelled/rejected at broker
            update_intent(
                self.db, intent_id, status="CANCELLED",
                error_message=f"Broker status: {broker_status}",
            )
            result["cancelled"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": f"BROKER_{broker_status}",
                "kite_order_id": kite_order_id,
            })
            logger.info(
                "Intent %s: broker already %s for order %s",
                intent_id, broker_status, kite_order_id,
            )
        else:
            # Unknown status — log and mark error
            update_intent(
                self.db, intent_id, status="RECOVERY_ERROR",
                error_message=f"Unknown broker status: {broker_status}",
            )
            result["errors"] += 1
            result["details"].append({
                "intent_id": str(intent_id),
                "signal_name": intent["signal_name"],
                "action": "UNKNOWN_STATUS",
                "broker_status": broker_status,
            })

    # ------------------------------------------------------------------
    # Position recovery
    # ------------------------------------------------------------------

    def _record_recovered_position(self, intent: Dict, kite_order: Dict) -> None:
        """
        Insert into trades table so the position is tracked locally,
        and log to order_audit for the audit trail.
        """
        cur = self.db.cursor()
        try:
            # Insert into trades
            cur.execute(
                """
                INSERT INTO trades
                    (signal_id, tradingsymbol, transaction_type, quantity,
                     lots, lot_size, entry_price, instrument, option_type,
                     strike, mode, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    intent["signal_name"],
                    kite_order.get("tradingsymbol", intent["instrument"]),
                    intent["direction"],
                    kite_order.get("filled_quantity", intent["qty"]),
                    1,  # lots — will be corrected by reconciler
                    max(kite_order.get("filled_quantity", intent["qty"]), 1),
                    kite_order.get("average_price", 0),
                    intent["instrument"],
                    "",  # option_type extracted from instrument if needed
                    0,   # strike extracted from instrument if needed
                    "CRASH_RECOVERY",
                ),
            )

            # Log to order_audit
            cur.execute(
                """
                INSERT INTO order_audit
                    (signal_id, tradingsymbol, transaction_type, quantity,
                     product, order_type, price, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    intent["signal_name"],
                    kite_order.get("tradingsymbol", intent["instrument"]),
                    intent["direction"],
                    kite_order.get("filled_quantity", intent["qty"]),
                    kite_order.get("product", "MIS"),
                    kite_order.get("order_type", "LIMIT"),
                    kite_order.get("average_price", 0),
                    "CRASH_RECOVERED",
                ),
            )
            self.db.commit()
            logger.info(
                "Recorded recovered position for %s in trades + order_audit",
                intent["signal_name"],
            )
        except Exception as e:
            self.db.rollback()
            logger.exception(
                "Failed to record recovered position for %s: %s",
                intent["signal_name"], e,
            )
            raise

    # ------------------------------------------------------------------
    # Cancel stale order
    # ------------------------------------------------------------------

    def _cancel_stale_order(self, kite_order_id: str) -> None:
        """
        Cancel an order that is still open from a crashed session.
        Handles the case where the order was already cancelled.
        """
        try:
            self.kite.cancel_order(
                variety="regular", order_id=kite_order_id
            )
            logger.info("Cancelled stale order %s", kite_order_id)
        except Exception as e:
            err_msg = str(e).lower()
            # "order is already cancelled" or similar — safe to ignore
            if "already" in err_msg or "cancel" in err_msg or "complete" in err_msg:
                logger.info(
                    "Order %s already terminal, cancel was no-op: %s",
                    kite_order_id, e,
                )
            else:
                logger.error(
                    "Failed to cancel stale order %s: %s",
                    kite_order_id, e,
                )
                raise

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _fetch_orphaned_intents(self) -> List[Dict]:
        """Fetch order_intents stuck in PENDING or PLACED."""
        cur = self.db.cursor()
        cur.execute(
            """
            SELECT intent_id, signal_name, direction, instrument, qty,
                   kite_order_id, status, created_at
            FROM order_intents
            WHERE status IN ('PENDING', 'PLACED')
            ORDER BY created_at
            """
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _send_recovery_alert(self, result: Dict) -> None:
        """Send Telegram alert summarizing recovery actions."""
        total = (
            result["recovered"]
            + result["cancelled"]
            + result["crashed_before_place"]
            + result["errors"]
        )
        if total == 0:
            return

        lines = [
            f"CRASH RECOVERY ({EXECUTION_MODE})",
            f"  Recovered fills: {result['recovered']}",
            f"  Cancelled stale: {result['cancelled']}",
            f"  Crashed before place: {result['crashed_before_place']}",
            f"  Errors: {result['errors']}",
        ]
        if result["details"]:
            lines.append("Details:")
            for d in result["details"][:10]:  # cap at 10
                lines.append(
                    f"  {d.get('signal_name', '?')} -> {d.get('action', '?')}"
                )

        msg = "\n".join(lines)
        logger.warning(msg)

        if self.alerter:
            try:
                self.alerter.send(msg)
            except Exception as e:
                logger.error("Failed to send recovery alert: %s", e)


# ======================================================================
# Static helpers — used by kite_bridge before/after placing orders
# ======================================================================

def log_intent(db_conn, signal_name: str, direction: str,
               instrument: str, qty: int) -> str:
    """
    Create an order intent BEFORE placing the order.
    Returns the intent_id (UUID as string).

    Call this before kite.place_order() so that if the process crashes
    mid-flight, the intent row is already in the database.
    """
    intent_id = str(uuid.uuid4())
    cur = db_conn.cursor()
    cur.execute(
        """
        INSERT INTO order_intents
            (intent_id, signal_name, direction, instrument, qty,
             status, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, 'PENDING', NOW(), NOW())
        """,
        (intent_id, signal_name, direction, instrument, qty),
    )
    db_conn.commit()
    logger.info(
        "Logged order intent %s: %s %s %s x%d",
        intent_id, signal_name, direction, instrument, qty,
    )
    return intent_id


def update_intent(db_conn, intent_id, kite_order_id=None,
                  status=None, fill_price=None, error_message=None) -> None:
    """
    Update an existing order intent row.
    Only non-None fields are updated.
    """
    sets = []
    params = []

    if kite_order_id is not None:
        sets.append("kite_order_id = %s")
        params.append(kite_order_id)
    if status is not None:
        sets.append("status = %s")
        params.append(status)
    if fill_price is not None:
        sets.append("fill_price = %s")
        params.append(fill_price)
    if error_message is not None:
        sets.append("error_message = %s")
        params.append(error_message)

    if not sets:
        return

    sets.append("updated_at = NOW()")
    params.append(str(intent_id))

    cur = db_conn.cursor()
    cur.execute(
        f"UPDATE order_intents SET {', '.join(sets)} WHERE intent_id = %s",
        params,
    )
    db_conn.commit()


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    """
    Self-test: create test intents and simulate crash recovery.
    Requires DATABASE_DSN to point at a database with the order_intents table.
    """
    import psycopg2
    from config.settings import DATABASE_DSN

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=== Crash Recovery Self-Test ===\n")

    conn = psycopg2.connect(DATABASE_DSN)

    # Clean up any previous test data
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM order_intents WHERE signal_name LIKE 'TEST_CR_%'"
    )
    conn.commit()

    # --- Create test intents ---
    # 1. PENDING intent (crash before place_order)
    id1 = log_intent(conn, "TEST_CR_PENDING", "BUY", "NIFTY25MAR22000CE", 50)
    print(f"Created PENDING intent: {id1}")

    # 2. PLACED intent (crash after place_order but before fill tracking)
    id2 = log_intent(conn, "TEST_CR_PLACED", "SELL", "NIFTY25MAR21500PE", 50)
    update_intent(conn, id2, kite_order_id="FAKE_ORDER_123", status="PLACED")
    print(f"Created PLACED intent: {id2}")

    # 3. Another PENDING
    id3 = log_intent(conn, "TEST_CR_PENDING2", "BUY", "BANKNIFTY25MAR50000CE", 25)
    print(f"Created PENDING intent: {id3}")

    print(f"\nOrphaned intents created. Running recovery in PAPER mode...\n")

    # --- Run recovery in PAPER mode (kite=None) ---
    recovery = CrashRecovery(kite=None, db_conn=conn, alerter=None)
    result = recovery.recover_on_startup()

    print(f"Recovery result:")
    print(f"  Recovered:            {result['recovered']}")
    print(f"  Cancelled:            {result['cancelled']}")
    print(f"  Crashed before place: {result['crashed_before_place']}")
    print(f"  Errors:               {result['errors']}")
    print(f"  Details:")
    for d in result["details"]:
        print(f"    {d}")

    # Verify intents are no longer orphaned
    cur.execute(
        """
        SELECT intent_id, signal_name, status
        FROM order_intents
        WHERE signal_name LIKE 'TEST_CR_%'
        ORDER BY created_at
        """
    )
    print(f"\nFinal intent states:")
    for row in cur.fetchall():
        print(f"  {row[1]:25s} -> {row[2]}")

    # Clean up test data
    cur.execute(
        "DELETE FROM order_intents WHERE signal_name LIKE 'TEST_CR_%'"
    )
    conn.commit()
    conn.close()

    print("\n=== Self-Test Complete ===")
