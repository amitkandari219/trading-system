"""
Session Reconciler — intra-session position reconciliation every 15 minutes.

Detects position drift between Kite broker state and internal DB during
market hours (9:15 AM to 3:30 PM IST). Catches discrepancies early:
  - ORPHANED: position in Kite but not in local DB
  - PHANTOM:  position in local DB but not in Kite
  - DRIFT:    quantity mismatch between Kite and local DB

Kite is always source of truth. Local DB is adjusted to match.

In PAPER mode (kite is None): reconciliation is skipped entirely.

Usage:
    from execution.session_reconciler import SessionReconciler
    reconciler = SessionReconciler(kite, db_conn, alerter)
    report = reconciler.reconcile()

    # Or run as background daemon during market hours:
    reconciler.start_background()
    # ... later ...
    reconciler.stop()
"""

import json
import logging
import threading
import time
import traceback
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Market hours IST
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)


class SessionReconciler:
    """
    Intra-session reconciliation that runs every N minutes during market hours.
    Compares Kite positions against the internal trades table and alerts on drift.
    """

    def __init__(
        self,
        kite,
        db_conn,
        alerter=None,
        interval_minutes: int = 15,
    ):
        """
        Args:
            kite:              authenticated KiteConnect instance (None in PAPER mode)
            db_conn:           psycopg2 connection
            alerter:           TelegramAlerter instance (optional)
            interval_minutes:  reconciliation frequency in minutes
        """
        self.kite = kite
        self.db = db_conn
        self.alerter = alerter
        self.interval_minutes = interval_minutes
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ==================================================================
    # PUBLIC: Single reconciliation pass
    # ==================================================================

    def reconcile(self) -> Dict[str, Any]:
        """
        Run one reconciliation pass.

        Returns:
            {orphaned: int, phantom: int, drift: int, details: list}
            Empty result in PAPER mode.
        """
        # PAPER mode: kite is None, nothing to reconcile against
        if self.kite is None:
            logger.info("PAPER mode: reconciliation skipped")
            return {"orphaned": 0, "phantom": 0, "drift": 0, "details": []}

        result = {"orphaned": 0, "phantom": 0, "drift": 0, "details": []}

        try:
            # Step 1: Fetch Kite positions (NFO with non-zero qty)
            kite_positions = self._fetch_kite_positions()

            # Step 2: Fetch local open positions for today
            local_positions = self._fetch_local_positions()

            # Step 3: Compare and detect discrepancies
            self._detect_orphaned(kite_positions, local_positions, result)
            self._detect_phantom(kite_positions, local_positions, result)
            self._detect_drift(kite_positions, local_positions, result)

            # Step 4: Log to session_reconciliation_log
            self._log_to_db(result)

            logger.info(
                f"Session reconciliation complete: "
                f"orphaned={result['orphaned']} phantom={result['phantom']} "
                f"drift={result['drift']}"
            )

        except Exception as e:
            logger.error(f"Session reconciliation failed: {e}\n{traceback.format_exc()}")
            if self.alerter:
                try:
                    self.alerter.send(
                        "CRITICAL",
                        f"Session reconciliation FAILED: {e}",
                    )
                except Exception:
                    pass

        return result

    # ==================================================================
    # PUBLIC: Background daemon
    # ==================================================================

    def start_background(self):
        """Start daemon thread that reconciles every interval_minutes during market hours."""
        if self._running:
            logger.warning("Session reconciler already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop,
            name="session-reconciler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Session reconciler started (interval={self.interval_minutes}m, "
            f"market hours {MARKET_OPEN.strftime('%H:%M')}-{MARKET_CLOSE.strftime('%H:%M')})"
        )

    def stop(self):
        """Stop the background reconciliation thread."""
        self._running = False
        logger.info("Session reconciler stop requested")

    # ==================================================================
    # PRIVATE: Background loop
    # ==================================================================

    def _background_loop(self):
        """Run reconcile() every interval_minutes, only during market hours."""
        while self._running:
            now = datetime.now()
            current_time = now.time()

            if MARKET_OPEN <= current_time <= MARKET_CLOSE:
                try:
                    self.reconcile()
                except Exception as e:
                    logger.error(f"Background reconciliation error: {e}")

            # Sleep until next interval
            sleep_seconds = self.interval_minutes * 60
            # Sleep in small increments so stop() is responsive
            for _ in range(sleep_seconds):
                if not self._running:
                    break
                time.sleep(1)

    # ==================================================================
    # PRIVATE: Fetch positions
    # ==================================================================

    def _fetch_kite_positions(self) -> Dict[str, Dict]:
        """Fetch NFO positions with non-zero quantity from Kite."""
        positions = self.kite.positions()
        net = positions.get("net", [])

        result = {}
        for pos in net:
            qty = pos.get("quantity", 0)
            exchange = pos.get("exchange", "")
            if qty != 0 and exchange == "NFO":
                symbol = pos["tradingsymbol"]
                result[symbol] = {
                    "tradingsymbol": symbol,
                    "quantity": qty,
                    "average_price": pos.get("average_price", 0),
                    "pnl": pos.get("pnl", 0),
                }
        return result

    def _fetch_local_positions(self) -> Dict[str, Dict]:
        """Fetch open positions from trades table for today."""
        cur = self.db.cursor()
        cur.execute(
            """
            SELECT trade_id, signal_id, tradingsymbol, transaction_type,
                   quantity, entry_price
            FROM trades
            WHERE exit_date IS NULL
              AND DATE(created_at) = CURRENT_DATE
            """
        )
        rows = cur.fetchall()

        # Aggregate net quantity per tradingsymbol
        positions: Dict[str, Dict] = {}
        for trade_id, signal_id, symbol, tx_type, qty, price in rows:
            if symbol not in positions:
                positions[symbol] = {
                    "tradingsymbol": symbol,
                    "quantity": 0,
                    "trade_ids": [],
                    "signal_ids": [],
                    "entry_price": price or 0,
                }
            # BUY adds, SELL subtracts
            if tx_type == "BUY":
                positions[symbol]["quantity"] += qty
            else:
                positions[symbol]["quantity"] -= qty
            positions[symbol]["trade_ids"].append(trade_id)
            positions[symbol]["signal_ids"].append(signal_id)

        # Filter to non-zero net positions
        return {sym: pos for sym, pos in positions.items() if pos["quantity"] != 0}

    # ==================================================================
    # PRIVATE: Discrepancy detection
    # ==================================================================

    def _detect_orphaned(
        self,
        kite_pos: Dict[str, Dict],
        local_pos: Dict[str, Dict],
        result: Dict,
    ):
        """ORPHANED: in Kite but not in local DB."""
        for symbol, kpos in kite_pos.items():
            if symbol not in local_pos:
                result["orphaned"] += 1
                detail = {
                    "type": "ORPHANED",
                    "tradingsymbol": symbol,
                    "kite_qty": kpos["quantity"],
                    "kite_avg_price": kpos["average_price"],
                    "action": "added_to_local",
                }
                result["details"].append(detail)

                logger.warning(
                    f"ORPHANED: {symbol} qty={kpos['quantity']} in Kite but not in local DB"
                )
                if self.alerter:
                    self.alerter.send(
                        "CRITICAL",
                        f"ORPHANED position detected: {symbol} qty={kpos['quantity']}\n"
                        f"In Kite but NOT in local DB. Adding to local tracking.",
                    )

                # Add to local tracking
                self._add_orphan_to_local(symbol, kpos)

    def _detect_phantom(
        self,
        kite_pos: Dict[str, Dict],
        local_pos: Dict[str, Dict],
        result: Dict,
    ):
        """PHANTOM: in local DB but not in Kite."""
        for symbol, lpos in local_pos.items():
            if symbol not in kite_pos:
                result["phantom"] += 1
                detail = {
                    "type": "PHANTOM",
                    "tradingsymbol": symbol,
                    "local_qty": lpos["quantity"],
                    "trade_ids": lpos["trade_ids"],
                    "action": "marked_closed_in_local",
                }
                result["details"].append(detail)

                logger.warning(
                    f"PHANTOM: {symbol} qty={lpos['quantity']} in local DB but not in Kite"
                )
                if self.alerter:
                    self.alerter.send(
                        "WARNING",
                        f"PHANTOM position detected: {symbol} qty={lpos['quantity']}\n"
                        f"In local DB but NOT in Kite. Marking as closed.",
                    )

                # Mark as closed in local
                self._close_phantom_in_local(lpos["trade_ids"])

    def _detect_drift(
        self,
        kite_pos: Dict[str, Dict],
        local_pos: Dict[str, Dict],
        result: Dict,
    ):
        """DRIFT: quantity mismatch between Kite and local."""
        for symbol in kite_pos:
            if symbol in local_pos:
                kite_qty = kite_pos[symbol]["quantity"]
                local_qty = local_pos[symbol]["quantity"]

                if kite_qty != local_qty:
                    result["drift"] += 1
                    detail = {
                        "type": "DRIFT",
                        "tradingsymbol": symbol,
                        "kite_qty": kite_qty,
                        "local_qty": local_qty,
                        "diff": kite_qty - local_qty,
                        "action": "local_adjusted_to_kite",
                    }
                    result["details"].append(detail)

                    logger.warning(
                        f"DRIFT: {symbol} kite_qty={kite_qty} local_qty={local_qty} "
                        f"diff={kite_qty - local_qty}"
                    )
                    if self.alerter:
                        self.alerter.send(
                            "INFO",
                            f"Position DRIFT detected: {symbol}\n"
                            f"Kite qty={kite_qty}, Local qty={local_qty}\n"
                            f"Adjusting local to match Kite.",
                        )

                    # Adjust local to match Kite (Kite is source of truth)
                    self._adjust_local_quantity(
                        local_pos[symbol]["trade_ids"],
                        kite_qty,
                        local_qty,
                    )

    # ==================================================================
    # PRIVATE: DB corrections
    # ==================================================================

    def _add_orphan_to_local(self, symbol: str, kpos: Dict):
        """Insert an orphaned Kite position into the trades table."""
        try:
            cur = self.db.cursor()
            tx_type = "BUY" if kpos["quantity"] > 0 else "SELL"
            cur.execute(
                """
                INSERT INTO trades
                    (signal_id, tradingsymbol, transaction_type, quantity,
                     entry_price, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                """,
                (
                    f"RECON_ORPHAN_{datetime.now().strftime('%H%M%S')}",
                    symbol,
                    tx_type,
                    abs(kpos["quantity"]),
                    kpos.get("average_price", 0),
                ),
            )
            self.db.commit()
            logger.info(f"Added orphan {symbol} to local trades table")
        except Exception as e:
            logger.error(f"Failed to add orphan {symbol} to local: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

    def _close_phantom_in_local(self, trade_ids: List):
        """Mark phantom trades as closed with exit_date = now."""
        try:
            cur = self.db.cursor()
            for trade_id in trade_ids:
                cur.execute(
                    """
                    UPDATE trades
                    SET exit_date = CURRENT_DATE
                    WHERE trade_id = %s AND exit_date IS NULL
                    """,
                    (trade_id,),
                )
            self.db.commit()
            logger.info(f"Marked phantom trade_ids {trade_ids} as closed")
        except Exception as e:
            logger.error(f"Failed to close phantom trades {trade_ids}: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

    def _adjust_local_quantity(
        self, trade_ids: List, kite_qty: int, local_qty: int
    ):
        """
        Adjust local position quantity to match Kite.
        Updates the most recent trade record's quantity to absorb the diff.
        """
        if not trade_ids:
            return
        try:
            diff = kite_qty - local_qty
            # Apply adjustment to the last trade entry
            last_trade_id = trade_ids[-1]
            cur = self.db.cursor()
            cur.execute(
                """
                UPDATE trades
                SET quantity = quantity + %s
                WHERE trade_id = %s
                """,
                (diff, last_trade_id),
            )
            self.db.commit()
            logger.info(
                f"Adjusted trade_id={last_trade_id} quantity by {diff:+d} "
                f"to match Kite (was {local_qty}, now {kite_qty})"
            )
        except Exception as e:
            logger.error(f"Failed to adjust quantity for trade_ids {trade_ids}: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

    # ==================================================================
    # PRIVATE: Logging to DB
    # ==================================================================

    def _log_to_db(self, result: Dict):
        """Log reconciliation result to session_reconciliation_log table."""
        try:
            cur = self.db.cursor()
            cur.execute(
                """
                INSERT INTO session_reconciliation_log
                    (run_time, orphaned, phantom, drift, details_json)
                VALUES (NOW(), %s, %s, %s, %s)
                """,
                (
                    result["orphaned"],
                    result["phantom"],
                    result["drift"],
                    json.dumps(result["details"], default=str),
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to log session reconciliation: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

    # ==================================================================
    # PUBLIC: Status
    # ==================================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current reconciler state."""
        return {
            "running": self._running,
            "interval_minutes": self.interval_minutes,
            "paper_mode": self.kite is None,
            "market_open": MARKET_OPEN.strftime("%H:%M"),
            "market_close": MARKET_CLOSE.strftime("%H:%M"),
            "in_market_hours": MARKET_OPEN <= datetime.now().time() <= MARKET_CLOSE,
        }
