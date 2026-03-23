"""
Kite Bridge — core order placement with audit trail, retry logic, and PAPER mode.

Translates options_executor output dicts into kite.place_order() calls.
Logs every order to the database BEFORE calling the Kite API (audit trail).

Usage:
    from execution.kite_bridge import KiteBridge
    bridge = KiteBridge(kite, db, alerter)
    order_id = bridge.place_order(order_dict)
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# Rate limiter: max 10 orders/second
_MAX_ORDERS_PER_SECOND = 10
_rate_lock = Lock()
_order_timestamps: list = []

# LTP buffer for limit orders — add 0.5% for quick fills
LTP_BUFFER_PCT = 0.005


class KiteBridge:
    """
    Places orders through Kite Connect with full audit trail.

    In PAPER mode, logs everything but never touches the Kite API.
    In LIVE mode, places real orders with retry-on-failure.
    """

    def __init__(self, kite, db, alerter, logger_override=None):
        """
        Args:
            kite:    authenticated KiteConnect instance (can be None in PAPER mode)
            db:      database connection with execute() method
            alerter: TelegramAlerter instance
        """
        self.kite = kite
        self.db = db
        self.alerter = alerter
        self.log = logger_override or logger

    def place_order(self, order_dict: Dict) -> Optional[str]:
        """
        Place an order through Kite Connect.

        Args:
            order_dict: output from options_executor.signal_to_orders()
                Required keys: tradingsymbol, transaction_type, lots, lot_size,
                               signal_id, instrument
                Optional keys: limit_price, option_type, strike

        Returns:
            order_id (str) on success, None on failure.
        """
        signal_id = order_dict.get("signal_id", "UNKNOWN")
        tradingsymbol = order_dict.get("tradingsymbol", "")
        transaction_type = order_dict.get("transaction_type", "BUY")
        lots = int(order_dict.get("lots", 1))
        lot_size = int(order_dict.get("lot_size", 25))
        quantity = lots * lot_size  # Kite wants units, not lots

        if quantity <= 0:
            self.log.warning(f"[{signal_id}] Zero quantity — skipping order")
            return None

        # ── 2x Safety Cap (Layer 7) ──
        # Reject if final lots exceed 2x the system-calculated base.
        # system_lots is set by CompoundSizer before any behavioral adjustments.
        system_lots = int(order_dict.get("system_lots", 0))
        if system_lots > 0 and lots > 2 * system_lots:
            self.log.warning(
                f"[{signal_id}] 2x SAFETY CAP: requested {lots} lots "
                f"> 2x system {system_lots} — REJECTING order"
            )
            return None

        # --- Fetch LTP and compute limit price ---
        limit_price = self._compute_limit_price(
            tradingsymbol, transaction_type, order_dict
        )

        # --- Rate limit ---
        self._enforce_rate_limit()

        # --- Build Kite order params ---
        tag = signal_id[:20] if signal_id else ""
        kite_params = {
            "variety": "regular",
            "exchange": "NFO",
            "tradingsymbol": tradingsymbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "product": "MIS",
            "order_type": "LIMIT",
            "price": limit_price,
            "validity": "DAY",
            "tag": tag,
        }

        # --- Log to DB BEFORE placing (audit trail) ---
        audit_id = self._log_order_to_db(signal_id, kite_params, "PENDING")

        # --- PAPER mode: skip Kite API ---
        if EXECUTION_MODE == "PAPER":
            fake_order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
            self.log.info(
                f"PAPER ORDER | {transaction_type} {tradingsymbol} "
                f"x{quantity} ({lots}L) @ {limit_price:.2f} | "
                f"signal={signal_id} | order_id={fake_order_id}"
            )
            self._update_order_status(audit_id, "PAPER_PLACED", fake_order_id)
            return fake_order_id

        # --- LIVE mode: place order with retry ---
        return self._place_live_order(
            kite_params, signal_id, audit_id, lots, lot_size
        )

    # ------------------------------------------------------------------
    # Private: live order placement with retry
    # ------------------------------------------------------------------

    def _place_live_order(
        self,
        kite_params: Dict,
        signal_id: str,
        audit_id: Optional[int],
        lots: int,
        lot_size: int,
    ) -> Optional[str]:
        """Place order via Kite API. On failure, retry once with MARKET order."""

        # Attempt 1: LIMIT order
        try:
            order_id = self.kite.place_order(**kite_params)
            self.log.info(
                f"LIMIT ORDER PLACED | {kite_params['transaction_type']} "
                f"{kite_params['tradingsymbol']} x{kite_params['quantity']} "
                f"@ {kite_params['price']:.2f} | order_id={order_id}"
            )
            self._update_order_status(audit_id, "PLACED", str(order_id))
            self.alerter.send(
                "INFO",
                f"Order placed: {kite_params['transaction_type']} "
                f"{kite_params['tradingsymbol']} x{lots}L "
                f"@ {kite_params['price']:.2f}",
                signal_id=signal_id,
            )
            return str(order_id)

        except Exception as e:
            error_msg = str(e)
            self.log.warning(
                f"LIMIT order failed for {signal_id}: {error_msg} — retrying as MARKET"
            )

            # Check for margin errors — no point retrying
            if "margin" in error_msg.lower() or "insufficient" in error_msg.lower():
                self.log.error(f"Margin error for {signal_id}: {error_msg}")
                self._update_order_status(audit_id, "MARGIN_REJECTED", error=error_msg)
                self.alerter.send(
                    "CRITICAL",
                    f"Margin rejected: {kite_params['tradingsymbol']} — {error_msg}",
                    signal_id=signal_id,
                )
                return None

        # Attempt 2: MARKET order (fallback)
        try:
            market_params = dict(kite_params)
            market_params["order_type"] = "MARKET"
            market_params.pop("price", None)

            self._enforce_rate_limit()
            order_id = self.kite.place_order(**market_params)
            self.log.info(
                f"MARKET ORDER PLACED (retry) | {market_params['transaction_type']} "
                f"{market_params['tradingsymbol']} x{market_params['quantity']} "
                f"| order_id={order_id}"
            )
            self._update_order_status(audit_id, "PLACED_MARKET", str(order_id))
            return str(order_id)

        except Exception as e2:
            error_msg = str(e2)
            self.log.error(
                f"MARKET order also failed for {signal_id}: {error_msg}"
            )
            self._update_order_status(audit_id, "FAILED", error=error_msg)
            self.alerter.send(
                "CRITICAL",
                f"Order FAILED (both LIMIT and MARKET): "
                f"{kite_params['tradingsymbol']} — {error_msg}",
                signal_id=signal_id,
            )
            return None

    # ------------------------------------------------------------------
    # Private: LTP and pricing
    # ------------------------------------------------------------------

    def _compute_limit_price(
        self, tradingsymbol: str, transaction_type: str, order_dict: Dict
    ) -> float:
        """
        Compute limit price: fetch LTP and add buffer for quick fills.

        BUY orders  → LTP * (1 + 0.5%)  (pay slightly more)
        SELL orders → LTP * (1 - 0.5%)  (accept slightly less)
        """
        # Try to get live LTP
        ltp = None
        if EXECUTION_MODE == "LIVE" and self.kite:
            try:
                quote_key = f"NFO:{tradingsymbol}"
                quote = self.kite.ltp(quote_key)
                ltp = quote[quote_key]["last_price"]
            except Exception as e:
                self.log.warning(f"LTP fetch failed for {tradingsymbol}: {e}")

        # Fallback to limit_price from order_dict
        if ltp is None:
            ltp = float(order_dict.get("limit_price", 0) or order_dict.get("entry_premium", 0))

        if ltp <= 0:
            self.log.warning(f"No price available for {tradingsymbol} — using 0")
            return 0.0

        # Apply buffer
        if transaction_type == "BUY":
            price = ltp * (1 + LTP_BUFFER_PCT)
        else:
            price = ltp * (1 - LTP_BUFFER_PCT)

        # Round to tick size (0.05)
        price = round(price / 0.05) * 0.05
        return round(price, 2)

    # ------------------------------------------------------------------
    # Private: rate limiting
    # ------------------------------------------------------------------

    def _enforce_rate_limit(self):
        """Block until we are under 10 orders/second."""
        with _rate_lock:
            now = time.time()
            # Purge timestamps older than 1 second
            _order_timestamps[:] = [t for t in _order_timestamps if now - t < 1.0]

            if len(_order_timestamps) >= _MAX_ORDERS_PER_SECOND:
                sleep_time = 1.0 - (now - _order_timestamps[0])
                if sleep_time > 0:
                    self.log.debug(f"Rate limit — sleeping {sleep_time:.3f}s")
                    time.sleep(sleep_time)

            _order_timestamps.append(time.time())

    # ------------------------------------------------------------------
    # Private: database audit trail
    # ------------------------------------------------------------------

    def _log_order_to_db(
        self, signal_id: str, kite_params: Dict, status: str
    ) -> Optional[int]:
        """
        Log order to database BEFORE placing. Returns audit row ID.
        """
        try:
            cur = self.db.cursor()
            cur.execute(
                """
                INSERT INTO order_audit
                    (signal_id, tradingsymbol, transaction_type, quantity,
                     product, order_type, price, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
                """,
                (
                    signal_id,
                    kite_params.get("tradingsymbol", ""),
                    kite_params.get("transaction_type", ""),
                    kite_params.get("quantity", 0),
                    kite_params.get("product", "MIS"),
                    kite_params.get("order_type", "LIMIT"),
                    kite_params.get("price", 0),
                    status,
                ),
            )
            self.db.commit()
            row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            self.log.error(f"Failed to log order to DB: {e}")
            return None

    def _update_order_status(
        self,
        audit_id: Optional[int],
        status: str,
        order_id: str = None,
        error: str = None,
    ):
        """Update the audit row with final status and broker order_id."""
        if audit_id is None:
            return
        try:
            cur = self.db.cursor()
            cur.execute(
                """
                UPDATE order_audit
                SET status = %s,
                    broker_order_id = %s,
                    error_message = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (status, order_id, error, audit_id),
            )
            self.db.commit()
        except Exception as e:
            self.log.error(f"Failed to update order audit: {e}")
            try:
                self.db.rollback()
            except:
                pass
