"""
Async Fill Monitor — non-blocking order fill tracking.

Monitors multiple orders concurrently via a single background polling
thread.  Callers submit an order ID + callback and return immediately.
The callback fires when the order reaches a terminal state (COMPLETE,
CANCELLED, REJECTED, TIMEOUT).

Usage:
    from execution.fill_monitor import AsyncFillMonitor

    monitor = AsyncFillMonitor(kite, alerter, db)
    monitor.start()

    # Non-blocking — returns immediately
    monitor.submit(order_id, expected_price=200.0,
                   tradingsymbol='NIFTY2630524500CE',
                   signal_id='ORB_TUNED',
                   callback=on_fill)

    # Legacy blocking API (backwards-compatible)
    result = monitor.monitor_fill(order_id, timeout_seconds=90, ...)
"""

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

POLL_INTERVAL_SECONDS = 2
DEFAULT_TIMEOUT = 90
SLIPPAGE_ALERT_THRESHOLD = 0.002  # alert if slippage > 0.2%

# Paper mode slippage simulation
PAPER_SLIPPAGE_MIN = -0.002
PAPER_SLIPPAGE_MAX = 0.005


@dataclass
class FillRequest:
    """Pending fill monitoring request."""
    order_id: str
    callback: Optional[Callable]
    expected_price: float = 0.0
    tradingsymbol: str = ""
    signal_id: str = ""
    timeout: float = DEFAULT_TIMEOUT
    submitted_at: float = 0.0
    _event: threading.Event = field(default_factory=threading.Event)
    _result: Optional[Dict] = None


class AsyncFillMonitor:
    """
    Non-blocking fill monitor.  A single daemon thread polls all pending
    orders every POLL_INTERVAL_SECONDS and fires callbacks on terminal states.
    """

    # Alias for backwards compatibility
    FillMonitor = None  # set after class definition

    def __init__(self, kite, alerter, db=None, logger_override=None,
                 poll_interval: float = POLL_INTERVAL_SECONDS,
                 max_wait: float = DEFAULT_TIMEOUT):
        self.kite = kite
        self.alerter = alerter
        self.db = db
        self.log = logger_override or logger
        self.poll_interval = poll_interval
        self.max_wait = max_wait

        self._pending: Dict[str, FillRequest] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Aggregate slippage tracker
        self._slippage_log: List[Dict] = []

    # ──────────────────────────────────────────────────────────
    # PUBLIC: Non-blocking API
    # ──────────────────────────────────────────────────────────

    def start(self):
        """Start the background polling thread (call once at startup)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="fill-monitor", daemon=True
        )
        self._thread.start()
        self.log.info("AsyncFillMonitor started (poll=%ss)", self.poll_interval)

    def stop(self):
        """Stop the background thread."""
        self._running = False

    def submit(
        self,
        order_id: str,
        callback: Optional[Callable] = None,
        timeout: float = DEFAULT_TIMEOUT,
        expected_price: float = 0.0,
        tradingsymbol: str = "",
        signal_id: str = "",
    ) -> FillRequest:
        """
        Non-blocking: register an order for fill monitoring.
        Returns immediately.  *callback(fill_result)* is invoked from the
        background thread when the order reaches a terminal state.

        If callback is None, use req.wait() to block or req.result to poll.
        """
        # Paper mode: resolve immediately
        if EXECUTION_MODE == "PAPER" or order_id.startswith("PAPER-"):
            result = self._simulate_paper_fill(
                order_id, expected_price, tradingsymbol, signal_id
            )
            req = FillRequest(
                order_id=order_id, callback=callback,
                expected_price=expected_price,
                tradingsymbol=tradingsymbol, signal_id=signal_id,
                submitted_at=time.time(),
            )
            req._result = result
            req._event.set()
            if callback:
                callback(result)
            return req

        req = FillRequest(
            order_id=order_id,
            callback=callback,
            expected_price=expected_price,
            tradingsymbol=tradingsymbol,
            signal_id=signal_id,
            timeout=timeout,
            submitted_at=time.time(),
        )
        with self._lock:
            self._pending[order_id] = req

        self.log.info(
            "Submitted fill monitor: %s %s timeout=%ss",
            order_id, tradingsymbol, timeout,
        )

        # Auto-start if not running
        if not self._running:
            self.start()

        return req

    def pending_count(self) -> int:
        """Number of orders currently being monitored."""
        with self._lock:
            return len(self._pending)

    # ──────────────────────────────────────────────────────────
    # PUBLIC: Legacy blocking API (backwards-compatible)
    # ──────────────────────────────────────────────────────────

    def monitor_fill(
        self,
        order_id: str,
        timeout_seconds: int = 90,
        expected_price: float = 0.0,
        tradingsymbol: str = "",
        signal_id: str = "",
    ) -> Optional[Dict]:
        """
        Blocking API (backwards-compatible with old FillMonitor).
        Submits to async monitor and waits for result.
        """
        req = self.submit(
            order_id=order_id,
            callback=None,
            timeout=timeout_seconds,
            expected_price=expected_price,
            tradingsymbol=tradingsymbol,
            signal_id=signal_id,
        )
        # Block until result is ready
        req._event.wait(timeout=timeout_seconds + 10)
        return req._result

    # ──────────────────────────────────────────────────────────
    # BACKGROUND: Polling loop
    # ──────────────────────────────────────────────────────────

    def _poll_loop(self):
        """
        Single thread polls ALL pending orders every poll_interval.
        Much more efficient than one blocking wait per order.
        """
        while self._running:
            with self._lock:
                snapshot = list(self._pending.items())

            for order_id, req in snapshot:
                try:
                    self._check_order(order_id, req)
                except Exception as e:
                    self.log.warning("Fill poll error for %s: %s", order_id, e)

            time.sleep(self.poll_interval)

    def _check_order(self, order_id: str, req: FillRequest):
        """Check a single order's status via Kite API."""
        elapsed = time.time() - req.submitted_at

        # Timeout check
        if elapsed >= req.timeout:
            self._handle_timeout(order_id, req)
            return

        # Query Kite
        history = self.kite.order_history(order_id)
        if not history:
            return

        latest = history[-1]
        status = latest.get("status", "").upper()

        if status == "COMPLETE":
            fill_price = float(latest.get("average_price", 0))
            fill_qty = int(latest.get("filled_quantity", 0))
            fill_time = latest.get("exchange_timestamp") or datetime.now()
            slippage = self._calc_slippage(req.expected_price, fill_price)

            result = {
                "status": "FILLED",
                "order_id": order_id,
                "fill_price": fill_price,
                "fill_quantity": fill_qty,
                "fill_time": str(fill_time),
                "slippage": round(slippage, 4),
                "slippage_rupees": round(
                    (fill_price - req.expected_price) * fill_qty, 2
                ) if req.expected_price > 0 else 0,
                "elapsed_seconds": round(elapsed, 1),
            }

            self.log.info(
                "FILLED | %s x%d @ %.2f | slippage=%+.2f%% | %.1fs",
                req.tradingsymbol, fill_qty, fill_price, slippage * 100, elapsed,
            )

            # Track slippage
            self._record_slippage(req, result)

            self._resolve(order_id, req, result)

        elif status in ("CANCELLED", "REJECTED"):
            reason = latest.get("status_message", "unknown")
            self.log.warning("Order %s: %s — %s", status, req.tradingsymbol, reason)
            self.alerter.send(
                "WARNING",
                f"Order {status}: {req.tradingsymbol} — {reason}",
                signal_id=req.signal_id,
            )
            self._resolve(order_id, req, None)

    def _handle_timeout(self, order_id: str, req: FillRequest):
        """Cancel unfilled order after timeout, check for partials."""
        self.log.warning(
            "TIMEOUT (%.0fs) — cancelling %s for %s",
            req.timeout, order_id, req.tradingsymbol,
        )

        try:
            history = self.kite.order_history(order_id)
            latest = history[-1] if history else {}
            filled_qty = int(latest.get("filled_quantity", 0))

            # Cancel pending portion
            try:
                self.kite.cancel_order(variety="regular", order_id=order_id)
            except Exception as ce:
                self.log.warning("Cancel failed (may already be terminal): %s", ce)

            if filled_qty > 0:
                fill_price = float(latest.get("average_price", 0))
                slippage = self._calc_slippage(req.expected_price, fill_price)
                result = {
                    "status": "PARTIAL",
                    "order_id": order_id,
                    "fill_price": fill_price,
                    "fill_quantity": filled_qty,
                    "fill_time": str(datetime.now()),
                    "slippage": round(slippage, 4),
                    "slippage_rupees": round(
                        (fill_price - req.expected_price) * filled_qty, 2
                    ) if req.expected_price > 0 else 0,
                    "elapsed_seconds": round(time.time() - req.submitted_at, 1),
                }
                self.alerter.send(
                    "WARNING",
                    f"Partial fill after timeout: {req.tradingsymbol} "
                    f"x{filled_qty} @ {fill_price:.2f}",
                    signal_id=req.signal_id,
                    priority="critical",
                )
                self._resolve(order_id, req, result)
                return

        except Exception as e:
            self.log.error("Error during timeout handling for %s: %s", order_id, e)

        # Nothing filled
        self.alerter.send(
            "WARNING",
            f"UNFILLED after {req.timeout:.0f}s: {req.tradingsymbol} — cancelled",
            signal_id=req.signal_id,
        )
        self._resolve(order_id, req, None)

    def _resolve(self, order_id: str, req: FillRequest, result: Optional[Dict]):
        """Remove from pending, set result, fire callback."""
        with self._lock:
            self._pending.pop(order_id, None)

        req._result = result
        req._event.set()

        if req.callback:
            try:
                req.callback(result)
            except Exception as e:
                self.log.error("Fill callback error for %s: %s", order_id, e)

    # ──────────────────────────────────────────────────────────
    # Slippage tracking
    # ──────────────────────────────────────────────────────────

    def _record_slippage(self, req: FillRequest, result: Dict):
        """Track slippage and alert if excessive."""
        slippage = result.get("slippage", 0)
        entry = {
            "order_id": req.order_id,
            "tradingsymbol": req.tradingsymbol,
            "signal_id": req.signal_id,
            "expected_price": req.expected_price,
            "fill_price": result.get("fill_price", 0),
            "slippage_pct": slippage,
            "timestamp": datetime.now().isoformat(),
        }
        self._slippage_log.append(entry)

        # Keep bounded
        if len(self._slippage_log) > 500:
            self._slippage_log = self._slippage_log[-500:]

        # Alert on excessive slippage
        if abs(slippage) > SLIPPAGE_ALERT_THRESHOLD:
            self.alerter.send(
                "WARNING",
                f"High slippage: {req.tradingsymbol} "
                f"{slippage:+.2%} ({result['fill_price']:.2f} vs "
                f"{req.expected_price:.2f})",
                signal_id=req.signal_id,
            )

        # Log to DB if available
        if self.db:
            try:
                cur = self.db.cursor()
                cur.execute(
                    "INSERT INTO slippage_log "
                    "(order_id, tradingsymbol, signal_id, expected_price, "
                    " fill_price, slippage_pct, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, NOW())",
                    (req.order_id, req.tradingsymbol, req.signal_id,
                     req.expected_price, result["fill_price"], slippage),
                )
                self.db.commit()
            except Exception:
                pass  # table may not exist yet

    def get_slippage_stats(self) -> Dict:
        """Aggregate slippage statistics."""
        if not self._slippage_log:
            return {"count": 0, "avg_slippage": 0, "max_slippage": 0}
        slips = [e["slippage_pct"] for e in self._slippage_log]
        return {
            "count": len(slips),
            "avg_slippage": round(sum(slips) / len(slips), 4),
            "max_slippage": round(max(slips, key=abs), 4),
            "alerts": sum(1 for s in slips if abs(s) > SLIPPAGE_ALERT_THRESHOLD),
        }

    # ──────────────────────────────────────────────────────────
    # Paper mode
    # ──────────────────────────────────────────────────────────

    def _simulate_paper_fill(
        self, order_id, expected_price, tradingsymbol, signal_id,
    ) -> Dict:
        """Simulate fill in PAPER mode with random slippage."""
        if expected_price <= 0:
            expected_price = 200.0

        slippage_pct = random.uniform(PAPER_SLIPPAGE_MIN, PAPER_SLIPPAGE_MAX)
        fill_price = expected_price * (1 + slippage_pct)
        fill_price = round(fill_price / 0.05) * 0.05
        fill_price = round(fill_price, 2)

        result = {
            "status": "FILLED",
            "order_id": order_id,
            "fill_price": fill_price,
            "fill_quantity": 0,
            "fill_time": datetime.now().isoformat(),
            "slippage": round(slippage_pct, 4),
            "slippage_rupees": 0,
            "elapsed_seconds": 0.1,
        }

        self.log.info(
            "PAPER FILL | %s @ %.2f (expected %.2f, slippage %+.2f%%) | %s",
            tradingsymbol, fill_price, expected_price, slippage_pct * 100, order_id,
        )
        return result

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _calc_slippage(expected: float, actual: float) -> float:
        if expected <= 0:
            return 0.0
        return (actual - expected) / expected


# Backwards-compatible alias
FillMonitor = AsyncFillMonitor
