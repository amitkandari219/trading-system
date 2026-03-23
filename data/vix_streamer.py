"""
Real-time India VIX streaming via Kite WebSocket (KiteTicker).

Provides continuous VIX updates for dynamic regime management.
Falls back to kite.ltp() polling on WebSocket disconnect, and to
DB-based static VIX in PAPER mode.

Usage:
    from data.vix_streamer import VIXStreamer

    streamer = VIXStreamer(kite, on_update_callback=my_callback)
    streamer.start()
    vix = streamer.get_vix()         # thread-safe
    change = streamer.get_vix_change()  # session stats
    streamer.stop()
"""

import logging
import threading
import time as time_mod
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

import psycopg2

from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

# India VIX instrument details
VIX_EXCHANGE = "NSE"
VIX_TRADINGSYMBOL = "INDIA VIX"
VIX_LTP_KEY = f"{VIX_EXCHANGE}:{VIX_TRADINGSYMBOL}"

# Staleness thresholds
STALE_WARN_SECONDS = 120   # warn after 2 min without update
STALE_POLL_SECONDS = 60    # poll kite.ltp if no tick for 60s
LTP_POLL_INTERVAL = 30     # poll every 30s when WS is down

# History buffer for session stats
MAX_HISTORY_LEN = 500  # ~8 hrs of 1-per-min ticks

# Minimum VIX change (pts) to trigger on_update callback
VIX_CHANGE_THRESHOLD = 0.5


class VIXStreamer:
    """
    Real-time India VIX streamer using KiteTicker WebSocket.

    Thread-safe: all shared state guarded by self._lock.
    PAPER mode: reads static VIX from nifty_daily table (no WebSocket).
    """

    def __init__(self, kite=None, on_update_callback: Optional[Callable] = None):
        """
        Args:
            kite: authenticated KiteConnect instance (None for PAPER mode)
            on_update_callback: called with (vix: float, timestamp: datetime)
                                when VIX changes by >= 0.5 pts
        """
        self._kite = kite
        self._on_update = on_update_callback

        # Thread-safe state
        self._lock = threading.Lock()
        self._current_vix: Optional[float] = None
        self._last_update: Optional[datetime] = None
        self._session_open_vix: Optional[float] = None
        self._session_high: Optional[float] = None
        self._session_low: Optional[float] = None
        self._history: deque = deque(maxlen=MAX_HISTORY_LEN)
        self._last_callback_vix: Optional[float] = None

        # VIX instrument token (resolved at start)
        self._vix_token: Optional[int] = None

        # Ticker and threads
        self._ticker = None
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_running = False
        self._connected = False
        self._stopped = False

        # Paper mode flag
        self._paper_mode = (kite is None)

        if self._paper_mode:
            logger.info("VIXStreamer: PAPER mode — static VIX from DB")
        else:
            logger.info("VIXStreamer: LIVE mode — WebSocket streaming")

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════

    def start(self):
        """
        Start VIX streaming.

        LIVE: resolve VIX token, start KiteTicker, connect(threaded=True).
        PAPER: load static VIX from nifty_daily.
        """
        if self._paper_mode:
            self._load_static_vix()
            return

        # Resolve India VIX instrument token
        self._vix_token = self._resolve_vix_token()
        if self._vix_token is None:
            logger.error(
                "Could not resolve India VIX instrument token. "
                "Falling back to LTP polling."
            )
            self._start_poll_thread()
            return

        # Start KiteTicker
        try:
            from kiteconnect import KiteTicker

            api_key = self._kite.api_key
            access_token = self._kite.access_token
            self._ticker = KiteTicker(api_key, access_token)

            self._ticker.on_ticks = self._on_ticks
            self._ticker.on_connect = self._on_connect
            self._ticker.on_close = self._on_close
            self._ticker.on_error = self._on_error

            self._ticker.connect(threaded=True)
            logger.info(
                f"VIXStreamer: KiteTicker connecting "
                f"(token={self._vix_token})"
            )
        except Exception as e:
            logger.error(f"VIXStreamer: KiteTicker start failed: {e}")
            self._start_poll_thread()

    def stop(self):
        """Stop streaming and clean up."""
        self._stopped = True
        self._poll_running = False

        if self._ticker is not None:
            try:
                self._ticker.close()
            except Exception as e:
                logger.debug(f"VIXStreamer: ticker close error: {e}")
            self._ticker = None

        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)

        logger.info("VIXStreamer stopped")

    def get_vix(self) -> Optional[float]:
        """
        Get current VIX value (thread-safe).

        Returns fresh value if updated within 60s.
        Falls back to kite.ltp() polling if stale.
        Returns None only if no data at all.
        """
        with self._lock:
            now = datetime.now()

            # Check freshness
            if self._current_vix is not None and self._last_update is not None:
                age = (now - self._last_update).total_seconds()

                if age < STALE_POLL_SECONDS:
                    return self._current_vix

                # Stale — try LTP poll
                if age >= STALE_WARN_SECONDS:
                    logger.warning(
                        f"VIXStreamer: VIX data stale "
                        f"({age:.0f}s old), using last known value"
                    )

        # Try LTP poll (outside lock to avoid blocking)
        if not self._paper_mode and self._kite is not None:
            vix = self._poll_ltp()
            if vix is not None:
                return vix

        # Return last known value
        with self._lock:
            return self._current_vix

    def get_vix_change(self) -> Dict:
        """
        Get VIX session statistics.

        Returns:
            dict with current, open, high, low, change_pct, change_abs
        """
        with self._lock:
            current = self._current_vix
            open_vix = self._session_open_vix
            high = self._session_high
            low = self._session_low

        if current is None:
            return {
                'current': None, 'open': None,
                'high': None, 'low': None,
                'change_pct': 0.0, 'change_abs': 0.0,
            }

        change_abs = (current - open_vix) if open_vix else 0.0
        change_pct = (change_abs / open_vix * 100) if open_vix and open_vix > 0 else 0.0

        return {
            'current': round(current, 2),
            'open': round(open_vix, 2) if open_vix else None,
            'high': round(high, 2) if high else None,
            'low': round(low, 2) if low else None,
            'change_pct': round(change_pct, 2),
            'change_abs': round(change_abs, 2),
        }

    # ══════════════════════════════════════════════════════════
    # KITETICKER CALLBACKS
    # ══════════════════════════════════════════════════════════

    def _on_connect(self, ws, response):
        """Called when WebSocket connects. Subscribe to VIX token."""
        logger.info("VIXStreamer: WebSocket connected")
        self._connected = True
        if self._vix_token is not None:
            ws.subscribe([self._vix_token])
            ws.set_mode(ws.MODE_LTP, [self._vix_token])
            logger.info(f"VIXStreamer: subscribed to VIX token {self._vix_token}")

    def _on_ticks(self, ws, ticks):
        """Called on every tick. Update VIX value."""
        for tick in ticks:
            if tick.get('instrument_token') == self._vix_token:
                vix_val = tick.get('last_price')
                if vix_val is not None and vix_val > 0:
                    self._update_vix(float(vix_val), datetime.now())

    def _on_close(self, ws, code, reason):
        """Called when WebSocket disconnects."""
        logger.warning(
            f"VIXStreamer: WebSocket closed (code={code}, reason={reason})"
        )
        self._connected = False

        # Start fallback polling if not shutting down
        if not self._stopped:
            logger.info("VIXStreamer: starting LTP poll fallback")
            self._start_poll_thread()

    def _on_error(self, ws, code, reason):
        """Called on WebSocket error."""
        logger.error(
            f"VIXStreamer: WebSocket error (code={code}, reason={reason})"
        )

    # ══════════════════════════════════════════════════════════
    # INTERNAL
    # ══════════════════════════════════════════════════════════

    def _update_vix(self, vix: float, timestamp: datetime):
        """Thread-safe VIX update with callback trigger."""
        with self._lock:
            old_vix = self._current_vix
            self._current_vix = vix
            self._last_update = timestamp

            # Session stats
            if self._session_open_vix is None:
                self._session_open_vix = vix
            if self._session_high is None or vix > self._session_high:
                self._session_high = vix
            if self._session_low is None or vix < self._session_low:
                self._session_low = vix

            self._history.append((timestamp, vix))

            # Check if callback should fire
            should_callback = False
            if self._on_update is not None:
                if self._last_callback_vix is None:
                    should_callback = True
                elif abs(vix - self._last_callback_vix) >= VIX_CHANGE_THRESHOLD:
                    should_callback = True

            if should_callback:
                self._last_callback_vix = vix

        # Fire callback outside lock to avoid deadlocks
        if should_callback and self._on_update is not None:
            try:
                self._on_update(vix, timestamp)
            except Exception as e:
                logger.error(f"VIXStreamer: callback error: {e}")

    def _resolve_vix_token(self) -> Optional[int]:
        """Resolve India VIX instrument token via kite.instruments()."""
        try:
            instruments = self._kite.instruments(VIX_EXCHANGE)
            for inst in instruments:
                if inst.get('tradingsymbol') == VIX_TRADINGSYMBOL:
                    token = inst['instrument_token']
                    logger.info(
                        f"VIXStreamer: resolved India VIX token = {token}"
                    )
                    return token

            # Try partial match
            for inst in instruments:
                ts = inst.get('tradingsymbol', '')
                if 'INDIA' in ts and 'VIX' in ts:
                    token = inst['instrument_token']
                    logger.info(
                        f"VIXStreamer: resolved India VIX token = {token} "
                        f"(matched '{ts}')"
                    )
                    return token

            logger.error("VIXStreamer: India VIX not found in instruments")
            return None
        except Exception as e:
            logger.error(f"VIXStreamer: instruments lookup failed: {e}")
            return None

    def _poll_ltp(self) -> Optional[float]:
        """Poll VIX via kite.ltp() — fallback when WS is down."""
        try:
            data = self._kite.ltp(VIX_LTP_KEY)
            vix = data.get(VIX_LTP_KEY, {}).get('last_price')
            if vix is not None and vix > 0:
                self._update_vix(float(vix), datetime.now())
                return float(vix)
        except Exception as e:
            logger.debug(f"VIXStreamer: LTP poll failed: {e}")
        return None

    def _start_poll_thread(self):
        """Start background thread for LTP polling (WS fallback)."""
        if self._poll_running:
            return

        self._poll_running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="VIXStreamer-LTPPoll",
            daemon=True,
        )
        self._poll_thread.start()

    def _poll_loop(self):
        """Background LTP poll loop — runs every 30s until stopped or WS reconnects."""
        logger.info("VIXStreamer: LTP poll loop started")
        while self._poll_running and not self._stopped:
            # Stop polling if WS reconnected
            if self._connected:
                logger.info("VIXStreamer: WS reconnected, stopping poll loop")
                break

            self._poll_ltp()
            # Sleep in small increments to check stop flag
            for _ in range(LTP_POLL_INTERVAL):
                if not self._poll_running or self._stopped:
                    break
                time_mod.sleep(1)

        self._poll_running = False
        logger.info("VIXStreamer: LTP poll loop ended")

    def _load_static_vix(self):
        """PAPER mode: load VIX from nifty_daily table (static for session)."""
        try:
            conn = psycopg2.connect(DATABASE_DSN)
            cur = conn.cursor()
            cur.execute("""
                SELECT india_vix FROM nifty_daily
                WHERE india_vix IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            conn.close()

            if row and row[0]:
                vix = float(row[0])
                self._update_vix(vix, datetime.now())
                logger.info(f"VIXStreamer: PAPER mode VIX = {vix:.2f} (from DB)")
            else:
                logger.warning(
                    "VIXStreamer: no VIX data in nifty_daily, defaulting to 15.0"
                )
                self._update_vix(15.0, datetime.now())

        except Exception as e:
            logger.error(f"VIXStreamer: DB VIX load failed: {e}")
            self._update_vix(15.0, datetime.now())
