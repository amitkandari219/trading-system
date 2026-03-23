"""
NFO Symbol Resolver — maps (instrument, strike, expiry, option_type) to Kite tradingsymbol.

Caches kite.instruments("NFO") at startup for fast lookups.
Handles Nifty/BankNifty strike rounding and monthly vs weekly expiry formats.

Usage:
    from execution.symbol_resolver import SymbolResolver
    resolver = SymbolResolver(kite)
    info = resolver.resolve_option_symbol("NIFTY", 23500, "2603027", "CE")
    strike = resolver.find_atm_strike("NIFTY", 23520.75)
"""

import logging
import os
import time
from datetime import datetime, date
from typing import Dict, Optional

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# Strike intervals per instrument
STRIKE_INTERVALS = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
}

# Lot sizes per instrument
LOT_SIZES = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
}


class SymbolResolver:
    """
    Resolves NFO option symbols using cached instrument master.
    Thread-safe for reads after initial load.
    """

    def __init__(self, kite=None):
        """
        Args:
            kite: authenticated KiteConnect instance (None in PAPER mode).
        """
        self.kite = kite
        self._instruments: Dict[str, dict] = {}   # tradingsymbol -> instrument row
        self._by_key: Dict[tuple, dict] = {}       # (name, strike, expiry, type) -> row
        self._loaded = False
        self._load_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self):
        """
        Fetch and cache NFO instrument master from Kite.
        Call once at startup or when the cache is stale (> 6 hours).
        In PAPER mode, creates a minimal stub cache.
        """
        if EXECUTION_MODE == "PAPER" and (self.kite is None):
            logger.info("PAPER mode — symbol resolver using stub cache")
            self._loaded = True
            self._load_time = time.time()
            return

        if self.kite is None:
            raise RuntimeError("SymbolResolver requires an authenticated KiteConnect instance")

        try:
            logger.info("Loading NFO instrument master from Kite...")
            start = time.time()
            instruments = self.kite.instruments("NFO")
            elapsed = time.time() - start

            self._instruments.clear()
            self._by_key.clear()

            for inst in instruments:
                tsym = inst.get("tradingsymbol", "")
                self._instruments[tsym] = inst

                # Build lookup key: (name, strike, expiry_date, instrument_type)
                name = inst.get("name", "")
                strike = inst.get("strike", 0)
                expiry = inst.get("expiry")  # date object from Kite
                inst_type = inst.get("instrument_type", "")  # CE / PE / FUT

                if name and expiry:
                    key = (name, float(strike), expiry, inst_type)
                    self._by_key[key] = inst

            self._loaded = True
            self._load_time = time.time()
            logger.info(
                f"Loaded {len(self._instruments)} NFO instruments in {elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load NFO instruments: {e}")
            raise

    def ensure_loaded(self):
        """Load instruments if not already loaded or cache is stale (> 6h)."""
        if not self._loaded:
            self.load()
            return
        if self._load_time and (time.time() - self._load_time) > 6 * 3600:
            logger.info("Instrument cache stale (>6h) — reloading")
            self.load()

    def resolve_option_symbol(
        self,
        instrument: str,
        strike: int,
        expiry,
        option_type: str,
    ) -> Optional[Dict]:
        """
        Resolve an option to its Kite tradingsymbol and instrument_token.

        Args:
            instrument: "NIFTY" or "BANKNIFTY"
            strike: strike price (e.g. 23500)
            expiry: expiry date — either a date object, or str in "YYMMDD" format
            option_type: "CE" or "PE"

        Returns:
            dict with keys: tradingsymbol, instrument_token, lot_size, tick_size
            or None if not found.
        """
        self.ensure_loaded()

        # Normalise expiry to a date object
        if isinstance(expiry, str):
            expiry_date = datetime.strptime(expiry, "%y%m%d").date()
        elif isinstance(expiry, datetime):
            expiry_date = expiry.date()
        elif isinstance(expiry, date):
            expiry_date = expiry
        else:
            logger.error(f"Invalid expiry type: {type(expiry)}")
            return None

        option_type = option_type.upper()

        # --- Try cached lookup first ---
        key = (instrument.upper(), float(strike), expiry_date, option_type)
        row = self._by_key.get(key)

        if row:
            return self._row_to_result(row, instrument)

        # --- Fallback: build tradingsymbol and search ---
        tsym = self._build_tradingsymbol(instrument, strike, expiry_date, option_type)
        row = self._instruments.get(tsym)
        if row:
            return self._row_to_result(row, instrument)

        # --- PAPER mode fallback: synthesise a result ---
        if EXECUTION_MODE == "PAPER":
            lot_size = LOT_SIZES.get(instrument.upper(), 25)
            result = {
                "tradingsymbol": tsym,
                "instrument_token": hash(tsym) & 0x7FFFFFFF,  # deterministic fake token
                "lot_size": lot_size,
                "tick_size": 0.05,
            }
            logger.debug(f"PAPER mode — synthesised symbol: {tsym}")
            return result

        logger.warning(f"Symbol not found: {instrument} {strike} {expiry_date} {option_type}")
        return None

    def find_atm_strike(self, instrument: str, price: float) -> int:
        """
        Round price to the nearest ATM strike.

        Nifty  → round to nearest 50
        BankNifty → round to nearest 100

        Args:
            instrument: "NIFTY" or "BANKNIFTY"
            price: current underlying price

        Returns:
            ATM strike as int.
        """
        interval = STRIKE_INTERVALS.get(instrument.upper(), 50)
        return int(round(price / interval) * interval)

    def get_lot_size(self, instrument: str) -> int:
        """Return the lot size for an instrument."""
        return LOT_SIZES.get(instrument.upper(), 25)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tradingsymbol(
        self, instrument: str, strike: int, expiry_date: date, option_type: str
    ) -> str:
        """
        Build Kite-format tradingsymbol.

        Weekly format:  NIFTY2530623500CE   (NIFTY + YYMDD + strike + CE/PE)
        Monthly format: NIFTY25MAR23500CE   (NIFTY + YYMMM + strike + CE/PE)

        Monthly expiry is the last Thursday of the month.
        """
        yy = expiry_date.strftime("%y")

        if self._is_monthly_expiry(expiry_date):
            # Monthly: NIFTY25MAR23500CE
            month_str = expiry_date.strftime("%b").upper()
            return f"{instrument.upper()}{yy}{month_str}{strike}{option_type}"
        else:
            # Weekly: NIFTY2530623500CE (YYMDD — single-char month codes not used post-2024)
            mmdd = expiry_date.strftime("%m%d")
            return f"{instrument.upper()}{yy}{mmdd}{strike}{option_type}"

    def _is_monthly_expiry(self, expiry_date: date) -> bool:
        """
        Check if a date is the monthly expiry (last Thursday of the month).
        """
        import calendar
        year, month = expiry_date.year, expiry_date.month
        # Find last Thursday
        last_day = calendar.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != 3:  # Thursday = 3
            d = d.replace(day=d.day - 1)
        return expiry_date == d

    def _row_to_result(self, row: dict, instrument: str) -> dict:
        """Convert an instrument master row to our standard result dict."""
        lot_size = row.get("lot_size", LOT_SIZES.get(instrument.upper(), 25))
        return {
            "tradingsymbol": row["tradingsymbol"],
            "instrument_token": row.get("instrument_token", 0),
            "lot_size": lot_size,
            "tick_size": row.get("tick_size", 0.05),
        }
