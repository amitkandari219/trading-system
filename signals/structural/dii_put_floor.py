"""
DII Put Floor Detection Signal.

Domestic Institutional Investors (DIIs) — primarily mutual funds and
insurance companies — routinely write large put positions at strikes
they consider strong support levels.  When DII put writing OI at a
single strike exceeds ~20 lakh, that strike acts as a "floor" — the
put writers will hedge/defend it aggressively.

Signal logic:
    1. Scan participant-wise OI for the strike where DII put writing
       OI > 20 lakh AND is increasing (building floor).
    2. Strike must be within 3 % of current Nifty spot.
    3. LONG if Nifty approaches the floor from above (within 50 pts).
    4. SHORT if Nifty breaks below floor by 30+ pts (floor broke).
    5. Skip if FII put OI > 2x DII put OI at that strike (FII dominates).
    6. Skip on event days (RBI policy, Budget, etc.).

Risk management:
    LONG:
        SL  = floor_strike - 60 pts
        TGT = entry + 100 pts
    SHORT (floor break):
        SL  = floor_strike + 20 pts
        TGT = next major put OI cluster below, or entry - 100 pts

Data source:
    - Live: NSE participant-wise OI data (available daily EOD).
    - Backtest: approximate using total put OI (DII typically ≈ 30-40 %
      of total put writing).

Interface:
    evaluate(trade_date, nifty_price, dii_put_oi_by_strike,
             fii_put_oi_by_strike, vix, is_event_day) → dict | None

    backtest_evaluate(trade_date, nifty_price, option_chain_df) → dict | None

Usage:
    from signals.structural.dii_put_floor import DIIPutFloorSignal

    sig = DIIPutFloorSignal()
    dii_oi = {24000: 2500000, 24100: 1800000, 24200: 3200000}
    result = sig.evaluate(date.today(), 24250, dii_oi)

Academic basis: Informed institutional hedging creates sticky support
levels; Ni, Pearson & Poteshman (2005) — "Stock price clustering on
option expiration dates".
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = "DII_PUT_FLOOR"

# OI thresholds
MIN_DII_PUT_OI = 2_000_000          # 20 lakh contracts
DII_SHARE_OF_TOTAL = 0.35           # Approximate DII share for backtest proxy

# Proximity thresholds
MAX_FLOOR_DISTANCE_PCT = 0.03       # Floor strike must be within 3 % of spot
APPROACH_DISTANCE_PTS = 50          # LONG zone: price within 50 pts above floor
BREAK_DISTANCE_PTS = 30             # SHORT zone: price ≥ 30 pts below floor

# FII dominance filter
FII_DOMINANCE_RATIO = 2.0           # Skip if FII put OI > 2x DII at that strike

# Risk management (Nifty points)
LONG_SL_PTS = 60                    # SL below floor
LONG_TGT_PTS = 100                  # Target above entry
SHORT_SL_PTS = 20                   # SL above floor (tight — floor just broke)
SHORT_TGT_PTS = 100                 # Target below entry (default if no cluster)

# Max hold
MAX_HOLD_BARS = 48                  # 48 × 5-min = 4 hours

# Confidence
BASE_CONF_APPROACH = 0.57           # Approaching floor from above
BASE_CONF_BREAK = 0.50              # Floor break (lower — contrarian)
OI_MASSIVE_BOOST = 0.06             # OI > 40 lakh
OI_INCREASING_BOOST = 0.04          # OI increasing day-on-day
FII_ABSENT_BOOST = 0.03             # No FII presence at floor
VIX_LOW_BOOST = 0.03                # VIX < 14


# ================================================================
# Helpers
# ================================================================

def _safe_float(val: Any, default: float = float("nan")) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely cast to int."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _find_dii_floor(
    dii_put_oi_by_strike: Dict[int, int],
    nifty_price: float,
    min_oi: int = MIN_DII_PUT_OI,
    max_distance_pct: float = MAX_FLOOR_DISTANCE_PCT,
) -> Optional[Dict[str, Any]]:
    """
    Find the DII put floor strike.

    Parameters
    ----------
    dii_put_oi_by_strike : dict {strike: oi_value}
        DII put writing OI at each strike.
    nifty_price : float
        Current Nifty spot / futures price.
    min_oi : int
        Minimum OI threshold to qualify as a floor.
    max_distance_pct : float
        Maximum distance from spot as a fraction (e.g. 0.03 = 3 %).

    Returns
    -------
    dict with {strike, oi, distance_pts, distance_pct} or None.
    """
    if not dii_put_oi_by_strike or nifty_price <= 0:
        return None

    max_distance_pts = nifty_price * max_distance_pct
    best_strike = None
    best_oi = 0

    for strike, oi in dii_put_oi_by_strike.items():
        strike = _safe_int(strike)
        oi = _safe_int(oi)

        if strike <= 0 or oi < min_oi:
            continue

        distance_pts = abs(nifty_price - strike)
        if distance_pts > max_distance_pts:
            continue

        # Among qualifying strikes, pick the one with highest OI
        if oi > best_oi:
            best_oi = oi
            best_strike = strike

    if best_strike is None:
        return None

    distance_pts = nifty_price - best_strike  # positive = price above floor
    distance_pct = distance_pts / nifty_price if nifty_price > 0 else 0.0

    return {
        "strike": best_strike,
        "oi": best_oi,
        "distance_pts": round(distance_pts, 2),
        "distance_pct": round(distance_pct * 100, 4),
    }


def _is_floor_break(nifty_price: float, floor_strike: int) -> bool:
    """
    True if price has broken below the floor by BREAK_DISTANCE_PTS.

    A floor break means Nifty is trading ≥ 30 pts below the DII floor
    strike — suggesting the institutional support has failed.
    """
    return (floor_strike - nifty_price) >= BREAK_DISTANCE_PTS


def _find_next_put_cluster(
    dii_put_oi_by_strike: Dict[int, int],
    floor_strike: int,
    min_oi: int = MIN_DII_PUT_OI,
) -> Optional[int]:
    """
    Find the next significant put OI cluster below the broken floor.

    Used to set a SHORT target when the floor breaks.

    Returns
    -------
    Next lower strike with OI ≥ min_oi, or None.
    """
    candidates = []
    for strike, oi in dii_put_oi_by_strike.items():
        strike = _safe_int(strike)
        oi = _safe_int(oi)
        if strike < floor_strike and oi >= min_oi:
            candidates.append((strike, oi))

    if not candidates:
        return None

    # Pick the highest strike below the floor (nearest cluster)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][0]


# ================================================================
# Signal Class
# ================================================================

class DIIPutFloorSignal:
    """
    DII put floor detection signal.

    Identifies strikes where DII put writing OI creates a support floor,
    then generates LONG signals when Nifty approaches the floor and
    SHORT signals when the floor breaks.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        self._daily_trade_count: int = 0
        self._max_daily_trades: int = 2
        logger.info("DIIPutFloorSignal initialised")

    # ----------------------------------------------------------------
    # Public: live evaluate
    # ----------------------------------------------------------------

    def evaluate(
        self,
        trade_date: date,
        nifty_price: float,
        dii_put_oi_by_strike: Dict[int, int],
        fii_put_oi_by_strike: Optional[Dict[int, int]] = None,
        vix: float = 15.0,
        is_event_day: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate DII put floor signal.

        Parameters
        ----------
        trade_date            : Current trading date.
        nifty_price           : Current Nifty spot / futures price.
        dii_put_oi_by_strike  : dict {strike: oi_value} for DII put writing.
        fii_put_oi_by_strike  : dict {strike: oi_value} for FII put writing
                                (optional — used for dominance check).
        vix                   : India VIX value.
        is_event_day          : True for RBI policy, Budget, etc.

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(
                trade_date, nifty_price, dii_put_oi_by_strike,
                fii_put_oi_by_strike, vix, is_event_day,
            )
        except Exception as e:
            logger.error(
                "DIIPutFloorSignal.evaluate error: %s", e, exc_info=True
            )
            return None

    def _evaluate_inner(
        self,
        trade_date: date,
        nifty_price: float,
        dii_put_oi_by_strike: Dict[int, int],
        fii_put_oi_by_strike: Optional[Dict[int, int]],
        vix: float,
        is_event_day: bool,
    ) -> Optional[Dict[str, Any]]:
        # ── Reset daily counter ────────────────────────────────────
        if self._last_fire_date != trade_date:
            self._daily_trade_count = 0

        # ── Max trades per day ─────────────────────────────────────
        if self._daily_trade_count >= self._max_daily_trades:
            return None

        # ── Event day filter ───────────────────────────────────────
        if is_event_day:
            logger.debug("%s: event day — skip", trade_date)
            return None

        # ── Validate price ─────────────────────────────────────────
        nifty_price = _safe_float(nifty_price)
        if math.isnan(nifty_price) or nifty_price <= 0:
            logger.debug("Invalid nifty_price: %s", nifty_price)
            return None

        # ── Validate OI data ───────────────────────────────────────
        if not dii_put_oi_by_strike:
            logger.debug("Empty dii_put_oi_by_strike")
            return None

        # ── Find DII floor ─────────────────────────────────────────
        floor = _find_dii_floor(dii_put_oi_by_strike, nifty_price)
        if floor is None:
            logger.debug(
                "%s: no qualifying DII floor near %.1f",
                self.SIGNAL_ID, nifty_price,
            )
            return None

        floor_strike = floor["strike"]
        floor_oi = floor["oi"]

        # ── FII dominance filter ───────────────────────────────────
        if fii_put_oi_by_strike:
            fii_oi_at_floor = _safe_int(
                fii_put_oi_by_strike.get(floor_strike, 0)
            )
            if fii_oi_at_floor > floor_oi * FII_DOMINANCE_RATIO:
                logger.debug(
                    "%s: FII dominates strike %d (FII=%d vs DII=%d)",
                    self.SIGNAL_ID, floor_strike, fii_oi_at_floor, floor_oi,
                )
                return None
            fii_absent = fii_oi_at_floor == 0
        else:
            fii_absent = True
            fii_oi_at_floor = 0

        # ── Determine signal type ──────────────────────────────────
        is_break = _is_floor_break(nifty_price, floor_strike)
        dist_above_floor = nifty_price - floor_strike  # positive = above

        if is_break:
            # Floor broken → SHORT
            direction = "SHORT"
            signal_type = "FLOOR_BREAK"
        elif 0 <= dist_above_floor <= APPROACH_DISTANCE_PTS:
            # Approaching floor from above → LONG (bounce)
            direction = "LONG"
            signal_type = "FLOOR_APPROACH"
        else:
            # Price too far from floor — no signal
            return None

        # ── VIX context (not a hard filter, but affects confidence) ─
        vix = _safe_float(vix, default=15.0)
        if math.isnan(vix):
            vix = 15.0

        # ── SL / TGT ──────────────────────────────────────────────
        if direction == "LONG":
            stop_loss = floor_strike - LONG_SL_PTS
            target = round(nifty_price + LONG_TGT_PTS, 2)
        else:
            stop_loss = floor_strike + SHORT_SL_PTS
            # Try to find next put cluster for a smarter target
            next_cluster = _find_next_put_cluster(
                dii_put_oi_by_strike, floor_strike
            )
            if next_cluster is not None:
                target = float(next_cluster)
            else:
                target = round(nifty_price - SHORT_TGT_PTS, 2)

        # ── Confidence ─────────────────────────────────────────────
        if signal_type == "FLOOR_APPROACH":
            confidence = BASE_CONF_APPROACH
        else:
            confidence = BASE_CONF_BREAK

        if floor_oi > MIN_DII_PUT_OI * 2:
            confidence += OI_MASSIVE_BOOST
        if fii_absent:
            confidence += FII_ABSENT_BOOST
        if vix < 14.0:
            confidence += VIX_LOW_BOOST

        confidence = min(0.85, max(0.10, confidence))

        # ── Risk / reward ──────────────────────────────────────────
        risk = abs(nifty_price - stop_loss)
        reward = abs(target - nifty_price)
        rr = reward / risk if risk > 0 else 0.0

        # ── Reason ─────────────────────────────────────────────────
        reason_parts = [
            f"DII_PUT_FLOOR ({signal_type})",
            f"Dir={direction}",
            f"Floor={floor_strike}",
            f"DII_OI={floor_oi:,}",
            f"FII_OI={fii_oi_at_floor:,}",
            f"Spot={nifty_price:.2f}",
            f"Dist={floor['distance_pts']}pts",
            f"VIX={vix:.1f}",
            f"R:R={rr:.1f}",
        ]

        self._last_fire_date = trade_date
        self._daily_trade_count += 1

        logger.info(
            "%s signal: %s %s %s floor=%d oi=%d price=%.1f conf=%.3f",
            self.SIGNAL_ID, signal_type, direction, trade_date,
            floor_strike, floor_oi, nifty_price, confidence,
        )

        return {
            "signal_id": self.SIGNAL_ID,
            "signal_type": signal_type,
            "direction": direction,
            "confidence": round(confidence, 3),
            "trade_date": trade_date.isoformat(),
            "entry_price": round(nifty_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "floor_strike": floor_strike,
            "floor_oi": floor_oi,
            "fii_oi_at_floor": fii_oi_at_floor,
            "distance_pts": floor["distance_pts"],
            "distance_pct": floor["distance_pct"],
            "max_hold_bars": MAX_HOLD_BARS,
            "vix": round(vix, 2),
            "risk_reward": round(rr, 2),
            "reason": " | ".join(reason_parts),
        }

    # ----------------------------------------------------------------
    # Floor detection (public for external use)
    # ----------------------------------------------------------------

    @staticmethod
    def _find_dii_floor(
        dii_put_oi_by_strike: Dict[int, int],
        nifty_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the DII put floor strike.

        Parameters
        ----------
        dii_put_oi_by_strike : dict {strike: oi_value}
        nifty_price          : Current Nifty price.

        Returns
        -------
        {strike, oi, distance_pts, distance_pct} or None.
        """
        return _find_dii_floor(dii_put_oi_by_strike, nifty_price)

    # ----------------------------------------------------------------
    # Floor break detection (public)
    # ----------------------------------------------------------------

    @staticmethod
    def _is_floor_break(nifty_price: float, floor_strike: int) -> bool:
        """True if price has broken below floor by ≥ 30 pts."""
        return _is_floor_break(nifty_price, floor_strike)

    # ----------------------------------------------------------------
    # Backtest evaluate
    # ----------------------------------------------------------------

    def backtest_evaluate(
        self,
        trade_date: date,
        nifty_price: float,
        option_chain_df=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Backtest variant — approximates DII floor from total put OI.

        Since participant-wise OI breakdown is not available in the
        historical database, we use total put OI as a proxy and assume
        DII contributes ~35 % of total put writing.

        Parameters
        ----------
        trade_date      : Date being evaluated.
        nifty_price     : Nifty spot at evaluation time.
        option_chain_df : pandas DataFrame with columns:
                          strike, option_type ('CE'/'PE'), oi.

        Returns
        -------
        dict with signal details, or None.
        """
        try:
            return self._backtest_inner(trade_date, nifty_price, option_chain_df)
        except Exception as e:
            logger.error(
                "DIIPutFloorSignal.backtest_evaluate error: %s",
                e, exc_info=True,
            )
            return None

    def _backtest_inner(
        self,
        trade_date: date,
        nifty_price: float,
        option_chain_df,
    ) -> Optional[Dict[str, Any]]:
        # ── Validate price ─────────────────────────────────────────
        nifty_price = _safe_float(nifty_price)
        if math.isnan(nifty_price) or nifty_price <= 0:
            return None

        # ── Build approximate DII OI from total put OI ─────────────
        if option_chain_df is None:
            return None

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not available for backtest")
            return None

        if option_chain_df.empty:
            return None

        df = option_chain_df.copy()

        # Normalise column names
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("strike", "strike_price"):
                col_map[c] = "strike"
            elif cl in ("oi", "open_interest", "openinterest"):
                col_map[c] = "oi"
            elif cl in ("option_type", "optiontype", "type", "opt_type"):
                col_map[c] = "option_type"
        df = df.rename(columns=col_map)

        required = {"strike", "oi", "option_type"}
        if not required.issubset(set(df.columns)):
            logger.warning(
                "Option chain DF missing columns: %s",
                required - set(df.columns),
            )
            return None

        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype(int)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce").fillna(0).astype(int)
        df["option_type"] = df["option_type"].astype(str).str.upper().str.strip()
        df["option_type"] = df["option_type"].replace({
            "PUT": "PE", "P": "PE", "CALL": "CE", "C": "CE",
        })

        # Extract put OI by strike
        puts = df[df["option_type"] == "PE"]
        if puts.empty:
            return None

        # Build approximate DII OI map (total_put_oi * DII_SHARE)
        approx_dii_oi: Dict[int, int] = {}
        for _, row in puts.iterrows():
            strike = int(row["strike"])
            total_oi = int(row["oi"])
            dii_oi_approx = int(total_oi * DII_SHARE_OF_TOTAL)
            if dii_oi_approx > 0:
                approx_dii_oi[strike] = dii_oi_approx

        if not approx_dii_oi:
            return None

        # ── Find floor using approximated DII OI ───────────────────
        floor = _find_dii_floor(approx_dii_oi, nifty_price)
        if floor is None:
            return None

        floor_strike = floor["strike"]
        floor_oi = floor["oi"]

        # ── Determine signal type ──────────────────────────────────
        is_break = _is_floor_break(nifty_price, floor_strike)
        dist_above_floor = nifty_price - floor_strike

        if is_break:
            direction = "SHORT"
            signal_type = "FLOOR_BREAK"
        elif 0 <= dist_above_floor <= APPROACH_DISTANCE_PTS:
            direction = "LONG"
            signal_type = "FLOOR_APPROACH"
        else:
            return None

        # ── SL / TGT ──────────────────────────────────────────────
        if direction == "LONG":
            stop_loss = floor_strike - LONG_SL_PTS
            target = round(nifty_price + LONG_TGT_PTS, 2)
        else:
            stop_loss = floor_strike + SHORT_SL_PTS
            next_cluster = _find_next_put_cluster(approx_dii_oi, floor_strike)
            if next_cluster is not None:
                target = float(next_cluster)
            else:
                target = round(nifty_price - SHORT_TGT_PTS, 2)

        # ── Confidence (lower for backtest approximation) ──────────
        if signal_type == "FLOOR_APPROACH":
            confidence = BASE_CONF_APPROACH - 0.03  # penalty for proxy data
        else:
            confidence = BASE_CONF_BREAK - 0.03

        if floor_oi > MIN_DII_PUT_OI * 2:
            confidence += OI_MASSIVE_BOOST

        confidence = min(0.80, max(0.10, confidence))

        risk = abs(nifty_price - stop_loss)
        reward = abs(target - nifty_price)
        rr = reward / risk if risk > 0 else 0.0

        reason_parts = [
            f"DII_PUT_FLOOR_BT ({signal_type})",
            f"Dir={direction}",
            f"Floor={floor_strike}",
            f"ApproxDII_OI={floor_oi:,}",
            f"Spot={nifty_price:.2f}",
            f"Dist={floor['distance_pts']}pts",
            f"R:R={rr:.1f}",
            "proxy_data=True",
        ]

        logger.info(
            "%s backtest: %s %s %s floor=%d oi=%d price=%.1f conf=%.3f",
            self.SIGNAL_ID, signal_type, direction, trade_date,
            floor_strike, floor_oi, nifty_price, confidence,
        )

        return {
            "signal_id": self.SIGNAL_ID,
            "signal_type": signal_type,
            "direction": direction,
            "confidence": round(confidence, 3),
            "trade_date": trade_date.isoformat(),
            "entry_price": round(nifty_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "floor_strike": floor_strike,
            "floor_oi": floor_oi,
            "floor_oi_is_proxy": True,
            "distance_pts": floor["distance_pts"],
            "distance_pct": floor["distance_pct"],
            "max_hold_bars": MAX_HOLD_BARS,
            "risk_reward": round(rr, 2),
            "reason": " | ".join(reason_parts),
        }

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None
        self._daily_trade_count = 0

    def __repr__(self) -> str:
        return f"DIIPutFloorSignal(signal_id='{self.SIGNAL_ID}')"
