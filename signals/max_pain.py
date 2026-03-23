"""
Max Pain Calculator & Strike Clustering Gravitational Signal for Nifty weekly options.

Max pain theory: option writers (who hold the majority of capital) have an incentive
to move the underlying toward the strike where total option buyer payoff is minimized.
Near expiry (DTE <= 2), this gravitational pull intensifies as gamma pins the spot.

Data source:
- Historical: `nifty_options` table (692K rows, Jan 2024 – Mar 2026)
  Columns: date, expiry, strike, option_type, open, high, low, close, volume, oi,
           implied_volatility, delta, gamma, theta, vega
- Live: Kite Connect option chain via kite.instruments('NFO') + kite.quote()

Usage:
    from signals.max_pain import MaxPainCalculator, StrikeClusteringSignal

    calc = MaxPainCalculator(db_conn=conn)
    chain = calc.get_option_chain(expiry=date(2026, 3, 26), as_of=date(2026, 3, 25))
    result = calc.calculate_max_pain(chain, spot_price=23450)

    sig = StrikeClusteringSignal(calc)
    signal = sig.check(spot=23450, today=date(2026, 3, 25), vix=16.5)
"""

import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

NIFTY_STRIKE_INTERVAL = 50
ATM_RANGE_STRIKES = 10          # ±10 strikes around ATM for PCR calc
PIN_BASE_PROBABILITY = 0.35     # empirical base rate for pin to max pain
MAX_CONFIDENCE = 0.92           # cap confidence to avoid overfit
VIX_THRESHOLD = 22.0            # above this, pinning weakens
EVENT_DAY_PENALTY = 0.40        # multiplicative penalty on event days

# Signal thresholds
ENTRY_MIN_DTE = 0
ENTRY_MAX_DTE = 2
ENTRY_DIST_MIN_PCT = 0.004      # 0.4% — too close = no edge
ENTRY_DIST_MAX_PCT = 0.015      # 1.5% — too far = weak pull
ENTRY_MIN_CONFIDENCE = 0.60
SL_BUFFER_PCT = 0.008           # 0.8% beyond max pain on losing side
EXPIRY_TARGET_TIME = time(14, 0) # Thursday 14:00 IST


# ================================================================
# MaxPainCalculator
# ================================================================

class MaxPainCalculator:
    """Compute max pain strike and OI-based support/resistance levels."""

    def __init__(self, kite=None, db_conn=None):
        """
        Parameters
        ----------
        kite : KiteConnect instance (optional, for live option chain)
        db_conn : psycopg2 connection (optional, for historical queries)
        """
        self.kite = kite
        self.db_conn = db_conn

    # ----------------------------------------------------------
    # DB connection helper
    # ----------------------------------------------------------
    def _get_conn(self):
        """Return existing connection or create a new one."""
        if self.db_conn and not self.db_conn.closed:
            return self.db_conn
        try:
            self.db_conn = psycopg2.connect(DATABASE_DSN)
            return self.db_conn
        except Exception as e:
            logger.error("Cannot connect to database: %s", e)
            return None

    # ----------------------------------------------------------
    # Next expiry helper
    # ----------------------------------------------------------
    @staticmethod
    def next_weekly_expiry(as_of: date) -> date:
        """Return the next Nifty weekly expiry (Thursday, or Wednesday if Thu is holiday)."""
        d = as_of
        # Find next Thursday (weekday 3)
        days_ahead = (3 - d.weekday()) % 7
        if days_ahead == 0 and d.weekday() == 3:
            # Already Thursday — this is the expiry if market hasn't closed
            return d
        if days_ahead == 0:
            days_ahead = 7
        return d + timedelta(days=days_ahead)

    @staticmethod
    def dte(as_of: date, expiry: date) -> int:
        """Calendar days to expiry (0 on expiry day itself)."""
        return max((expiry - as_of).days, 0)

    # ----------------------------------------------------------
    # Option chain retrieval
    # ----------------------------------------------------------
    def get_option_chain(
        self,
        instrument: str = 'NIFTY',
        expiry: Optional[date] = None,
        as_of: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch option chain as list of dicts with keys:
          strike, call_oi, put_oi, call_premium, put_premium

        Live path: kite.instruments('NFO') + kite.quote()
        Historical path: query nifty_options table for (as_of, expiry)
        """
        if as_of is None:
            as_of = date.today()
        if expiry is None:
            expiry = self.next_weekly_expiry(as_of)

        # Try live first if kite is available and as_of is today
        if self.kite and as_of == date.today():
            chain = self._fetch_live_chain(instrument, expiry)
            if chain:
                return chain
            logger.warning("Live chain fetch failed, falling back to DB")

        # Historical / DB fallback
        return self._fetch_db_chain(instrument, expiry, as_of)

    def _fetch_live_chain(
        self, instrument: str, expiry: date
    ) -> List[Dict[str, Any]]:
        """Fetch live option chain from Kite Connect."""
        try:
            all_instruments = self.kite.instruments('NFO')
            expiry_dt = datetime.combine(expiry, time(0, 0))
            opts = [
                i for i in all_instruments
                if i['name'] == instrument
                and i['instrument_type'] in ('CE', 'PE')
                and i['expiry'] == expiry_dt.date()
            ]
            if not opts:
                return []

            # Group by strike
            strikes: Dict[float, Dict] = {}
            token_map: Dict[int, Tuple[float, str]] = {}
            for o in opts:
                s = float(o['strike'])
                otype = 'call' if o['instrument_type'] == 'CE' else 'put'
                if s not in strikes:
                    strikes[s] = {'strike': s, 'call_oi': 0, 'put_oi': 0,
                                  'call_premium': 0.0, 'put_premium': 0.0}
                token_map[o['instrument_token']] = (s, otype)

            # Batch quote (Kite allows up to 500 instruments per call)
            tokens = list(token_map.keys())
            for batch_start in range(0, len(tokens), 500):
                batch = tokens[batch_start:batch_start + 500]
                instrument_keys = [f"NFO:{t}" for t in batch]
                quotes = self.kite.quote(instrument_keys)
                for key, q in quotes.items():
                    token = int(key.split(':')[1]) if ':' in key else None
                    if token and token in token_map:
                        s, otype = token_map[token]
                        oi_val = q.get('oi', 0) or 0
                        ltp = q.get('last_price', 0) or 0
                        if otype == 'call':
                            strikes[s]['call_oi'] = oi_val
                            strikes[s]['call_premium'] = ltp
                        else:
                            strikes[s]['put_oi'] = oi_val
                            strikes[s]['put_premium'] = ltp

            chain = sorted(strikes.values(), key=lambda x: x['strike'])
            logger.info("Live chain: %d strikes for %s expiry %s",
                        len(chain), instrument, expiry)
            return chain

        except Exception as e:
            logger.error("Live chain fetch error: %s", e)
            return []

    def _fetch_db_chain(
        self, instrument: str, expiry: date, as_of: date
    ) -> List[Dict[str, Any]]:
        """Fetch option chain from nifty_options table."""
        conn = self._get_conn()
        if not conn:
            return []

        query = """
            SELECT strike, option_type, oi, close
            FROM nifty_options
            WHERE date = %s AND expiry = %s
            ORDER BY strike, option_type
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query, (as_of, expiry))
                rows = cur.fetchall()

            if not rows:
                logger.warning("No option data for date=%s expiry=%s", as_of, expiry)
                return []

            strikes: Dict[float, Dict] = {}
            for strike, otype, oi, close_px in rows:
                s = float(strike)
                if s not in strikes:
                    strikes[s] = {'strike': s, 'call_oi': 0, 'put_oi': 0,
                                  'call_premium': 0.0, 'put_premium': 0.0}
                oi_val = int(oi) if oi else 0
                px_val = float(close_px) if close_px else 0.0
                if otype in ('CE', 'CALL', 'call'):
                    strikes[s]['call_oi'] = oi_val
                    strikes[s]['call_premium'] = px_val
                else:
                    strikes[s]['put_oi'] = oi_val
                    strikes[s]['put_premium'] = px_val

            chain = sorted(strikes.values(), key=lambda x: x['strike'])
            logger.info("DB chain: %d strikes for date=%s expiry=%s",
                        len(chain), as_of, expiry)
            return chain

        except Exception as e:
            logger.error("DB chain fetch error: %s", e)
            return []

    # ----------------------------------------------------------
    # Max pain calculation
    # ----------------------------------------------------------
    def calculate_max_pain(
        self,
        option_chain: List[Dict[str, Any]],
        spot_price: float,
    ) -> Dict[str, Any]:
        """
        Compute the max pain strike — the settlement price that minimizes
        total payout to option buyers (= minimizes writer losses).

        For each candidate settlement S:
          call_loss = sum over strikes K where K < S: (S - K) * call_oi_at_K
          put_loss  = sum over strikes K where K > S: (K - S) * put_oi_at_K
          total_loss = call_loss + put_loss

        Max pain = S that minimizes total_loss.

        Returns
        -------
        dict with keys: max_pain_strike, distance_from_spot, distance_pct,
                        direction, confidence, writer_loss_curve
        """
        if not option_chain:
            return {'max_pain_strike': None, 'distance_from_spot': 0,
                    'distance_pct': 0, 'direction': 'NEUTRAL',
                    'confidence': 0, 'writer_loss_curve': []}

        strikes = [r['strike'] for r in option_chain]
        call_oi = {r['strike']: r['call_oi'] for r in option_chain}
        put_oi = {r['strike']: r['put_oi'] for r in option_chain}

        min_loss = float('inf')
        max_pain_strike = strikes[0]
        loss_curve = []

        for s in strikes:
            total_loss = 0.0
            # Call writer losses: for all strikes K < S, call buyers gain (S - K)
            for k in strikes:
                if k < s and call_oi.get(k, 0) > 0:
                    total_loss += (s - k) * call_oi[k]
            # Put writer losses: for all strikes K > S, put buyers gain (K - S)
            for k in strikes:
                if k > s and put_oi.get(k, 0) > 0:
                    total_loss += (k - s) * put_oi[k]

            loss_curve.append({'strike': s, 'total_loss': total_loss})

            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = s

        distance = spot_price - max_pain_strike
        distance_pct = abs(distance) / spot_price if spot_price > 0 else 0

        if abs(distance) < NIFTY_STRIKE_INTERVAL * 0.5:
            direction = 'NEUTRAL'
        elif distance > 0:
            direction = 'BEARISH'    # spot above max pain → gravitational pull down
        else:
            direction = 'BULLISH'    # spot below max pain → gravitational pull up

        # Confidence: higher when loss curve has a sharp valley
        confidence = self._curve_confidence(loss_curve, max_pain_strike)

        return {
            'max_pain_strike': max_pain_strike,
            'distance_from_spot': round(distance, 2),
            'distance_pct': round(distance_pct, 5),
            'direction': direction,
            'confidence': round(confidence, 3),
            'writer_loss_curve': loss_curve,
        }

    @staticmethod
    def _curve_confidence(
        loss_curve: List[Dict], max_pain_strike: float
    ) -> float:
        """
        Confidence based on how peaked the loss valley is.
        Sharp valley → high confidence that spot will pin.
        """
        if len(loss_curve) < 3:
            return 0.3

        losses = np.array([x['total_loss'] for x in loss_curve])
        min_loss = losses.min()
        max_loss = losses.max()

        if max_loss == 0 or min_loss == max_loss:
            return 0.3

        # Normalized depth: how deep is the valley relative to the range
        depth_ratio = (max_loss - min_loss) / max_loss

        # Sharpness: how quickly does loss rise near the min?
        min_idx = int(np.argmin(losses))
        neighbors = []
        for offset in [-2, -1, 1, 2]:
            idx = min_idx + offset
            if 0 <= idx < len(losses):
                neighbors.append(losses[idx])
        if neighbors:
            avg_neighbor = np.mean(neighbors)
            sharpness = (avg_neighbor - min_loss) / (max_loss - min_loss + 1e-10)
        else:
            sharpness = 0.0

        confidence = 0.3 + 0.35 * depth_ratio + 0.35 * min(sharpness, 1.0)
        return min(confidence, MAX_CONFIDENCE)

    # ----------------------------------------------------------
    # OI Buildup analysis
    # ----------------------------------------------------------
    def get_oi_buildup(
        self, option_chain: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identify key OI-based support/resistance levels.

        Returns
        -------
        dict with keys:
          highest_put_oi_strike (support), highest_call_oi_strike (resistance),
          pcr_atm, oi_skew (BULLISH / BEARISH / NEUTRAL),
          top_call_strikes, top_put_strikes
        """
        if not option_chain:
            return {
                'highest_put_oi_strike': None,
                'highest_call_oi_strike': None,
                'pcr_atm': 0.0,
                'oi_skew': 'NEUTRAL',
                'top_call_strikes': [],
                'top_put_strikes': [],
            }

        # Overall highest OI strikes
        max_call_oi = 0
        max_put_oi = 0
        highest_call_strike = option_chain[0]['strike']
        highest_put_strike = option_chain[0]['strike']

        total_call_oi = 0
        total_put_oi = 0

        for r in option_chain:
            if r['call_oi'] > max_call_oi:
                max_call_oi = r['call_oi']
                highest_call_strike = r['strike']
            if r['put_oi'] > max_put_oi:
                max_put_oi = r['put_oi']
                highest_put_strike = r['strike']
            total_call_oi += r['call_oi']
            total_put_oi += r['put_oi']

        # PCR around ATM: use middle N strikes
        n = len(option_chain)
        mid = n // 2
        lo = max(0, mid - ATM_RANGE_STRIKES)
        hi = min(n, mid + ATM_RANGE_STRIKES + 1)
        atm_call_oi = sum(r['call_oi'] for r in option_chain[lo:hi])
        atm_put_oi = sum(r['put_oi'] for r in option_chain[lo:hi])
        pcr_atm = atm_put_oi / atm_call_oi if atm_call_oi > 0 else 0.0

        # OI skew
        overall_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0
        if overall_pcr > 1.3:
            oi_skew = 'BULLISH'    # heavy put writing = support = bullish
        elif overall_pcr < 0.7:
            oi_skew = 'BEARISH'    # heavy call writing = resistance = bearish
        else:
            oi_skew = 'NEUTRAL'

        # Top 3 strikes by OI
        sorted_by_call = sorted(option_chain, key=lambda x: x['call_oi'], reverse=True)
        sorted_by_put = sorted(option_chain, key=lambda x: x['put_oi'], reverse=True)
        top_call = [{'strike': r['strike'], 'oi': r['call_oi']} for r in sorted_by_call[:3]]
        top_put = [{'strike': r['strike'], 'oi': r['put_oi']} for r in sorted_by_put[:3]]

        return {
            'highest_put_oi_strike': highest_put_strike,
            'highest_call_oi_strike': highest_call_strike,
            'pcr_atm': round(pcr_atm, 3),
            'oi_skew': oi_skew,
            'top_call_strikes': top_call,
            'top_put_strikes': top_put,
        }

    # ----------------------------------------------------------
    # Pin probability
    # ----------------------------------------------------------
    def get_pin_probability(
        self,
        spot: float,
        max_pain: float,
        dte: int,
    ) -> float:
        """
        Probability that spot pins to max pain by expiry.

        Higher when:
        - DTE <= 2 (gamma intensifies)
        - Distance < 1% (within gravitational range)
        - OI concentrated (handled via confidence in caller)

        Returns float in [0, 1].
        """
        if spot <= 0 or max_pain is None:
            return 0.0

        distance_pct = abs(spot - max_pain) / spot

        # DTE decay: strongest on expiry day, weakens sharply beyond 2d
        if dte == 0:
            dte_factor = 1.0
        elif dte == 1:
            dte_factor = 0.80
        elif dte == 2:
            dte_factor = 0.55
        else:
            dte_factor = max(0.1, 0.55 * math.exp(-0.5 * (dte - 2)))

        # Distance decay: exponential fall-off beyond 1%
        if distance_pct <= 0.005:
            dist_factor = 1.0
        elif distance_pct <= 0.01:
            dist_factor = 1.0 - (distance_pct - 0.005) / 0.005 * 0.3  # 1.0 → 0.7
        elif distance_pct <= 0.02:
            dist_factor = 0.7 - (distance_pct - 0.01) / 0.01 * 0.4    # 0.7 → 0.3
        else:
            dist_factor = max(0.05, 0.3 * math.exp(-10 * (distance_pct - 0.02)))

        probability = PIN_BASE_PROBABILITY * dte_factor * dist_factor
        return round(min(probability, 0.85), 3)


# ================================================================
# StrikeClusteringSignal
# ================================================================

class StrikeClusteringSignal:
    """
    Gravitational fade signal: when spot is within striking distance of
    max pain near expiry, fade away from spot toward max pain.
    """

    SIGNAL_ID = 'STRIKE_CLUSTERING_GRAV'

    def __init__(self, max_pain_calc: MaxPainCalculator):
        self.calc = max_pain_calc

    def check(
        self,
        spot: float,
        today: date,
        vix: float,
        option_chain: Optional[List[Dict[str, Any]]] = None,
        is_event_day: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Check whether a strike-clustering gravitational signal fires.

        Parameters
        ----------
        spot : current Nifty spot price
        today : date for evaluation
        vix : India VIX level
        option_chain : pre-fetched chain (optional; fetched if None)
        is_event_day : True on RBI policy, budget, election result days

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        # ------ Fetch chain if not provided ------
        expiry = MaxPainCalculator.next_weekly_expiry(today)
        days_to_exp = MaxPainCalculator.dte(today, expiry)

        if option_chain is None:
            option_chain = self.calc.get_option_chain(
                instrument='NIFTY', expiry=expiry, as_of=today
            )
        if not option_chain:
            logger.debug("No option chain available for %s", today)
            return None

        # ------ Compute max pain ------
        mp = self.calc.calculate_max_pain(option_chain, spot)
        if mp['max_pain_strike'] is None:
            return None

        max_pain_strike = mp['max_pain_strike']
        distance_pct = mp['distance_pct']
        confidence = mp['confidence']
        direction = mp['direction']

        # ------ OI context ------
        oi_info = self.calc.get_oi_buildup(option_chain)

        # ------ Pin probability ------
        pin_prob = self.calc.get_pin_probability(spot, max_pain_strike, days_to_exp)

        # ------ Gate conditions ------
        if days_to_exp > ENTRY_MAX_DTE:
            logger.debug("DTE %d > %d, skip", days_to_exp, ENTRY_MAX_DTE)
            return None

        if distance_pct < ENTRY_DIST_MIN_PCT:
            logger.debug("Distance %.3f%% < min %.3f%%, too close",
                         distance_pct * 100, ENTRY_DIST_MIN_PCT * 100)
            return None

        if distance_pct > ENTRY_DIST_MAX_PCT:
            logger.debug("Distance %.3f%% > max %.3f%%, too far",
                         distance_pct * 100, ENTRY_DIST_MAX_PCT * 100)
            return None

        if confidence < ENTRY_MIN_CONFIDENCE:
            logger.debug("Confidence %.3f < %.3f, skip", confidence, ENTRY_MIN_CONFIDENCE)
            return None

        if vix > VIX_THRESHOLD:
            logger.debug("VIX %.1f > %.1f, pinning weakens in high vol", vix, VIX_THRESHOLD)
            return None

        if is_event_day:
            logger.debug("Event day — suppressing signal")
            return None

        # ------ Direction: fade toward max pain ------
        if direction == 'NEUTRAL':
            logger.debug("Spot ≈ max pain, no directional edge")
            return None

        # If spot > max pain → fade BEARISH (sell / short)
        # If spot < max pain → fade BULLISH (buy / long)
        fade_direction = 'BEARISH' if spot > max_pain_strike else 'BULLISH'

        # ------ SL / Target ------
        if fade_direction == 'BEARISH':
            sl_price = spot * (1 + SL_BUFFER_PCT)
            sl_pct = SL_BUFFER_PCT
        else:
            sl_price = spot * (1 - SL_BUFFER_PCT)
            sl_pct = SL_BUFFER_PCT

        tgt_price = max_pain_strike

        # ------ Size multiplier ------
        # Closer to max pain + higher confidence → bigger size
        if distance_pct <= 0.006:
            size_mult = 1.3
        elif distance_pct <= 0.010:
            size_mult = 1.0
        else:
            size_mult = 0.7

        # Boost for high pin probability
        if pin_prob >= 0.25:
            size_mult = min(size_mult * 1.1, 1.3)

        # ------ Build reason string ------
        reason_parts = [
            f"MaxPain={max_pain_strike:.0f}",
            f"dist={distance_pct * 100:.2f}%",
            f"DTE={days_to_exp}",
            f"conf={confidence:.2f}",
            f"pin={pin_prob:.2f}",
            f"PCR={oi_info['pcr_atm']:.2f}",
            f"skew={oi_info['oi_skew']}",
            f"VIX={vix:.1f}",
        ]
        if oi_info['highest_put_oi_strike']:
            reason_parts.append(f"PutWall={oi_info['highest_put_oi_strike']:.0f}")
        if oi_info['highest_call_oi_strike']:
            reason_parts.append(f"CallWall={oi_info['highest_call_oi_strike']:.0f}")

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': fade_direction,
            'price': round(spot, 2),
            'sl_pct': round(sl_pct, 4),
            'tgt_price': round(tgt_price, 2),
            'size_mult': round(size_mult, 2),
            'reason': ' | '.join(reason_parts),
            # Extra context for downstream consumers
            'max_pain_strike': max_pain_strike,
            'dte': days_to_exp,
            'pin_probability': pin_prob,
            'confidence': confidence,
            'oi_buildup': oi_info,
            'expiry': expiry,
            'expiry_target_time': EXPIRY_TARGET_TIME.strftime('%H:%M'),
        }


# ================================================================
# Self-test
# ================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s',
    )

    # ---- Connect to DB ----
    try:
        conn = psycopg2.connect(DATABASE_DSN)
        logger.info("Connected to %s", DATABASE_DSN)
    except Exception as e:
        logger.error("DB connection failed: %s", e)
        sys.exit(1)

    calc = MaxPainCalculator(db_conn=conn)

    # ---- Pick a sample date with data ----
    # Try a recent Thursday (expiry day) with DTE=0
    sample_date = date(2026, 3, 19)  # Thursday
    sample_expiry = date(2026, 3, 19)
    sample_spot = 23200.0  # approximate; adjust after reading data

    # Attempt to read spot from nifty_daily if available
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT close FROM nifty_daily WHERE date = %s",
                (sample_date,),
            )
            row = cur.fetchone()
            if row:
                sample_spot = float(row[0])
                logger.info("Spot from nifty_daily on %s: %.2f", sample_date, sample_spot)
            else:
                # Fallback: pick most recent date with option data
                cur.execute(
                    "SELECT DISTINCT date FROM nifty_options ORDER BY date DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    sample_date = row[0]
                    # Find the expiry for that date
                    cur.execute(
                        "SELECT DISTINCT expiry FROM nifty_options "
                        "WHERE date = %s ORDER BY expiry LIMIT 1",
                        (sample_date,),
                    )
                    exp_row = cur.fetchone()
                    if exp_row:
                        sample_expiry = exp_row[0]
                    # Read spot
                    cur.execute(
                        "SELECT close FROM nifty_daily WHERE date <= %s "
                        "ORDER BY date DESC LIMIT 1",
                        (sample_date,),
                    )
                    spot_row = cur.fetchone()
                    if spot_row:
                        sample_spot = float(spot_row[0])
                    logger.info("Using fallback date=%s expiry=%s spot=%.2f",
                                sample_date, sample_expiry, sample_spot)
    except Exception as e:
        logger.warning("Could not read spot from DB, using default: %s", e)

    # ---- Fetch chain ----
    chain = calc.get_option_chain(
        instrument='NIFTY', expiry=sample_expiry, as_of=sample_date
    )
    if not chain:
        logger.error("No option chain data found. Check nifty_options table.")
        conn.close()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Max Pain Analysis — {sample_date} (expiry {sample_expiry})")
    print(f"  Spot: {sample_spot:.2f}  |  Strikes in chain: {len(chain)}")
    print(f"{'='*60}\n")

    # ---- Max pain ----
    mp = calc.calculate_max_pain(chain, sample_spot)
    print(f"  Max Pain Strike : {mp['max_pain_strike']}")
    print(f"  Distance        : {mp['distance_from_spot']:+.2f} ({mp['distance_pct']*100:.2f}%)")
    print(f"  Direction       : {mp['direction']}")
    print(f"  Confidence      : {mp['confidence']:.3f}")

    # ---- OI buildup ----
    oi = calc.get_oi_buildup(chain)
    print(f"\n  OI Analysis:")
    print(f"    Put Wall  (support)    : {oi['highest_put_oi_strike']}")
    print(f"    Call Wall (resistance) : {oi['highest_call_oi_strike']}")
    print(f"    PCR (ATM)              : {oi['pcr_atm']:.3f}")
    print(f"    OI Skew                : {oi['oi_skew']}")
    print(f"    Top Call OI : {oi['top_call_strikes']}")
    print(f"    Top Put OI  : {oi['top_put_strikes']}")

    # ---- Pin probability ----
    dte_val = MaxPainCalculator.dte(sample_date, sample_expiry)
    pin_p = calc.get_pin_probability(sample_spot, mp['max_pain_strike'], dte_val)
    print(f"\n  Pin Probability : {pin_p:.3f}  (DTE={dte_val})")

    # ---- Signal check ----
    sig = StrikeClusteringSignal(calc)
    signal = sig.check(
        spot=sample_spot, today=sample_date, vix=15.0,
        option_chain=chain, is_event_day=False,
    )
    print(f"\n  Signal:")
    if signal:
        for k, v in signal.items():
            if k == 'oi_buildup':
                continue  # already printed
            print(f"    {k:22s} : {v}")
    else:
        print("    No signal fired (gate conditions not met)")

    print(f"\n{'='*60}\n")

    conn.close()
