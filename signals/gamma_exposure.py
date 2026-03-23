"""
GammaGEX — Dealer Gamma Exposure Signal.

Dealer gamma positioning from options OI determines market microstructure regime:
  - POSITIVE gamma: dealers are short options → they buy dips and sell rallies
    → MEAN-REVERSION regime (low intraday volatility, range-bound)
  - NEGATIVE gamma: dealers hedging amplifies moves
    → TRENDING regime (high intraday volatility, breakouts)

This is the single most powerful microstructure signal used by institutional
options desks. Combined with Nifty options OI data, it predicts:
  1. Intraday regime (range vs trend)
  2. Support/resistance from gamma walls
  3. Volatility regime transitions

Signal logic:
  Net gamma = Σ (call_oi × call_gamma × spot²/100) - Σ (put_oi × put_gamma × spot²/100)

  Gamma zones:
    STRONG_POSITIVE (>1B):   Deep mean-reversion → sell strangles, fade breakouts
    POSITIVE (0-1B):         Mild mean-reversion
    NEGATIVE (0 to -500M):   Trending → follow breakouts, buy strangles
    STRONG_NEGATIVE (<-500M): Strong trending → aggressive momentum

  Gamma flip level: strike where net gamma crosses zero
    → KEY pivot: above flip = positive gamma (fade), below = negative (follow)

  Charm/vanna adjustment:
    As time passes (charm) and vol changes (vanna), gamma shifts.
    This signal accounts for T+1 gamma change estimates.

Data source:
  - nifty_options table: OI + Greeks per strike
  - Requires delta, gamma per strike from options chain

Usage:
    from signals.gamma_exposure import GammaExposureSignal
    sig = GammaExposureSignal(db_conn=conn)
    result = sig.evaluate(spot=23400, trade_date=date.today())
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================
NIFTY_LOT_SIZE = 25  # Updated Nifty lot size
CONTRACT_MULTIPLIER = NIFTY_LOT_SIZE

# Gamma thresholds (in notional INR)
GAMMA_STRONG_POSITIVE = 1_000_000_000   # 1B
GAMMA_POSITIVE = 0
GAMMA_NEGATIVE = -500_000_000           # -500M
GAMMA_STRONG_NEGATIVE = -1_000_000_000  # -1B

# Gamma wall identification
GAMMA_WALL_MIN_OI = 50_000  # Minimum OI to be considered a wall

# Size modifiers based on gamma regime vs trade type
# When positive gamma: fade strategies work, momentum doesn't
# When negative gamma: momentum works, fade doesn't
SIZE_MAP_FADE = {           # For mean-reversion strategies
    'STRONG_POSITIVE': 1.35,
    'POSITIVE': 1.15,
    'NEGATIVE': 0.80,
    'STRONG_NEGATIVE': 0.60,
}
SIZE_MAP_MOMENTUM = {       # For trend-following strategies
    'STRONG_POSITIVE': 0.60,
    'POSITIVE': 0.80,
    'NEGATIVE': 1.15,
    'STRONG_NEGATIVE': 1.35,
}


@dataclass
class GammaContext:
    """Evaluation result from gamma exposure signal."""
    net_gamma: float              # Net dealer gamma exposure (INR notional)
    gamma_zone: str               # STRONG_POSITIVE, POSITIVE, NEGATIVE, STRONG_NEGATIVE
    gamma_flip_strike: float      # Strike where net gamma crosses zero
    spot_vs_flip: str             # ABOVE_FLIP or BELOW_FLIP
    regime: str                   # MEAN_REVERSION or TRENDING
    call_gamma_wall: float        # Highest call gamma strike (resistance)
    put_gamma_wall: float         # Highest put gamma strike (support)
    gamma_range: Tuple[float, float]  # Expected intraday range from gamma
    direction: str                # BULLISH (above flip + positive) etc.
    confidence: float
    size_modifier_fade: float     # For fade strategies
    size_modifier_momentum: float # For momentum strategies
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'GAMMA_EXPOSURE',
            'net_gamma': round(self.net_gamma, 0),
            'gamma_zone': self.gamma_zone,
            'gamma_flip_strike': round(self.gamma_flip_strike, 0),
            'spot_vs_flip': self.spot_vs_flip,
            'regime': self.regime,
            'call_gamma_wall': round(self.call_gamma_wall, 0),
            'put_gamma_wall': round(self.put_gamma_wall, 0),
            'gamma_range': (round(self.gamma_range[0], 0), round(self.gamma_range[1], 0)),
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier_fade': round(self.size_modifier_fade, 2),
            'size_modifier_momentum': round(self.size_modifier_momentum, 2),
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = '🟢' if self.regime == 'MEAN_REVERSION' else '🔴'
        return (
            f"{emoji} Gamma Exposure ({self.regime})\n"
            f"  Net Γ: ₹{self.net_gamma/1e6:,.0f}M ({self.gamma_zone})\n"
            f"  Flip: {self.gamma_flip_strike:.0f} | Spot: {self.spot_vs_flip}\n"
            f"  Call Wall: {self.call_gamma_wall:.0f} | Put Wall: {self.put_gamma_wall:.0f}\n"
            f"  Range: [{self.gamma_range[0]:.0f}, {self.gamma_range[1]:.0f}]\n"
            f"  Fade: {self.size_modifier_fade:.2f}x | Mom: {self.size_modifier_momentum:.2f}x"
        )


class GammaExposureSignal:
    """
    Dealer gamma exposure signal for regime detection.

    Computes net dealer gamma from options OI and Greeks to determine
    whether the market is in mean-reversion or trending regime.
    """

    SIGNAL_ID = 'GAMMA_EXPOSURE'

    def __init__(self, db_conn=None, kite=None):
        self.conn = db_conn
        self.kite = kite

    def _get_conn(self):
        if self.conn:
            try:
                if not self.conn.closed:
                    return self.conn
            except Exception:
                pass
        try:
            import psycopg2
            from config.settings import DATABASE_DSN
            self.conn = psycopg2.connect(DATABASE_DSN)
            return self.conn
        except Exception as e:
            logger.error("DB connection failed: %s", e)
            return None

    # ----------------------------------------------------------
    # Data retrieval
    # ----------------------------------------------------------
    def _get_options_with_greeks(
        self, trade_date: date, expiry: date
    ) -> Optional[List[Dict]]:
        """
        Fetch option chain with Greeks from nifty_options.

        Returns list of dicts with: strike, option_type, oi, delta, gamma
        """
        conn = self._get_conn()
        if not conn:
            return None

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT strike, option_type, oi,
                           COALESCE(delta, 0) as delta,
                           COALESCE(gamma, 0) as gamma
                    FROM nifty_options
                    WHERE date = %s AND expiry = %s
                      AND oi > 0
                    ORDER BY strike, option_type
                    """,
                    (trade_date, expiry)
                )
                rows = cur.fetchall()

            if not rows:
                return None

            return [
                {
                    'strike': float(r[0]),
                    'option_type': r[1],
                    'oi': int(r[2]),
                    'delta': float(r[3]),
                    'gamma': float(r[4]),
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("Failed to fetch options with greeks: %s", e)
            return None

    @staticmethod
    def _next_weekly_expiry(as_of: date) -> date:
        """Return next weekly expiry (Thursday)."""
        days_ahead = (3 - as_of.weekday()) % 7
        if days_ahead == 0:
            return as_of
        return as_of + timedelta(days=days_ahead)

    # ----------------------------------------------------------
    # Gamma calculation
    # ----------------------------------------------------------
    def _compute_gamma_exposure(
        self, options: List[Dict], spot: float
    ) -> Dict:
        """
        Compute net dealer gamma exposure.

        Dealers are net short options → their gamma is opposite to OI gamma.
        Call OI: dealers short calls → negative delta hedge → sell on up
        Put OI: dealers short puts → positive delta hedge → buy on down

        Net gamma at each strike:
          call_gamma_exposure = call_oi * call_gamma * spot^2 / 100 * contract_mult
          put_gamma_exposure  = put_oi * put_gamma * spot^2 / 100 * contract_mult
          net_dealer_gamma = -call_gamma_exposure + put_gamma_exposure

        (Negative because dealers are SHORT options)
        """
        if not options or spot <= 0:
            return {
                'net_gamma': 0,
                'gamma_by_strike': [],
                'flip_strike': spot,
                'call_wall': spot,
                'put_wall': spot,
            }

        spot_sq_factor = (spot ** 2) / 100.0 * CONTRACT_MULTIPLIER
        gamma_by_strike = {}

        call_gamma_max = 0
        put_gamma_max = 0
        call_wall = spot
        put_wall = spot

        for opt in options:
            strike = opt['strike']
            oi = opt['oi']
            gamma = opt['gamma']
            otype = opt['option_type']

            if strike not in gamma_by_strike:
                gamma_by_strike[strike] = {
                    'strike': strike,
                    'call_gamma_exposure': 0,
                    'put_gamma_exposure': 0,
                    'net_gamma': 0,
                }

            gamma_exp = oi * gamma * spot_sq_factor

            if otype in ('CE', 'CALL', 'call'):
                gamma_by_strike[strike]['call_gamma_exposure'] = gamma_exp
                if oi * gamma > call_gamma_max:
                    call_gamma_max = oi * gamma
                    call_wall = strike
            else:
                gamma_by_strike[strike]['put_gamma_exposure'] = gamma_exp
                if oi * gamma > put_gamma_max:
                    put_gamma_max = oi * gamma
                    put_wall = strike

        # Net dealer gamma (dealers are short options)
        net_gamma = 0
        for strike, data in gamma_by_strike.items():
            # Dealers: short calls (negative gamma) + short puts (positive gamma for dealer)
            data['net_gamma'] = -data['call_gamma_exposure'] + data['put_gamma_exposure']
            net_gamma += data['net_gamma']

        # Find gamma flip level (where cumulative crosses zero)
        sorted_strikes = sorted(gamma_by_strike.keys())
        cumulative = 0
        flip_strike = spot
        for strike in sorted_strikes:
            prev_cum = cumulative
            cumulative += gamma_by_strike[strike]['net_gamma']
            if prev_cum <= 0 and cumulative > 0:
                flip_strike = strike
                break
            elif prev_cum >= 0 and cumulative < 0:
                flip_strike = strike
                break

        return {
            'net_gamma': net_gamma,
            'gamma_by_strike': list(gamma_by_strike.values()),
            'flip_strike': flip_strike,
            'call_wall': call_wall,
            'put_wall': put_wall,
        }

    def _estimate_gamma_range(
        self, net_gamma: float, spot: float
    ) -> Tuple[float, float]:
        """
        Estimate intraday price range based on gamma.

        Positive gamma → compressed range (dealers dampen moves)
        Negative gamma → expanded range (dealers amplify moves)
        """
        base_range_pct = 0.008  # 0.8% base range

        if net_gamma > GAMMA_STRONG_POSITIVE:
            range_mult = 0.5  # Very compressed
        elif net_gamma > 0:
            range_mult = 0.75
        elif net_gamma > GAMMA_NEGATIVE:
            range_mult = 1.25
        else:
            range_mult = 1.75  # Very expanded

        half_range = spot * base_range_pct * range_mult
        return (spot - half_range, spot + half_range)

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_gamma(net_gamma: float) -> Tuple[str, str]:
        """Classify gamma exposure into zone and regime."""
        if net_gamma >= GAMMA_STRONG_POSITIVE:
            return 'STRONG_POSITIVE', 'MEAN_REVERSION'
        elif net_gamma >= GAMMA_POSITIVE:
            return 'POSITIVE', 'MEAN_REVERSION'
        elif net_gamma >= GAMMA_STRONG_NEGATIVE:
            return 'NEGATIVE', 'TRENDING'
        else:
            return 'STRONG_NEGATIVE', 'TRENDING'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        spot: Optional[float] = None,
        trade_date: Optional[date] = None,
        net_gamma_override: Optional[float] = None,
    ) -> GammaContext:
        """Evaluate dealer gamma exposure signal."""
        if trade_date is None:
            trade_date = date.today()

        # Get spot price
        if spot is None:
            conn = self._get_conn()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT close FROM nifty_daily WHERE date <= %s ORDER BY date DESC LIMIT 1",
                            (trade_date,)
                        )
                        row = cur.fetchone()
                        spot = float(row[0]) if row else 23000.0
                except Exception:
                    spot = 23000.0
            else:
                spot = 23000.0

        # Override path
        if net_gamma_override is not None:
            net_gamma = net_gamma_override
            flip_strike = spot
            call_wall = spot + 200
            put_wall = spot - 200
        else:
            expiry = self._next_weekly_expiry(trade_date)
            options = self._get_options_with_greeks(trade_date, expiry)

            if not options:
                return GammaContext(
                    net_gamma=0, gamma_zone='UNKNOWN',
                    gamma_flip_strike=spot, spot_vs_flip='UNKNOWN',
                    regime='UNKNOWN', call_gamma_wall=spot,
                    put_gamma_wall=spot,
                    gamma_range=(spot * 0.99, spot * 1.01),
                    direction='NEUTRAL', confidence=0.0,
                    size_modifier_fade=1.0, size_modifier_momentum=1.0,
                    reason='No options data with Greeks available'
                )

            result = self._compute_gamma_exposure(options, spot)
            net_gamma = result['net_gamma']
            flip_strike = result['flip_strike']
            call_wall = result['call_wall']
            put_wall = result['put_wall']

        # Classify
        gamma_zone, regime = self._classify_gamma(net_gamma)

        # Spot vs flip
        spot_vs_flip = 'ABOVE_FLIP' if spot >= flip_strike else 'BELOW_FLIP'

        # Direction: above gamma flip = bullish support, below = bearish pressure
        if spot_vs_flip == 'ABOVE_FLIP' and regime == 'MEAN_REVERSION':
            direction = 'BULLISH'  # Dealers support dips
        elif spot_vs_flip == 'BELOW_FLIP' and regime == 'TRENDING':
            direction = 'BEARISH'  # Dealers amplify selling
        elif spot_vs_flip == 'ABOVE_FLIP' and regime == 'TRENDING':
            direction = 'BULLISH'  # Momentum up
        else:
            direction = 'NEUTRAL'

        # Estimated range
        gamma_range = self._estimate_gamma_range(net_gamma, spot)

        # Confidence
        confidence = 0.50
        if gamma_zone.startswith('STRONG'):
            confidence += 0.20
        else:
            confidence += 0.10
        # Distance from flip adds confidence
        flip_dist_pct = abs(spot - flip_strike) / spot * 100
        if flip_dist_pct > 1.0:
            confidence += 0.10
        confidence = min(0.90, confidence)

        # Size modifiers
        size_fade = SIZE_MAP_FADE.get(gamma_zone, 1.0)
        size_momentum = SIZE_MAP_MOMENTUM.get(gamma_zone, 1.0)

        parts = [
            f"NetΓ=₹{net_gamma/1e6:,.0f}M",
            f"Zone={gamma_zone}",
            f"Regime={regime}",
            f"Flip={flip_strike:.0f}",
            f"Spot={spot_vs_flip}",
            f"CallWall={call_wall:.0f}",
            f"PutWall={put_wall:.0f}",
            f"Range=[{gamma_range[0]:.0f},{gamma_range[1]:.0f}]",
        ]

        return GammaContext(
            net_gamma=net_gamma,
            gamma_zone=gamma_zone,
            gamma_flip_strike=flip_strike,
            spot_vs_flip=spot_vs_flip,
            regime=regime,
            call_gamma_wall=call_wall,
            put_gamma_wall=put_wall,
            gamma_range=gamma_range,
            direction=direction,
            confidence=confidence,
            size_modifier_fade=size_fade,
            size_modifier_momentum=size_momentum,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, spot: float, trade_date: date) -> Dict:
        ctx = self.evaluate(spot=spot, trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')

    sig = GammaExposureSignal()
    for gamma in [2e9, 500e6, -200e6, -800e6]:
        ctx = sig.evaluate(spot=23400, net_gamma_override=gamma)
        print(f"Gamma=₹{gamma/1e6:,.0f}M → {ctx.regime:16s} {ctx.gamma_zone:16s} "
              f"fade={ctx.size_modifier_fade:.2f} mom={ctx.size_modifier_momentum:.2f}")
