"""
PCR (Put-Call Ratio) Autotrender Signal for Nifty F&O.

Uses real-time and historical PCR data from NSE bhavcopy to generate
contrarian and trend-following signals based on PCR extremes and momentum.

Signal logic:
  EXTREME_HIGH PCR (> 1.35):  Contrarian BULLISH — heavy put writing = support
  HIGH PCR (1.1 - 1.35):      Mild BULLISH bias — put writers confident
  NEUTRAL PCR (0.75 - 1.1):   No signal
  LOW PCR (0.55 - 0.75):      Mild BEARISH bias — call writers confident
  EXTREME_LOW PCR (< 0.55):   Contrarian BEARISH — heavy call writing = resistance

  PCR MOMENTUM (5-day EMA crossover of PCR itself):
    Rising PCR from low → BULLISH confirmation
    Falling PCR from high → BEARISH confirmation

Data source:
  - Historical: nifty_pcr table (via pcr_loader.py)
  - Live: Computed from nifty_options table (ATM ± 10 strikes)

Integration:
  - Overlay signal: modifies sizing 0.7x–1.3x
  - Standalone contrarian entries at extremes (PCR > 1.5 or < 0.45)

Usage:
    from signals.pcr_signal import PCRAutotrender
    sig = PCRAutotrender(db_conn=conn)
    result = sig.evaluate(trade_date=date.today())
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# PCR THRESHOLDS
# ================================================================
PCR_EXTREME_HIGH = 1.35    # Contrarian bullish — too many puts
PCR_HIGH = 1.10            # Mild bullish bias
PCR_NEUTRAL_LOW = 0.75     # Below this = bearish lean
PCR_LOW = 0.55             # Contrarian bearish — too many calls
PCR_EXTREME_LOW = 0.45     # Extreme bearish — standalone entry

# Standalone contrarian thresholds (fires trades, not just overlay)
PCR_CONTRARIAN_BULL = 1.50  # Extreme pessimism → buy
PCR_CONTRARIAN_BEAR = 0.45  # Extreme optimism → sell

# Momentum: 5-day EMA of PCR
PCR_EMA_PERIOD = 5
PCR_MOMENTUM_THRESHOLD = 0.05  # Min EMA change over 3 days for momentum signal

# VIX-adjusted PCR scaling
VIX_PCR_SCALE = {
    'CALM':     {'high': 1.25, 'low': 0.65},   # VIX < 13: tighter bands
    'NORMAL':   {'high': 1.35, 'low': 0.55},   # VIX 13-18: base
    'ELEVATED': {'high': 1.50, 'low': 0.50},   # VIX 18-24: wider
    'HIGH_VOL': {'high': 1.70, 'low': 0.45},   # VIX 24-32: much wider
    'CRISIS':   {'high': 2.00, 'low': 0.40},   # VIX > 32: extreme
}

# Size modifiers
SIZE_MAP = {
    'EXTREME_BULLISH': 1.30,
    'BULLISH': 1.15,
    'NEUTRAL': 1.00,
    'BEARISH': 0.85,
    'EXTREME_BEARISH': 0.70,
}

# Expiry proximity boost
EXPIRY_PROXIMITY_DAYS = 2    # Within 2 DTE, PCR signal strengthens
EXPIRY_BOOST = 1.15          # 15% boost near expiry


@dataclass
class PCRContext:
    """Evaluation result from PCR autotrender."""
    pcr_current: float
    pcr_ema5: float
    pcr_zone: str           # EXTREME_HIGH, HIGH, NEUTRAL, LOW, EXTREME_LOW
    pcr_momentum: str       # RISING, FALLING, FLAT
    direction: str          # BULLISH, BEARISH, NEUTRAL
    confidence: float       # 0.0 - 1.0
    size_modifier: float    # 0.7 - 1.3
    is_contrarian: bool     # True if at extremes → standalone entry
    vix_regime: str
    dte: int
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'PCR_AUTOTRENDER',
            'pcr_current': round(self.pcr_current, 3),
            'pcr_ema5': round(self.pcr_ema5, 3),
            'pcr_zone': self.pcr_zone,
            'pcr_momentum': self.pcr_momentum,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'is_contrarian': self.is_contrarian,
            'vix_regime': self.vix_regime,
            'dte': self.dte,
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        ctr = ' ⚡CONTRARIAN' if self.is_contrarian else ''
        return (
            f"{emoji} PCR Signal{ctr}\n"
            f"  PCR: {self.pcr_current:.3f} (EMA5: {self.pcr_ema5:.3f})\n"
            f"  Zone: {self.pcr_zone} | Mom: {self.pcr_momentum}\n"
            f"  Dir: {self.direction} | Conf: {self.confidence:.0%}\n"
            f"  Size: {self.size_modifier:.2f}x | VIX: {self.vix_regime}\n"
            f"  {self.reason}"
        )


class PCRAutotrender:
    """
    PCR-based autotrender signal.

    Combines absolute PCR level with PCR momentum (EMA crossover)
    to generate directional bias and sizing overlay.
    """

    SIGNAL_ID = 'PCR_AUTOTRENDER'

    def __init__(self, db_conn=None):
        self.conn = db_conn

    def _get_conn(self):
        if self.conn and not getattr(self.conn, 'closed', True) == False:
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
    # VIX regime classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_vix(vix: float) -> str:
        if vix < 13:
            return 'CALM'
        elif vix < 18:
            return 'NORMAL'
        elif vix < 24:
            return 'ELEVATED'
        elif vix < 32:
            return 'HIGH_VOL'
        else:
            return 'CRISIS'

    # ----------------------------------------------------------
    # PCR data retrieval
    # ----------------------------------------------------------
    def _get_pcr_history(
        self, trade_date: date, lookback: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch PCR history from nifty_pcr or compute from nifty_options.

        Returns DataFrame with columns: [date, pcr]
        """
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback * 2)

        # Try nifty_pcr table first
        try:
            df = pd.read_sql(
                """
                SELECT date, pcr FROM nifty_pcr
                WHERE date BETWEEN %s AND %s
                ORDER BY date
                """,
                conn, params=(start_date, trade_date)
            )
            if len(df) >= 5:
                return df
        except Exception:
            pass

        # Fallback: compute from nifty_options
        try:
            df = pd.read_sql(
                """
                SELECT date,
                       SUM(CASE WHEN option_type IN ('PE','PUT','put') THEN oi ELSE 0 END) as put_oi,
                       SUM(CASE WHEN option_type IN ('CE','CALL','call') THEN oi ELSE 0 END) as call_oi
                FROM nifty_options
                WHERE date BETWEEN %s AND %s
                GROUP BY date
                ORDER BY date
                """,
                conn, params=(start_date, trade_date)
            )
            if len(df) < 5:
                return None

            df['pcr'] = df['put_oi'] / df['call_oi'].replace(0, np.nan)
            df = df.dropna(subset=['pcr'])
            return df[['date', 'pcr']].reset_index(drop=True)
        except Exception as e:
            logger.error("Failed to compute PCR from options: %s", e)
            return None

    def _get_vix(self, trade_date: date) -> float:
        """Fetch India VIX for trade_date."""
        conn = self._get_conn()
        if not conn:
            return 15.0  # default

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT close FROM india_vix
                    WHERE date <= %s ORDER BY date DESC LIMIT 1
                    """,
                    (trade_date,)
                )
                row = cur.fetchone()
                return float(row[0]) if row else 15.0
        except Exception:
            return 15.0

    def _get_dte(self, trade_date: date) -> int:
        """Days to next weekly expiry (Thursday)."""
        days_ahead = (3 - trade_date.weekday()) % 7
        if days_ahead == 0:
            return 0
        return days_ahead

    # ----------------------------------------------------------
    # PCR classification
    # ----------------------------------------------------------
    def _classify_pcr(
        self, pcr: float, vix_regime: str
    ) -> Tuple[str, str]:
        """
        Classify PCR into zone and direction, adjusted for VIX regime.

        Returns (zone, direction)
        """
        thresholds = VIX_PCR_SCALE.get(vix_regime, VIX_PCR_SCALE['NORMAL'])
        high = thresholds['high']
        low = thresholds['low']

        # Scale intermediate thresholds proportionally
        scale_factor = high / PCR_EXTREME_HIGH
        adj_high = PCR_HIGH * scale_factor
        adj_low = PCR_NEUTRAL_LOW / scale_factor

        if pcr >= high:
            return 'EXTREME_HIGH', 'BULLISH'
        elif pcr >= adj_high:
            return 'HIGH', 'BULLISH'
        elif pcr <= low:
            return 'EXTREME_LOW', 'BEARISH'
        elif pcr <= adj_low:
            return 'LOW', 'BEARISH'
        else:
            return 'NEUTRAL', 'NEUTRAL'

    def _compute_momentum(self, pcr_series: pd.Series) -> Tuple[str, float]:
        """
        PCR momentum via EMA crossover.

        Returns (momentum_label, momentum_value)
        """
        if len(pcr_series) < PCR_EMA_PERIOD + 3:
            return 'FLAT', 0.0

        ema = pcr_series.ewm(span=PCR_EMA_PERIOD, adjust=False).mean()
        recent_change = ema.iloc[-1] - ema.iloc[-4]  # 3-day EMA change

        if recent_change > PCR_MOMENTUM_THRESHOLD:
            return 'RISING', float(recent_change)
        elif recent_change < -PCR_MOMENTUM_THRESHOLD:
            return 'FALLING', float(recent_change)
        else:
            return 'FLAT', float(recent_change)

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        pcr_override: Optional[float] = None,
        vix_override: Optional[float] = None,
    ) -> PCRContext:
        """
        Evaluate PCR autotrender signal.

        Parameters
        ----------
        trade_date : evaluation date (default: today)
        pcr_override : manual PCR value for testing
        vix_override : manual VIX for testing

        Returns
        -------
        PCRContext with direction, sizing, and classification
        """
        if trade_date is None:
            trade_date = date.today()

        # Fetch VIX
        vix = vix_override if vix_override is not None else self._get_vix(trade_date)
        vix_regime = self._classify_vix(vix)
        dte = self._get_dte(trade_date)

        # Fetch PCR history
        pcr_hist = self._get_pcr_history(trade_date)

        if pcr_override is not None:
            pcr_current = pcr_override
            pcr_ema5 = pcr_override
            momentum_label, momentum_val = 'FLAT', 0.0
        elif pcr_hist is not None and len(pcr_hist) >= 5:
            pcr_current = float(pcr_hist['pcr'].iloc[-1])
            pcr_ema5 = float(
                pcr_hist['pcr'].ewm(span=PCR_EMA_PERIOD, adjust=False).mean().iloc[-1]
            )
            momentum_label, momentum_val = self._compute_momentum(pcr_hist['pcr'])
        else:
            # No data available
            return PCRContext(
                pcr_current=0.0, pcr_ema5=0.0, pcr_zone='UNKNOWN',
                pcr_momentum='FLAT', direction='NEUTRAL',
                confidence=0.0, size_modifier=1.0, is_contrarian=False,
                vix_regime=vix_regime, dte=dte,
                reason='No PCR data available'
            )

        # Classify
        zone, direction = self._classify_pcr(pcr_current, vix_regime)

        # Momentum confirmation/contradiction
        if direction == 'BULLISH' and momentum_label == 'RISING':
            # PCR high + rising → strong bullish confirmation
            confidence_boost = 0.10
        elif direction == 'BEARISH' and momentum_label == 'FALLING':
            # PCR low + falling → strong bearish confirmation
            confidence_boost = 0.10
        elif direction != 'NEUTRAL' and (
            (direction == 'BULLISH' and momentum_label == 'FALLING') or
            (direction == 'BEARISH' and momentum_label == 'RISING')
        ):
            # Momentum contradicts level → reduce confidence
            confidence_boost = -0.10
        else:
            confidence_boost = 0.0

        # Base confidence by zone
        confidence_base = {
            'EXTREME_HIGH': 0.80,
            'HIGH': 0.65,
            'NEUTRAL': 0.30,
            'LOW': 0.65,
            'EXTREME_LOW': 0.80,
            'UNKNOWN': 0.0,
        }.get(zone, 0.3)

        confidence = min(0.95, max(0.0, confidence_base + confidence_boost))

        # Size modifier
        size_key = {
            'EXTREME_HIGH': 'EXTREME_BULLISH',
            'HIGH': 'BULLISH',
            'NEUTRAL': 'NEUTRAL',
            'LOW': 'BEARISH',
            'EXTREME_LOW': 'EXTREME_BEARISH',
        }.get(zone, 'NEUTRAL')
        size_modifier = SIZE_MAP[size_key]

        # Expiry proximity boost
        if dte <= EXPIRY_PROXIMITY_DAYS and direction != 'NEUTRAL':
            size_modifier = min(1.3, size_modifier * EXPIRY_BOOST)
            confidence = min(0.95, confidence + 0.05)

        # Contrarian standalone?
        is_contrarian = (
            pcr_current >= PCR_CONTRARIAN_BULL or
            pcr_current <= PCR_CONTRARIAN_BEAR
        )

        # Build reason
        parts = [
            f"PCR={pcr_current:.3f}",
            f"EMA5={pcr_ema5:.3f}",
            f"Zone={zone}",
            f"Mom={momentum_label}({momentum_val:+.3f})",
            f"VIX={vix:.1f}({vix_regime})",
            f"DTE={dte}",
        ]
        if is_contrarian:
            parts.append("CONTRARIAN_ENTRY")

        return PCRContext(
            pcr_current=pcr_current,
            pcr_ema5=pcr_ema5,
            pcr_zone=zone,
            pcr_momentum=momentum_label,
            direction=direction,
            confidence=confidence,
            size_modifier=size_modifier,
            is_contrarian=is_contrarian,
            vix_regime=vix_regime,
            dte=dte,
            reason=' | '.join(parts),
        )

    # ----------------------------------------------------------
    # Backtest evaluation
    # ----------------------------------------------------------
    def evaluate_backtest(
        self, trade_date: date, pcr: float, vix: float
    ) -> Dict:
        """Evaluate for backtest engine — returns dict."""
        ctx = self.evaluate(
            trade_date=trade_date,
            pcr_override=pcr,
            vix_override=vix,
        )
        return ctx.to_dict()


# ================================================================
# Self-test
# ================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        conn = psycopg2.connect(DATABASE_DSN)
    except Exception:
        conn = None

    sig = PCRAutotrender(db_conn=conn)

    # Test with override values
    for pcr_val in [0.40, 0.55, 0.80, 1.10, 1.40, 1.60]:
        ctx = sig.evaluate(pcr_override=pcr_val, vix_override=16.0)
        print(f"PCR={pcr_val:.2f} → {ctx.direction:8s} zone={ctx.pcr_zone:14s} "
              f"conf={ctx.confidence:.2f} size={ctx.size_modifier:.2f} "
              f"ctr={ctx.is_contrarian}")

    if conn:
        # Test with live data
        ctx = sig.evaluate()
        print(f"\nLive: {ctx.to_telegram()}")
        conn.close()
