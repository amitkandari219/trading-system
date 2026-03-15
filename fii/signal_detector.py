"""
FII signal detector — detects 4 FII positioning patterns from NSE OI data.
Runs at 7:30 PM after downloading day's OI data.
Produces signals valid for NEXT trading day.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FIISignalResult:
    date: object
    signal_id: Optional[str]    # None if no signal
    direction: Optional[str]    # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    confidence: float           # 0.0 to 1.0
    pattern_name: str
    raw_metrics: dict
    valid_for_date: object      # Trade on THIS date (next day)
    notes: str


class FIISignalDetector:
    """
    Detects 4 FII positioning patterns.
    Runs at 7:30 PM after downloading day's OI data.
    Produces signals valid for NEXT trading day.
    """

    # ================================================================
    # PATTERN THRESHOLDS
    # ================================================================

    # Pattern 1: Bearish — FII short futures + long puts
    P1_MIN_FUTURE_NET_SHORT = -50_000
    P1_MIN_PUT_LONG_RATIO   = 1.5
    P1_MIN_FUTURE_PERCENTILE = 75

    # Pattern 2: Bullish — FII long futures + covering puts
    P2_MIN_FUTURE_NET_LONG  = 30_000
    P2_MAX_PUT_LONG_RATIO   = 0.8
    P2_MIN_FUTURE_PERCENTILE = 70

    # Pattern 3: Large futures shift (continuation signal)
    P3_CRORE_THRESHOLD      = 3000
    P3_LOT_SIZE             = 75

    @staticmethod
    def compute_p3_threshold(nifty_level: float,
                              crore_threshold: float = 3000) -> int:
        """
        Dynamic Pattern 3 threshold based on current Nifty level.
        """
        contract_value_rupees = nifty_level * 75
        rupee_threshold       = crore_threshold * 1e7
        return int(rupee_threshold / contract_value_rupees)

    # Pattern 4: Pure put buying (hedging — IGNORE, not a signal)
    P4_MIN_PUT_LONG_CONTRACTS = 200_000
    P4_MAX_FUTURE_CHANGE    = 10_000

    MIN_HISTORY_DAYS = 60

    def __init__(self, db_connection):
        self.db = db_connection

    def detect(self, today_data: pd.DataFrame,
               for_date: object) -> FIISignalResult:
        """
        Runs all 4 pattern checks in order.
        Returns first matching pattern, or NEUTRAL if none.
        """
        metrics = self._extract_metrics(today_data)

        history = self._load_history(days=252)

        history_with_today = pd.concat([
            history,
            pd.DataFrame([metrics])
        ], ignore_index=True)

        if len(history_with_today) < self.MIN_HISTORY_DAYS:
            return FIISignalResult(
                date=metrics['date'],
                signal_id=None,
                direction='NEUTRAL',
                confidence=0.0,
                pattern_name='INSUFFICIENT_HISTORY',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=f"Only {len(history_with_today)} days of history. "
                      f"Need {self.MIN_HISTORY_DAYS}."
            )

        # Check Pattern 4 first (hedging — suppress other signals)
        if self._is_pattern_4_hedging(metrics, history_with_today):
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_004',
                direction='NEUTRAL',
                confidence=0.9,
                pattern_name='FII_PURE_HEDGE',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes='FII buying puts without shorting futures. '
                      'Hedging behavior. Do not trade directionally.'
            )

        # Pattern 1: Bearish setup
        p1_result = self._check_pattern_1_bearish(
            metrics, history_with_today
        )
        if p1_result['detected']:
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_001',
                direction='BEARISH',
                confidence=p1_result['confidence'],
                pattern_name='FII_SHORT_FUTURES_LONG_PUTS',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=p1_result['notes']
            )

        # Pattern 2: Bullish setup
        p2_result = self._check_pattern_2_bullish(
            metrics, history_with_today
        )
        if p2_result['detected']:
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_002',
                direction='BULLISH',
                confidence=p2_result['confidence'],
                pattern_name='FII_LONG_FUTURES_COVERING_PUTS',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=p2_result['notes']
            )

        # Pattern 3: Large shift continuation
        p3_result = self._check_pattern_3_shift(
            metrics, history_with_today,
            current_nifty=metrics.get('nifty_close', 24000.0)
        )
        if p3_result['detected']:
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_003',
                direction=p3_result['direction'],
                confidence=p3_result['confidence'],
                pattern_name='FII_LARGE_FUTURES_SHIFT',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=p3_result['notes']
            )

        # No pattern detected
        return FIISignalResult(
            date=metrics['date'],
            signal_id=None,
            direction='NEUTRAL',
            confidence=0.0,
            pattern_name='NO_PATTERN',
            raw_metrics=metrics,
            valid_for_date=for_date,
            notes='No FII positioning pattern detected today.'
        )

    def _extract_metrics(self, fii_row: pd.DataFrame) -> dict:
        """Extract and compute derived metrics from raw OI row."""
        row = fii_row.iloc[0]
        fut_long  = float(row.get('future_long_contracts', 0))
        fut_short = float(row.get('future_short_contracts', 0))
        put_long  = float(row.get('put_long_contracts', 0))
        put_short = float(row.get('put_short_contracts', 0))
        call_long = float(row.get('call_long_contracts', 0))
        call_short= float(row.get('call_short_contracts', 0))

        return {
            'date':             row.get('date'),
            'nifty_close':      float(row.get('nifty_close', 24000.0)),
            'fut_long':         fut_long,
            'fut_short':        fut_short,
            'fut_net':          fut_long - fut_short,
            'put_long':         put_long,
            'put_short':        put_short,
            'put_net':          put_long - put_short,
            'put_ratio':        put_long / put_short
                                if put_short > 0 else 999,
            'call_long':        call_long,
            'call_short':       call_short,
            'call_net':         call_long - call_short,
            'pcr':              (put_long + put_short) /
                                (call_long + call_short)
                                if (call_long + call_short) > 0 else 1,
        }

    def _check_pattern_1_bearish(self, m: dict,
                                  history: pd.DataFrame) -> dict:
        """
        Pattern NSE_001: FII net short futures AND buying puts.
        Historical accuracy: 68% over 1-3 trading days.
        """
        if m['fut_net'] >= self.P1_MIN_FUTURE_NET_SHORT:
            return {'detected': False}

        if m['put_ratio'] < self.P1_MIN_PUT_LONG_RATIO:
            return {'detected': False}

        fut_net_pct = self._percentile_rank(
            history['fut_net'], m['fut_net']
        )
        if fut_net_pct > (100 - self.P1_MIN_FUTURE_PERCENTILE):
            return {'detected': False}

        confidence = min(0.95, 0.60 + (
            (100 - self.P1_MIN_FUTURE_PERCENTILE - fut_net_pct) / 100
        ))

        return {
            'detected': True,
            'confidence': confidence,
            'notes': (
                f"FII net futures: {m['fut_net']:,.0f} contracts "
                f"(bottom {fut_net_pct:.0f}th percentile). "
                f"Put ratio: {m['put_ratio']:.2f}. "
                f"Bearish signal with {confidence:.0%} confidence."
            )
        }

    def _check_pattern_2_bullish(self, m: dict,
                                  history: pd.DataFrame) -> dict:
        """
        Pattern NSE_002: FII net long futures AND reducing put longs.
        Historical accuracy: 65% over 1-3 trading days.
        """
        if m['fut_net'] < self.P2_MIN_FUTURE_NET_LONG:
            return {'detected': False}

        if m['put_ratio'] > self.P2_MAX_PUT_LONG_RATIO:
            return {'detected': False}

        if len(history) < 2:
            return {'detected': False}

        yesterday_put_ratio = history.iloc[-1].get('put_ratio', 999)
        if m['put_ratio'] >= yesterday_put_ratio:
            return {'detected': False}

        fut_net_pct = self._percentile_rank(
            history['fut_net'], m['fut_net']
        )
        if fut_net_pct < self.P2_MIN_FUTURE_PERCENTILE:
            return {'detected': False}

        confidence = min(0.90, 0.55 + (fut_net_pct - 70) / 200)

        return {
            'detected': True,
            'confidence': confidence,
            'notes': (
                f"FII net futures: {m['fut_net']:,.0f} contracts "
                f"(top {fut_net_pct:.0f}th percentile). "
                f"Put ratio declining: {yesterday_put_ratio:.2f} → "
                f"{m['put_ratio']:.2f}. "
                f"Bullish signal with {confidence:.0%} confidence."
            )
        }

    def _check_pattern_3_shift(self, m: dict,
                                history: pd.DataFrame,
                                current_nifty: float = 24000.0) -> dict:
        """
        Pattern NSE_003: Large single-day futures position shift.
        Threshold scales with Nifty level.
        """
        if len(history) < 2:
            return {'detected': False}

        p3_threshold = self.compute_p3_threshold(current_nifty,
                                                  self.P3_CRORE_THRESHOLD)

        yesterday_fut_net = history.iloc[-1].get('fut_net', 0)
        day_change = m['fut_net'] - yesterday_fut_net

        if abs(day_change) < p3_threshold:
            return {'detected': False}

        direction = 'BULLISH' if day_change > 0 else 'BEARISH'

        if 'fut_net_change' in history.columns:
            pct = self._percentile_rank(
                history['fut_net_change'].abs(),
                abs(day_change)
            )
        else:
            pct = 80

        confidence = min(0.85, 0.55 + pct / 400)

        return {
            'detected': True,
            'direction': direction,
            'confidence': confidence,
            'notes': (
                f"FII futures shift: {day_change:+,.0f} contracts "
                f"(threshold: {p3_threshold:,} at Nifty {current_nifty:,.0f}). "
                f"{pct:.0f}th percentile of historical shifts. "
                f"{direction} continuation expected."
            )
        }

    def _is_pattern_4_hedging(self, m: dict,
                               history: pd.DataFrame) -> bool:
        """
        Pattern NSE_004: Pure hedging — puts bought, futures unchanged.
        """
        if m['put_long'] < self.P4_MIN_PUT_LONG_CONTRACTS:
            return False

        if len(history) < 2:
            return False

        yesterday_fut_net = history.iloc[-1].get('fut_net', 0)
        fut_change = abs(m['fut_net'] - yesterday_fut_net)

        if fut_change > self.P4_MAX_FUTURE_CHANGE:
            return False

        return True

    def _percentile_rank(self, series: pd.Series,
                          value: float) -> float:
        """Returns percentile rank of value in series (0-100)."""
        series = series.dropna()
        if len(series) == 0:
            return 50.0
        return float((series <= value).mean() * 100)

    def _load_history(self, days: int) -> pd.DataFrame:
        """Load historical FII metrics from DB."""
        rows = self.db.execute(
            """
            SELECT date, fut_net, put_ratio, put_long, fut_net_change
            FROM fii_daily_metrics
            ORDER BY date DESC
            LIMIT %s
            """,
            (days,)
        ).fetchall()
        df = pd.DataFrame(rows, columns=[
            'date', 'fut_net', 'put_ratio',
            'put_long', 'fut_net_change'
        ])
        return df.sort_values('date').reset_index(drop=True)
