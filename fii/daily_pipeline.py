"""
FII daily pipeline — nightly orchestrator that runs at 7:30 PM IST.

Downloads NSE FII OI data, computes derived metrics (z-scores, percentiles,
5-day flow, FII-DII divergence), detects 6 signal patterns, stores results,
and sends Telegram evening digest.

Signal patterns detected:
  1. NSE_001: Bearish hedge (short futures + long puts)
  2. NSE_002: Bullish accumulation (long futures, covering puts)
  3. NSE_003: Large futures shift (momentum continuation)
  4. NSE_004: Pure hedging (puts bought, futures unchanged) — NEUTRAL
  5. NSE_005: FII-DII divergence (FII bearish, DII bullish or vice versa)
  6. NSE_006: Extreme positioning (contrarian, only when VIX < 20)

Signals are OVERLAY only — they modify sizing (never standalone trades).
Stale data (>1 day) falls back to 1.0x modifier with WARNING.

Usage:
    venv/bin/python3 -m fii.daily_pipeline                  # today
    venv/bin/python3 -m fii.daily_pipeline --date 2026-03-21
    venv/bin/python3 -m fii.daily_pipeline --init            # bootstrap 6 months
    venv/bin/python3 -m fii.daily_pipeline --status          # show recent signals
    venv/bin/python3 -m fii.daily_pipeline --signals         # show active signals
    venv/bin/python3 -m fii.daily_pipeline --download-only   # just download, no detect

Cron (add to crontab -e):
    30 19 * * 1-5  cd /Users/amitkandari/Desktop/trading-system && venv/bin/python3 -m fii.daily_pipeline >> logs/fii_pipeline.log 2>&1
    # Retry at 8 PM for late publishing:
    00 20 * * 1-5  cd /Users/amitkandari/Desktop/trading-system && venv/bin/python3 -m fii.daily_pipeline --retry >> logs/fii_pipeline.log 2>&1
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
import requests

from config.settings import DATABASE_DSN, REDIS_HOST, REDIS_PORT, REDIS_DB
from fii.downloader import FIIDataDownloader, DataNotAvailableError
from fii.signal_detector import FIISignalDetector, FIISignalResult

logger = logging.getLogger(__name__)


# ================================================================
# PIPELINE RESULT
# ================================================================

@dataclass
class FIIPipelineResult:
    """Result of a single pipeline run."""
    as_of_date: date
    valid_for_date: Optional[date] = None
    download_ok: bool = False
    signal_detected: bool = False
    signal: Optional[FIISignalResult] = None
    additional_signals: List[FIISignalResult] = field(default_factory=list)
    derived_metrics: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    digest_sent: bool = False

    @property
    def summary(self) -> str:
        parts = [f"FII Pipeline {self.as_of_date}:"]
        parts.append(f"  download={'OK' if self.download_ok else 'FAIL'}")
        if self.signal:
            parts.append(
                f"  signal={self.signal.pattern_name} "
                f"({self.signal.direction}, {self.signal.confidence:.0%})"
            )
        else:
            parts.append("  signal=NONE")
        if self.additional_signals:
            parts.append(
                f"  additional_signals={len(self.additional_signals)}"
            )
        if self.derived_metrics:
            dm = self.derived_metrics
            parts.append(
                f"  z_score={dm.get('z_score', 0):.2f} "
                f"pctile={dm.get('percentile', 0):.0f} "
                f"flow_5d={dm.get('flow_5d', 0):,.0f}"
            )
        if self.errors:
            parts.append(f"  errors={self.errors}")
        return '\n'.join(parts)


# ================================================================
# PIPELINE
# ================================================================

class FIIDailyPipeline:
    """
    Complete nightly pipeline:
    1. Download NSE FII OI data
    2. Compute derived: z_score, percentile, 5d flow, FII-DII divergence
    3. Detect signals (6 patterns)
    4. Store to fii_signals table
    5. Pre-load signals into Redis for morning
    6. Telegram evening digest
    """

    MAX_RETRIES = 3
    RETRY_WAIT_MINUTES = 15

    # Pattern 5: FII-DII divergence thresholds
    P5_FII_NET_THRESHOLD = -30_000    # FII net short
    P5_DII_NET_THRESHOLD = 20_000     # DII net long (opposite)

    # Pattern 6: Extreme positioning thresholds
    P6_PERCENTILE_EXTREME_LOW = 5     # bottom 5th percentile
    P6_PERCENTILE_EXTREME_HIGH = 95   # top 95th percentile
    P6_VIX_MAX = 20.0                 # only contrarian when VIX calm

    def __init__(self, db=None, redis_client=None, alerter=None,
                 truedata_token: str = None):
        """
        db:             psycopg2 connection
        redis_client:   redis.Redis instance (optional)
        alerter:        TelegramAlerter instance (optional)
        truedata_token: TrueData API token for Nifty close (optional)
        """
        self.db = db
        self.redis = redis_client
        self.alerter = alerter
        self.truedata_token = truedata_token or os.environ.get(
            'TRUEDATA_TOKEN', ''
        )
        self.downloader = FIIDataDownloader(db_conn=db)
        self.detector = FIISignalDetector(db) if db else None

    # ================================================================
    # PUBLIC: run
    # ================================================================

    def run(self, as_of_date: date = None) -> FIIPipelineResult:
        """
        Main pipeline: download, compute, detect, store, alert.

        Args:
            as_of_date: trading date to process (defaults to last trading day)

        Returns:
            FIIPipelineResult with all outputs
        """
        if as_of_date is None:
            as_of_date = self._get_last_trading_day()

        trade_date = self._get_next_trading_day(as_of_date)
        result = FIIPipelineResult(
            as_of_date=as_of_date,
            valid_for_date=trade_date,
        )

        logger.info(
            f"FII Pipeline starting: data_date={as_of_date} "
            f"valid_for={trade_date}"
        )

        # ── Purge stale Redis signals ──
        if self.redis:
            self._purge_stale_redis_signals()

        # ── STEP 1: Download FII OI ──
        raw_data = self._download_with_retries(as_of_date)
        if raw_data is None:
            result.errors.append(
                f"FII data not available for {as_of_date}"
            )
            logger.warning(result.errors[-1])
            return result
        result.download_ok = True

        # ── STEP 1b: Download DII for divergence ──
        dii_data = None
        try:
            dii_data = self.downloader.download_dii_date(as_of_date)
        except Exception as e:
            logger.debug(f"DII download failed (non-critical): {e}")

        # ── STEP 2: Fetch Nifty close ──
        nifty_close = self._fetch_nifty_close(as_of_date)
        raw_data['nifty_close'] = nifty_close

        # ── STEP 3: Compute derived metrics ──
        derived = self._compute_derived_metrics(raw_data, dii_data)
        result.derived_metrics = derived

        # ── STEP 3b: Update fut_net_change in DB ──
        self._update_fut_net_change(as_of_date, derived)

        # ── STEP 4: Detect signals (primary 4 patterns) ──
        if self.detector:
            signal = self.detector.detect(raw_data, for_date=trade_date)
            result.signal = signal
            result.signal_detected = (
                signal.signal_id is not None
                and signal.direction != 'NEUTRAL'
            )

            # ── STEP 4b: Additional patterns (5 & 6) ──
            extra_signals = self._detect_additional_patterns(
                derived, dii_data, trade_date, nifty_close
            )
            result.additional_signals = extra_signals

            # ── STEP 5: Store all signals ──
            all_signals = [signal] + extra_signals
            for sig in all_signals:
                self._store_signal(sig)

            # ── STEP 6: Pre-load to Redis ──
            for sig in all_signals:
                if sig.signal_id and sig.direction != 'NEUTRAL':
                    self._preload_redis(sig)

        # ── STEP 7: Telegram digest ──
        self._send_digest(result)
        result.digest_sent = True

        logger.info(result.summary)
        return result

    # ================================================================
    # PUBLIC: download_only
    # ================================================================

    def download_only(self, as_of_date: date) -> bool:
        """Just download and store data, no signal detection."""
        try:
            data = self.downloader.download_date(as_of_date)
            return data is not None
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    # ================================================================
    # PUBLIC: status / signals
    # ================================================================

    def show_status(self):
        """Print recent FII data and pipeline health."""
        if not self.db:
            print("No DB connection")
            return

        cur = self.db.cursor()

        print(f"\n{'='*70}")
        print("  FII PIPELINE STATUS")
        print(f"{'='*70}")

        # Latest data
        cur.execute("""
            SELECT date, fut_net, put_ratio, pcr, nifty_close
            FROM fii_daily_metrics
            ORDER BY date DESC LIMIT 5
        """)
        rows = cur.fetchall()
        if rows:
            print("\n  Recent FII data:")
            print(f"  {'Date':<12s} {'FutNet':>10s} {'PutRatio':>10s} "
                  f"{'PCR':>8s} {'Nifty':>8s}")
            for r in rows:
                print(f"  {r[0]!s:<12s} {r[1]:>10,.0f} {r[2]:>10.3f} "
                      f"{r[3]:>8.3f} {r[4]:>8.0f}")
        else:
            print("  No FII data found. Run --init to bootstrap.")

        # Data freshness
        if rows:
            latest_date = rows[0][0]
            staleness = (date.today() - latest_date).days
            status = "FRESH" if staleness <= 1 else (
                "STALE" if staleness <= 3 else "CRITICAL"
            )
            print(f"\n  Data freshness: {latest_date} ({staleness}d ago) [{status}]")

        # Recent signals
        cur.execute("""
            SELECT data_date, valid_for_date, signal_id, direction,
                   confidence, pattern_name
            FROM fii_signal_results
            ORDER BY data_date DESC LIMIT 10
        """)
        sig_rows = cur.fetchall()
        if sig_rows:
            print(f"\n  Recent signals:")
            print(f"  {'DataDate':<12s} {'ValidFor':<12s} {'Signal':<10s} "
                  f"{'Dir':<10s} {'Conf':>6s} {'Pattern':<30s}")
            for r in sig_rows:
                print(f"  {r[0]!s:<12s} {r[1]!s:<12s} {r[2] or 'None':<10s} "
                      f"{r[3] or '-':<10s} {r[4]:>5.0%} {r[5]:<30s}")

        print(f"{'='*70}\n")

    def show_signals(self):
        """Print currently active FII signals."""
        if not self.db:
            print("No DB connection")
            return

        cur = self.db.cursor()
        today = date.today()

        cur.execute("""
            SELECT signal_id, direction, confidence, pattern_name,
                   valid_for_date, notes
            FROM fii_signal_results
            WHERE valid_for_date >= %s
              AND signal_id IS NOT NULL
              AND direction != 'NEUTRAL'
            ORDER BY valid_for_date, confidence DESC
        """, (today,))
        rows = cur.fetchall()

        print(f"\n{'='*70}")
        print(f"  ACTIVE FII SIGNALS (valid for {today}+)")
        print(f"{'='*70}")

        if not rows:
            print("  No active FII signals.")
        else:
            for r in rows:
                print(f"\n  {r[0]} ({r[1]}, {r[2]:.0%})")
                print(f"  Pattern: {r[3]}")
                print(f"  Valid for: {r[4]}")
                print(f"  Notes: {r[5]}")

        print(f"{'='*70}\n")

    # ================================================================
    # PRIVATE: Download with retries
    # ================================================================

    def _download_with_retries(self, as_of_date: date) -> Optional[pd.DataFrame]:
        """Download FII data with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                data = self.downloader.download_date(as_of_date)
                if data is not None:
                    return data
            except DataNotAvailableError:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_WAIT_MINUTES * 60
                    logger.info(
                        f"Data not available, retry {attempt + 1}/"
                        f"{self.MAX_RETRIES} in {self.RETRY_WAIT_MINUTES}m"
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"FII data not available for {as_of_date} "
                        f"after {self.MAX_RETRIES} attempts"
                    )
            except Exception as e:
                logger.error(f"Download error attempt {attempt + 1}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(30)
        return None

    # ================================================================
    # PRIVATE: Derived metrics
    # ================================================================

    def _compute_derived_metrics(self, raw_data: pd.DataFrame,
                                  dii_data: Optional[pd.DataFrame]) -> Dict:
        """
        Compute derived metrics using 20-day rolling history:
        - z_score: (today_fut_net - 20d_mean) / 20d_std
        - percentile: rank of today's fut_net in 252-day history
        - flow_5d: sum of fut_net_change over last 5 days
        - fii_dii_divergence: FII net - DII net
        """
        if self.db is None:
            return {}

        row = raw_data.iloc[0]
        fut_net = float(row.get('fut_net', 0))
        result = {'fut_net': fut_net}

        try:
            cur = self.db.cursor()

            # Load 252-day history for percentile and z-score
            cur.execute("""
                SELECT date, fut_net, fut_net_change
                FROM fii_daily_metrics
                ORDER BY date DESC LIMIT 252
            """)
            hist_rows = cur.fetchall()
            if not hist_rows:
                return result

            hist_df = pd.DataFrame(
                hist_rows, columns=['date', 'fut_net', 'fut_net_change']
            )

            # Z-score (20-day rolling)
            recent_20 = hist_df.head(20)['fut_net']
            if len(recent_20) >= 10:
                mean_20 = recent_20.mean()
                std_20 = recent_20.std()
                result['z_score'] = (
                    (fut_net - mean_20) / std_20
                    if std_20 > 0 else 0.0
                )
            else:
                result['z_score'] = 0.0

            # Percentile (252-day)
            all_fut_net = hist_df['fut_net'].dropna()
            if len(all_fut_net) > 0:
                result['percentile'] = float(
                    (all_fut_net <= fut_net).mean() * 100
                )
            else:
                result['percentile'] = 50.0

            # 5-day flow
            recent_5 = hist_df.head(5)['fut_net_change'].dropna()
            result['flow_5d'] = float(recent_5.sum())

            # FII-DII divergence
            if dii_data is not None and not dii_data.empty:
                dii_row = dii_data.iloc[0]
                dii_fut_long = float(
                    dii_row.get('future_long_contracts', 0)
                )
                dii_fut_short = float(
                    dii_row.get('future_short_contracts', 0)
                )
                dii_net = dii_fut_long - dii_fut_short
                result['dii_net'] = dii_net
                result['fii_dii_divergence'] = fut_net - dii_net
            else:
                result['dii_net'] = None
                result['fii_dii_divergence'] = None

            # 20-day dynamic thresholds (for signal detector)
            result['mean_20d'] = float(recent_20.mean()) if len(recent_20) > 0 else 0
            result['std_20d'] = float(recent_20.std()) if len(recent_20) > 1 else 1

        except Exception as e:
            logger.error(f"Derived metrics computation failed: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

        return result

    def _update_fut_net_change(self, as_of_date: date, derived: Dict):
        """Update fut_net_change and nifty_close in DB."""
        if self.db is None:
            return

        try:
            cur = self.db.cursor()

            # Compute day-over-day change
            cur.execute("""
                SELECT fut_net FROM fii_daily_metrics
                WHERE date < %s
                ORDER BY date DESC LIMIT 1
            """, (as_of_date,))
            prev = cur.fetchone()
            fut_net_change = (
                derived.get('fut_net', 0) - float(prev[0])
                if prev and prev[0] is not None else 0.0
            )

            cur.execute("""
                UPDATE fii_daily_metrics
                SET fut_net_change = %s
                WHERE date = %s
            """, (fut_net_change, as_of_date))
            self.db.commit()

            derived['fut_net_change'] = fut_net_change

        except Exception as e:
            logger.error(f"fut_net_change update failed: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

    # ================================================================
    # PRIVATE: Additional pattern detection (5 & 6)
    # ================================================================

    def _detect_additional_patterns(
        self, derived: Dict, dii_data: Optional[pd.DataFrame],
        trade_date: date, nifty_close: float
    ) -> List[FIISignalResult]:
        """
        Detect patterns 5 (FII-DII divergence) and 6 (extreme positioning).
        These supplement the 4 patterns from FIISignalDetector.
        """
        signals = []

        # ── Pattern 5: FII-DII Divergence ──
        fii_net = derived.get('fut_net', 0)
        dii_net = derived.get('dii_net')

        if dii_net is not None:
            if (fii_net < self.P5_FII_NET_THRESHOLD
                    and dii_net > self.P5_DII_NET_THRESHOLD):
                # FII bearish, DII bullish — conflicting, reduce sizing
                signals.append(FIISignalResult(
                    date=derived.get('date', date.today()),
                    signal_id='NSE_005',
                    direction='NEUTRAL',  # conflicting = reduce
                    confidence=0.70,
                    pattern_name='FII_DII_DIVERGENCE',
                    raw_metrics={
                        'fii_net': fii_net,
                        'dii_net': dii_net,
                    },
                    valid_for_date=trade_date,
                    notes=(
                        f"FII net: {fii_net:,.0f} (bearish) vs "
                        f"DII net: {dii_net:,.0f} (bullish). "
                        f"Conflicting institutional positioning — "
                        f"reduce sizing to 0.7x."
                    ),
                ))
            elif (fii_net > abs(self.P5_FII_NET_THRESHOLD)
                  and dii_net < -abs(self.P5_DII_NET_THRESHOLD)):
                # FII bullish, DII bearish — still conflicting
                signals.append(FIISignalResult(
                    date=derived.get('date', date.today()),
                    signal_id='NSE_005',
                    direction='NEUTRAL',
                    confidence=0.65,
                    pattern_name='FII_DII_DIVERGENCE',
                    raw_metrics={
                        'fii_net': fii_net,
                        'dii_net': dii_net,
                    },
                    valid_for_date=trade_date,
                    notes=(
                        f"FII net: {fii_net:,.0f} (bullish) vs "
                        f"DII net: {dii_net:,.0f} (bearish). "
                        f"Conflicting — reduce sizing to 0.7x."
                    ),
                ))

        # ── Pattern 6: Extreme positioning (contrarian) ──
        percentile = derived.get('percentile', 50)
        vix = self._get_current_vix()

        if vix is not None and vix < self.P6_VIX_MAX:
            if percentile <= self.P6_PERCENTILE_EXTREME_LOW:
                # Extremely bearish positioning, contrarian bullish
                signals.append(FIISignalResult(
                    date=derived.get('date', date.today()),
                    signal_id='NSE_006',
                    direction='BULLISH',
                    confidence=0.60,
                    pattern_name='FII_EXTREME_POSITIONING_CONTRARIAN',
                    raw_metrics={
                        'fii_net': fii_net,
                        'percentile': percentile,
                        'vix': vix,
                    },
                    valid_for_date=trade_date,
                    notes=(
                        f"FII net at {percentile:.0f}th percentile "
                        f"(extreme bearish). VIX={vix:.1f} (calm). "
                        f"Contrarian bullish bias. Sizing +1.2x for longs."
                    ),
                ))
            elif percentile >= self.P6_PERCENTILE_EXTREME_HIGH:
                # Extremely bullish positioning, contrarian bearish
                signals.append(FIISignalResult(
                    date=derived.get('date', date.today()),
                    signal_id='NSE_006',
                    direction='BEARISH',
                    confidence=0.60,
                    pattern_name='FII_EXTREME_POSITIONING_CONTRARIAN',
                    raw_metrics={
                        'fii_net': fii_net,
                        'percentile': percentile,
                        'vix': vix,
                    },
                    valid_for_date=trade_date,
                    notes=(
                        f"FII net at {percentile:.0f}th percentile "
                        f"(extreme bullish). VIX={vix:.1f} (calm). "
                        f"Contrarian bearish bias. Sizing +1.2x for shorts."
                    ),
                ))

        return signals

    # ================================================================
    # PRIVATE: Store signals
    # ================================================================

    def _store_signal(self, signal: FIISignalResult):
        """Store signal to fii_signal_results table."""
        if self.db is None:
            return

        try:
            cur = self.db.cursor()
            cur.execute("""
                INSERT INTO fii_signal_results
                    (data_date, valid_for_date, signal_id, direction,
                     confidence, pattern_name, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                signal.date,
                signal.valid_for_date,
                signal.signal_id,
                signal.direction,
                signal.confidence,
                signal.pattern_name,
                signal.notes,
            ))
            self.db.commit()
        except Exception as e:
            logger.error(f"Signal store failed: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

    def _preload_redis(self, signal: FIISignalResult):
        """Pre-load actionable signal into Redis for morning."""
        if self.redis is None or not signal.signal_id:
            return

        try:
            self.redis.xadd(
                'SIGNAL_QUEUE_PRELOADED',
                {
                    'signal_id': signal.signal_id,
                    'direction': signal.direction,
                    'confidence': str(signal.confidence),
                    'valid_for': str(signal.valid_for_date),
                    'valid_until': str(signal.valid_for_date),
                    'pattern': signal.pattern_name,
                    'notes': signal.notes,
                    'source': 'FII_OVERNIGHT',
                    'requires_confirmation': 'true',
                }
            )
        except Exception as e:
            logger.debug(f"Redis preload failed: {e}")

    def _purge_stale_redis_signals(self):
        """Remove expired FII signals from Redis."""
        if self.redis is None:
            return

        try:
            today = date.today().isoformat()
            messages = self.redis.xrange('SIGNAL_QUEUE_PRELOADED', '-', '+')
            purged = 0
            for msg_id, fields in messages:
                valid_until = fields.get(b'valid_until', b'').decode()
                source = fields.get(b'source', b'').decode()
                if source == 'FII_OVERNIGHT' and valid_until < today:
                    self.redis.xdel('SIGNAL_QUEUE_PRELOADED', msg_id)
                    purged += 1
            if purged:
                logger.info(f"Purged {purged} stale FII signals from Redis")
        except Exception as e:
            logger.debug(f"Redis purge failed: {e}")

    # ================================================================
    # PRIVATE: Telegram digest
    # ================================================================

    def _send_digest(self, result: FIIPipelineResult):
        """Send evening Telegram digest with FII analysis."""
        if self.alerter is None:
            return

        try:
            dm = result.derived_metrics
            lines = [
                f"FII EVENING DIGEST — {result.as_of_date}",
                "",
            ]

            if not result.download_ok:
                lines.append("Data: NOT AVAILABLE (holiday?)")
            else:
                lines.append(
                    f"FII Net Futures: {dm.get('fut_net', 0):+,.0f}"
                )
                lines.append(
                    f"Z-Score (20d): {dm.get('z_score', 0):+.2f}"
                )
                lines.append(
                    f"Percentile (252d): {dm.get('percentile', 50):.0f}th"
                )
                lines.append(
                    f"5-Day Flow: {dm.get('flow_5d', 0):+,.0f}"
                )

                if dm.get('dii_net') is not None:
                    lines.append(
                        f"DII Net: {dm['dii_net']:+,.0f} | "
                        f"FII-DII Gap: {dm.get('fii_dii_divergence', 0):+,.0f}"
                    )

                lines.append("")

                # Primary signal
                if result.signal:
                    sig = result.signal
                    lines.append(
                        f"Signal: {sig.pattern_name} "
                        f"({sig.direction}, {sig.confidence:.0%})"
                    )
                    lines.append(f"Valid for: {result.valid_for_date}")
                    if sig.notes:
                        lines.append(f"Detail: {sig.notes[:200]}")
                else:
                    lines.append("Signal: NONE")

                # Additional signals
                for extra in result.additional_signals:
                    lines.append(
                        f"\nExtra: {extra.pattern_name} "
                        f"({extra.direction}, {extra.confidence:.0%})"
                    )
                    if extra.notes:
                        lines.append(f"  {extra.notes[:150]}")

            self.alerter.send('INFO', '\n'.join(lines))

        except Exception as e:
            logger.error(f"Telegram digest failed: {e}")

    # ================================================================
    # PRIVATE: Helpers
    # ================================================================

    def _fetch_nifty_close(self, for_date: date) -> float:
        """Fetch Nifty 50 closing price. Falls back to DB."""
        # Try TrueData API
        if self.truedata_token:
            try:
                date_str = for_date.strftime('%Y-%m-%d')
                resp = requests.get(
                    'https://api.truedata.in/gethistdata',
                    params={
                        'symbol': 'NIFTY50-I',
                        'startdate': date_str,
                        'enddate': date_str,
                        'bar': 'EOD',
                    },
                    headers={
                        'Authorization': f'Bearer {self.truedata_token}'
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                if data and isinstance(data, list):
                    return float(data[0][4])
            except Exception as e:
                logger.debug(f"TrueData Nifty fetch failed: {e}")

        # Fallback: DB
        if self.db:
            try:
                cur = self.db.cursor()
                cur.execute("""
                    SELECT close FROM nifty_daily
                    WHERE date <= %s
                    ORDER BY date DESC LIMIT 1
                """, (for_date,))
                row = cur.fetchone()
                if row and row[0]:
                    return float(row[0])
            except Exception as e:
                logger.debug(f"DB Nifty close fallback failed: {e}")
                try:
                    self.db.rollback()
                except Exception:
                    pass

        # Last resort: hardcoded sensible default
        logger.warning("Using fallback Nifty close 24000.0")
        return 24000.0

    def _get_current_vix(self) -> Optional[float]:
        """Get current India VIX from DB."""
        if self.db is None:
            return None

        try:
            cur = self.db.cursor()
            cur.execute("""
                SELECT india_vix FROM nifty_daily
                WHERE india_vix IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            return float(row[0]) if row and row[0] else None
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass
            return None

    def _get_last_trading_day(self) -> date:
        """Get the last NSE trading day (on or before today)."""
        if self.db:
            try:
                cur = self.db.cursor()
                cur.execute("""
                    SELECT MAX(trading_date)
                    FROM market_calendar
                    WHERE trading_date <= CURRENT_DATE
                      AND is_trading_day = TRUE
                """)
                row = cur.fetchone()
                if row and row[0]:
                    return row[0]
            except Exception:
                try:
                    self.db.rollback()
                except Exception:
                    pass

        # Fallback: skip weekends
        today = date.today()
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        return today

    def _get_next_trading_day(self, from_date: date) -> date:
        """Get the next NSE trading day after from_date."""
        if self.db:
            try:
                cur = self.db.cursor()
                cur.execute("""
                    SELECT MIN(trading_date)
                    FROM market_calendar
                    WHERE trading_date > %s
                      AND is_trading_day = TRUE
                """, (from_date,))
                row = cur.fetchone()
                if row and row[0]:
                    return row[0]
            except Exception:
                try:
                    self.db.rollback()
                except Exception:
                    pass

        # Fallback: skip weekends
        next_day = from_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FII overnight signal pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  venv/bin/python3 -m fii.daily_pipeline                  # today
  venv/bin/python3 -m fii.daily_pipeline --date 2026-03-21
  venv/bin/python3 -m fii.daily_pipeline --init            # bootstrap 6 months
  venv/bin/python3 -m fii.daily_pipeline --status
  venv/bin/python3 -m fii.daily_pipeline --signals
  venv/bin/python3 -m fii.daily_pipeline --download-only

Cron (7:30 PM IST daily, Mon-Fri):
  30 19 * * 1-5  cd /Users/amitkandari/Desktop/trading-system && venv/bin/python3 -m fii.daily_pipeline >> logs/fii_pipeline.log 2>&1
  00 20 * * 1-5  cd /Users/amitkandari/Desktop/trading-system && venv/bin/python3 -m fii.daily_pipeline --retry >> logs/fii_pipeline.log 2>&1
        """,
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='Process specific date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--init', action='store_true',
        help='Bootstrap 6 months of historical FII data',
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show pipeline status and recent data',
    )
    parser.add_argument(
        '--signals', action='store_true',
        help='Show currently active FII signals',
    )
    parser.add_argument(
        '--download-only', action='store_true',
        help='Download data only (no signal detection)',
    )
    parser.add_argument(
        '--retry', action='store_true',
        help='Retry mode — skip if signal already exists for today',
    )

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # DB connection
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_DSN)
        conn.autocommit = False
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        if args.status or args.signals:
            print(f"ERROR: Cannot connect to DB: {e}")
            sys.exit(1)

    # Redis (optional)
    redis_client = None
    try:
        import redis
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB
        )
        redis_client.ping()
    except Exception:
        redis_client = None
        logger.info("Redis not available — signals won't be pre-loaded")

    # Telegram (optional)
    alerter = None
    tg_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    tg_chat = os.environ.get('TELEGRAM_CHAT_ID')
    if tg_token and tg_chat:
        from monitoring.telegram_alerter import TelegramAlerter
        alerter = TelegramAlerter(tg_token, tg_chat)

    # Build pipeline
    pipeline = FIIDailyPipeline(
        db=conn, redis_client=redis_client, alerter=alerter,
    )

    try:
        if args.status:
            pipeline.show_status()
        elif args.signals:
            pipeline.show_signals()
        elif args.init:
            print("Bootstrapping 6 months of FII data...")
            downloader = FIIDataDownloader(db_conn=conn)
            downloaded = downloader.initial_load(months=6)
            print(f"Downloaded {len(downloaded)} days of FII data.")
        elif args.download_only:
            as_of = date.fromisoformat(args.date) if args.date else date.today()
            ok = pipeline.download_only(as_of)
            print(f"Download {'OK' if ok else 'FAILED'} for {as_of}")
        else:
            as_of = date.fromisoformat(args.date) if args.date else None

            # Retry mode: check if signal already exists
            if args.retry and conn:
                check_date = as_of or pipeline._get_last_trading_day()
                cur = conn.cursor()
                cur.execute("""
                    SELECT COUNT(*) FROM fii_signal_results
                    WHERE data_date = %s
                """, (check_date,))
                count = cur.fetchone()[0]
                if count > 0:
                    logger.info(
                        f"Signal already exists for {check_date} — "
                        f"skipping retry"
                    )
                    sys.exit(0)

            result = pipeline.run(as_of_date=as_of)
            print(result.summary)

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if conn and not conn.closed:
            conn.close()


if __name__ == '__main__':
    main()
