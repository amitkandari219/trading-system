"""
FII daily pipeline — nightly orchestrator that runs at 7:30 PM.
Downloads NSE FII OI data, detects patterns, pre-loads signals into Redis.
"""

import time
import requests
from datetime import date, timedelta

from fii.downloader import FIIDataDownloader, DataNotAvailableError
from fii.signal_detector import FIISignalDetector


class FIIDailyPipeline:
    """
    Complete nightly pipeline:
    1. Download NSE FII OI data
    2. Fetch Nifty closing price
    3. Extract metrics and store to DB
    4. Detect FII signal pattern
    5. Store signal result to DB
    6. Pre-load signal into tomorrow's Redis queue
    7. Alert on signal detection
    """

    MAX_RETRIES = 3
    RETRY_WAIT_MINUTES = 15

    def __init__(self, db, redis_client, alerter, logger,
                 truedata_token: str):
        """
        db:             database connection
        redis_client:   redis.Redis instance
        alerter:        TelegramAlerter instance
        logger:         Python Logger
        truedata_token: TrueData API Bearer token for Nifty price fetch
        """
        self.db              = db
        self.redis           = redis_client
        self.alerter         = alerter
        self.logger          = logger
        self.truedata_token  = truedata_token
        self.downloader      = FIIDataDownloader()
        self.detector        = FIISignalDetector(db)

    def run(self, today_trading_date=None):
        """
        Main pipeline: download today's NSE FII OI data, detect patterns,
        pre-load signals into Redis for next morning's SignalSelector.
        """
        if today_trading_date is None:
            today_trading_date = self._get_last_trading_day()

        trade_date = self._get_next_trading_day(today_trading_date)

        # Purge stale FII signals from Redis before adding new ones.
        self._purge_stale_redis_signals()

        # STEP 1: Download with retries
        raw_data = None
        for attempt in range(self.MAX_RETRIES):
            try:
                raw_data = self.downloader.download_participant_oi(
                    today_trading_date
                )
                break
            except DataNotAvailableError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_WAIT_MINUTES * 60)
                else:
                    self.logger.warning(
                        f"FII data not available for {today_trading_date} "
                        f"after {self.MAX_RETRIES} attempts"
                    )
                    return  # Skip — no data today

        if raw_data is None:
            return

        # STEP 2: Fetch today's Nifty closing price and inject into raw_data.
        try:
            nifty_close = self._fetch_nifty_close(today_trading_date)
        except Exception as e:
            self.logger.warning(
                f"Could not fetch Nifty close for {today_trading_date}: {e}. "
                f"Falling back to yesterday's stored value."
            )
            row = self.db.execute(
                "SELECT nifty_close FROM fii_daily_metrics "
                "ORDER BY date DESC LIMIT 1"
            ).fetchone()
            nifty_close = row['nifty_close'] if row and row['nifty_close'] else 24000.0
        raw_data['nifty_close'] = nifty_close

        # STEP 3: Extract and store metrics
        metrics = self.detector._extract_metrics(raw_data)
        metrics['fut_net_change'] = self._compute_day_change(
            metrics['fut_net']
        )
        self._store_fii_metrics(metrics)

        # STEP 4: Detect signal
        result = self.detector.detect(raw_data, for_date=trade_date)

        # STEP 5: Store signal result
        self._store_fii_signal_result(result)

        # STEP 6: If signal detected, pre-load into Redis queue
        if result.signal_id and result.direction != 'NEUTRAL':
            self.redis.xadd(
                'SIGNAL_QUEUE_PRELOADED',
                {
                    'signal_id':             result.signal_id,
                    'direction':             result.direction,
                    'confidence':            str(result.confidence),
                    'valid_for':             str(result.valid_for_date),
                    'valid_until':           str(result.valid_for_date),
                    'pattern':               result.pattern_name,
                    'notes':                 result.notes,
                    'source':                'FII_OVERNIGHT',
                    'requires_confirmation': 'true',
                }
            )
            self.alerter.send(
                'INFO',
                f"FII signal pre-loaded: {result.pattern_name} "
                f"({result.direction}, {result.confidence:.0%} confidence) "
                f"valid for {result.valid_for_date}"
            )

        self.logger.info(
            f"FII pipeline complete for {today_trading_date}: "
            f"{result.pattern_name} ({result.direction})"
        )

    def _get_last_trading_day(self):
        """Query market_calendar for the last actual NSE trading day."""
        result = self.db.execute("""
            SELECT MAX(trading_date)
            FROM market_calendar
            WHERE trading_date <= CURRENT_DATE
              AND is_trading_day = TRUE
        """).fetchone()
        return result[0] if result else date.today()

    def _get_next_trading_day(self, from_date):
        """Returns the next NSE trading day after from_date."""
        result = self.db.execute("""
            SELECT MIN(trading_date)
            FROM market_calendar
            WHERE trading_date > %s
              AND is_trading_day = TRUE
        """, (from_date,)).fetchone()
        return result[0] if result else from_date + timedelta(days=1)

    def _purge_stale_redis_signals(self):
        """
        Remove FII signals from Redis that are past their valid_until date.
        """
        today = date.today().isoformat()
        messages = self.redis.xrange('SIGNAL_QUEUE_PRELOADED', '-', '+')
        purged = 0
        for msg_id, fields in messages:
            valid_until = fields.get('valid_until', '')
            if valid_until and valid_until < today:
                self.redis.xdel('SIGNAL_QUEUE_PRELOADED', msg_id)
                purged += 1
        if purged:
            self.logger.info(f"Purged {purged} stale FII signal(s) from Redis")

    def _fetch_nifty_close(self, for_date) -> float:
        """Fetch Nifty 50 closing price for for_date from TrueData."""
        date_str = for_date.strftime('%Y-%m-%d')
        resp = requests.get(
            'https://api.truedata.in/gethistdata',
            params={
                'symbol':   'NIFTY50-I',
                'startdate': date_str,
                'enddate':   date_str,
                'bar':       'EOD',
            },
            headers={'Authorization': f'Bearer {self.truedata_token}'},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list):
            return float(data[0][4])
        raise ValueError(f"Empty TrueData response for {date_str}")

    def _compute_day_change(self, today_fut_net: float) -> float:
        """Compute day-over-day change in FII net futures."""
        yesterday = self.db.execute(
            "SELECT fut_net FROM fii_daily_metrics "
            "ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if not yesterday:
            return 0.0
        return today_fut_net - yesterday['fut_net']

    def _store_fii_metrics(self, metrics: dict):
        """Store extracted FII metrics to fii_daily_metrics table."""
        self.db.execute("""
            INSERT INTO fii_daily_metrics
                (date, fut_long, fut_short, fut_net,
                 put_long, put_short, put_net, put_ratio,
                 call_long, call_short, call_net, pcr,
                 fut_net_change, nifty_close)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                fut_long = EXCLUDED.fut_long,
                fut_short = EXCLUDED.fut_short,
                fut_net = EXCLUDED.fut_net,
                put_long = EXCLUDED.put_long,
                put_short = EXCLUDED.put_short,
                put_net = EXCLUDED.put_net,
                put_ratio = EXCLUDED.put_ratio,
                call_long = EXCLUDED.call_long,
                call_short = EXCLUDED.call_short,
                call_net = EXCLUDED.call_net,
                pcr = EXCLUDED.pcr,
                fut_net_change = EXCLUDED.fut_net_change,
                nifty_close = EXCLUDED.nifty_close
        """, (
            metrics['date'],
            metrics['fut_long'], metrics['fut_short'], metrics['fut_net'],
            metrics['put_long'], metrics['put_short'], metrics['put_net'],
            metrics['put_ratio'],
            metrics['call_long'], metrics['call_short'], metrics['call_net'],
            metrics['pcr'],
            metrics.get('fut_net_change', 0),
            metrics.get('nifty_close', 24000.0),
        ))

    def _store_fii_signal_result(self, result):
        """Store FII signal detection result to fii_signal_results table."""
        self.db.execute("""
            INSERT INTO fii_signal_results
                (data_date, valid_for_date, signal_id, direction,
                 confidence, pattern_name, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            result.date,
            result.valid_for_date,
            result.signal_id,
            result.direction,
            result.confidence,
            result.pattern_name,
            result.notes,
        ))
