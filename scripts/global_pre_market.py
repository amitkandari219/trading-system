"""
Global pre-market data pipeline — runs at 8:30 AM IST.

This script:
  1. Fetches latest global market data (S&P, VIX, DXY, Brent, GIFT Nifty)
  2. Stores snapshot in global_market_snapshots table
  3. Evaluates the global composite signal
  4. Sends Telegram alert with pre-market context
  5. Stores evaluation for the daily runner to consume

Designed to run before the main pre-market job (8:45 AM).

Cron entry:
  30 8 * * 1-5  cd /path/to/trading-system && python -m scripts.global_pre_market

Usage:
    python -m scripts.global_pre_market
    python -m scripts.global_pre_market --dry-run
    python -m scripts.global_pre_market --backfill 365  # backfill N days
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.global_market_fetcher import GlobalMarketFetcher
from signals.global_composite import GlobalCompositeSignal

logger = logging.getLogger(__name__)


def run_pre_market(dry_run: bool = False):
    """Execute the full global pre-market pipeline."""
    logger.info("=" * 60)
    logger.info("GLOBAL PRE-MARKET PIPELINE")
    logger.info(f"Date: {date.today()}, Time: {datetime.now().strftime('%H:%M:%S IST')}")
    logger.info("=" * 60)

    # ── DB Connection ─────────────────────────────────────────
    db = None
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        db = psycopg2.connect(DATABASE_DSN)
        logger.info("Database connected")
    except Exception as e:
        logger.warning(f"No DB connection: {e}")

    # ── Step 1: Fetch global data ─────────────────────────────
    logger.info("\n[1/4] Fetching global market data...")
    fetcher = GlobalMarketFetcher(db_conn=db)

    try:
        snapshot = fetcher.fetch_pre_market_snapshot()
        logger.info(f"  S&P 500:  {snapshot.get('sp500_close', 'N/A')} ({snapshot.get('sp500_change_pct', 'N/A'):+.2f}%)" if snapshot.get('sp500_change_pct') else "  S&P 500:  N/A")
        logger.info(f"  US VIX:   {snapshot.get('us_vix_close', 'N/A')}")
        logger.info(f"  DXY:      {snapshot.get('dxy_close', 'N/A')} ({snapshot.get('dxy_change_pct', 'N/A')}%)" if snapshot.get('dxy_change_pct') else "  DXY:      N/A")
        logger.info(f"  Brent:    {snapshot.get('brent_close', 'N/A')}")
        logger.info(f"  GIFT gap: {snapshot.get('gift_nifty_gap_pct', 'N/A')}%")
        logger.info(f"  Risk:     {snapshot.get('global_risk_score', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to fetch global data: {e}")
        _send_alert(f"⚠️ Global pre-market fetch FAILED: {e}")
        return

    # ── Step 2: Store snapshot ────────────────────────────────
    if not dry_run and db:
        logger.info("\n[2/4] Storing snapshot...")
        fetcher.store_snapshot(snapshot)
    else:
        logger.info("\n[2/4] Skipping store (dry-run or no DB)")

    # ── Step 3: Evaluate composite signal ─────────────────────
    logger.info("\n[3/4] Evaluating global composite signal...")
    composite = GlobalCompositeSignal(db_conn=db)

    try:
        ctx = composite.evaluate(snapshot=snapshot)
        logger.info(f"  Direction:  {ctx.direction or 'NEUTRAL'}")
        logger.info(f"  Score:      {ctx.composite_score:+.3f}")
        logger.info(f"  Confidence: {ctx.confidence:.0%}")
        logger.info(f"  Size mod:   {ctx.size_modifier:.2f}x")
        logger.info(f"  Risk-off:   {ctx.risk_off}")
        logger.info(f"  Reason:     {ctx.reason}")

        if ctx.regime_warning:
            logger.warning(f"  ⚠️ REGIME WARNING: {ctx.regime_warning.get('warning', '')}")

        # Store evaluation
        if not dry_run and db:
            composite.store_evaluation(ctx)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        ctx = None

    # ── Step 4: Telegram alert ────────────────────────────────
    logger.info("\n[4/4] Sending Telegram alert...")
    if ctx and not dry_run:
        alert_text = ctx.to_telegram()
        _send_alert(alert_text)
        logger.info("  Alert sent")
    elif dry_run:
        if ctx:
            logger.info(f"  [DRY RUN] Would send:\n{ctx.to_telegram()}")
    else:
        logger.info("  No context to alert")

    if db:
        db.close()

    logger.info("\nGlobal pre-market pipeline complete.")


def run_backfill(days: int):
    """Backfill global market history."""
    logger.info(f"Backfilling {days} days of global market data...")

    db = None
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        db = psycopg2.connect(DATABASE_DSN)
    except Exception as e:
        logger.error(f"DB connection required for backfill: {e}")
        return

    fetcher = GlobalMarketFetcher(db_conn=db)
    years = max(1, days // 365)
    results = fetcher.backfill_all(years=years)

    for key, rows in results.items():
        logger.info(f"  {key}: {rows} rows")

    db.close()
    logger.info("Backfill complete.")


def _send_alert(message: str):
    """Send Telegram alert via the system alerter."""
    try:
        from monitoring.telegram_alerter import TelegramAlerter
        alerter = TelegramAlerter()
        alerter.send('INFO', message)
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Global pre-market pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Skip DB writes and Telegram')
    parser.add_argument('--backfill', type=int, help='Backfill N days of history')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    if args.backfill:
        run_backfill(args.backfill)
    else:
        run_pre_market(dry_run=args.dry_run)
