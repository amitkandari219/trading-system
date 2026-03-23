"""
BankNifty Weekly Signal Review — cron-callable Sunday script.

Evaluates all 10 BN_ SHADOW signals, generates a weekly report,
flags signals READY_TO_PROMOTE or due for demotion review.

Promotion requires human confirmation via CLI:
    venv/bin/python3 -m paper_trading.bn_weekly_review --promote BN_KAUFMAN_BB_MR
    venv/bin/python3 -m paper_trading.bn_weekly_review --demote BN_ORB_BREAKOUT
    venv/bin/python3 -m paper_trading.bn_weekly_review --status

Cron (Sunday 10:00 AM):
    0 10 * * 0 cd /path/to/trading-system && venv/bin/python3 -m paper_trading.bn_weekly_review >> logs/bn_weekly_review.log 2>&1

Usage:
    venv/bin/python3 -m paper_trading.bn_weekly_review           # generate report
    venv/bin/python3 -m paper_trading.bn_weekly_review --status   # show current status
    venv/bin/python3 -m paper_trading.bn_weekly_review --promote BN_KAUFMAN_BB_MR
    venv/bin/python3 -m paper_trading.bn_weekly_review --demote BN_ORB_BREAKOUT
"""

import argparse
import logging
import os
import sys
from datetime import date

from paper_trading.bn_promotion_tracker import (
    BNPromotionTracker, BN_SIGNAL_IDS, PROMOTION_CRITERIA,
)

logger = logging.getLogger(__name__)


def run_weekly_review(as_of: date = None):
    """Generate and send the weekly BN signal report."""
    as_of = as_of or date.today()

    tracker = BNPromotionTracker()
    try:
        report_text = tracker.generate_weekly_report(as_of)
        print(report_text)

        # Send via Telegram
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        if token and chat_id:
            from monitoring.telegram_alerter import TelegramAlerter
            alerter = TelegramAlerter(token, chat_id)
            alerter.send('INFO', report_text)
            logger.info("Weekly report sent to Telegram")
        else:
            logger.info("Telegram not configured — report printed to stdout only")

        # Check for demotion flags on SCORING signals
        reports = tracker.evaluate_all_bn_signals(as_of)
        for signal_id, rpt in reports.items():
            if rpt.current_status == 'SCORING' and rpt.demotion_flagged:
                logger.warning(
                    f"DEMOTION FLAG: {signal_id} — "
                    f"{', '.join(rpt.demotion_reasons)}"
                )
    finally:
        tracker.close()


def show_status():
    """Print current status of all BN signals."""
    tracker = BNPromotionTracker()
    try:
        reports = tracker.evaluate_all_bn_signals()

        print(f"\n{'='*80}")
        print(f"  BANKNIFTY SIGNAL STATUS — {date.today()}")
        print(f"{'='*80}")
        print(f"\n  Promotion criteria: {PROMOTION_CRITERIA}\n")

        header = (
            f"  {'Signal':<22s} {'Status':>6s} {'Trd':>4s} {'Shrp':>5s} "
            f"{'WR':>5s} {'PF':>5s} {'DD':>5s} {'W/L':>4s} "
            f"{'P&L':>8s} {'Verdict'}"
        )
        print(header)
        print(f"  {'-'*76}")

        for signal_id in BN_SIGNAL_IDS:
            rpt = reports.get(signal_id)
            if rpt is None:
                print(f"  {signal_id:<22s}  ERROR")
                continue
            print(f"  {rpt.summary_line()}")

        total_pnl = sum(r.total_pnl for r in reports.values())
        total_trades = sum(r.trade_count for r in reports.values())
        print(f"\n  Total: {total_trades} trades | P&L: {total_pnl:+,.0f}")
        print(f"{'='*80}\n")
    finally:
        tracker.close()


def promote_signal(signal_id: str):
    """Promote a BN signal to SCORING (human-confirmed)."""
    if signal_id not in BN_SIGNAL_IDS:
        print(f"ERROR: {signal_id} is not a valid BN signal.")
        print(f"Valid signals: {', '.join(BN_SIGNAL_IDS)}")
        sys.exit(1)

    tracker = BNPromotionTracker()
    try:
        report = tracker.evaluate_signal(signal_id)
        if not report.promotion_eligible:
            print(f"WARNING: {signal_id} does NOT meet promotion criteria:")
            for reason in report.fail_reasons:
                print(f"  - {reason}")
            confirm = input("Promote anyway? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Promotion cancelled.")
                return

        success = tracker.promote_signal(signal_id)
        if success:
            print(f"PROMOTED: {signal_id} -> SCORING")
        else:
            print(f"ERROR: Promotion failed for {signal_id}")
            sys.exit(1)
    finally:
        tracker.close()


def demote_signal(signal_id: str):
    """Demote a BN signal from SCORING to DEMOTED."""
    if signal_id not in BN_SIGNAL_IDS:
        print(f"ERROR: {signal_id} is not a valid BN signal.")
        sys.exit(1)

    tracker = BNPromotionTracker()
    try:
        report = tracker.evaluate_signal(signal_id)
        if report.current_status != 'SCORING':
            print(f"WARNING: {signal_id} is not SCORING (current: {report.current_status})")
            confirm = input("Demote anyway? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Demotion cancelled.")
                return

        reason = input("Demotion reason (or Enter for auto): ").strip()
        if not reason:
            reason = '; '.join(report.demotion_reasons) if report.demotion_reasons else 'Manual demotion'

        success = tracker.demote_signal(signal_id, reason)
        if success:
            print(f"DEMOTED: {signal_id} -> DEMOTED ({reason})")
        else:
            print(f"ERROR: Demotion failed for {signal_id}")
            sys.exit(1)
    finally:
        tracker.close()


def main():
    parser = argparse.ArgumentParser(
        description='BankNifty Weekly Signal Review',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  venv/bin/python3 -m paper_trading.bn_weekly_review           # weekly report
  venv/bin/python3 -m paper_trading.bn_weekly_review --status   # signal status
  venv/bin/python3 -m paper_trading.bn_weekly_review --promote BN_KAUFMAN_BB_MR
  venv/bin/python3 -m paper_trading.bn_weekly_review --demote BN_ORB_BREAKOUT
        """,
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show current status of all BN signals',
    )
    parser.add_argument(
        '--promote', type=str, default=None,
        help='Promote SIGNAL_ID from SHADOW to SCORING',
    )
    parser.add_argument(
        '--demote', type=str, default=None,
        help='Demote SIGNAL_ID from SCORING to DEMOTED',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.promote:
        promote_signal(args.promote)
    elif args.demote:
        demote_signal(args.demote)
    elif args.status:
        show_status()
    else:
        run_weekly_review()


if __name__ == '__main__':
    main()
