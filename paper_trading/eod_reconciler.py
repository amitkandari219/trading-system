"""
EOD Reconciler — scheduler entry point for end-of-day reconciliation.

Orchestrates the full EOD pipeline:
  1. Check market open (skip weekends/holidays)
  2. Run order verification
  3. Run position reconciliation
  4. Log report as JSON to logs/reconciliation/
  5. Send Telegram digest
  6. Update weekly counters (Friday reset)
  7. Update compound_sizer equity
  8. Generate next-day prep (lots, halt status)

Usage:
    python -m paper_trading.eod_reconciler                    # today
    python -m paper_trading.eod_reconciler --date 2026-03-20  # specific date
    python -m paper_trading.eod_reconciler --dry-run          # no DB writes
    python -m paper_trading.eod_reconciler --status           # show last report

Schedule via cron:
    35 15 * * 1-5 cd /path/to/trading-system && venv/bin/python3 -m paper_trading.eod_reconciler >> logs/eod_reconciler.log 2>&1
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import psycopg2

from config.settings import DATABASE_DSN, TOTAL_CAPITAL, NIFTY_LOT_SIZE
from execution.position_reconciler import PositionReconciler, ReconciliationReport
from execution.order_verifier import OrderVerifier
from risk.daily_loss_limiter import DailyLossLimiter
from risk.compound_sizer import CompoundSizer

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# Project root for log directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "reconciliation"


def _is_trading_day(conn, as_of_date: date) -> bool:
    """Check if the given date is a trading day (not weekend/holiday)."""
    # Weekend check
    if as_of_date.weekday() >= 5:
        return False

    # Check market_calendar table
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT is_trading_day FROM market_calendar WHERE trading_date = %s",
            (as_of_date,),
        )
        row = cur.fetchone()
        if row:
            return bool(row[0])
        # If not in calendar, assume trading day (weekday)
        return True
    except Exception as e:
        logger.warning(f"market_calendar lookup failed: {e}")
        # Fallback: weekday = trading day
        return as_of_date.weekday() < 5


def _get_alerter():
    """Get TelegramAlerter if credentials are available."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        from monitoring.telegram_alerter import TelegramAlerter
        return TelegramAlerter(token, chat_id)

    # Return a no-op alerter
    class NoopAlerter:
        def send(self, level, message, **kwargs):
            logger.info(f"[NOOP ALERT] [{level}] {message}")
    return NoopAlerter()


def _get_kite():
    """Get authenticated Kite instance (None in PAPER mode)."""
    if EXECUTION_MODE == "PAPER":
        return None
    try:
        from data.kite_auth import get_kite
        return get_kite()
    except Exception as e:
        logger.warning(f"Kite auth failed: {e}")
        return None


def _save_report_json(report: ReconciliationReport):
    """Save report as JSON to logs/reconciliation/YYYY-MM-DD.json."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    filepath = LOG_DIR / f"{report.date}.json"

    try:
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info(f"Report saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save report JSON: {e}")


def _generate_next_day_prep(
    conn, as_of_date: date, compound_sizer: CompoundSizer,
    loss_limiter: DailyLossLimiter
) -> dict:
    """Generate next-day prep summary: lots available, halt status, etc."""
    prep = {
        "for_date": str(as_of_date),
        "equity": compound_sizer.equity,
        "nifty_lots": compound_sizer.get_lots("NIFTY", premium=200, today=as_of_date),
        "drawdown_pct": round(compound_sizer.drawdown_pct * 100, 2),
        "in_drawdown": compound_sizer.in_drawdown,
        "loss_limiter_tier": loss_limiter.tier,
        "weekly_halt": loss_limiter.weekly_halt,
        "can_trade": loss_limiter.can_trade(as_of_date),
        "size_factor": loss_limiter.size_factor,
    }

    # Check for open positions carrying over (should be 0 for MIS)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM trades
            WHERE exit_date IS NULL AND entry_date <= %s
            """,
            (as_of_date,),
        )
        row = cur.fetchone()
        prep["open_positions_carryover"] = row[0] if row else 0
    except Exception:
        prep["open_positions_carryover"] = -1

    return prep


def _get_bn_shadow_summary(conn, as_of_date: date) -> dict:
    """
    Fetch BankNifty SHADOW trade P&L for today's EOD digest.

    Returns dict with:
        trade_count: int
        total_pnl: float
        top_signals: list of (signal_id, pnl, count) tuples
    """
    result = {'trade_count': 0, 'total_pnl': 0.0, 'top_signals': []}
    try:
        cur = conn.cursor()
        # Total BN shadow trades today
        cur.execute("""
            SELECT COUNT(*), COALESCE(SUM(pnl), 0)
            FROM trades
            WHERE instrument = 'BANKNIFTY'
              AND trade_type = 'SHADOW'
              AND entry_date = %s
              AND exit_date IS NOT NULL
        """, (as_of_date,))
        row = cur.fetchone()
        if row:
            result['trade_count'] = row[0] or 0
            result['total_pnl'] = float(row[1] or 0)

        # Per-signal breakdown
        cur.execute("""
            SELECT signal_id, COALESCE(SUM(pnl), 0), COUNT(*)
            FROM trades
            WHERE instrument = 'BANKNIFTY'
              AND trade_type = 'SHADOW'
              AND entry_date = %s
              AND exit_date IS NOT NULL
            GROUP BY signal_id
            ORDER BY COALESCE(SUM(pnl), 0) DESC
        """, (as_of_date,))
        rows = cur.fetchall()
        result['top_signals'] = [
            (r[0], float(r[1]), r[2]) for r in rows
        ]
    except Exception as e:
        logger.warning(f"BN shadow summary failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass

    return result


def run_eod(as_of_date: Optional[date] = None, dry_run: bool = False) -> dict:
    """
    Main EOD reconciliation entry point.

    Args:
        as_of_date: date to reconcile (default: today)
        dry_run:    if True, no DB writes or Kite calls

    Returns:
        dict with report, order_issues, fill_issues, next_day_prep
    """
    as_of_date = as_of_date or date.today()
    result = {
        "date": str(as_of_date),
        "mode": EXECUTION_MODE,
        "dry_run": dry_run,
        "skipped": False,
    }

    conn = psycopg2.connect(DATABASE_DSN)
    conn.autocommit = False

    try:
        # Step 1: Check if trading day
        if not _is_trading_day(conn, as_of_date):
            logger.info(f"{as_of_date} is not a trading day — skipping reconciliation")
            result["skipped"] = True
            result["skip_reason"] = "Not a trading day"
            return result

        # Initialize components
        kite = None if dry_run else _get_kite()
        alerter = _get_alerter()
        loss_limiter = DailyLossLimiter(equity=TOTAL_CAPITAL)
        compound_sizer = CompoundSizer(initial_equity=TOTAL_CAPITAL)

        # Load latest equity from portfolio_state
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT total_capital FROM portfolio_state
                ORDER BY snapshot_time DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            if row and row[0]:
                latest_equity = float(row[0])
                loss_limiter = DailyLossLimiter(equity=latest_equity)
                compound_sizer = CompoundSizer(initial_equity=latest_equity)
        except Exception as e:
            logger.warning(f"Could not load latest equity: {e}")
            conn.rollback()

        # Step 2: Order verification
        verifier = OrderVerifier(kite, conn, alerter)
        order_issues = verifier.verify_todays_orders(as_of_date)
        fill_issues = verifier.verify_fills(as_of_date)
        result["order_issues"] = order_issues
        result["fill_issues"] = fill_issues

        # Step 3: Position reconciliation
        reconciler = PositionReconciler(
            kite=kite,
            db=conn,
            alerter=alerter,
            loss_limiter=loss_limiter if not dry_run else None,
            compound_sizer=compound_sizer if not dry_run else None,
        )
        report = reconciler.reconcile(as_of_date)
        result["report"] = report.to_dict()
        result["status"] = report.status

        # Step 4: Save report JSON
        _save_report_json(report)

        # Step 5: Send Telegram digest (including BN shadow P&L)
        bn_shadow_summary = _get_bn_shadow_summary(conn, as_of_date)
        try:
            summary = report.to_telegram_summary()

            # Append BN shadow P&L attribution
            if bn_shadow_summary['trade_count'] > 0:
                summary += (
                    f"\n\nBN Shadow: {bn_shadow_summary['trade_count']} trades "
                    f"| P&L: {bn_shadow_summary['total_pnl']:+,.0f}"
                )
                if bn_shadow_summary['top_signals']:
                    for sig_id, pnl, cnt in bn_shadow_summary['top_signals'][:3]:
                        summary += f"\n  {sig_id}: {pnl:+,.0f} ({cnt}t)"

            if order_issues:
                summary += f"\n\nOrder issues: {len(order_issues)}"
                for oi in order_issues[:3]:
                    summary += f"\n  - {oi['type']}: {oi.get('tradingsymbol', '?')}"
            if fill_issues:
                summary += f"\n\nFill issues: {len(fill_issues)}"
                for fi in fill_issues[:3]:
                    summary += (
                        f"\n  - {fi.get('tradingsymbol', '?')} "
                        f"slippage={fi.get('adverse_slippage_pct', 0):+.2f}%"
                    )

            alert_level = {
                "CLEAN": "INFO",
                "MINOR": "WARNING",
                "MAJOR": "CRITICAL",
                "CRITICAL": "EMERGENCY",
            }.get(report.status, "WARNING")

            if not dry_run:
                alerter.send(alert_level, summary)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

        # Step 6: Weekly counter update (Friday)
        if as_of_date.weekday() == 4 and not dry_run:
            logger.info("Friday — weekly counter reset will happen on next Monday")

        # Step 7: Compound sizer equity update (already done in reconciler)

        # Step 7b: Daily decay quick check
        try:
            from paper_trading.decay_auto_manager import DecayAutoManager
            decay_mgr = DecayAutoManager(conn=conn)
            decay_result = decay_mgr.run_daily_quick_check(
                as_of=as_of_date, dry_run=dry_run
            )
            result["decay_quick_check"] = {
                "total_signals": decay_result.total_signals,
                "warning": decay_result.warning,
                "critical": decay_result.critical,
                "actions": len(decay_result.actions_taken),
            }
            if decay_result.warning > 0 or decay_result.critical > 0:
                logger.info(
                    f"Decay quick check: {decay_result.warning} warnings, "
                    f"{decay_result.critical} critical"
                )
        except Exception as e:
            logger.warning(f"Decay quick check failed (non-fatal): {e}")
            result["decay_quick_check"] = {"error": str(e)}

        # Step 8: Next-day prep
        next_day_prep = _generate_next_day_prep(
            conn, as_of_date, compound_sizer, loss_limiter
        )
        result["next_day_prep"] = next_day_prep

        if not dry_run:
            conn.commit()
        else:
            conn.rollback()

    except Exception as e:
        logger.critical(f"EOD reconciler failed: {e}", exc_info=True)
        result["error"] = str(e)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()

    return result


def show_last_status(conn) -> dict:
    """Show the last reconciliation report from the DB."""
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT run_date, run_time, mode, status, discrepancy_count,
                   broker_pnl, internal_pnl, pnl_discrepancy, eod_equity
            FROM reconciliation_log
            ORDER BY run_time DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if row:
            return {
                "run_date": str(row[0]),
                "run_time": str(row[1]),
                "mode": row[2],
                "status": row[3],
                "discrepancy_count": row[4],
                "broker_pnl": row[5],
                "internal_pnl": row[6],
                "pnl_discrepancy": row[7],
                "eod_equity": row[8],
            }
        else:
            return {"message": "No reconciliation records found"}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="EOD Position Reconciliation")
    parser.add_argument("--date", type=str, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes or Kite calls")
    parser.add_argument("--status", action="store_true", help="Show last reconciliation status")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    if args.status:
        conn = psycopg2.connect(DATABASE_DSN)
        status = show_last_status(conn)
        conn.close()
        print(json.dumps(status, indent=2, default=str))
        return

    as_of = date.fromisoformat(args.date) if args.date else date.today()

    print(f"{'='*60}")
    print(f"  EOD RECONCILIATION — {as_of}")
    print(f"  Mode: {EXECUTION_MODE} | Dry run: {args.dry_run}")
    print(f"{'='*60}")

    result = run_eod(as_of_date=as_of, dry_run=args.dry_run)

    if result.get("skipped"):
        print(f"\nSkipped: {result.get('skip_reason', 'unknown')}")
        return

    print(f"\nStatus: {result.get('status', 'UNKNOWN')}")
    print(f"Order issues: {len(result.get('order_issues', []))}")
    print(f"Fill issues: {len(result.get('fill_issues', []))}")

    report = result.get("report", {})
    print(f"\nBroker positions: {report.get('broker_positions', 0)}")
    print(f"Internal positions: {report.get('internal_positions', 0)}")
    print(f"Matched: {report.get('matched', 0)}")
    print(f"Total discrepancies: {report.get('total_discrepancies', 0)}")
    print(f"Broker P&L: {report.get('broker_pnl', 0):+,.0f}")
    print(f"Internal P&L: {report.get('internal_pnl', 0):+,.0f}")
    print(f"EOD Equity: {report.get('eod_equity', 0):,.0f}")

    prep = result.get("next_day_prep", {})
    if prep:
        print(f"\n--- Next Day Prep ---")
        print(f"  Nifty lots: {prep.get('nifty_lots', 0)}")
        print(f"  Can trade: {prep.get('can_trade', '?')}")
        print(f"  Drawdown: {prep.get('drawdown_pct', 0):.1f}%")
        print(f"  Open carryover: {prep.get('open_positions_carryover', 0)}")

    if result.get("error"):
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print(f"\nEOD reconciliation complete.")


if __name__ == "__main__":
    main()
