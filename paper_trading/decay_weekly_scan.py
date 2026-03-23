"""
CLI entry point for decay weekly scan.

Usage:
    venv/bin/python3 -m paper_trading.decay_weekly_scan              # weekly scan
    venv/bin/python3 -m paper_trading.decay_weekly_scan --daily      # daily quick check
    venv/bin/python3 -m paper_trading.decay_weekly_scan --status     # active overrides
    venv/bin/python3 -m paper_trading.decay_weekly_scan --signal KAUFMAN_DRY_20
    venv/bin/python3 -m paper_trading.decay_weekly_scan --history KAUFMAN_DRY_20
    venv/bin/python3 -m paper_trading.decay_weekly_scan --override KAUFMAN_DRY_20 0.5 7
    venv/bin/python3 -m paper_trading.decay_weekly_scan --clear-override KAUFMAN_DRY_20
    venv/bin/python3 -m paper_trading.decay_weekly_scan --dry-run

Schedule (cron):
    0 18 * * 5  cd /path/to/trading-system && venv/bin/python3 -m paper_trading.decay_weekly_scan >> logs/decay_scan.log 2>&1
    0 16 * * 1-5 cd /path/to/trading-system && venv/bin/python3 -m paper_trading.decay_weekly_scan --daily >> logs/decay_daily.log 2>&1
"""

import argparse
import json
import logging
import sys
from datetime import date

from paper_trading.decay_auto_manager import DecayAutoManager
from models.signal_decay_detector import SignalDecayDetector


def main():
    parser = argparse.ArgumentParser(
        description='Signal Decay Auto-Detection Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--daily', action='store_true',
        help='Run daily quick check (loss streaks, Sharpe) instead of weekly',
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show all active size overrides',
    )
    parser.add_argument(
        '--signal', type=str, default=None,
        help='Analyze a single signal (e.g., KAUFMAN_DRY_20)',
    )
    parser.add_argument(
        '--history', type=str, default=None,
        help='Show decay history for a signal',
    )
    parser.add_argument(
        '--override', nargs=3, metavar=('SIGNAL', 'FACTOR', 'DAYS'),
        help='Set manual override: --override SIGNAL_ID 0.5 7',
    )
    parser.add_argument(
        '--clear-override', type=str, default=None,
        metavar='SIGNAL',
        help='Clear all overrides for a signal',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run scan without DB writes',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S',
    )

    manager = DecayAutoManager()

    try:
        # ── Status: show active overrides ──
        if args.status:
            overrides = manager.get_override_status()
            print(f"\nActive Size Overrides ({len(overrides)}):")
            if not overrides:
                print("  (none)")
            else:
                print(f"  {'Signal':<28s} {'Type':<14s} {'Factor':>6s} "
                      f"{'Expires':<20s} {'By':<8s} Reason")
                print(f"  {'─' * 90}")
                for o in overrides:
                    print(
                        f"  {o['signal_id']:<28s} {o['override_type']:<14s} "
                        f"{o['factor']:>6.2f} {o['expires_at']:<20s} "
                        f"{o['created_by']:<8s} {o['reason'][:40]}"
                    )
            return

        # ── History for a single signal ──
        if args.history:
            history = manager.get_signal_history(args.history, limit=15)
            print(f"\nDecay History: {args.history} ({len(history)} entries)")
            if not history:
                print("  (no history)")
            else:
                print(f"  {'Date':<12s} {'Type':<8s} {'Status':<12s} "
                      f"{'Sev':>3s} {'Shrp':>5s} {'WR':>5s} {'DD':>5s} "
                      f"{'CL':>3s} {'P(cp)':>5s} {'Action':<16s}")
                print(f"  {'─' * 85}")
                for h in history:
                    print(
                        f"  {h['scan_date']:<12s} {h['scan_type']:<8s} "
                        f"{h['status']:<12s} {h['severity']:>3d} "
                        f"{(h['sharpe_20'] or 0):>5.2f} "
                        f"{(h['win_rate_20'] or 0):>5.1%} "
                        f"{(h['max_drawdown_20'] or 0):>5.1%} "
                        f"{(h['consec_losses'] or 0):>3d} "
                        f"{(h['bocd_cp_prob'] or 0):>5.3f} "
                        f"{(h['action'] or 'NONE'):<16s}"
                    )
            return

        # ── Manual override ──
        if args.override:
            signal_id = args.override[0]
            factor = float(args.override[1])
            days = int(args.override[2])
            manager.set_manual_override(signal_id, factor, days)
            print(f"Override set: {signal_id} factor={factor} for {days} days")
            return

        # ── Clear override ──
        if args.clear_override:
            manager.clear_manual_override(args.clear_override)
            print(f"Cleared overrides for {args.clear_override}")
            return

        # ── Single signal analysis ──
        if args.signal:
            detector = SignalDecayDetector()
            try:
                report = detector.analyze_signal(args.signal, lookback_trades=50)
                print(f"\nDecay Report: {args.signal}")
                print(f"{'─' * 60}")
                print(report.summary_line())
                print(f"\n  Status: {report.status}")
                print(f"  Regime: {report.regime}")
                if report.flags:
                    print(f"  Flags:")
                    for f in report.flags:
                        print(f"    - {f}")
                if report.regime_decay_reason:
                    print(f"  Regime decay: {report.regime_decay_reason}")
                print(f"\n  Full report:")
                print(json.dumps(report.to_dict(), indent=2))
            finally:
                detector.close()
            return

        # ── Daily quick check ──
        if args.daily:
            result = manager.run_daily_quick_check(dry_run=args.dry_run)
            print(f"\n{result.summary()}")
            return

        # ── Weekly scan (default) ──
        result = manager.run_weekly_scan(dry_run=args.dry_run)
        print(f"\n{result.summary()}")

        # Print report table
        if result.reports:
            print(f"\n{'Signal':<28s} {'Typ':>3s} {'St':>3s} "
                  f"{'Trd':>3s} {'Shrp':>5s} {'WR':>5s} {'DD':>5s} "
                  f"{'CL':>2s} {'P(cp)':>5s} {'CU':>2s} {'Flg':>3s}")
            print('─' * 80)
            for sid in sorted(result.reports.keys()):
                r = result.reports[sid]
                print(r.summary_line())

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        manager.close()


if __name__ == '__main__':
    main()
