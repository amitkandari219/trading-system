"""
Daily paper trading pipeline — run after market close.

Runs two parallel modes:
  PAPER_CONTROL:  DRY_20 alone (baseline)
  PAPER_SCORING:  Scoring system (DRY_20 weighted + DRY_12 + DRY_16)

Shadow signals (GUJRAL_DRY_7) tracked for regime detection.

Usage:
    python -m paper_trading.daily_run              # full run
    python -m paper_trading.daily_run --dry-run    # no DB writes
    python -m paper_trading.daily_run --date 2026-03-14
    python -m paper_trading.daily_run --backdate 2025-09-01  # replay from date

Schedule via cron:
    35 15 * * 1-5 cd /path/to/trading-system && venv/bin/python3 -m paper_trading.daily_run >> logs/paper_trading.log 2>&1
"""

import json
import logging
import os
from datetime import date, datetime, timedelta

import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN
from paper_trading.signal_compute import SignalComputer
from paper_trading.pnl_tracker import PnLTracker
from paper_trading.vix_monitor import VIXMonitor
from paper_trading.scoring_engine import ScoringEngine
from paper_trading.regime_detector import GujralRegimeDetector
from paper_trading.combination_engine import CombinationEngine

logger = logging.getLogger(__name__)


def format_telegram_digest(as_of, compute_result, scoring_result,
                           control_action, combo_result, regime_info,
                           vix_result, pnl_control, pnl_scoring):
    """Format the daily Telegram digest message."""
    indicators = compute_result.get('indicators', {})
    close = indicators.get('close', 0)
    prev_close = indicators.get('prev_close', 0)
    change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0

    lines = [
        f"*NIFTY F&O DAILY DIGEST*",
        f"Date: {as_of} | Nifty: {close:.0f} ({change_pct:+.1f}%)",
        f"VIX: {indicators.get('india_vix', 0):.1f}",
        "",
        f"*--- SCORING SYSTEM ---*",
        f"Score: {scoring_result.get('score', 0)} | "
        f"Position: {scoring_result.get('position', 'FLAT')}",
    ]

    action = scoring_result.get('action')
    if action:
        lines.append(f"Action: {action} (size: {scoring_result.get('size', 0)}x)")
        lines.append(f"Reason: {scoring_result.get('reason', '')}")
    else:
        lines.append("No action today")

    lines.append(f"Regime: {regime_info.get('regime', 'UNKNOWN')} "
                 f"(Gujral streak: {regime_info.get('streak', 'N/A')})")
    lines.append(f"Confidence: {regime_info.get('confidence', 'MEDIUM')}")
    lines.append("")

    lines.append(f"*--- CONTROL (DRY_20 ALONE) ---*")
    if control_action:
        lines.append(f"Action: {control_action}")
    else:
        lines.append("No action")
    lines.append("")

    # Combination
    lines.append(f"*--- COMBINATION (Grimes+Kaufman SEQ_5) ---*")
    if combo_result:
        combo_action = combo_result.get('action')
        if combo_action:
            lines.append(f"Action: {combo_action}")
            lines.append(f"Reason: {combo_result.get('reason', '')}")
        elif combo_result.get('grimes_fired'):
            lines.append(f"Grimes fired {combo_result.get('grimes_direction')} — waiting Kaufman")
            lines.append(f"Pending: {combo_result.get('pending_days_remaining', 0)} days remaining")
        else:
            pending = combo_result.get('pending_days_remaining', 0)
            if pending > 0:
                lines.append(f"Pending confirmation: {pending} days remaining")
            else:
                lines.append("Idle (no pending signal)")
    lines.append("")

    # Shadow signals
    actions = compute_result.get('signal_actions', {})
    lines.append(f"*--- SHADOW SIGNALS ---*")
    dry12_action = actions.get('KAUFMAN_DRY_12', {}).get('action', 'silent')
    dry16_action = actions.get('KAUFMAN_DRY_16', {}).get('action', 'silent')
    guj7_action = actions.get('GUJRAL_DRY_7', {}).get('action', 'silent')
    lines.append(f"DRY_12: {dry12_action or 'silent'}")
    lines.append(f"DRY_16: {dry16_action or 'silent'}")
    lines.append(f"GUJRAL_DRY_7: {guj7_action or 'silent'} → streak: {regime_info.get('streak', 'N/A')}")
    lines.append("")

    # Open positions
    lines.append(f"*--- OPEN POSITIONS ---*")
    lines.append(f"Control: {pnl_control.get('positions_open', 0)} | "
                 f"Scoring: {pnl_scoring.get('positions_open', 0)}")
    lines.append("")

    # P&L
    lines.append(f"*--- TODAY'S P&L ---*")
    lines.append(f"Control: {pnl_control.get('day_pnl_pts', 0):+.1f} pts | "
                 f"Scoring: {pnl_scoring.get('day_pnl_pts', 0):+.1f} pts")

    if vix_result and vix_result.get('alert_level'):
        lines.append(f"\n⚠️ *VIX Alert:* {vix_result['message']}")

    return '\n'.join(lines)


def run_single_day(conn, as_of, dry_run=False, scoring_engine=None):
    """Run the full pipeline for a single day."""
    computer = SignalComputer(db_conn=conn)
    vix_monitor = VIXMonitor(db_conn=conn)
    detector = GujralRegimeDetector(db_conn=conn)

    if scoring_engine is None:
        scoring_engine = ScoringEngine()

    combo_engine = CombinationEngine()
    if not dry_run:
        try:
            combo_engine.load_state(conn)
        except Exception:
            pass

    # Step 1: Compute all signals
    compute_result = computer.run(as_of_date=as_of, dry_run=dry_run)
    signal_actions = compute_result.get('signal_actions', {})

    # Step 2: Scoring engine
    scoring_input = {}
    for sid in ['KAUFMAN_DRY_20', 'KAUFMAN_DRY_12', 'KAUFMAN_DRY_16']:
        key = sid.replace('KAUFMAN_', '')
        scoring_input[key] = signal_actions.get(sid, {'action': None})
    scoring_result = scoring_engine.update(scoring_input)

    # Step 3: Regime detection
    regime_info = detector.get_streak()
    confidence = detector.get_confidence_label(
        regime_info['regime'], scoring_result.get('score', 0))
    size_mult = detector.get_size_multiplier(confidence)
    regime_info['confidence'] = confidence
    regime_info['size_multiplier'] = size_mult

    # Step 4: Control mode (DRY_20 alone)
    dry20_action = signal_actions.get('KAUFMAN_DRY_20', {}).get('action')
    control_action = None
    if dry20_action == 'ENTER_LONG':
        control_action = f"ENTER_LONG @ {compute_result['indicators']['close']:.0f}"
    elif dry20_action == 'EXIT':
        control_action = "EXIT"

    # Step 5: Record scoring trades
    scoring_action = scoring_result.get('action')
    if scoring_action and not dry_run:
        if scoring_action in ('ENTER_LONG', 'ENTER_SHORT'):
            direction = scoring_action.replace('ENTER_', '')
            computer._record_entry(
                'SCORING_SYSTEM',
                {'direction': direction, 'price': compute_result['indicators']['close']},
                None, as_of, compute_result.get('indicators', {}),
                trade_type='PAPER_SCORING',
                size_multiplier=scoring_result.get('size', 1.0) * size_mult,
                confidence_label=confidence,
            )

    # Step 6: Record control trades
    if control_action and not dry_run:
        for entry in compute_result.get('entries', []):
            if entry['signal_id'] == 'KAUFMAN_DRY_20':
                computer._record_entry(
                    'KAUFMAN_DRY_20', entry, None, as_of,
                    compute_result.get('indicators', {}),
                    trade_type='PAPER_CONTROL',
                    size_multiplier=size_mult,
                    confidence_label=confidence,
                )

    # Step 6b: Combination engine (GRIMES+KAUFMAN SEQ_5)
    combo_result = {'action': None, 'grimes_fired': False, 'pending_days_remaining': 0}
    indicators = compute_result.get('indicators', {})
    if indicators.get('close'):
        combo_row = pd.Series(indicators)
        combo_prev = pd.Series({
            'high': indicators.get('prev_close', 0) * 1.005,  # approximate
            'low': indicators.get('prev_close', 0) * 0.995,
            'close': indicators.get('prev_close', 0),
            'volume': indicators.get('prev_volume', 0),
            'prev_high': 0, 'prev_low': 0,
        })

        # Use the actual df for accurate prev values
        df_data = computer._load_market_data(as_of)
        if df_data is not None and len(df_data) >= 2:
            from backtest.indicators import add_all_indicators, historical_volatility
            df_ind = add_all_indicators(df_data)
            df_ind['hvol_6'] = historical_volatility(df_ind['close'], period=6)
            df_ind['hvol_100'] = historical_volatility(df_ind['close'], period=100)
            df_ind['date'] = df_data['date']
            df_ind['india_vix'] = df_data['india_vix']
            from regime_labeler import RegimeLabeler
            regime_dict = RegimeLabeler().label_full_history(df_data)
            df_ind['regime'] = df_ind['date'].map(regime_dict).fillna('UNKNOWN')

            today_row = df_ind.iloc[-1]
            prev_row = df_ind.iloc[-2]
            combo_result = combo_engine.update(as_of, today_row, prev_row)

            # Record combination trades
            combo_action = combo_result.get('action')
            if combo_action and not dry_run:
                if combo_action in ('ENTER_LONG', 'ENTER_SHORT'):
                    direction = combo_action.replace('ENTER_', '')
                    computer._record_entry(
                        'COMBO_GRIMES_KAUFMAN',
                        {'direction': direction, 'price': float(today_row['close'])},
                        None, as_of, indicators,
                        trade_type='PAPER_COMBINATION',
                    )

    if not dry_run:
        try:
            combo_engine.save_state(conn)
        except Exception:
            pass

    # Step 7: VIX check
    vix_result = vix_monitor.check(
        current_vix=compute_result.get('indicators', {}).get('india_vix'))

    # Step 8: P&L
    tracker = PnLTracker(db_conn=conn)
    pnl_control = {'day_pnl_pts': 0, 'positions_open': 0}
    pnl_scoring = {'day_pnl_pts': 0, 'positions_open': 0}

    return {
        'date': str(as_of),
        'compute_result': compute_result,
        'scoring_result': scoring_result,
        'combo_result': combo_result,
        'control_action': control_action,
        'regime_info': regime_info,
        'vix_result': vix_result,
        'pnl_control': pnl_control,
        'pnl_scoring': pnl_scoring,
        'digest': format_telegram_digest(
            as_of, compute_result, scoring_result, control_action,
            combo_result, regime_info, vix_result, pnl_control, pnl_scoring),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Daily paper trading pipeline')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--date', type=str, help='YYYY-MM-DD')
    parser.add_argument('--backdate', type=str, help='Replay from YYYY-MM-DD to today')
    parser.add_argument('--no-telegram', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    conn = psycopg2.connect(DATABASE_DSN)

    if args.backdate:
        # Replay mode
        import pandas as pd
        start = date.fromisoformat(args.backdate)
        cur = conn.cursor()
        cur.execute("SELECT date FROM nifty_daily WHERE date >= %s ORDER BY date", (start,))
        trading_days = [r[0] for r in cur.fetchall()]
        print(f"Backdate replay: {len(trading_days)} days from {start}")

        scoring_engine = ScoringEngine()
        control_pnl = 0
        scoring_pnl = 0

        for day in trading_days:
            result = run_single_day(conn, day, dry_run=True,
                                    scoring_engine=scoring_engine)
            sr = result['scoring_result']
            ca = result['control_action']

            # Track P&L from compute_result entries/exits
            for x in result['compute_result'].get('exits', []):
                if x['signal_id'] == 'KAUFMAN_DRY_20':
                    control_pnl += x.get('pnl', 0)

            if sr.get('score', 0) != 0 or sr.get('action'):
                print(f"  {day} Score={sr['score']:+d} Action={sr.get('action', '-'):12s} "
                      f"Size={sr.get('size', 0):.1f}x "
                      f"Regime={result['regime_info']['regime']} "
                      f"Conf={result['regime_info'].get('confidence', '?')}")

        print(f"\nBackdate complete.")
        print(f"Control (DRY_20) P&L: {control_pnl:+.1f} pts")

    else:
        # Single day mode
        as_of = date.fromisoformat(args.date) if args.date else date.today()
        print(f"\n{'='*60}")
        print(f"PAPER TRADING DAILY RUN — {as_of}")
        print(f"{'='*60}")

        result = run_single_day(conn, as_of, dry_run=args.dry_run)

        # Print digest
        print(result['digest'])

        # Print detailed compute result
        cr = result['compute_result']
        indicators = cr.get('indicators', {})
        print(f"\nDetailed indicators:")
        print(f"  stoch_k_5={indicators.get('stoch_k_5', 0):.1f} "
              f"sma_10={indicators.get('sma_10', 0):.0f} "
              f"hvol_6={indicators.get('hvol_6')}")
        print(f"  pivot={indicators.get('pivot', 0):.0f} "
              f"r1={indicators.get('r1', 0):.0f} "
              f"s1={indicators.get('s1', 0):.0f}")

        # Send Telegram
        if not args.dry_run and not args.no_telegram:
            token = os.environ.get('TELEGRAM_BOT_TOKEN')
            chat_id = os.environ.get('TELEGRAM_CHAT_ID')
            if token and chat_id:
                from monitoring.telegram_alerter import TelegramAlerter
                try:
                    alerter = TelegramAlerter(token, chat_id)
                    alerter.send('INFO', result['digest'])
                except Exception as e:
                    logger.error(f"Telegram send failed: {e}")

    conn.close()
    print(f"\nDaily run complete.")


if __name__ == '__main__':
    main()
