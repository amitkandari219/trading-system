"""
Enhanced Signal Pipeline — runs all 11 new signals daily.

Scheduled at 8:15 AM IST (before global pre-market at 8:30 AM).
Fetches data, evaluates signals, stores results, sends Telegram alert.

Usage:
    python -m scripts.enhanced_signal_pipeline
    python -m scripts.enhanced_signal_pipeline --dry-run
"""

import json
import logging
import os
import sys
import argparse
from datetime import date

import psycopg2
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)


def get_db_conn():
    dsn = os.environ.get('DATABASE_DSN', DATABASE_DSN)
    return psycopg2.connect(dsn)


def evaluate_all_signals(conn, trade_date=None):
    """
    Evaluate all enhanced signals and return results dict.
    """
    if trade_date is None:
        trade_date = date.today()

    results = {}

    # 1. PCR Autotrender
    try:
        from signals.pcr_signal import PCRAutotrender
        sig = PCRAutotrender(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['PCR_AUTOTRENDER'] = ctx.to_dict()
        logger.info("PCR: %s dir=%s size=%.2f", ctx.pcr_zone, ctx.direction, ctx.size_modifier)
    except Exception as e:
        logger.error("PCR failed: %s", e)

    # 2. Rollover Analysis
    try:
        from signals.rollover_signal import RolloverSignal
        sig = RolloverSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['ROLLOVER_ANALYSIS'] = ctx.to_dict()
        logger.info("Rollover: %.1f%% %s dir=%s", ctx.rollover_pct, ctx.buildup_type, ctx.direction)
    except Exception as e:
        logger.error("Rollover failed: %s", e)

    # 3. FII Futures OI
    try:
        from signals.fii_futures_oi import FIIFuturesOI
        sig = FIIFuturesOI(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['FII_FUTURES_OI'] = ctx.to_dict()
        logger.info("FII OI: ratio=%.3f dir=%s size=%.2f", ctx.fii_ratio, ctx.direction, ctx.size_modifier)
    except Exception as e:
        logger.error("FII Futures OI failed: %s", e)

    # 4. Delivery %
    try:
        from signals.delivery_signal import DeliverySignal
        sig = DeliverySignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['DELIVERY_PCT'] = ctx.to_dict()
        logger.info("Delivery: %.1f%% %s dir=%s", ctx.delivery_pct, ctx.classification, ctx.direction)
    except Exception as e:
        logger.error("Delivery failed: %s", e)

    # 5. Sentiment (MMI + GTrends)
    try:
        from signals.sentiment_signal import SentimentSignal
        sig = SentimentSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['SENTIMENT_COMPOSITE'] = ctx.to_dict()
        logger.info("Sentiment: MMI=%.0f %s dir=%s", ctx.mmi_value, ctx.mmi_zone, ctx.combined_direction)
    except Exception as e:
        logger.error("Sentiment failed: %s", e)

    # 6. Bond Yield Spread
    try:
        from signals.bond_yield_signal import BondYieldSignal
        sig = BondYieldSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['BOND_YIELD_SPREAD'] = ctx.to_dict()
        logger.info("Bond: spread=%.2f%% %s dir=%s", ctx.spread, ctx.spread_zone, ctx.direction)
    except Exception as e:
        logger.error("Bond Yield failed: %s", e)

    # 7. Gamma Exposure
    try:
        from signals.gamma_exposure import GammaExposureSignal
        sig = GammaExposureSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['GAMMA_EXPOSURE'] = ctx.to_dict()
        logger.info("Gamma: %s regime=%s", ctx.gamma_zone, ctx.regime)
    except Exception as e:
        logger.error("Gamma Exposure failed: %s", e)

    # 8. Vol Term Structure
    try:
        from signals.vol_term_structure import VolTermStructureSignal
        sig = VolTermStructureSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['VOL_TERM_STRUCTURE'] = ctx.to_dict()
        logger.info("VolTS: %s spread=%.1f%%", ctx.structure_state, ctx.term_spread * 100)
    except Exception as e:
        logger.error("Vol Term Structure failed: %s", e)

    # 9. RBI Macro Filter
    try:
        from signals.rbi_macro_filter import RBIMacroFilter
        filt = RBIMacroFilter(db_conn=conn)
        ctx = filt.evaluate(trade_date=trade_date)
        results['RBI_MACRO_FILTER'] = ctx.to_dict()
        logger.info("Macro: event=%s window=%s size=%.2f",
                     ctx.nearest_event, ctx.in_event_window, ctx.size_modifier)
    except Exception as e:
        logger.error("RBI Macro failed: %s", e)

    # 10. Order Flow (only available after market open)
    try:
        from signals.order_flow import OrderFlowSignal
        sig = OrderFlowSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        results['ORDER_FLOW_IMBALANCE'] = ctx.to_dict()
        logger.info("OrderFlow: OFI=%.3f %s dir=%s", ctx.ofi_15min, ctx.ofi_zone, ctx.direction)
    except Exception as e:
        logger.error("Order Flow failed: %s", e)

    # 11. XGBoost Meta-Learner (combines all above)
    try:
        from signals.xgboost_meta import XGBoostMetaLearner
        meta = XGBoostMetaLearner(db_conn=conn)
        meta_input = dict(results)
        meta_input['context'] = _build_context(conn, trade_date)
        meta_result = meta.predict(meta_input)
        results['XGBOOST_META_LEARNER'] = meta_result
        logger.info("Meta: dir=%s size=%.3f method=%s",
                     meta_result['direction'], meta_result['size_modifier'], meta_result['method'])
    except Exception as e:
        logger.error("Meta-learner failed: %s", e)

    return results


def _build_context(conn, trade_date):
    """Build context dict for meta-learner."""
    ctx = {
        'day_of_week': trade_date.weekday(),
        'dte_weekly': (3 - trade_date.weekday()) % 7,
        'dte_monthly': 15,
        'nifty_5d_return': 0.0,
        'nifty_20d_return': 0.0,
    }

    try:
        with conn.cursor() as cur:
            # VIX
            cur.execute("SELECT close FROM india_vix WHERE date <= %s ORDER BY date DESC LIMIT 1",
                        (trade_date,))
            row = cur.fetchone()
            ctx['vix_level'] = float(row[0]) if row else 15.0

            # VIX regime
            vix = ctx['vix_level']
            if vix < 13: ctx['vix_regime'] = 'CALM'
            elif vix < 18: ctx['vix_regime'] = 'NORMAL'
            elif vix < 24: ctx['vix_regime'] = 'ELEVATED'
            elif vix < 32: ctx['vix_regime'] = 'HIGH_VOL'
            else: ctx['vix_regime'] = 'CRISIS'

            # Nifty returns
            cur.execute(
                "SELECT close FROM nifty_daily WHERE date <= %s ORDER BY date DESC LIMIT 21",
                (trade_date,))
            rows = cur.fetchall()
            if len(rows) >= 6:
                ctx['nifty_5d_return'] = (rows[0][0] - rows[5][0]) / rows[5][0] * 100
            if len(rows) >= 21:
                ctx['nifty_20d_return'] = (rows[0][0] - rows[20][0]) / rows[20][0] * 100
    except Exception as e:
        logger.warning("Context build partial: %s", e)
        ctx.setdefault('vix_level', 15.0)
        ctx.setdefault('vix_regime', 'NORMAL')

    return ctx


def store_results(conn, results, trade_date):
    """Store signal evaluations in database."""
    try:
        with conn.cursor() as cur:
            for signal_id, result in results.items():
                direction = result.get('direction', 'NEUTRAL')
                confidence = result.get('confidence', 0.0)
                size_mod = result.get('size_modifier', 1.0)

                cur.execute(
                    """
                    INSERT INTO signal_evaluations
                        (eval_date, signal_id, result_json, direction, confidence, size_modifier)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (eval_date, signal_id)
                    DO UPDATE SET result_json = EXCLUDED.result_json,
                                  direction = EXCLUDED.direction,
                                  confidence = EXCLUDED.confidence,
                                  size_modifier = EXCLUDED.size_modifier
                    """,
                    (trade_date, signal_id, json.dumps(result, default=str),
                     direction, float(confidence), float(size_mod))
                )
        conn.commit()
        logger.info("Stored %d signal evaluations for %s", len(results), trade_date)
    except Exception as e:
        logger.error("Failed to store results: %s", e)
        conn.rollback()


def send_telegram_summary(results, trade_date):
    """Send Telegram summary of all signals."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        logger.info("Telegram not configured — skipping alert")
        return

    # Build summary
    lines = [f"📊 Enhanced Signals — {trade_date}"]

    for sig_id, result in sorted(results.items()):
        direction = result.get('direction', '?')
        size = result.get('size_modifier', 1.0)
        conf = result.get('confidence', 0.0)
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(direction, '⚪')
        lines.append(f"  {emoji} {sig_id}: {direction} {size:.2f}x ({conf:.0%})")

    # Meta-learner summary
    meta = results.get('XGBOOST_META_LEARNER', {})
    if meta:
        lines.append(f"\n🤖 Meta: {meta.get('direction', '?')} "
                     f"{meta.get('size_modifier', 1.0):.3f}x "
                     f"[{meta.get('method', '?')}]")

    message = '\n'.join(lines)

    try:
        requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id': chat_id, 'text': message},
            timeout=10,
        )
        logger.info("Telegram alert sent")
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Signal Pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Evaluate without storing')
    parser.add_argument('--date', type=str, help='Trade date (YYYY-MM-DD)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s',
    )

    trade_date = date.fromisoformat(args.date) if args.date else date.today()
    logger.info("Enhanced Signal Pipeline — %s %s",
                trade_date, "(DRY RUN)" if args.dry_run else "")

    conn = get_db_conn()

    try:
        # Evaluate all signals
        results = evaluate_all_signals(conn, trade_date)
        logger.info("Evaluated %d signals", len(results))

        if not args.dry_run:
            store_results(conn, results, trade_date)
            send_telegram_summary(results, trade_date)
        else:
            print(json.dumps(results, indent=2, default=str))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
