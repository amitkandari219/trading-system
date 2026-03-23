"""
Tier 3 ML/AI Signal Pipeline.

Evaluates all 7 Tier 3 signals (Mamba, TFT, RL, GNN, NLP, AMFI, CC Spending)
and stores results. Runs after Tier 1+2 enhanced signals.

Run:
    python -m scripts.tier3_signal_pipeline
    python -m scripts.tier3_signal_pipeline --dry-run
    python -m scripts.tier3_signal_pipeline --date 2026-03-20

Schedule: Weekdays 8:25 AM IST (after enhanced_signal_pipeline at 8:15)
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_db_conn():
    """Get database connection."""
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        return psycopg2.connect(DATABASE_DSN)
    except Exception as e:
        logger.warning("DB connection failed: %s", e)
        return None


def evaluate_mamba(conn, trade_date: date) -> Dict:
    """Evaluate Mamba/S4 regime detection."""
    try:
        from models.mamba_regime import MambaRegimeDetector
        detector = MambaRegimeDetector(db_conn=conn)
        return detector.evaluate(trade_date=trade_date)
    except Exception as e:
        logger.error("Mamba evaluation failed: %s", e)
        return {'signal_id': 'MAMBA_REGIME', 'direction': 'NEUTRAL',
                'confidence': 0.0, 'size_modifier': 1.0, 'error': str(e)}


def evaluate_tft(conn, trade_date: date) -> Dict:
    """Evaluate TFT multi-horizon forecast."""
    try:
        from models.tft_forecaster import TFTForecaster
        tft = TFTForecaster(db_conn=conn)
        return tft.evaluate(trade_date=trade_date)
    except Exception as e:
        logger.error("TFT evaluation failed: %s", e)
        return {'signal_id': 'TFT_FORECAST', 'direction': 'NEUTRAL',
                'confidence': 0.0, 'size_modifier': 1.0, 'error': str(e)}


def evaluate_rl_sizer(conn, trade_date: date, all_signals: Dict) -> Dict:
    """Evaluate RL position sizer (needs other signal outputs)."""
    try:
        from models.rl_position_sizer import RLPositionSizer
        sizer = RLPositionSizer(db_conn=conn)
        return sizer.evaluate(signals=all_signals)
    except Exception as e:
        logger.error("RL sizer evaluation failed: %s", e)
        return {'signal_id': 'RL_POSITION_SIZER', 'size_modifier': 1.0,
                'confidence': 0.0, 'error': str(e)}


def evaluate_gnn(conn, trade_date: date) -> Dict:
    """Evaluate GNN sector rotation."""
    try:
        from models.gnn_sector_rotation import GNNSectorRotation
        gnn = GNNSectorRotation(db_conn=conn)
        return gnn.evaluate(trade_date=trade_date)
    except Exception as e:
        logger.error("GNN evaluation failed: %s", e)
        return {'signal_id': 'GNN_SECTOR_ROTATION', 'direction': 'NEUTRAL',
                'confidence': 0.0, 'size_modifier': 1.0, 'error': str(e)}


def evaluate_nlp(conn, trade_date: date) -> Dict:
    """Evaluate NLP sentiment."""
    try:
        from models.nlp_sentiment import NLPSentiment
        nlp = NLPSentiment(db_conn=conn)
        return nlp.evaluate(trade_date=trade_date)
    except Exception as e:
        logger.error("NLP evaluation failed: %s", e)
        return {'signal_id': 'NLP_SENTIMENT', 'direction': 'NEUTRAL',
                'confidence': 0.0, 'size_modifier': 1.0, 'error': str(e)}


def evaluate_amfi(conn, trade_date: date) -> Dict:
    """Evaluate AMFI mutual fund flows."""
    try:
        from data.amfi_mf_flows import AMFIMutualFundSignal
        sig = AMFIMutualFundSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        return ctx.to_dict()
    except Exception as e:
        logger.error("AMFI evaluation failed: %s", e)
        return {'signal_id': 'AMFI_MF_FLOW', 'direction': 'NEUTRAL',
                'confidence': 0.0, 'size_modifier': 1.0, 'error': str(e)}


def evaluate_cc_spending(conn, trade_date: date) -> Dict:
    """Evaluate credit card spending."""
    try:
        from data.credit_card_spending import CreditCardSpendingSignal
        sig = CreditCardSpendingSignal(db_conn=conn)
        ctx = sig.evaluate(trade_date=trade_date)
        return ctx.to_dict()
    except Exception as e:
        logger.error("CC spending evaluation failed: %s", e)
        return {'signal_id': 'CREDIT_CARD_SPENDING', 'direction': 'NEUTRAL',
                'confidence': 0.0, 'size_modifier': 1.0, 'error': str(e)}


def evaluate_all_tier3(trade_date: date, dry_run: bool = False) -> Dict[str, Dict]:
    """Evaluate all Tier 3 signals."""
    conn = get_db_conn()
    results = {}

    logger.info("=" * 60)
    logger.info("Tier 3 ML/AI Signal Pipeline — %s", trade_date)
    logger.info("=" * 60)

    # Independent signals first
    evaluators = [
        ('MAMBA_REGIME', evaluate_mamba),
        ('TFT_FORECAST', evaluate_tft),
        ('GNN_SECTOR_ROTATION', evaluate_gnn),
        ('NLP_SENTIMENT', evaluate_nlp),
        ('AMFI_MF_FLOW', evaluate_amfi),
        ('CREDIT_CARD_SPENDING', evaluate_cc_spending),
    ]

    for signal_id, evaluator in evaluators:
        logger.info("Evaluating %s...", signal_id)
        try:
            result = evaluator(conn, trade_date)
            results[signal_id] = result
            direction = result.get('direction', 'N/A')
            confidence = result.get('confidence', 0.0)
            size_mod = result.get('size_modifier', 1.0)
            logger.info("  %s → dir=%s conf=%.3f size=%.2f",
                       signal_id, direction, confidence, size_mod)
        except Exception as e:
            logger.error("  %s FAILED: %s", signal_id, e)
            results[signal_id] = {'error': str(e)}

    # RL sizer depends on other signals
    logger.info("Evaluating RL_POSITION_SIZER (depends on other signals)...")
    rl_result = evaluate_rl_sizer(conn, trade_date, results)
    results['RL_POSITION_SIZER'] = rl_result
    logger.info("  RL_POSITION_SIZER → size=%.3f method=%s",
               rl_result.get('size_modifier', 1.0),
               rl_result.get('method', 'unknown'))

    # Store results
    if not dry_run and conn:
        store_results(conn, trade_date, results)
        logger.info("Results stored to tier3_signal_log")
    elif dry_run:
        logger.info("[DRY RUN] Would store %d signal results", len(results))

    # Summary
    logger.info("-" * 60)
    logger.info("Tier 3 Summary:")
    for sig_id, r in results.items():
        if 'error' not in r:
            logger.info("  %-25s dir=%-8s conf=%.3f size=%.2f",
                       sig_id,
                       r.get('direction', 'N/A'),
                       r.get('confidence', 0.0),
                       r.get('size_modifier', 1.0))

    if conn:
        conn.close()

    return results


def store_results(conn, trade_date: date, results: Dict[str, Dict]):
    """Store Tier 3 results to database."""
    try:
        with conn.cursor() as cur:
            for signal_id, result in results.items():
                if 'error' in result:
                    continue
                cur.execute(
                    """
                    INSERT INTO tier3_signal_log
                        (trade_date, signal_id, direction, confidence, size_modifier, details)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        trade_date,
                        signal_id,
                        result.get('direction', 'NEUTRAL'),
                        result.get('confidence', 0.0),
                        result.get('size_modifier', 1.0),
                        json.dumps(result, default=str),
                    )
                )
        conn.commit()
    except Exception as e:
        logger.error("Failed to store results: %s", e)
        conn.rollback()


def send_telegram_summary(results: Dict[str, Dict]):
    """Send Tier 3 summary to Telegram."""
    try:
        from notifications.telegram_bot import send_message
        lines = ["📊 *Tier 3 ML/AI Signals*\n"]
        for sig_id, r in results.items():
            if 'error' in r:
                lines.append(f"❌ {sig_id}: ERROR")
                continue
            emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
                r.get('direction', 'NEUTRAL'), '⚪')
            lines.append(
                f"{emoji} {sig_id}: {r.get('direction', 'N/A')} "
                f"(conf={r.get('confidence', 0):.2f}, "
                f"size={r.get('size_modifier', 1.0):.2f})"
            )
        send_message('\n'.join(lines))
    except Exception as e:
        logger.debug("Telegram send failed: %s", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tier 3 ML/AI Signal Pipeline')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--date', type=str, default=None,
                       help='Trade date (YYYY-MM-DD)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s'
    )

    trade_date = date.fromisoformat(args.date) if args.date else date.today()
    results = evaluate_all_tier3(trade_date, dry_run=args.dry_run)

    if not args.dry_run:
        send_telegram_summary(results)
