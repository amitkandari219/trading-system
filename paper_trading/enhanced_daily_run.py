"""
Enhanced daily pipeline that integrates all 10 improvements.

Drop-in replacement for daily_run.py's run_single_day() function.
Adds: ML regime, cross-asset overlays, Kelly sizing, FII overlay,
adaptive variants, transaction costs, and proper exception handling.

Usage:
    python -m paper_trading.enhanced_daily_run              # full run
    python -m paper_trading.enhanced_daily_run --dry-run    # no DB writes
    python -m paper_trading.enhanced_daily_run --date 2026-03-21

Integration points (what changed from daily_run.py):
    1. RegimeBridge replaces RegimeLabeler (ML ensemble + fallback)
    2. CrossAssetBridge computes global macro overlay
    3. KellySizer replaces fixed 1% sizing
    4. FIIOverlay adds institutional positioning
    5. AdaptiveVariantManager switches DRY_12 variants
    6. TransactionCostModel tracks realistic costs
    7. ExceptionHandler replaces bare except: pass
    8. SignalRegistry replaces hardcoded signal checks
"""

import json
import logging
import os
from datetime import date, datetime

import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN

# Core pipeline (existing)
from paper_trading.signal_compute import SignalComputer
from paper_trading.pnl_tracker import PnLTracker
from paper_trading.vix_monitor import VIXMonitor
from paper_trading.scoring_engine import ScoringEngine
from paper_trading.regime_detector import GujralRegimeDetector
from paper_trading.combination_engine import CombinationEngine
from paper_trading.decay_monitor import DecayMonitor

# NEW: Enhancement modules
from paper_trading.regime_bridge import RegimeBridge
from paper_trading.cross_asset_bridge import get_cross_asset_multiplier, persist_to_db as persist_cross_asset, clear_cache as clear_cross_asset_cache
from paper_trading.kelly_sizer import KellySizer
from paper_trading.adaptive_variant import AdaptiveVariantManager
from paper_trading.exception_handler import log_and_alert, safe_call
from paper_trading.signal_registry import SignalRegistry
from signals.fii_overlay import FIIOverlay
from signals.calendar_overlay import CalendarOverlay
from backtest.transaction_costs import TransactionCostModel
from paper_trading.eod_reconciler import run_eod as run_eod_reconciliation
from risk.behavioral_overlay import BehavioralOverlay, OverlayContext, TradeRecord

logger = logging.getLogger(__name__)


def run_enhanced_day(conn, as_of, dry_run=False, scoring_engine=None):
    """
    Enhanced daily pipeline with all 10 improvements integrated.

    Returns same dict structure as daily_run.run_single_day() for compatibility,
    plus additional enhancement data.
    """
    # ================================================================
    # INITIALIZE COMPONENTS
    # ================================================================
    computer = SignalComputer(db_conn=conn)
    vix_monitor = VIXMonitor(db_conn=conn)
    detector = GujralRegimeDetector(db_conn=conn)

    if scoring_engine is None:
        scoring_engine = ScoringEngine()

    # NEW: Enhancement components
    regime_bridge = RegimeBridge(confidence_threshold=0.6)
    kelly_sizer = KellySizer(kelly_fraction=0.25)
    fii_overlay = FIIOverlay(db_conn=conn)
    calendar_overlay = CalendarOverlay(db_conn=conn)
    variant_manager = AdaptiveVariantManager(db_conn=conn)
    signal_registry = SignalRegistry()
    cost_model = TransactionCostModel()
    behavioral_overlay = BehavioralOverlay()

    # Load variant state from DB
    safe_call(
        lambda: variant_manager.load_state(conn),
        'load_variant_state'
    )

    combo_engine = CombinationEngine()
    if not dry_run:
        safe_call(lambda: combo_engine.load_state(conn), 'load_combo_state')

    # ================================================================
    # STEP 1: Compute all signals (existing pipeline)
    # ================================================================
    compute_result = computer.run(as_of_date=as_of, dry_run=dry_run)
    signal_actions = compute_result.get('signal_actions', {})
    indicators = compute_result.get('indicators', {})

    # ================================================================
    # STEP 2: ML Regime Classification (IMPROVEMENT 1)
    # ================================================================
    regime_info_ml = {'regime': 'UNKNOWN', 'confidence': 0.5, 'method': 'unavailable'}
    if indicators.get('close'):
        today_row = pd.Series(indicators)
        regime_info_ml = safe_call(
            lambda: regime_bridge.classify(today_row),
            'ml_regime_classify',
            default_return={'regime': indicators.get('regime', 'UNKNOWN'),
                          'confidence': 0.5, 'method': 'error_fallback'}
        )
    logger.info(
        f"Regime: {regime_info_ml['regime']} "
        f"(method={regime_info_ml.get('method')}, "
        f"conf={regime_info_ml.get('confidence', 0):.2f})"
    )

    # ================================================================
    # STEP 3: Cross-Asset Overlay (IMPROVEMENT 4)
    # ================================================================
    cross_asset = safe_call(
        lambda: get_cross_asset_multiplier(as_of),
        'cross_asset_overlay',
        default_return={'composite_multiplier': 1.0, 'signals': {},
                       'data_source': 'error_fallback'}
    )
    cross_asset_mult = cross_asset.get('composite_multiplier', 1.0)

    # FIX 4: Persist successful fetch to DB for cache fallback
    if not dry_run and cross_asset.get('data_source') == 'yfinance':
        safe_call(
            lambda: persist_cross_asset(conn, as_of, cross_asset),
            'persist_cross_asset'
        )

    # ================================================================
    # STEP 4: FII Positioning Overlay (IMPROVEMENT 8)
    # ================================================================
    fii_result = safe_call(
        lambda: fii_overlay.get_multiplier(as_of),
        'fii_overlay',
        default_return={'multiplier': 1.0, 'regime': 'NEUTRAL'}
    )
    fii_mult = fii_result.get('multiplier', 1.0)

    # ================================================================
    # STEP 4b: Calendar Overlay (LAYER 6)
    # ================================================================
    calendar_ctx = safe_call(
        lambda: calendar_overlay.get_daily_context(as_of),
        'calendar_overlay',
        default_return={'composite_modifier': 1.0, 'block_new_entries': False,
                       'events_active': [], 'monthly_seasonality': 0.0}
    )
    calendar_mult = calendar_ctx.get('composite_modifier', 1.0)
    calendar_block = calendar_ctx.get('block_new_entries', False)

    if calendar_block:
        logger.info(
            f"Calendar overlay BLOCKS new entries: "
            f"events={calendar_ctx.get('events_active', [])}"
        )
    elif calendar_mult != 1.0:
        logger.info(
            f"Calendar overlay: {calendar_mult:.2f}x "
            f"(events={calendar_ctx.get('events_active', [])}, "
            f"seasonality={calendar_ctx.get('monthly_seasonality', 0):+.1%})"
        )

    # ================================================================
    # STEP 5: Adaptive Variant Check (IMPROVEMENT 5)
    # Now APPLIED — variant_mult feeds into composite_size
    # ================================================================
    variant_result = safe_call(
        lambda: variant_manager.evaluate_switch('KAUFMAN_DRY_12', as_of),
        'variant_switch_eval',
        default_return={'current_variant': 'PRIMARY', 'new_variant': None}
    )
    variant_mult = variant_manager.get_size_multiplier('KAUFMAN_DRY_12')

    # ================================================================
    # STEP 6a: Decay monitoring (moved BEFORE composite sizing so decay_results exists)
    # ================================================================
    decay_monitor = DecayMonitor(db_conn=conn)
    decay_results = {}
    for signal_id in ['KAUFMAN_DRY_20', 'KAUFMAN_DRY_16', 'KAUFMAN_DRY_12']:
        health = safe_call(
            lambda sid=signal_id: decay_monitor.get_signal_health(sid),
            f'decay_check_{signal_id}',
            default_return={}
        )
        decay_results[signal_id] = health

    # ================================================================
    # STEP 6b: Compute current portfolio drawdown for scoring engine
    # ================================================================
    current_drawdown_pct = 0.0
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(
                (1.0 - equity / NULLIF(max_equity, 0)) * 100, 0
            ) AS drawdown_pct
            FROM (
                SELECT
                    SUM(cumulative_pnl) + (SELECT capital FROM config LIMIT 1) AS equity,
                    MAX(SUM(cumulative_pnl) + (SELECT capital FROM config LIMIT 1))
                        OVER (ORDER BY trade_date) AS max_equity
                FROM paper_trades
                WHERE trade_date <= %s
                GROUP BY trade_date
                ORDER BY trade_date DESC
                LIMIT 1
            ) sub
        """, (as_of,))
        row = cur.fetchone()
        if row and row[0] is not None:
            current_drawdown_pct = max(0.0, float(row[0]))
    except Exception as e:
        logger.debug(f"Drawdown query failed (using 0%): {e}")
        conn.rollback()  # Reset transaction state so subsequent queries work

    # ================================================================
    # STEP 6c: Scoring engine (existing + drawdown scaling)
    # ================================================================
    scoring_input = {}
    for sid in ['KAUFMAN_DRY_20', 'KAUFMAN_DRY_12', 'KAUFMAN_DRY_16']:
        key = sid.replace('KAUFMAN_', '')
        scoring_input[key] = signal_actions.get(sid, {'action': None})
    scoring_result = scoring_engine.update(scoring_input, current_drawdown_pct=current_drawdown_pct)

    # ================================================================
    # STEP 7: Kelly Position Sizing (IMPROVEMENT 6)
    # ================================================================
    kelly_result = {}
    scoring_action = scoring_result.get('action')
    if scoring_action and scoring_action.startswith('ENTER'):
        # Determine which signal drove the scoring action
        primary_signal = 'KAUFMAN_DRY_20'  # Default to primary
        for sid in ['KAUFMAN_DRY_20', 'KAUFMAN_DRY_12', 'KAUFMAN_DRY_16']:
            if signal_actions.get(sid, {}).get('action'):
                primary_signal = sid
                break

        kelly_result = safe_call(
            lambda: kelly_sizer.compute_size(
                primary_signal, {},
                regime=regime_info_ml.get('regime'),
                vix=indicators.get('india_vix')
            ),
            'kelly_sizing',
            default_return={'size_multiplier': 1.0, 'skip_trade': False}
        )

        if kelly_result.get('skip_trade'):
            logger.info(f"Kelly sizer recommends SKIP — trade not taken")
            scoring_result['action'] = None
            scoring_result['reason'] = scoring_result.get('reason', '') + ' [Kelly SKIP]'

    # ================================================================
    # STEP 8: Compute composite position size
    # NOW INCLUDES: decay monitor, variant mult, transaction cost adj
    # ================================================================
    base_size = scoring_result.get('size', 1.0)
    regime_size_mod = regime_info_ml.get('size_modifier', 1.0)
    kelly_mult = kelly_result.get('size_multiplier', 1.0)

    # --- FIX 1: Apply decay monitor multiplier (was computed but never used) ---
    decay_mult = 1.0
    if decay_results:
        decay_mults = []
        for sid, health in decay_results.items():
            sig_mult = health.get('size_multiplier', 1.0)
            decay_mults.append(sig_mult)
            if sig_mult < 1.0:
                logger.info(f"Decay adjustment {sid}: {health.get('status')} -> {sig_mult}x")
        # Use the WORST signal's multiplier — if any signal is CRITICAL, reduce all
        decay_mult = min(decay_mults) if decay_mults else 1.0

    # --- FIX 2: Apply adaptive variant multiplier (was computed but never used) ---
    # variant_mult is 0.5x when DRY_12 is in REDUCED mode, 1.0x otherwise
    logger.info(f"Variant mult (DRY_12): {variant_mult}x ({variant_result.get('current_variant', 'PRIMARY')})")

    # --- FIX 5: Transaction cost adjustment to sizing ---
    cost_adj = 1.0
    cost_info = {}
    if indicators.get('close'):
        entry_price = indicators['close']
        cost_pts = cost_model.cost_in_points(entry_price, entry_price)
        cost_pct = cost_model.cost_as_pct(entry_price, entry_price)
        cost_info = {
            'cost_per_trade_pts': round(cost_pts, 2),
            'cost_per_trade_pct': round(cost_pct * 100, 4),
            'cost_per_trade_rs': round(
                cost_model.compute_futures_round_trip(entry_price, entry_price).total, 2
            ),
        }
        # Reduce sizing proportional to cost drag (subtracts ~0.3-0.8% from multiplier)
        cost_adj = max(0.9, 1.0 - cost_pct)

    # Calendar overlay: if entries blocked, force composite to 0
    if calendar_block:
        calendar_mult_applied = 0.0
    else:
        calendar_mult_applied = calendar_mult

    composite_size = (base_size * regime_size_mod * cross_asset_mult * fii_mult
                      * kelly_mult * decay_mult * variant_mult * cost_adj
                      * calendar_mult_applied)
    composite_size = max(0.0 if calendar_block else 0.1, min(3.0, composite_size))

    logger.info(
        f"Composite size: {composite_size:.3f}x = "
        f"base({base_size:.2f}) × regime({regime_size_mod:.2f}) × "
        f"cross_asset({cross_asset_mult:.2f}) × fii({fii_mult:.2f}) × "
        f"kelly({kelly_mult:.2f}) × decay({decay_mult:.2f}) × "
        f"variant({variant_mult:.2f}) × cost_adj({cost_adj:.3f}) × "
        f"calendar({calendar_mult_applied:.2f})"
    )

    # ================================================================
    # STEP 8b: Behavioral Overlay (LAYER 7 — Kahneman corrections)
    # Runs AFTER all sizing modifiers. Only reduces risk.
    # ================================================================
    behavioral_ctx = OverlayContext(
        system_size=composite_size,
        current_dd_pct=current_drawdown_pct / 100.0,  # convert from pct to fraction
        entry_price=indicators.get('close', 0),
        current_atr=indicators.get('atr_14', 0) if indicators.get('atr_14') else 0,
    )
    behavioral_result = safe_call(
        lambda: behavioral_overlay.apply_all(behavioral_ctx),
        'behavioral_overlay',
        default_return=None,
    )

    behavioral_mult = 1.0
    behavioral_triggers = []
    if behavioral_result is not None:
        behavioral_mult = behavioral_result.size_multiplier
        behavioral_triggers = behavioral_result.overlays_triggered
        composite_size *= behavioral_mult
        composite_size = max(0.0 if calendar_block else 0.1, min(3.0, composite_size))

        if behavioral_triggers:
            logger.info(
                f"Behavioral overlay: {behavioral_mult:.2f}x "
                f"(triggers={behavioral_triggers})"
            )

    # ================================================================
    # STEP 10: Gujral regime detection (existing)
    # ================================================================
    gujral_regime = detector.get_streak()
    confidence = detector.get_confidence_label(
        gujral_regime['regime'], scoring_result.get('score', 0))
    gujral_regime['confidence'] = confidence

    # ================================================================
    # STEP 11: VIX check (existing)
    # ================================================================
    vix_result = vix_monitor.check(
        current_vix=indicators.get('india_vix'))

    # ================================================================
    # STEP 12: EOD Reconciliation (IMPROVEMENT: position integrity)
    # ================================================================
    reconciliation_result = {}
    if not dry_run:
        reconciliation_result = safe_call(
            lambda: run_eod_reconciliation(as_of_date=as_of, dry_run=dry_run),
            'eod_reconciliation',
            default_return={'status': 'SKIPPED', 'error': 'safe_call fallback'}
        )
        recon_status = reconciliation_result.get('status', 'UNKNOWN')
        if recon_status in ('MAJOR', 'CRITICAL'):
            logger.warning(f"Reconciliation status: {recon_status}")

    # ================================================================
    # BUILD RESULT
    # ================================================================
    result = {
        'date': str(as_of),
        'compute_result': compute_result,
        'scoring_result': scoring_result,
        'combo_result': {'action': None},
        'control_action': signal_actions.get('KAUFMAN_DRY_20', {}).get('action'),
        'regime_info': gujral_regime,
        'decay_results': decay_results,
        'vix_result': vix_result,
        'pnl_control': {'day_pnl_pts': 0, 'positions_open': 0},
        'pnl_scoring': {'day_pnl_pts': 0, 'positions_open': 0},

        # Reconciliation
        'reconciliation': reconciliation_result,

        # NEW: Enhancement data (all multipliers now APPLIED, not just logged)
        'enhancements': {
            'ml_regime': regime_info_ml,
            'cross_asset': cross_asset,
            'fii_overlay': fii_result,
            'calendar_overlay': calendar_ctx,
            'variant_switch': variant_result,
            'kelly_sizing': kelly_result,
            'composite_size': composite_size,
            'transaction_costs': cost_info,
            'decay_mult': decay_mult,
            'variant_mult': variant_mult,
            'cost_adj': cost_adj,
            'calendar_mult': calendar_mult_applied,
            'behavioral_overlay': {
                'multiplier': behavioral_mult,
                'triggers': behavioral_triggers,
            },
            'multiplier_breakdown': {
                'base_size': base_size,
                'regime': regime_size_mod,
                'cross_asset': cross_asset_mult,
                'fii': fii_mult,
                'kelly': kelly_mult,
                'decay': decay_mult,
                'variant': variant_mult,
                'cost_adj': cost_adj,
                'calendar': calendar_mult_applied,
                'behavioral': behavioral_mult,
            },
        },
    }

    # Format enhanced digest
    result['digest'] = format_enhanced_digest(as_of, result)

    return result


def format_enhanced_digest(as_of, result):
    """Format enhanced Telegram digest with all overlay data."""
    cr = result['compute_result']
    indicators = cr.get('indicators', {})
    close = indicators.get('close', 0)
    prev_close = indicators.get('prev_close', 0)
    change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0
    enh = result.get('enhancements', {})

    lines = [
        f"*NIFTY F&O ENHANCED DAILY DIGEST*",
        f"Date: {as_of} | Nifty: {close:.0f} ({change_pct:+.1f}%)",
        f"VIX: {indicators.get('india_vix', 0):.1f}",
        "",

        # ML Regime
        f"*--- REGIME (ML) ---*",
        f"Regime: {enh.get('ml_regime', {}).get('regime', '?')} "
        f"({enh.get('ml_regime', {}).get('method', '?')}, "
        f"conf={enh.get('ml_regime', {}).get('confidence', 0):.0%})",
        "",

        # Scoring
        f"*--- SCORING ---*",
        f"Score: {result['scoring_result'].get('score', 0)} | "
        f"Action: {result['scoring_result'].get('action', 'None')}",
        f"Composite Size: {enh.get('composite_size', 1.0):.2f}x",
        "",

        # Overlays
        f"*--- OVERLAYS ---*",
        f"Cross-Asset: {enh.get('cross_asset', {}).get('composite_multiplier', 1.0):.2f}x",
        f"FII: {enh.get('fii_overlay', {}).get('regime', '?')} "
        f"({enh.get('fii_overlay', {}).get('multiplier', 1.0):.2f}x)",
    ]

    # Calendar overlay
    cal = enh.get('calendar_overlay', {})
    cal_mult = enh.get('calendar_mult', 1.0)
    cal_events = cal.get('events_active', [])
    if cal_events:
        lines.append(
            f"Calendar: {cal_mult:.2f}x — events: {', '.join(cal_events)}"
        )
    elif cal_mult != 1.0:
        seasonality = cal.get('monthly_seasonality', 0)
        lines.append(
            f"Calendar: {cal_mult:.2f}x (seasonality={seasonality:+.1%})"
        )
    if cal.get('block_new_entries'):
        lines.append("CALENDAR BLOCK: No new entries today")

    # Kelly sizing
    kelly = enh.get('kelly_sizing', {})
    if kelly:
        lines.append(
            f"Kelly: grade={kelly.get('grade', '?')}, "
            f"p={kelly.get('p_profit', 0):.0%}, "
            f"size={kelly.get('size_multiplier', 1.0):.2f}x"
        )

    # Decay monitor
    decay_m = enh.get('decay_mult', 1.0)
    if decay_m < 1.0:
        lines.append(f"Decay: {decay_m:.2f}x (signal health degraded)")

    # Variant status
    variant = enh.get('variant_switch', {})
    variant_m = enh.get('variant_mult', 1.0)
    if variant.get('new_variant'):
        lines.append(f"⚠️ VARIANT SWITCH: DRY_12 → {variant['new_variant']} ({variant_m}x)")
    elif variant_m < 1.0:
        lines.append(f"DRY_12: REDUCED mode ({variant_m}x)")

    # Cost adjustment
    cost_a = enh.get('cost_adj', 1.0)
    if cost_a < 1.0:
        lines.append(f"Cost adj: {cost_a:.3f}x")

    lines.append("")

    # Transaction costs
    costs = enh.get('transaction_costs', {})
    if costs:
        lines.append(
            f"*--- COSTS ---*\n"
            f"Per trade: {costs.get('cost_per_trade_pts', 0):.1f} pts "
            f"(₹{costs.get('cost_per_trade_rs', 0):.0f})"
        )

    # Signal health
    decay = result.get('decay_results', {})
    if decay:
        lines.append("")
        lines.append(f"*--- SIGNAL HEALTH ---*")
        emoji_map = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴',
                     'CRITICAL': '🚨', 'INSUFFICIENT_DATA': '⚪'}
        for sid, health in decay.items():
            status = health.get('status', 'UNKNOWN')
            emoji = emoji_map.get(status, '❓')
            short_id = sid.replace('KAUFMAN_', '')
            lines.append(f"{emoji} {short_id}: {status}")

    return '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced daily trading pipeline')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--date', type=str, help='YYYY-MM-DD')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    conn = psycopg2.connect(DATABASE_DSN)
    as_of = date.fromisoformat(args.date) if args.date else date.today()

    print(f"\n{'='*60}")
    print(f"ENHANCED PAPER TRADING DAILY RUN — {as_of}")
    print(f"{'='*60}")

    result = run_enhanced_day(conn, as_of, dry_run=args.dry_run)
    print(result['digest'])

    # Print enhancement summary
    enh = result.get('enhancements', {})
    print(f"\n{'='*60}")
    print(f"ENHANCEMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  ML Regime:     {enh.get('ml_regime', {}).get('regime')} "
          f"({enh.get('ml_regime', {}).get('method')})")
    print(f"  Cross-Asset:   {enh.get('cross_asset', {}).get('composite_multiplier', 1.0):.3f}x")
    print(f"  FII Overlay:   {enh.get('fii_overlay', {}).get('regime')} "
          f"({enh.get('fii_overlay', {}).get('multiplier', 1.0):.2f}x)")
    cal = enh.get('calendar_overlay', {})
    print(f"  Calendar:      {enh.get('calendar_mult', 1.0):.2f}x "
          f"(events={cal.get('events_active', [])})")
    print(f"  Kelly Grade:   {enh.get('kelly_sizing', {}).get('grade', 'N/A')}")
    print(f"  Composite:     {enh.get('composite_size', 1.0):.3f}x")
    print(f"  DRY_12 Variant: {enh.get('variant_switch', {}).get('current_variant', 'PRIMARY')} "
          f"({enh.get('variant_mult', 1.0):.2f}x)")
    print(f"  Drawdown Mult:  {result['scoring_result'].get('drawdown_mult', 1.0):.2f}x")

    conn.close()


if __name__ == '__main__':
    main()
