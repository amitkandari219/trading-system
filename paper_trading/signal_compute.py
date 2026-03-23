"""
Daily signal computation pipeline for paper trading.

Reads latest market data from DB, computes indicators for 3 confirmed signals,
checks entry/exit conditions against open positions, and logs fired signals.

Designed to run daily after market close (3:35 PM IST) or on EOD data update.

Usage:
    python -m paper_trading.signal_compute          # run once
    python -m paper_trading.signal_compute --dry-run # check without DB writes
"""

import json
import logging
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import (
    add_all_indicators, historical_volatility,
)
from config.settings import DATABASE_DSN
from extraction.dsl_schema_sizing import PositionSizingRule, ScaleCondition
from paper_trading.pcr_signals import PCRSignals
from regime_labeler import RegimeLabeler
from signals.regime_filter import IntradayRegimeFilter, REGIMES, SIGNAL_REGIME_MATRIX
from signals.expiry_day_detector import ExpiryDayDetector

logger = logging.getLogger(__name__)

# ================================================================
# SIGNAL DEFINITIONS
#
# Three categories:
#   SCORING_SIGNALS  — fire trades, full sizing, tracked P&L
#   SHADOW_SIGNALS   — paper trade only, evaluate for promotion
#   OVERLAY_SIGNALS  — modify sizing/confidence, never fire standalone
# ================================================================

OVERLAY_SIGNALS = {
    'GUJRAL_DRY_7': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_gujral7',
        'check_exit': '_check_exit_gujral7',
        'trade_type': 'OVERLAY',
        'overlay_type': 'REGIME_CONFIDENCE',
    },
    'CRISIS_SHORT': {
        'direction': 'SHORT',
        'stop_loss_pct': 0.04,
        'take_profit_pct': 0.0,
        'hold_days_max': 15,
        'check_entry': '_check_entry_crisis_short',
        'check_exit': '_check_exit_crisis_short',
        'trade_type': 'OVERLAY',
        'overlay_type': 'CRISIS_MODE',
        # When active: block new LONG entries, boost SHORT sizing 2x
        # V1: VIX>25 + close<sma_50 + ADX>25
        # COVID: 15 trades, +1,752 pts. Outside crisis: loses.
        # Use as regime modifier only, not standalone.
    },
    'SSRN_JANUARY_EFFECT': {
        'direction': 'LONG',
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.0,
        'hold_days_max': 22,
        'check_entry': '_check_entry_january',
        'check_exit': '_check_exit_january',
        'trade_type': 'OVERLAY',
        'overlay_type': 'CALENDAR_BOOST',
        # When active (January): boost LONG sizing by 1.25x
        # DSR 1.00, WF 73% pass, but 35.9% DD as standalone
    },
    'GLOBAL_OVERNIGHT_COMPOSITE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_global_composite',
        'check_exit': '_check_exit_global_composite',
        'trade_type': 'OVERLAY',
        'overlay_type': 'GLOBAL_SENTIMENT',
        # WF: 30/37 windows pass (81%), Avg Sharpe 3.59, Avg PF 2.04
        # Combines: GIFT Nifty gap (35%), US overnight S&P+VIX+DXY (35%),
        #           Crude oil impact (15%), Asian session (15%)
        # Runs at 8:30 AM IST via global_pre_market cron job
        # Output: size_modifier 0.3x–1.3x, directional bias, risk-off flag
        # When risk_off=True: reduce ALL new entries to 30% size
        # When direction agrees with daily signal: boost size by modifier
        # When direction conflicts: reduce size to 70%
    },
    'PCR_AUTOTRENDER': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_pcr',
        'check_exit': '_check_exit_pcr',
        'trade_type': 'OVERLAY',
        'overlay_type': 'OPTIONS_FLOW',
        # PCR level + momentum → contrarian at extremes, confirmation in middle
        # Size modifier 0.70x–1.30x, standalone contrarian at PCR>1.5 or <0.45
    },
    'ROLLOVER_ANALYSIS': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_rollover',
        'check_exit': '_check_exit_rollover',
        'trade_type': 'OVERLAY',
        'overlay_type': 'INSTITUTIONAL_FLOW',
        # OI migration across expiries: long buildup, short buildup, etc.
        # Active in last 5 days before monthly expiry, size 0.60x–1.35x
    },
    'FII_FUTURES_OI': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_fii_oi',
        'check_exit': '_check_exit_fii_oi',
        'trade_type': 'OVERLAY',
        'overlay_type': 'INSTITUTIONAL_FLOW',
        # FII long/short ratio in index futures — strongest institutional signal
        # Ratio > 0.65 = strong bullish, < 0.35 = strong bearish
        # Size modifier 0.60x–1.35x
    },
    'DELIVERY_PCT': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_delivery',
        'check_exit': '_check_exit_delivery',
        'trade_type': 'OVERLAY',
        'overlay_type': 'INSTITUTIONAL_FLOW',
        # Nifty aggregate delivery % — institutional activity proxy
        # High delivery + up = accumulation, Low delivery + up = speculative rally
    },
    'SENTIMENT_COMPOSITE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_sentiment',
        'check_exit': '_check_exit_sentiment',
        'trade_type': 'OVERLAY',
        'overlay_type': 'SENTIMENT',
        # Tickertape MMI + Google Trends → contrarian at extremes
        # MMI < 25 = bullish, MMI > 75 = bearish, size 0.70x–1.30x
    },
    'BOND_YIELD_SPREAD': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_bond',
        'check_exit': '_check_exit_bond',
        'trade_type': 'OVERLAY',
        'overlay_type': 'MACRO_FLOW',
        # India-US 10Y spread: wide = FII inflow bullish, narrow = outflow bearish
        # Spread > 4.5% = bullish, < 3% = bearish, DXY amplifies
    },
    'GAMMA_EXPOSURE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_gamma',
        'check_exit': '_check_exit_gamma',
        'trade_type': 'OVERLAY',
        'overlay_type': 'MICROSTRUCTURE',
        # Dealer gamma → positive = mean-reversion, negative = trending
        # Adjusts fade vs momentum strategy sizing
    },
    'VOL_TERM_STRUCTURE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_vol_ts',
        'check_exit': '_check_exit_vol_ts',
        'trade_type': 'OVERLAY',
        'overlay_type': 'VOLATILITY',
        # VIX term structure: contango = complacent, backwardation = stressed
        # Transitions are early regime warnings, size 0.60x–1.15x
    },
    'RBI_MACRO_FILTER': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_macro',
        'check_exit': '_check_exit_macro',
        'trade_type': 'OVERLAY',
        'overlay_type': 'EVENT_FILTER',
        # Reduces sizing around RBI MPC, Budget, FOMC, elections
        # GST collection as macro regime signal
    },
    'ORDER_FLOW_IMBALANCE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_orderflow',
        'check_exit': '_check_exit_orderflow',
        'trade_type': 'OVERLAY',
        'overlay_type': 'MICROSTRUCTURE',
        # First 15-min buy/sell imbalance → intraday direction predictor
        # Strong OFI + volume spike = high conviction, size 0.70x–1.30x
    },
    'XGBOOST_META_LEARNER': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_meta',
        'check_exit': '_check_exit_meta',
        'trade_type': 'OVERLAY',
        'overlay_type': 'META_COMBINER',
        # XGBoost meta-learner: optimal signal weighting by regime
        # Heuristic fallback until training data accumulates
        # Replaces fixed weights with adaptive regime-based weights
    },
    # ----------------------------------------------------------
    # Tier 3 ML/AI Overlays
    # ----------------------------------------------------------
    'MAMBA_REGIME': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_mamba',
        'check_exit': '_check_exit_mamba',
        'trade_type': 'OVERLAY',
        'overlay_type': 'REGIME_ML',
        # Mamba/S4 state-space model for regime detection
        # 5 regimes: CALM_BULL, VOLATILE_BULL, NEUTRAL, VOLATILE_BEAR, CALM_BEAR
    },
    'TFT_FORECAST': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_tft',
        'check_exit': '_check_exit_tft',
        'trade_type': 'OVERLAY',
        'overlay_type': 'FORECAST_ML',
        # TFT multi-horizon forecaster (1d, 3d, 5d, 10d, 20d)
        # Quantile outputs: p10, p50, p90
    },
    'RL_POSITION_SIZER': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_rl_sizer',
        'check_exit': '_check_exit_rl_sizer',
        'trade_type': 'OVERLAY',
        'overlay_type': 'SIZING_ML',
        # SAC RL agent for dynamic position sizing [0.1, 2.0]
        # Replaces heuristic size_modifier once trained
    },
    'GNN_SECTOR_ROTATION': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_gnn',
        'check_exit': '_check_exit_gnn',
        'trade_type': 'OVERLAY',
        'overlay_type': 'SECTOR_ML',
        # GNN on stock correlation graph for regime + sector rotation
        # RISK_ON / RISK_OFF / MIXED detection
    },
    'NLP_SENTIMENT': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_nlp',
        'check_exit': '_check_exit_nlp',
        'trade_type': 'OVERLAY',
        'overlay_type': 'SENTIMENT_ML',
        # NLP earnings + RBI speech sentiment
        # Weighted: earnings 60% + RBI 40%
    },
    'AMFI_MF_FLOW': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_mf_flow',
        'check_exit': '_check_exit_mf_flow',
        'trade_type': 'OVERLAY',
        'overlay_type': 'STRUCTURAL_FLOW',
        # AMFI SIP + equity net flow as structural floor/ceiling
        # Monthly data — low-frequency structural signal
    },
    'CREDIT_CARD_SPENDING': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_cc_spending',
        'check_exit': '_check_exit_cc_spending',
        'trade_type': 'OVERLAY',
        'overlay_type': 'CONSUMPTION_MACRO',
        # Credit card spending as consumption proxy
        # Monthly data — cross-checked with GST collections
    },
    # Future overlays:
    # 'EXPIRY_WEEK':     sizing reducer 0.60x — confirmed, calendar-based
    # Zero-trade signals: intensity scorer inputs (51 signals)
    # KAHNEMAN behavioral overlays (4 planned)
}

SHADOW_SIGNALS = {
    'CHAN_AT_DRY_4': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 10,
        'check_entry': '_check_entry_chan_at4',
        'check_exit': '_check_exit_chan_at4',
        'trade_type': 'SHADOW',
    },
    'CANDLESTICK_DRY_0': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 10,
        'check_entry': '_check_entry_candlestick0',
        'check_exit': '_check_exit_candlestick0',
        'trade_type': 'SHADOW',
    },
    'KAUFMAN_DRY_7': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_dry7',
        'check_exit': '_check_exit_dry7',
        'trade_type': 'SHADOW',
    },
    'GUJRAL_DRY_9': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_gujral9',
        'check_exit': '_check_exit_gujral9',
        'trade_type': 'SHADOW',
    },
    'BULKOWSKI_CUP_HANDLE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.0,
        'hold_days_max': 30,
        'check_entry': '_check_entry_cup_handle',
        'check_exit': '_check_exit_cup_handle',
        'trade_type': 'SHADOW',
    },
    # BULKOWSKI_ADAM_AND_EVE_OR: PROMOTED TO SCORING
    # WF 85% (22/26), Sharpe 2.34 in 6-sig portfolio, PF 1.52
    # Zero correlation with existing signals (max 0.008)
    # 6-sig portfolio: +₹84K P&L, -0.5% MaxDD, +0.13 Sharpe vs 5-sig
    'BULKOWSKI_ROUND_BOTTOM_RDB_PATTERN': {
        'direction': 'LONG',
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.0,
        'hold_days_max': 30,
        'check_entry': '_check_entry_round_bottom',
        'check_exit': '_check_exit_round_bottom',
        'trade_type': 'SHADOW',
    },
    'BULKOWSKI_EADT_BUSTED_PATTERN': {
        'direction': 'LONG',
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.0,
        'hold_days_max': 20,
        'check_entry': '_check_entry_eadt_busted',
        'check_exit': '_check_exit_eadt_busted',
        'trade_type': 'SHADOW',
    },
    'BULKOWSKI_FALLING_VOLUME_TREND_IN': {
        'direction': 'LONG',
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.0,
        'hold_days_max': 20,
        'check_entry': '_check_entry_falling_vol',
        'check_exit': '_check_exit_falling_vol',
        'trade_type': 'SHADOW',
    },
    'BULKOWSKI_EADB_EARLY_ATTEMPT_TO': {
        'direction': 'SHORT',
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.0,
        'hold_days_max': 10,
        'check_entry': '_check_entry_eadb',
        'check_exit': '_check_exit_eadb',
        'trade_type': 'SHADOW',
    },
    'SSRN_WEEKLY_MOM': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 5,
        'check_entry': '_check_entry_weekly_mom',
        'check_exit': '_check_exit_weekly_mom',
        'trade_type': 'SHADOW',
    },
    # ── Intraday signals (5-min bar based) ──────────────────────
    # These fire from real-time 5-min data via intraday_runner.py.
    # In daily pipeline they're logged as SHADOW for tracking only.
    # ID_KAUFMAN_BB_MR: REMOVED — PF 0.22 on real Nifty 5-min data (17 configs tested, all lose)
    # ID_GUJRAL_RANGE: REMOVED — PF 0.36 on real Nifty 5-min data (all configs lose)
    'ID_GAMMA_BREAKOUT': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.002,
        'take_profit_pct': 0.004,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'expiry_only': True,    # only fires on expiry days
        'regime_filter': True,
    },
    'ID_GAMMA_REVERSAL': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.003,
        'take_profit_pct': 0.005,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'expiry_only': True,
        'regime_filter': True,
    },
    'ID_GAMMA_SQUEEZE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.002,
        'take_profit_pct': 0.004,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'expiry_only': True,
        'regime_filter': True,
    },
    'L8_BEAR_CALL_RESISTANCE': {
        'direction': 'SHORT',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 21,
        'check_entry': '_check_entry_bear_call',
        'check_exit': '_check_exit_bear_call',
        'trade_type': 'SHADOW',
        'instrument_type': 'OPTIONS',
    },
    # ================================================================
    # BANKNIFTY L9 SIGNALS — REMOVED after real-data WF (2026-03-22)
    # Synthetic WF: 4/8 passed. Real 5-min Kite data: 0/8 passed.
    # ================================================================
    # ================================================================
    # INDIA-SPECIFIC INTRADAY — validated on real Kite 5-min data
    # EXPIRY_PIN_FADE: 91% WR, PF 14.53, 100% WF (45 trades, expiry days only)
    # ORR_REVERSION: 71% WR, PF 2.89, 94% WF (189 trades, gap fade)
    # ================================================================
    # EXPIRY_PIN_FADE: PROMOTED TO SCORING (2026-03-22)
    # 45 real trades, 91% WR, PF 14.53, 100% WF pass rate
    # Moved to SCORING_SIGNALS below
    # ORR_REVERSION: PROMOTED TO SCORING (2026-03-22)
    # 189 real trades, 71% WR, PF 2.89, 94% WF pass rate
    # Moved to SCORING_SIGNALS (SIGNALS dict) below
    # ================================================================
    # BANKNIFTY VARIANTS (SHADOW — need 30 BN-specific trades)
    # ================================================================
    'BN_ORR_REVERSION': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.006,     # wider SL for BN volatility
        'take_profit_pct': 0.010,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
    },
    'BN_EXPIRY_PIN_FADE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.004,     # wider for BN
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'expiry_only': True,        # monthly expiry only (last Thursday)
    },
    # ================================================================
    # MAX PAIN + STRANGLE SIGNALS (SHADOW pending WF)
    # ================================================================
    'STRIKE_CLUSTERING_GRAV': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.008,
        'take_profit_pct': 0.0,
        'hold_days_max': 3,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': 'daily',
        'instrument': 'NIFTY',
    },
    'WEEKLY_STRANGLE_SELL': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.0,   # managed by strangle_seller
        'take_profit_pct': 0.0,
        'hold_days_max': 5,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'instrument_type': 'OPTIONS',
        'instrument': 'NIFTY',
    },
    # ================================================================
    # VALIDATED COMBINATIONS (OOS Sharpe 5.08-6.95)
    # ================================================================
    'COMBO_GRIMES32_DRY12_SEQ5': {
        # GRIMES_DRY_3_2 + KAUFMAN_DRY_12 (SEQ_5): DRY_12 fires,
        # then GRIMES_3_2 confirms within 5 days → enter
        # OOS Sharpe 5.08, 91 trades, 46% WR
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 10,
        'check_entry': '_check_entry_combo_grimes32_dry12',
        'check_exit': '_check_exit_combo_grimes32',
        'trade_type': 'SHADOW',
        'combo_logic': 'SEQ_5',
        'combo_window': 5,
    },
    'COMBO_GRIMES60_DRY8_AND': {
        # GRIMES_DRY_6_0 + GUJRAL_DRY_8 (AND): Both fire on same day
        # OOS Sharpe 6.95, 46 trades, 85% WR
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'check_entry': '_check_entry_combo_grimes60_dry8',
        'check_exit': '_check_exit_combo_grimes60',
        'trade_type': 'SHADOW',
        'combo_logic': 'AND',
    },

    'COMBO_GRIMES50_DRY16_SEQ3': {
        # GRIMES_DRY_5_0 + KAUFMAN_DRY_16 (SEQ_3): WF 10/13, Sharpe 4.69, PF 2.62
        # Grimes breakout (range > ATR, vol surge) confirmed by DRY_16 pivot breakout
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 10,
        'check_entry': '_check_entry_combo_grimes50_dry16',
        'check_exit': '_check_exit_combo_grimes50',
        'trade_type': 'SHADOW',
        'combo_logic': 'SEQ_3',
        'combo_window': 3,
    },
    'COMBO_GRIMES50_DRY16_SEQ5': {
        # Same pair, wider window: WF 10/13, Sharpe 5.16, PF 2.67
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 10,
        'check_entry': '_check_entry_combo_grimes50_dry16',
        'check_exit': '_check_exit_combo_grimes50',
        'trade_type': 'SHADOW',
        'combo_logic': 'SEQ_5',
        'combo_window': 5,
    },

    # ================================================================
    # BANKNIFTY SHADOW SIGNALS (intraday 5-min, wider SL/TGT for BN vol)
    # Computed by BankNiftySignalComputer in intraday_runner.py.
    # Registered here for tracking/promotion via bn_promotion_tracker.
    # ================================================================
    # BN_KAUFMAN_BB_MR: REMOVED — PF 0.26 on real BankNifty 5-min data
    # BN_GUJRAL_RANGE: REMOVED — PF 0.49 on real BankNifty 5-min data
    'BN_ORB_BREAKOUT': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.008,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
    'BN_VWAP_RECLAIM': {
        'direction': 'LONG',
        'stop_loss_pct': 0.004,
        'take_profit_pct': 0.007,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
    'BN_VWAP_REJECTION': {
        'direction': 'SHORT',
        'stop_loss_pct': 0.004,
        'take_profit_pct': 0.007,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
    'BN_FIRST_PULLBACK': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.008,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
    'BN_FAILED_BREAKOUT': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.008,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
    'BN_GAP_FILL': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.004,
        'take_profit_pct': 0.007,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
    'BN_TREND_BAR': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.008,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
    'BN_EOD_TREND': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.004,
        'take_profit_pct': 0.006,
        'hold_days_max': 0,
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'trade_type': 'SHADOW',
        'timeframe': '5min',
        'instrument': 'BANKNIFTY',
        'regime_filter': True,
    },
}

# ================================================================
# POSITION SIZING RULES — top 5 from THARP/VINCE DSL translation
# ================================================================

SIZING_RULES = {
    # 1. Baseline: fixed 1% risk per trade (Tharp)
    'FIXED_1PCT': PositionSizingRule(
        signal_id='THARP_FIXED_1PCT',
        sizing_method='FIXED_FRACTIONAL',
        base_risk_pct=0.01,
        r_multiple_stop=1.0,
        r_multiple_target=3.0,
        confidence='HIGH',
    ),
    # 2. Anti-Martingale: increase 25% after 3 consecutive wins (Tharp)
    'ANTI_MARTINGALE': PositionSizingRule(
        signal_id='THARP_ANTI_MART',
        sizing_method='ANTI_MARTINGALE',
        base_risk_pct=0.01,
        scale_up_condition=ScaleCondition(
            trigger='consecutive_wins', threshold=3, multiplier=1.25),
        confidence='HIGH',
    ),
    # 3. Drawdown scaling: halve size at 10% DD, quarter at 20% (Tharp)
    'DRAWDOWN_SCALED': PositionSizingRule(
        signal_id='THARP_DD_SCALE',
        sizing_method='DRAWDOWN_SCALED',
        base_risk_pct=0.01,
        scale_down_condition=ScaleCondition(
            trigger='drawdown_pct', threshold=0.10, multiplier=0.50),
        confidence='HIGH',
    ),
    # 4. Half Kelly optimal f (Vince)
    'HALF_KELLY': PositionSizingRule(
        signal_id='VINCE_HALF_KELLY',
        sizing_method='KELLY',
        base_risk_pct=0.01,
        kelly_fraction=0.10,  # half of typical 0.20 optimal_f
        confidence='HIGH',
    ),
    # 5. Volatility-scaled sizing (Moreira-validated, Vince)
    'VOL_SCALED': PositionSizingRule(
        signal_id='VINCE_VOL_SCALED',
        sizing_method='VOLATILITY_SCALED',
        base_risk_pct=0.01,
        confidence='HIGH',
    ),
}

# Active sizing rule — start with baseline, switch via config
ACTIVE_SIZING_RULE = 'FIXED_1PCT'

# ================================================================
# REGIME C — GRADUATED VIX + ADX WEIGHTS
#
# No hard cutoffs. Graduated VIX scale:
#   VIX < 15:  1.2x (low vol — size up slightly)
#   VIX 15-20: 1.0x (normal)
#   VIX 20-25: 0.7x (elevated — reduce but don't stop)
#   VIX 25-30: 0.3x (high — minimal LONG exposure)
#   VIX > 30:  0.0x (crisis — stop LONGS entirely)
#
# ADX-based trend/range weight per signal family.
# SHORT signals (EADB) get boosted in crisis, not penalized.
# ================================================================

# ADX weights: how each signal performs in trending vs ranging markets
ADX_WEIGHTS = {
    'KAUFMAN_DRY_20':       {'TRENDING': 1.5, 'RANGING': 0.5},
    'KAUFMAN_DRY_16':       {'TRENDING': 1.5, 'RANGING': 0.5},
    'KAUFMAN_DRY_12':       {'TRENDING': 1.5, 'RANGING': 0.5},
    'GUJRAL_DRY_8':         {'TRENDING': 1.5, 'RANGING': 1.0},
    'GUJRAL_DRY_13':        {'TRENDING': 1.5, 'RANGING': 0.8},
    'CHAN_AT_DRY_4':         {'TRENDING': 0.5, 'RANGING': 1.5},
    'CANDLESTICK_DRY_0':    {'TRENDING': 1.0, 'RANGING': 1.0},
    'KAUFMAN_DRY_7':        {'TRENDING': 1.5, 'RANGING': 0.5},
    'GUJRAL_DRY_9':         {'TRENDING': 1.5, 'RANGING': 1.0},
    'BULKOWSKI_CUP_HANDLE': {'TRENDING': 0.8, 'RANGING': 1.5},
    'BULKOWSKI_ADAM_AND_EVE_OR': {'TRENDING': 0.8, 'RANGING': 1.5},
    'SSRN_WEEKLY_MOM':      {'TRENDING': 1.5, 'RANGING': 0.5},
    'PCR_ELEVATED_FEAR':    {'TRENDING': 0.8, 'RANGING': 1.0},  # contrarian — works better in ranging
}


def vix_multiplier(vix):
    """Graduated VIX scale — Scenario C. No cliff edges."""
    if vix < 15:  return 1.2
    if vix < 20:  return 1.0
    if vix < 25:  return 0.7
    if vix < 30:  return 0.3
    return 0.0


def compute_regime_multiplier(signal_id, vix, adx):
    """Scenario C: graduated VIX scale + ADX trend/range weight."""

    # SHORT signals get boosted in crisis, not penalized
    if signal_id == 'BULKOWSKI_EADB_EARLY_ATTEMPT_TO':
        adx_regime = 'TRENDING' if adx > 25 else 'RANGING'
        adx_mult = ADX_WEIGHTS.get(signal_id, {}).get(adx_regime, 1.0)
        if vix >= 25:
            return adx_mult * 2.0
        return adx_mult

    # ADX weight
    adx_regime = 'TRENDING' if adx > 25 else 'RANGING'
    adx_mult = ADX_WEIGHTS.get(signal_id, {}).get(adx_regime, 1.0)

    # VIX weight (graduated)
    vix_mult = vix_multiplier(vix)

    return adx_mult * vix_mult


SIGNALS = {
    'KAUFMAN_DRY_20': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 10,  # was 0 (unlimited) - capped to prevent stale positions
        'check_entry': '_check_entry_dry20',
        'check_exit': '_check_exit_dry20',
    },
    'KAUFMAN_DRY_16': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 8,  # was 0 (unlimited) - capped to prevent stale positions
        'check_entry': '_check_entry_dry16',
        'check_exit': '_check_exit_dry16',
    },
    'KAUFMAN_DRY_12': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 7,  # was 6 - set to 7 per spec
        'check_entry': '_check_entry_dry12',
        'check_exit': '_check_exit_dry12',
    },
    'GUJRAL_DRY_8': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 5,  # was 0 (unlimited) - capped to prevent stale positions
        'check_entry': '_check_entry_gujral8',
        'check_exit': '_check_exit_gujral8',
    },
    'GUJRAL_DRY_13': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 7,  # was 10 - tightened to match signal edge decay
        'check_entry': '_check_entry_gujral13',
        'check_exit': '_check_exit_gujral13',
    },
    # ── PROMOTED FROM SHADOW (2026-03-22) ──
    # EXPIRY_PIN_FADE: 45 real trades, 91% WR, PF 14.53, 100% WF pass
    # Exploits NSE expiry-day gamma pinning near round strikes
    'EXPIRY_PIN_FADE': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.003,
        'take_profit_pct': 0.0,  # TGT is the pin strike (dynamic)
        'hold_days_max': 0,      # intraday only
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'timeframe': '5min',
        'instrument': 'NIFTY',
        'expiry_only': True,
    },
    # Promoted from SHADOW (2026-03-22) — 189 trades, 71% WR, PF 2.89, 94% WF
    # Gap fade: overnight gaps > 0.5% revert in first 2 hours
    'ORR_REVERSION': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.008,
        'hold_days_max': 0,      # intraday only
        'check_entry': '_check_entry_intraday_stub',
        'check_exit': '_check_exit_intraday_stub',
        'timeframe': '5min',
        'instrument': 'NIFTY',
    },

    # CANDLESTICK_DRY_0: Tested for SCORING but loses ₹65K when combined
    # with other signals (position limit conflicts). Stays in SHADOW.

    # Promoted from SHADOW — WF 85% (22/26), zero correlation (0.008 max)
    # 6-sig portfolio: Sharpe 2.34 (+0.13), MaxDD 4.8% (-0.5%), P&L +₹84K
    # Bulkowski Adam & Eve pattern: hammer at range low with vol expansion
    'BULKOWSKI_ADAM_AND_EVE_OR': {
        'direction': 'LONG',
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.0,
        'hold_days_max': 5,
        'check_entry': '_check_entry_adam_eve',
        'check_exit': '_check_exit_adam_eve',
    },
}


class SignalComputer:
    """Computes daily signals for paper trading."""

    def __init__(self, db_conn=None, sizing_rule_key: str = None):
        self.conn = db_conn or psycopg2.connect(DATABASE_DSN)
        self.sizing_rule = SIZING_RULES.get(
            sizing_rule_key or ACTIVE_SIZING_RULE,
            SIZING_RULES['FIXED_1PCT'],
        )
        self.pcr_signals = PCRSignals()
        self.regime_labeler = RegimeLabeler()

        # Decay auto-manager for size overrides
        from paper_trading.decay_auto_manager import DecayAutoManager
        self.decay_manager = DecayAutoManager(conn=self.conn)

        # Intraday regime filter — uses VIX-based regime detection
        # to gate intraday signals (ID_KAUFMAN_BB_MR, ID_GUJRAL_RANGE, gamma)
        self.intraday_regime_filter = IntradayRegimeFilter()
        self.expiry_detector = ExpiryDayDetector()

        # Calendar overlay for event-driven sizing
        from signals.calendar_overlay import CalendarOverlay
        self.calendar_overlay = CalendarOverlay(db_conn=self.conn)

        # Combo signal state: tracks when first signal fires for SEQ logic
        # {signal_id: {'direction': 'LONG', 'fire_date': date, 'price': float}}
        self._combo_pending = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dynamic_stoch_threshold(self, base=50, vix=15):
        """VIX-adaptive stochastic threshold.

        When VIX is elevated (>15), raise the threshold so that only
        stronger momentum signals pass. When VIX is low, lower slightly.
        Clamped to [base-5, base+10].
        """
        adjustment = (vix - 15) * 0.5
        adjustment = max(-5, min(adjustment, 10))
        return base + adjustment

    def run(self, as_of_date: date = None, dry_run: bool = False) -> Dict:
        """
        Run signal computation for a given date.

        Returns dict with:
            {
                'date': ...,
                'entries': [{'signal_id': ..., 'direction': ..., 'price': ...}, ...],
                'exits': [{'signal_id': ..., 'reason': ..., 'price': ...}, ...],
                'indicators': {'vix': ..., 'stoch_k_5': ..., ...},
            }
        """
        as_of_date = as_of_date or date.today()

        # Load market data (last 250 days is enough for all indicators)
        df = self._load_market_data(as_of_date)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient market data for {as_of_date}")
            return {'date': str(as_of_date), 'entries': [], 'exits': [], 'error': 'insufficient_data'}

        # Compute all indicators
        df = add_all_indicators(df)
        df['hvol_6'] = historical_volatility(df['close'], period=6)
        df['hvol_100'] = historical_volatility(df['close'], period=100)

        # Get today's and yesterday's rows
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        # Compute regime for today
        current_regime = self.regime_labeler.label_single_day(df, today['date'])
        logger.info(f"Regime: {current_regime}")

        # ── Intraday regime bridge ──────────────────────────────
        # Map VIX-based regime → intraday regime for filtering
        # This uses the same thresholds as regime_labeler.py:
        #   VIX ≥ 25 → CRISIS,  VIX ≥ 18 → HIGH_VOL
        # Plus ATR ratio from daily bars for CALM/NORMAL/ELEVATED
        vix_val_raw = float(today.get('india_vix', 0)) if pd.notna(today.get('india_vix')) else 0
        atr_14_today = float(today.get('atr_14', 0)) if pd.notna(today.get('atr_14')) else 0
        if atr_14_today > 0:
            self.intraday_regime_filter.update_daily_atr(atr_14_today)
        intraday_regime = self._vix_to_intraday_regime(vix_val_raw, current_regime)
        logger.info(f"Intraday regime: {intraday_regime} (VIX={vix_val_raw:.1f})")

        # Expiry day info
        expiry_info = self.expiry_detector.get_expiry_info(as_of_date)
        if expiry_info['is_any_expiry']:
            logger.info(f"EXPIRY DAY: {expiry_info['expiry_instruments']}")

        # Extract key indicator values for logging
        indicators = {
            'date': str(as_of_date),
            'close': float(today['close']),
            'prev_close': float(yesterday['close']),
            'volume': float(today['volume']),
            'prev_volume': float(yesterday['volume']),
            'india_vix': float(today.get('india_vix', 0)),
            'sma_10': float(today['sma_10']),
            'stoch_k_5': float(today['stoch_k_5']),
            'stoch_d_5': float(today['stoch_d_5']),
            'pivot': float(today['pivot']),
            'r1': float(today['r1']),
            's1': float(today['s1']),
            'hvol_6': float(today['hvol_6']) if pd.notna(today['hvol_6']) else None,
            'hvol_100': float(today['hvol_100']) if pd.notna(today['hvol_100']) else None,
            'vol_ratio_20': float(today['vol_ratio_20']) if pd.notna(today['vol_ratio_20']) else None,
            'adx_14': float(today['adx_14']) if pd.notna(today['adx_14']) else None,
            'regime': current_regime,
            'intraday_regime': intraday_regime,
            'is_expiry_day': expiry_info['is_any_expiry'],
            'expiry_instruments': expiry_info.get('expiry_instruments', []),
        }

        # Load open positions (PAPER + SHADOW)
        open_positions = self._load_open_positions()
        open_shadow = self._load_open_positions(trade_type='SHADOW')

        # Check exits for PAPER positions
        exits = []
        for pos in open_positions:
            exit_signal = self._check_exit(pos, today, yesterday, as_of_date)
            if exit_signal:
                exits.append(exit_signal)
                if not dry_run:
                    self._record_exit(pos, exit_signal, today)

        # Check exits for SHADOW positions
        shadow_exits = []
        for pos in open_shadow:
            exit_signal = self._check_exit(pos, today, yesterday, as_of_date)
            if exit_signal:
                shadow_exits.append(exit_signal)
                if not dry_run:
                    self._record_exit(pos, exit_signal, today)

        # CROSS-SIGNAL EXIT: Tested but disabled.
        # Walk-forward at fixed lots showed +1,733 pts, but full-scale test
        # with position sizing showed -4,501 pts (OLD wins 7/13 windows).
        # The exit cuts winners short more than it cuts losers in practice.

        # Check PAPER entries
        entries = []
        active_signal_ids = {pos['signal_id'] for pos in open_positions
                            if pos['signal_id'] not in {e['signal_id'] for e in exits}}

        # Pre-check overlay flags before entry loop
        crisis_mode = False
        january_boost = False
        pcr_extreme_fear_active = False
        for signal_id, config in OVERLAY_SIGNALS.items():
            if config.get('overlay_type') == 'CRISIS_MODE':
                entry = self._check_entry(signal_id, config, today, yesterday)
                if entry:
                    crisis_mode = True
                    logger.warning(f"CRISIS MODE ACTIVE: {entry['reason']}")
                    break

        # Compute sizing context from recent trade history
        sizing_ctx = self._get_sizing_context()

        for signal_id, config in SIGNALS.items():
            if signal_id in active_signal_ids:
                continue

            # DECAY OVERRIDE: check size overrides before entry
            decay_factor = self.decay_manager.apply_size_overrides(signal_id)
            if decay_factor <= 0:
                logger.info(f"BLOCKED {signal_id}: decay override (factor=0)")
                continue

            # CRISIS MODE: block new LONG entries, boost SHORT sizing 2x
            direction = config.get('direction', 'BOTH')
            if crisis_mode and direction == 'LONG':
                continue  # block LONG during crisis

            entry = self._check_entry(signal_id, config, today, yesterday)
            if entry:
                # Compute position size multiplier from sizing rule
                size_mult = self._compute_size_multiplier(today, sizing_ctx)

                # Apply Regime C graduated weights
                vix_val = float(today.get('india_vix', 15)) if pd.notna(today.get('india_vix')) else 15.0
                adx_val = float(today.get('adx_14', 20)) if pd.notna(today.get('adx_14')) else 20.0
                regime_mult = compute_regime_multiplier(signal_id, vix_val, adx_val)
                size_mult *= regime_mult

                # If regime weight is 0, skip this entry entirely
                if regime_mult == 0:
                    continue

                # Crisis boost: 2x SHORT sizing when crisis overlay active
                if crisis_mode and entry.get('direction') == 'SHORT':
                    size_mult *= 2.0

                # January boost: 1.25x LONG sizing in January
                if january_boost and entry.get('direction') == 'LONG':
                    size_mult *= 1.25

                # PCR extreme fear overlay: 1.3x LONG when market fear extreme
                if pcr_extreme_fear_active and entry.get('direction') == 'LONG':
                    size_mult *= 1.3

                # Decay override: reduce size (factor already checked > 0 above)
                if decay_factor < 1.0:
                    size_mult *= decay_factor
                    logger.info(
                        f"Decay override {signal_id}: "
                        f"factor={decay_factor:.2f} -> size_mult={size_mult:.3f}"
                    )

                # Calendar overlay: apply event-driven modifier
                try:
                    calendar_mod = self.calendar_overlay.get_size_modifier(as_of_date)
                    if calendar_mod == 0.0:
                        logger.info(
                            f"Calendar BLOCKS entry for {signal_id} on {as_of_date}"
                        )
                        continue
                    elif calendar_mod != 1.0:
                        size_mult *= calendar_mod
                        logger.info(
                            f"Calendar modifier {signal_id}: "
                            f"{calendar_mod:.2f}x -> size_mult={size_mult:.3f}"
                        )
                except Exception as e:
                    logger.debug(f"Calendar overlay failed: {e}")

                entry['size_multiplier'] = size_mult
                entry['sizing_rule'] = self.sizing_rule.signal_id
                entry['crisis_mode'] = crisis_mode
                entry['regime'] = current_regime
                entry['regime_mult'] = regime_mult
                entries.append(entry)
                if not dry_run:
                    self._record_entry(signal_id, entry, today, as_of_date, indicators,
                                       size_multiplier=size_mult)

        # Check SHADOW entries
        shadow_entries = []
        active_shadow_ids = {pos['signal_id'] for pos in open_shadow
                            if pos['signal_id'] not in {e['signal_id'] for e in shadow_exits}}

        for signal_id, config in SHADOW_SIGNALS.items():
            if signal_id in active_shadow_ids:
                continue

            # ── Intraday regime filter for ID_* signals ──
            if config.get('regime_filter'):
                # Check signal-regime compatibility
                # Map ID_ prefix signal names to regime filter names
                rf_signal = signal_id.replace('ID_', '')
                regime_policy = SIGNAL_REGIME_MATRIX.get(rf_signal, None)
                if regime_policy and not regime_policy.get(intraday_regime, True):
                    logger.debug(f"SHADOW {signal_id} blocked by regime filter "
                                 f"({intraday_regime})")
                    continue
                # Apply size factor from regime
                regime_cfg = REGIMES.get(intraday_regime, REGIMES['NORMAL'])
                if regime_cfg['size_factor'] <= 0:
                    logger.debug(f"SHADOW {signal_id} blocked: {intraday_regime} "
                                 f"regime size=0")
                    continue

            # Expiry-only signals: skip on non-expiry days
            if config.get('expiry_only') and not expiry_info['is_any_expiry']:
                continue

            # Intraday signals use stub entries in daily pipeline
            # (real entries come from intraday_runner.py on 5-min bars)
            entry = self._check_entry(signal_id, config, today, yesterday)
            if entry:
                # Annotate with regime info
                if config.get('regime_filter'):
                    regime_cfg = REGIMES.get(intraday_regime, REGIMES['NORMAL'])
                    entry['intraday_regime'] = intraday_regime
                    entry['regime_size_factor'] = regime_cfg['size_factor']
                    entry['regime_sl_mult'] = regime_cfg['sl_mult']
                shadow_entries.append(entry)
                if not dry_run:
                    self._record_entry(signal_id, entry, today, as_of_date, indicators,
                                       trade_type='SHADOW')

        # Check PCR contrarian signals
        pcr_entries = self.pcr_signals.compute(df)
        pcr_extreme_fear_active = False
        for pcr_sig in pcr_entries:
            sid = pcr_sig['signal_id']
            if sid not in active_signal_ids:
                if sid == 'PCR_ELEVATED_FEAR':
                    # PROMOTED TO SCORING — Tier A, DSR 4.12, 100% WF pass
                    size_mult = self._compute_size_multiplier(today, sizing_ctx)
                    regime_mult = compute_regime_multiplier(sid, vix_val, adx_val)
                    size_mult *= regime_mult
                    if regime_mult > 0:
                        pcr_sig['size_multiplier'] = size_mult
                        entries.append(pcr_sig)
                        if not dry_run:
                            self._record_entry(sid, pcr_sig, today, as_of_date, indicators,
                                               size_multiplier=size_mult)
                elif sid == 'PCR_EXTREME_FEAR':
                    # OVERLAY — boost LONG sizing 1.3x when extreme fear
                    pcr_extreme_fear_active = True
                else:
                    # GREED signals dropped from active use
                    pass
        indicators['pcr_oi'] = float(today.get('pcr_oi', 0)) if pd.notna(today.get('pcr_oi', None)) else None

        # Check OVERLAY signals (modify confidence/sizing, don't open positions)
        overlay_states = {}
        crisis_mode = False
        january_boost = False
        for signal_id, config in OVERLAY_SIGNALS.items():
            entry = self._check_entry(signal_id, config, today, yesterday)
            if entry:
                overlay_states[signal_id] = {
                    'active': True,
                    'direction': entry['direction'],
                    'overlay_type': config.get('overlay_type', 'GENERIC'),
                }
                if config.get('overlay_type') == 'CRISIS_MODE':
                    crisis_mode = True
                    logger.warning(f"CRISIS MODE ACTIVE: {entry['reason']}")
                elif config.get('overlay_type') == 'CALENDAR_BOOST':
                    january_boost = True
                    logger.info(f"CALENDAR BOOST: {entry['reason']}")
                if not dry_run:
                    self._record_entry(signal_id, entry, today, as_of_date, indicators,
                                       trade_type='OVERLAY')
            else:
                overlay_states[signal_id] = {'active': False}

        # Build signal actions dict for scoring engine
        signal_actions = {}
        for signal_id in list(SIGNALS.keys()) + list(SHADOW_SIGNALS.keys()):
            action = None
            for e in entries + shadow_entries:
                if e['signal_id'] == signal_id:
                    action = f"ENTER_{e['direction']}"
            for x in exits + shadow_exits:
                if x['signal_id'] == signal_id:
                    action = 'EXIT'
            signal_actions[signal_id] = {'action': action}

        result = {
            'date': str(as_of_date),
            'entries': entries,
            'exits': exits,
            'shadow_entries': shadow_entries,
            'shadow_exits': shadow_exits,
            'overlay_states': overlay_states,
            'signal_actions': signal_actions,
            'indicators': indicators,
            'open_positions': len(open_positions) - len(exits) + len(entries),
            'open_shadow': len(open_shadow) - len(shadow_exits) + len(shadow_entries),
            'intraday_regime': intraday_regime,
            'expiry_info': {
                'is_expiry': expiry_info['is_any_expiry'],
                'instruments': expiry_info.get('expiry_instruments', []),
                'nifty_dte': expiry_info.get('days_to_nifty_expiry'),
                'bn_dte': expiry_info.get('days_to_banknifty_expiry'),
            },
        }

        # Log summary
        logger.info(f"Signal compute {as_of_date}: "
                    f"{len(entries)} entries, {len(exits)} exits, "
                    f"{len(shadow_entries)} shadow entries, "
                    f"VIX={indicators['india_vix']:.1f}")
        for e in entries:
            logger.info(f"  ENTRY: {e['signal_id']} {e['direction']} @ {e['price']:.0f}")
        for x in exits:
            logger.info(f"  EXIT:  {x['signal_id']} reason={x['reason']} @ {x['price']:.0f}")
        for e in shadow_entries:
            logger.info(f"  SHADOW ENTRY: {e['signal_id']} {e['direction']} @ {e['price']:.0f}")

        return result

    # ================================================================
    # ENTRY CHECKS
    # ================================================================

    def _check_entry(self, signal_id: str, config: dict,
                     today, yesterday) -> Optional[Dict]:
        """Dispatch to signal-specific entry check."""
        method = getattr(self, config['check_entry'], None)
        if method is None:
            return None  # Signal method not yet implemented
        return method(signal_id, config, today, yesterday)

    def _check_entry_dry20(self, signal_id, config, today, yesterday):
        """KAUFMAN_DRY_20: sma_10 < prev_close AND stoch_k_5 > dynamic threshold"""
        if pd.isna(today['sma_10']) or pd.isna(today['stoch_k_5']):
            return None

        prev_close = float(yesterday['close'])
        sma_10 = float(today['sma_10'])
        stoch_k_5 = float(today['stoch_k_5'])

        # VIX-adaptive stochastic threshold
        vix = float(today.get('india_vix', 15)) if today is not None and pd.notna(today.get('india_vix')) else 15
        threshold = self._dynamic_stoch_threshold(50, vix)

        if sma_10 < prev_close and stoch_k_5 > threshold:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': float(today['close']),
                'reason': f'sma_10={sma_10:.0f} < prev_close={prev_close:.0f} '
                          f'AND stoch_k_5={stoch_k_5:.1f} > {threshold:.1f} (VIX={vix:.1f})',
            }
        return None

    def _check_entry_dry16(self, signal_id, config, today, yesterday):
        """KAUFMAN_DRY_16: pivot breakout + volatility filter.
        LONG requires low vol (hvol_6 < hvol_100) — breakout in calm market.
        SHORT requires high vol (hvol_6 >= hvol_100) — momentum in panic."""
        if pd.isna(today['hvol_6']) or pd.isna(today['hvol_100']):
            return None

        close = float(today['close'])
        low = float(today['low'])
        high = float(today['high'])
        r1 = float(today['r1'])
        s1 = float(today['s1'])
        pivot = float(today['pivot'])
        hvol_6 = float(today['hvol_6'])
        hvol_100 = float(today['hvol_100'])

        low_vol = hvol_6 < hvol_100
        high_vol = hvol_6 >= hvol_100

        # Long entry: close > r1 AND low >= pivot (low vol = calm breakout)
        if low_vol and close > r1 and low >= pivot:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} > r1={r1:.0f} AND low={low:.0f} >= pivot={pivot:.0f} '
                          f'AND hvol_6={hvol_6:.4f} < hvol_100={hvol_100:.4f}',
            }

        # Short entry: close < s1 (high vol = panic momentum confirms)
        if high_vol and close < s1 and high <= pivot:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'close={close:.0f} < s1={s1:.0f} AND high={high:.0f} <= pivot={pivot:.0f} '
                          f'AND hvol_6={hvol_6:.4f} >= hvol_100={hvol_100:.4f} (panic momentum)',
            }

        return None

    def _check_entry_dry12(self, signal_id, config, today, yesterday):
        """KAUFMAN_DRY_12: price/volume divergence OR pure price momentum.

        Primary edge: price moves on declining volume = weak conviction,
        likely to continue (momentum in trending markets).

        Fallback: when volume is unavailable (0 or NaN), use pure price
        momentum with stronger ADX gate (>30) to compensate.

        Filters:
          - ADX gate: skip choppy markets (adx_14 < 22, or < 30 if no volume)
          - Volume floor: skip very low volume (volume < volume_sma_20 * 0.8)
        """
        close = float(today['close'])
        prev_close = float(yesterday['close'])
        volume = float(today['volume']) if pd.notna(today.get('volume')) else 0
        prev_volume = float(yesterday['volume']) if pd.notna(yesterday.get('volume')) else 0

        # ADX gate: skip choppy markets
        adx_14 = float(today['adx_14']) if pd.notna(today.get('adx_14')) else 0

        has_volume = volume > 0 and prev_volume > 0

        if has_volume:
            # With volume data: standard divergence logic, ADX > 22
            if adx_14 < 22:
                return None
            volume_sma_20 = float(today['volume_sma_20']) if pd.notna(today.get('volume_sma_20')) else 0
            if volume_sma_20 > 0 and volume < volume_sma_20 * 0.8:
                return None

            if close > prev_close and volume < prev_volume:
                return {
                    'signal_id': signal_id,
                    'direction': 'LONG',
                    'price': close,
                    'reason': f'close={close:.0f} > prev={prev_close:.0f} '
                              f'vol={volume:.0f} < prev_vol={prev_volume:.0f} (ADX={adx_14:.0f})',
                }
            if close < prev_close and volume < prev_volume:
                return {
                    'signal_id': signal_id,
                    'direction': 'SHORT',
                    'price': close,
                    'reason': f'close={close:.0f} < prev={prev_close:.0f} '
                              f'vol={volume:.0f} < prev_vol={prev_volume:.0f} (ADX={adx_14:.0f})',
                }
        else:
            # No volume: pure price momentum with stricter ADX > 30
            if adx_14 < 30:
                return None
            pct_change = (close - prev_close) / prev_close
            if pct_change > 0.005:  # > 0.5% move up
                return {
                    'signal_id': signal_id,
                    'direction': 'LONG',
                    'price': close,
                    'reason': f'close={close:.0f} > prev={prev_close:.0f} ({pct_change:+.2%}) '
                              f'ADX={adx_14:.0f} (no vol data, price momentum)',
                }
            if pct_change < -0.005:  # > 0.5% move down
                return {
                    'signal_id': signal_id,
                    'direction': 'SHORT',
                    'price': close,
                    'reason': f'close={close:.0f} < prev={prev_close:.0f} ({pct_change:+.2%}) '
                              f'ADX={adx_14:.0f} (no vol data, price momentum)',
                }

        return None

    def _check_entry_gujral7(self, signal_id, config, today, yesterday):
        """GUJRAL_DRY_7: pivot crossover (shadow signal for regime detection)"""
        close = float(today['close'])
        pivot = float(today['pivot'])
        prev_close = float(yesterday['close'])

        # Long: close > pivot AND prev_close <= pivot (crosses above)
        if close > pivot and prev_close <= pivot:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} > pivot={pivot:.0f} '
                          f'AND prev_close={prev_close:.0f} <= pivot',
            }

        # Short: close < pivot AND prev_close >= pivot (crosses below)
        if close < pivot and prev_close >= pivot:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'close={close:.0f} < pivot={pivot:.0f} '
                          f'AND prev_close={prev_close:.0f} >= pivot',
            }

        return None

    def _check_entry_weekly_mom(self, signal_id, config, today, yesterday):
        """SSRN_WEEKLY_MOM: Weekly momentum — buy if close > sma_5 AND returns > 0.
        Tier B — Sharpe 1.69, 81% pass, 4/4 last windows. India-specific."""
        if pd.isna(today.get('sma_5')):
            return None

        close = float(today['close'])
        sma_5 = float(today['sma_5'])
        returns = float(today['returns']) if pd.notna(today.get('returns')) else 0

        if close > sma_5 and returns > 0:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'weekly_mom close={close:.0f} > sma_5={sma_5:.0f} ret={returns:.4f}',
            }
        return None

    def _check_exit_weekly_mom(self, pos, today, yesterday):
        """SSRN_WEEKLY_MOM exit: close < sma_5"""
        if pd.isna(today.get('sma_5')):
            return None
        price = float(today['close'])
        sma_5 = float(today['sma_5'])
        if price < sma_5:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < sma_5={sma_5:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_january(self, signal_id, config, today, yesterday):
        """SSRN_JANUARY_EFFECT: Calendar overlay — boost LONG sizing in January.
        DSR 1.00, WF 73%. Not standalone — modifies sizing for other signals."""
        bar_date = today.get('date', today.name if hasattr(today, 'name') else None)
        if bar_date is None:
            return None

        month = bar_date.month if hasattr(bar_date, 'month') else pd.Timestamp(bar_date).month

        if month == 1:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': float(today['close']),
                'reason': 'January effect active — LONG sizing boost',
            }
        return None

    def _check_exit_january(self, pos, today, yesterday):
        """JANUARY_EFFECT exit: end of January"""
        bar_date = today.get('date', today.name if hasattr(today, 'name') else None)
        if bar_date is None:
            return None
        month = bar_date.month if hasattr(bar_date, 'month') else pd.Timestamp(bar_date).month
        if month != 1:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': float(today['close']),
                'reason': 'January ended',
                'pnl': float(today['close']) - pos['entry_price'],
            }
        return None

    def _check_entry_bear_call(self, signal_id, config, today, yesterday):
        """L8_BEAR_CALL_RESISTANCE: sell call spread above resistance.
        DSR 1.80, Sharpe 1.87, 59% WR. Options Tier 2 survivor.
        Entry: IV rank > 40, VIX 12-25, RANGING regime."""
        vix = today.get('india_vix')
        if pd.isna(vix):
            return None
        vix = float(vix)

        # VIX filter
        if vix < 12 or vix > 25:
            return None

        # Price near resistance (close > R1 or near range high)
        if pd.isna(today.get('price_pos_20')):
            return None
        price_pos = float(today['price_pos_20'])

        # Only sell calls when price is in upper range (above 0.7)
        if price_pos > 0.7:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': float(today['close']),
                'reason': f'bear_call: price_pos={price_pos:.2f} VIX={vix:.0f} (sell call spread above resistance)',
            }
        return None

    def _check_exit_bear_call(self, pos, today, yesterday):
        """L8_BEAR_CALL exit: price drops below mid-range OR 60% profit target hit."""
        if pd.isna(today.get('price_pos_20')):
            return None
        price_pos = float(today['price_pos_20'])
        price = float(today['close'])

        # Exit when price drops (spread becomes profitable)
        if price_pos < 0.4:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (price_pos={price_pos:.2f} < 0.4)',
                'pnl': pos['entry_price'] - price,
            }
        return None

    # ================================================================
    # VALIDATED COMBO SIGNALS
    # ================================================================

    def _check_entry_combo_grimes32_dry12(self, signal_id, config, today, yesterday):
        """COMBO: KAUFMAN_DRY_12 fires first, then GRIMES_DRY_3_2 confirms within 5 days.
        GRIMES_3_2: higher high + higher low + ADX > 25 (trending confirmation).
        OOS Sharpe 5.08, 91 trades."""
        close = float(today['close'])
        as_of = today.get('date', today.name) if hasattr(today, 'name') else today.get('date')

        # Check if DRY_12 fired recently (within 5 days)
        # DRY_12 fires when: close > prev_close AND volume < prev_volume
        dry12_fires = (float(today['close']) > float(yesterday['close']) and
                       float(today['volume']) < float(yesterday['volume']))
        dry12_fires_short = (float(today['close']) < float(yesterday['close']) and
                             float(today['volume']) < float(yesterday['volume']))

        if dry12_fires:
            self._combo_pending['GRIMES32_DRY12_LONG'] = {
                'direction': 'LONG', 'fire_date': as_of, 'price': close, 'ttl': 5
            }
        if dry12_fires_short:
            self._combo_pending['GRIMES32_DRY12_SHORT'] = {
                'direction': 'SHORT', 'fire_date': as_of, 'price': close, 'ttl': 5
            }

        # Decrement TTL and check GRIMES_3_2 confirmation
        for key in list(self._combo_pending.keys()):
            if not key.startswith('GRIMES32_DRY12'):
                continue
            pending = self._combo_pending[key]
            pending['ttl'] -= 1
            if pending['ttl'] < 0:
                del self._combo_pending[key]
                continue

            # GRIMES_3_2: higher high + higher low + ADX > 25
            adx = float(today['adx_14']) if pd.notna(today.get('adx_14')) else 20
            if adx <= 25:
                continue

            if (pending['direction'] == 'LONG' and
                    float(today['high']) > float(yesterday['high']) and
                    float(today['low']) > float(yesterday['low'])):
                del self._combo_pending[key]
                return {
                    'signal_id': signal_id,
                    'direction': 'LONG',
                    'price': close,
                    'reason': f'COMBO SEQ_5: DRY_12 + GRIMES_3_2 LONG (ADX={adx:.0f})',
                }

            if (pending['direction'] == 'SHORT' and
                    float(today['low']) < float(yesterday['low']) and
                    float(today['high']) < float(yesterday['high'])):
                del self._combo_pending[key]
                return {
                    'signal_id': signal_id,
                    'direction': 'SHORT',
                    'price': close,
                    'reason': f'COMBO SEQ_5: DRY_12 + GRIMES_3_2 SHORT (ADX={adx:.0f})',
                }

        return None

    def _check_exit_combo_grimes32(self, pos, today, yesterday):
        """GRIMES_3_2 exit: lower low (LONG) or higher high (SHORT)."""
        price = float(today['close'])
        if pos['direction'] == 'LONG' and float(today['low']) < float(yesterday['low']):
            return {
                'signal_id': pos['signal_id'], 'direction': 'LONG',
                'price': price, 'reason': 'signal_exit (lower low)',
                'pnl': price - pos['entry_price'],
            }
        elif pos['direction'] == 'SHORT' and float(today['high']) > float(yesterday['high']):
            return {
                'signal_id': pos['signal_id'], 'direction': 'SHORT',
                'price': price, 'reason': 'signal_exit (higher high)',
                'pnl': pos['entry_price'] - price,
            }
        return None

    def _check_entry_combo_grimes60_dry8(self, signal_id, config, today, yesterday):
        """COMBO AND: GRIMES_DRY_6_0 + GUJRAL_DRY_8 both fire same day.
        GRIMES_6_0: ADX>25 + close>sma_20 + ATR_14<ATR_20 + RSI<70 (pullback in trend)
        GUJRAL_8: close>pivot AND open>pivot
        OOS Sharpe 6.95, 46 trades, 85% WR. LONG only."""
        close = float(today['close'])

        # GUJRAL_DRY_8 conditions
        if pd.isna(today.get('pivot')): return None
        pivot = float(today['pivot'])
        if close <= pivot or float(today['open']) <= pivot:
            return None

        # GRIMES_DRY_6_0 conditions
        adx = float(today['adx_14']) if pd.notna(today.get('adx_14')) else 20
        if adx <= 25: return None
        if pd.isna(today.get('sma_20')): return None
        sma_20 = float(today['sma_20'])
        if close <= sma_20: return None

        atr_14 = float(today['atr_14']) if pd.notna(today.get('atr_14')) else 999
        atr_20 = float(today['atr_20']) if pd.notna(today.get('atr_20')) else 0
        if atr_14 >= atr_20: return None  # Not a pullback (vol expanding)

        rsi = float(today['rsi_14']) if pd.notna(today.get('rsi_14')) else 50
        if rsi >= 70: return None  # Overbought

        return {
            'signal_id': signal_id,
            'direction': 'LONG',
            'price': close,
            'reason': f'COMBO AND: GRIMES_6_0 + GUJRAL_8 (ADX={adx:.0f} RSI={rsi:.0f} pivot={pivot:.0f})',
        }

    def _check_exit_combo_grimes60(self, pos, today, yesterday):
        """GRIMES_6_0 exit: close<sma_20 AND RSI>80 (exhaustion)."""
        price = float(today['close'])
        if pd.isna(today.get('sma_20')): return None
        sma_20 = float(today['sma_20'])
        rsi = float(today['rsi_14']) if pd.notna(today.get('rsi_14')) else 50

        if price < sma_20:
            return {
                'signal_id': pos['signal_id'], 'direction': pos['direction'],
                'price': price, 'reason': f'signal_exit (close < sma_20)',
                'pnl': price - pos['entry_price'],
            }
        if rsi > 80:
            return {
                'signal_id': pos['signal_id'], 'direction': pos['direction'],
                'price': price, 'reason': f'signal_exit (RSI={rsi:.0f} > 80)',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_combo_grimes50_dry16(self, signal_id, config, today, yesterday):
        """COMBO: GRIMES_DRY_5_0 fires (breakout: range>ATR + volume surge),
        then KAUFMAN_DRY_16 confirms within 3-5 days (pivot breakout + low vol).
        WF 10/13 windows, Sharpe 4.69-5.16, PF 2.62-2.67."""
        close = float(today['close'])
        as_of = today.get('date', today.name) if hasattr(today, 'name') else today.get('date')

        # GRIMES_DRY_5_0: close crosses above pivot + range > ATR + volume surge
        if pd.notna(today.get('pivot')) and pd.notna(today.get('atr_14')):
            pivot = float(today['pivot'])
            day_range = float(today['high']) - float(today['low'])
            atr_val = float(today['atr_14'])
            prev_close = float(yesterday['close'])

            grimes_long = (close > pivot and prev_close <= pivot and
                           day_range > atr_val)
            grimes_short = (close < pivot and prev_close >= pivot and
                            day_range > atr_val)

            if grimes_long:
                self._combo_pending['GRIMES50_DRY16_LONG'] = {
                    'direction': 'LONG', 'fire_date': as_of, 'price': close,
                    'ttl': config.get('combo_window', 5)
                }
            if grimes_short:
                self._combo_pending['GRIMES50_DRY16_SHORT'] = {
                    'direction': 'SHORT', 'fire_date': as_of, 'price': close,
                    'ttl': config.get('combo_window', 5)
                }

        # Check pending Grimes signals for DRY_16 confirmation
        for key in list(self._combo_pending.keys()):
            if not key.startswith('GRIMES50_DRY16'):
                continue
            pending = self._combo_pending[key]
            pending['ttl'] -= 1
            if pending['ttl'] < 0:
                del self._combo_pending[key]
                continue

            # DRY_16 confirmation: close > R1 + hvol_6 < hvol_100 (LONG)
            if (pd.isna(today.get('r1')) or pd.isna(today.get('hvol_6'))
                    or pd.isna(today.get('hvol_100'))):
                continue

            r1 = float(today['r1'])
            s1 = float(today['s1']) if pd.notna(today.get('s1')) else 0
            hvol_6 = float(today['hvol_6'])
            hvol_100 = float(today['hvol_100'])

            if hvol_6 >= hvol_100:
                continue  # DRY_16 needs low recent vol

            if pending['direction'] == 'LONG' and close > r1:
                del self._combo_pending[key]
                return {
                    'signal_id': signal_id, 'direction': 'LONG', 'price': close,
                    'reason': f'COMBO SEQ: GRIMES_5_0 breakout + DRY_16 pivot confirm',
                }
            if pending['direction'] == 'SHORT' and close < s1:
                del self._combo_pending[key]
                return {
                    'signal_id': signal_id, 'direction': 'SHORT', 'price': close,
                    'reason': f'COMBO SEQ: GRIMES_5_0 breakdown + DRY_16 pivot confirm',
                }

        return None

    def _check_exit_combo_grimes50(self, pos, today, yesterday):
        """GRIMES_5_0 exit: close crosses back through pivot + range contracts."""
        price = float(today['close'])
        if pd.isna(today.get('pivot')) or pd.isna(today.get('atr_14')):
            return None
        pivot = float(today['pivot'])
        day_range = float(today['high']) - float(today['low'])
        atr_val = float(today['atr_14'])

        if pos['direction'] == 'LONG' and price < pivot and day_range < atr_val:
            return {
                'signal_id': pos['signal_id'], 'direction': 'LONG',
                'price': price, 'reason': 'signal_exit (below pivot, range contracted)',
                'pnl': price - pos['entry_price'],
            }
        if pos['direction'] == 'SHORT' and price > pivot and day_range < atr_val:
            return {
                'signal_id': pos['signal_id'], 'direction': 'SHORT',
                'price': price, 'reason': 'signal_exit (above pivot, range contracted)',
                'pnl': pos['entry_price'] - price,
            }
        return None

    def _check_entry_crisis_short(self, signal_id, config, today, yesterday):
        """CRISIS_SHORT overlay: high vol + close<sma_50 + ADX>25.
        When active: blocks new LONG entries, boosts SHORT sizing 2x.
        COVID: +1,752 pts from 15 trades. Pure crisis signal.

        Accepts VIX>25 OR hvol_6 > 2*hvol_100 (annualized vol proxy)
        to handle missing VIX data."""
        if pd.isna(today.get('sma_50')) or pd.isna(today.get('adx_14')):
            return None

        # Check high-vol condition: VIX > 25 OR hvol_6 > 2x hvol_100
        vix = today.get('india_vix')
        hvol_6 = today.get('hvol_6')
        hvol_100 = today.get('hvol_100')

        high_vol = False
        vol_reason = ''
        if pd.notna(vix) and float(vix) > 25:
            high_vol = True
            vol_reason = f'VIX={float(vix):.0f}'
        elif pd.notna(hvol_6) and pd.notna(hvol_100) and float(hvol_100) > 0:
            if float(hvol_6) > 2.0 * float(hvol_100):
                high_vol = True
                vol_reason = f'hvol_6={float(hvol_6):.2%} > 2x hvol_100={float(hvol_100):.2%}'

        if not high_vol:
            return None

        close = float(today['close'])
        sma_50 = float(today['sma_50'])
        adx_val = float(today['adx_14'])

        if close < sma_50 and adx_val > 25:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'CRISIS: {vol_reason} close={close:.0f} < sma_50={sma_50:.0f} ADX={adx_val:.0f}',
            }
        return None

    def _check_exit_crisis_short(self, pos, today, yesterday):
        """CRISIS_SHORT exit: close > sma_20 (crisis easing)"""
        if pd.isna(today.get('sma_20')):
            return None
        price = float(today['close'])
        sma_20 = float(today['sma_20'])
        if price > sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'crisis_exit (close={price:.0f} > sma_20={sma_20:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    # ── GLOBAL_OVERNIGHT_COMPOSITE overlay ──────────────────────────

    def _check_entry_global_composite(self, signal_id, config, today, yesterday):
        """GLOBAL_OVERNIGHT_COMPOSITE: size modifier from global pre-market context.
        Reads pre-computed context from global_pre_market cron (8:30 AM).
        Uses S&P overnight return + VIX change + GIFT gap as proxy when
        the full composite table is unavailable."""
        # Try reading today's pre-market context from DB
        try:
            if hasattr(self, 'conn') and self.conn and not self.conn.closed:
                cur = self.conn.cursor()
                bar_date = today.get('date', today.name if hasattr(today, 'name') else None)
                cur.execute(
                    "SELECT direction, size_modifier, risk_off, composite_score "
                    "FROM global_pre_market_context WHERE date = %s",
                    (bar_date,)
                )
                row = cur.fetchone()
                cur.close()
                if row:
                    direction, size_mod, risk_off, score = row
                    return {
                        'signal_id': signal_id,
                        'direction': direction or 'LONG',
                        'price': float(today['close']),
                        'reason': f'global_composite: score={score:.2f} '
                                  f'size_mod={size_mod:.2f} risk_off={risk_off}',
                    }
        except Exception:
            pass  # Table may not exist yet — fall through to proxy

        # Proxy: use VIX level as simple risk-off indicator
        vix = today.get('india_vix')
        if pd.notna(vix) and float(vix) > 25:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': float(today['close']),
                'reason': f'global_composite_proxy: VIX={float(vix):.1f} > 25 (risk-off)',
            }

        return None

    def _check_exit_global_composite(self, pos, today, yesterday):
        """GLOBAL_OVERNIGHT_COMPOSITE exit: overlay resets daily, always exit."""
        return {
            'signal_id': pos['signal_id'],
            'direction': pos['direction'],
            'price': float(today['close']),
            'reason': 'global_composite: daily reset',
            'pnl': (float(today['close']) - pos['entry_price'])
                    * (1 if pos['direction'] == 'LONG' else -1),
        }

    # ── PCR_AUTOTRENDER overlay ─────────────────────────────────────

    def _check_entry_pcr(self, signal_id, config, today, yesterday):
        """PCR_AUTOTRENDER: contrarian at extremes, confirmation in middle.
        PCR > 1.5 → contrarian LONG, PCR < 0.7 → contrarian SHORT.
        Size modifier 0.70x–1.30x."""
        pcr = today.get('pcr_oi')
        if pd.isna(pcr):
            return None
        pcr = float(pcr)

        if pcr >= 1.5:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': float(today['close']),
                'reason': f'pcr_contrarian: PCR={pcr:.2f} >= 1.5 (extreme fear → buy)',
            }
        elif pcr <= 0.7:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': float(today['close']),
                'reason': f'pcr_contrarian: PCR={pcr:.2f} <= 0.7 (extreme greed → sell)',
            }
        return None

    def _check_exit_pcr(self, pos, today, yesterday):
        """PCR_AUTOTRENDER exit: PCR reverts to neutral zone (0.9–1.2)."""
        pcr = today.get('pcr_oi')
        if pd.isna(pcr):
            return None
        pcr = float(pcr)

        if 0.9 <= pcr <= 1.2:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': float(today['close']),
                'reason': f'pcr_exit: PCR={pcr:.2f} back to neutral',
                'pnl': (float(today['close']) - pos['entry_price'])
                        * (1 if pos['direction'] == 'LONG' else -1),
            }
        return None

    # ── ROLLOVER_ANALYSIS overlay ───────────────────────────────────

    def _check_entry_rollover(self, signal_id, config, today, yesterday):
        """ROLLOVER_ANALYSIS: OI migration across expiries.
        Long buildup = rollover_pct > 70% with positive basis → bullish overlay.
        Short buildup = rollover_pct > 70% with negative basis → bearish overlay."""
        rollover_pct = today.get('rollover_pct')
        if pd.isna(rollover_pct):
            return None
        rollover_pct = float(rollover_pct)

        basis = today.get('futures_basis')
        if pd.isna(basis):
            return None
        basis = float(basis)

        if rollover_pct > 70:
            direction = 'LONG' if basis > 0 else 'SHORT'
            return {
                'signal_id': signal_id,
                'direction': direction,
                'price': float(today['close']),
                'reason': f'rollover: {rollover_pct:.0f}% rollover, '
                          f'basis={basis:.1f} → {direction}',
            }
        return None

    def _check_exit_rollover(self, pos, today, yesterday):
        """ROLLOVER_ANALYSIS exit: overlay resets each expiry cycle."""
        return {
            'signal_id': pos['signal_id'],
            'direction': pos['direction'],
            'price': float(today['close']),
            'reason': 'rollover: expiry cycle reset',
            'pnl': (float(today['close']) - pos['entry_price'])
                    * (1 if pos['direction'] == 'LONG' else -1),
        }

    def _check_entry_gujral13(self, signal_id, config, today, yesterday):
        """GUJRAL_DRY_13: close > prev_high (breakout above previous candle)
        DSR 2.35, WF 85% pass rate, Sharpe 2.42"""
        close = float(today['close'])
        prev_high = float(yesterday['high'])

        if close > prev_high:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} > prev_high={prev_high:.0f}',
            }
        return None

    def _check_exit_gujral13(self, pos, today, yesterday):
        """GUJRAL_DRY_13 exit: close < prev_low"""
        price = float(today['close'])
        prev_low = float(yesterday['low'])
        if price < prev_low:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < prev_low={prev_low:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_chan_at4(self, signal_id, config, today, yesterday):
        """CHAN_AT_DRY_4: BB mean-reversion. DSR 4.27, WF 45% pass (wide windows).
        Long: close < sma_20 AND bb_pct_b < 0 (below lower BB)
        Short: close > sma_20 AND bb_pct_b > 1.0 (above upper BB)"""
        if pd.isna(today.get('sma_20')) or pd.isna(today.get('bb_pct_b')):
            return None

        close = float(today['close'])
        sma_20 = float(today['sma_20'])
        bb_pct_b = float(today['bb_pct_b'])

        # Long: price below lower BB (mean-reversion buy)
        if close < sma_20 and bb_pct_b < 0:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} < sma_20={sma_20:.0f} AND bb_pct_b={bb_pct_b:.3f} < 0',
            }

        # Short: price above upper BB (mean-reversion sell)
        if close > sma_20 and bb_pct_b > 1.0:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'close={close:.0f} > sma_20={sma_20:.0f} AND bb_pct_b={bb_pct_b:.3f} > 1.0',
            }

        return None

    def _check_exit_chan_at4(self, pos, today, yesterday):
        """CHAN_AT_DRY_4 exit: close crosses back to sma_20"""
        if pd.isna(today.get('sma_20')):
            return None
        price = float(today['close'])
        sma_20 = float(today['sma_20'])

        if pos['direction'] == 'LONG' and price >= sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} >= sma_20={sma_20:.0f})',
                'pnl': price - pos['entry_price'],
            }
        elif pos['direction'] == 'SHORT' and price <= sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} <= sma_20={sma_20:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    def _check_entry_gujral8(self, signal_id, config, today, yesterday):
        """GUJRAL_DRY_8: close > pivot AND open > pivot (LONG only).
        TIER A — Sharpe 2.82, 81% pass rate, 11.4% worst DD.
        Only Tier A survivor across all walk-forward runs."""
        close = float(today['close'])
        open_ = float(today['open'])
        pivot = float(today['pivot'])

        if close > pivot and open_ > pivot:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} > pivot={pivot:.0f} AND open={open_:.0f} > pivot',
            }
        return None

    def _check_exit_gujral8(self, pos, today, yesterday):
        """GUJRAL_DRY_8 exit: close < pivot"""
        price = float(today['close'])
        pivot = float(today['pivot'])
        if price < pivot:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < pivot={pivot:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_candlestick0(self, signal_id, config, today, yesterday):
        """CANDLESTICK_DRY_0: White candle — close > open, small wicks.
        Tier B — Sharpe 1.31, 85% pass rate. Independent source."""
        close = float(today['close'])
        open_ = float(today['open'])
        body = abs(close - open_)

        if body <= 0:
            return None

        upper_wick = float(today['high']) - max(close, open_)
        lower_wick = min(close, open_) - float(today['low'])

        # White candle: close > open, both wicks shorter than body
        if close > open_ and upper_wick < body and lower_wick < body:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'white_candle body={body:.0f} uwk={upper_wick:.0f} lwk={lower_wick:.0f}',
            }
        return None

    def _check_exit_candlestick0(self, pos, today, yesterday):
        """CANDLESTICK_DRY_0 exit: close < open (bearish candle)"""
        price = float(today['close'])
        open_ = float(today['open'])
        if price < open_:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < open={open_:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_dry7(self, signal_id, config, today, yesterday):
        """KAUFMAN_DRY_7: close crosses above sma_5.
        Tier B — Sharpe 3.40. Correlation risk with DRY_20/16/12."""
        if pd.isna(today['sma_5']):
            return None

        close = float(today['close'])
        prev_close = float(yesterday['close'])
        sma_5 = float(today['sma_5'])
        prev_sma_5 = float(yesterday['sma_5']) if pd.notna(yesterday.get('sma_5')) else sma_5

        # Crosses above: today > sma_5 AND yesterday <= sma_5
        if close > sma_5 and prev_close <= prev_sma_5:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'close={close:.0f} crosses_above sma_5={sma_5:.0f}',
            }
        return None

    def _check_exit_dry7(self, pos, today, yesterday):
        """KAUFMAN_DRY_7 exit: close crosses below sma_5"""
        if pd.isna(today['sma_5']):
            return None
        price = float(today['close'])
        prev_close = float(yesterday['close'])
        sma_5 = float(today['sma_5'])
        prev_sma_5 = float(yesterday['sma_5']) if pd.notna(yesterday.get('sma_5')) else sma_5

        if price < sma_5 and prev_close >= prev_sma_5:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} crosses_below sma_5={sma_5:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_gujral9(self, signal_id, config, today, yesterday):
        """GUJRAL_DRY_9: open < pivot AND low > s1 AND close > open.
        Tier B — Sharpe 1.18, 69% pass, 13% DD. Pivot retracement buy."""
        close = float(today['close'])
        open_ = float(today['open'])
        low = float(today['low'])
        pivot = float(today['pivot'])
        s1 = float(today['s1'])

        if open_ < pivot and low > s1 and close > open_:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'open={open_:.0f} < pivot={pivot:.0f} AND low={low:.0f} > s1={s1:.0f} AND close > open',
            }
        return None

    def _check_exit_gujral9(self, pos, today, yesterday):
        """GUJRAL_DRY_9 exit: close < s1"""
        price = float(today['close'])
        s1 = float(today['s1'])
        if price < s1:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < s1={s1:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    # ================================================================
    # BULKOWSKI PATTERN SIGNALS
    # ================================================================

    def _check_entry_falling_vol(self, signal_id, config, today, yesterday):
        """BULKOWSKI_FALLING_VOLUME_TREND_IN: volume declining = consolidation before breakout.
        Tier A — Sharpe 1.97, 85% pass, 17.5% DD. Works all regimes."""
        if pd.isna(today.get('vol_ratio_20')):
            return None
        volume = float(today['volume'])
        prev_volume = float(yesterday['volume'])
        vol_ratio = float(today['vol_ratio_20'])

        if volume < prev_volume and vol_ratio < 1.0:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': float(today['close']),
                'reason': f'vol={volume:.0f} < prev={prev_volume:.0f} AND vol_ratio={vol_ratio:.2f} < 1.0',
            }
        return None

    def _check_exit_falling_vol(self, pos, today, yesterday):
        """FALLING_VOLUME exit: volume > prev_volume (volume returns)"""
        volume = float(today['volume'])
        prev_volume = float(yesterday['volume'])
        if volume > prev_volume:
            price = float(today['close'])
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (vol={volume:.0f} > prev={prev_volume:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_eadb(self, signal_id, config, today, yesterday):
        """BULKOWSKI_EADB_EARLY_ATTEMPT_TO: SHORT on bearish reversal from upper range.
        Tier A — Sharpe 1.67, 65% pass, 10.2% DD. Only SHORT signal.
        REGIME GATE: blocked when HIGH_VOL (Sharpe -0.82 in high vol)."""
        # HIGH_VOL regime gate — do NOT short when VIX is spiking
        vix = today.get('india_vix')
        if pd.notna(vix) and float(vix) > 22:
            return None

        if pd.isna(today.get('price_pos_20')):
            return None

        close = float(today['close'])
        prev_close = float(yesterday['close'])
        volume = float(today['volume'])
        prev_volume = float(yesterday['volume'])
        price_pos = float(today['price_pos_20'])

        if close < prev_close and volume > prev_volume and price_pos > 0.6:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'close < prev AND vol_up AND price_pos={price_pos:.2f} > 0.6 (VIX={float(vix) if pd.notna(vix) else "?"})',
            }
        return None

    def _check_exit_eadb(self, pos, today, yesterday):
        """EADB exit: close > sma_20"""
        if pd.isna(today.get('sma_20')):
            return None
        price = float(today['close'])
        sma_20 = float(today['sma_20'])
        if price > sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} > sma_20={sma_20:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    def _check_entry_cup_handle(self, signal_id, config, today, yesterday):
        """BULKOWSKI_CUP_HANDLE: buy at range low with volume surge + trend.
        Tier B — Sharpe 2.53, 50% pass. Sharpe 4.21 in RANGING."""
        if pd.isna(today.get('price_pos_20')) or pd.isna(today.get('adx_14')):
            return None

        price_pos = float(today['price_pos_20'])
        vol_ratio = float(today['vol_ratio_20']) if pd.notna(today.get('vol_ratio_20')) else 0
        adx_val = float(today['adx_14'])
        close = float(today['close'])

        # Long: range low + volume + trend
        if price_pos < 0.2 and vol_ratio > 1.2 and adx_val > 20:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'cup_handle LONG pos={price_pos:.2f} vol_r={vol_ratio:.1f} adx={adx_val:.0f}',
            }
        # Short: range high + volume + trend
        if price_pos > 0.8 and vol_ratio > 1.2 and adx_val > 20:
            return {
                'signal_id': signal_id,
                'direction': 'SHORT',
                'price': close,
                'reason': f'cup_handle SHORT pos={price_pos:.2f} vol_r={vol_ratio:.1f} adx={adx_val:.0f}',
            }
        return None

    def _check_exit_cup_handle(self, pos, today, yesterday):
        """CUP_HANDLE exit: close crosses sma_20"""
        if pd.isna(today.get('sma_20')):
            return None
        price = float(today['close'])
        sma_20 = float(today['sma_20'])
        if pos['direction'] == 'LONG' and price < sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < sma_20={sma_20:.0f})',
                'pnl': price - pos['entry_price'],
            }
        elif pos['direction'] == 'SHORT' and price > sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} > sma_20={sma_20:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    def _check_entry_adam_eve(self, signal_id, config, today, yesterday):
        """BULKOWSKI_ADAM_AND_EVE_OR: hammer at range low with vol expansion.
        Tier B — Sharpe 1.43, 66% WR (highest), 7.6% DD (tightest).
        Fast-track SHADOW — 5-day hold accumulates data quickly."""
        close = float(today['close'])
        low = float(today['low'])
        high = float(today['high'])
        open_ = float(today['open'])

        if pd.isna(today.get('atr_14')) or pd.isna(today.get('price_pos_20')):
            return None

        atr = float(today['atr_14'])
        price_pos = float(today['price_pos_20'])
        bar_range = high - low
        body = abs(close - open_)
        lower_wick = min(close, open_) - low

        # Range > ATR, at range low, big lower wick (hammer)
        if bar_range > atr and price_pos < 0.2 and body > 0 and lower_wick > body:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'adam_eve range={bar_range:.0f} > atr={atr:.0f} pos={price_pos:.2f} lwk={lower_wick:.0f} > body={body:.0f}',
            }
        return None

    def _check_exit_adam_eve(self, pos, today, yesterday):
        """ADAM_EVE exit: next bullish close (close > open)"""
        price = float(today['close'])
        open_ = float(today['open'])
        if price > open_:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} > open={open_:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_round_bottom(self, signal_id, config, today, yesterday):
        """BULKOWSKI_ROUND_BOTTOM_RDB: buy at range low with mild trend, gap down open.
        Tier B — Sharpe 2.13, 46% pass."""
        if pd.isna(today.get('price_pos_20')) or pd.isna(today.get('adx_14')):
            return None

        price_pos = float(today['price_pos_20'])
        adx_val = float(today['adx_14'])
        open_ = float(today['open'])
        prev_close = float(yesterday['close'])

        if price_pos < 0.3 and adx_val > 15 and open_ <= prev_close:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': float(today['close']),
                'reason': f'round_bottom pos={price_pos:.2f} adx={adx_val:.0f} open <= prev_close',
            }
        return None

    def _check_exit_round_bottom(self, pos, today, yesterday):
        """ROUND_BOTTOM exit: close < sma_20"""
        if pd.isna(today.get('sma_20')):
            return None
        price = float(today['close'])
        sma_20 = float(today['sma_20'])
        if price < sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < sma_20={sma_20:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_entry_eadt_busted(self, signal_id, config, today, yesterday):
        """BULKOWSKI_EADT_BUSTED: buy on up day with ADX trend.
        Tier B — Sharpe 2.58, but 30.9% DD and 2/4 last windows. Monitor closely."""
        if pd.isna(today.get('adx_14')):
            return None

        close = float(today['close'])
        prev_close = float(yesterday['close'])
        adx_val = float(today['adx_14'])
        returns = (close - prev_close) / prev_close if prev_close > 0 else 0

        if close > prev_close and returns > -0.05 and adx_val > 20:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',
                'price': close,
                'reason': f'eadt_busted close > prev AND ret={returns:.3f} AND adx={adx_val:.0f}',
            }
        return None

    def _check_exit_eadt_busted(self, pos, today, yesterday):
        """EADT_BUSTED exit: close < sma_20"""
        if pd.isna(today.get('sma_20')):
            return None
        price = float(today['close'])
        sma_20 = float(today['sma_20'])
        if price < sma_20:
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < sma_20={sma_20:.0f})',
                'pnl': price - pos['entry_price'],
            }
        return None

    def _check_exit_gujral7(self, pos, today, yesterday):
        """GUJRAL_DRY_7 exit: close crosses back through pivot"""
        price = float(today['close'])
        pivot = float(today['pivot'])
        if pos['direction'] == 'LONG' and price < pivot:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'LONG',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < pivot={pivot:.0f})',
                'pnl': price - pos['entry_price'],
            }
        elif pos['direction'] == 'SHORT' and price > pivot:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'SHORT',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} > pivot={pivot:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    # ================================================================
    # EXIT CHECKS
    # ================================================================

    def _check_exit(self, pos: dict, today, yesterday,
                    as_of_date: date) -> Optional[Dict]:
        """Check if an open position should be exited."""
        signal_id = pos['signal_id']
        config = (SIGNALS.get(signal_id) or SHADOW_SIGNALS.get(signal_id)
                  or OVERLAY_SIGNALS.get(signal_id))
        if not config:
            return None

        entry_price = pos['entry_price']
        current_price = float(today['close'])
        direction = pos['direction']

        # Stop loss
        if direction == 'LONG':
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price

        if loss_pct >= config['stop_loss_pct']:
            return {
                'signal_id': signal_id,
                'direction': direction,
                'price': current_price,
                'reason': 'stop_loss',
                'pnl': -loss_pct * entry_price if direction == 'LONG' else loss_pct * entry_price,
            }

        # Take profit
        if config['take_profit_pct'] > 0:
            if direction == 'LONG':
                gain_pct = (current_price - entry_price) / entry_price
            else:
                gain_pct = (entry_price - current_price) / entry_price

            if gain_pct >= config['take_profit_pct']:
                return {
                    'signal_id': signal_id,
                    'direction': direction,
                    'price': current_price,
                    'reason': 'take_profit',
                    'pnl': gain_pct * entry_price,
                }

        # Hold days max
        if config['hold_days_max'] > 0:
            entry_date = pos['entry_date']
            if isinstance(entry_date, str):
                entry_date = datetime.strptime(entry_date, '%Y-%m-%d').date()
            days_held = (as_of_date - entry_date).days
            if days_held >= config['hold_days_max']:
                if direction == 'LONG':
                    pnl = current_price - entry_price
                else:
                    pnl = entry_price - current_price
                return {
                    'signal_id': signal_id,
                    'direction': direction,
                    'price': current_price,
                    'reason': 'hold_days_max',
                    'pnl': pnl,
                }

        # Signal-specific exit (pass days_held for signals that need it)
        if config['hold_days_max'] > 0:
            entry_date = pos['entry_date']
            if isinstance(entry_date, str):
                entry_date = datetime.strptime(entry_date, '%Y-%m-%d').date()
            pos['days_held'] = (as_of_date - entry_date).days
        method = getattr(self, config['check_exit'], None)
        if method is None:
            return None  # Exit method not yet implemented
        return method(pos, today, yesterday)

    def _check_exit_dry20(self, pos, today, yesterday):
        """KAUFMAN_DRY_20 exit: stoch_k_5 <= dynamic threshold"""
        if pd.isna(today['stoch_k_5']):
            return None
        stoch_k_5 = float(today['stoch_k_5'])

        # VIX-adaptive stochastic threshold
        vix = float(today.get('india_vix', 15)) if today is not None and pd.notna(today.get('india_vix')) else 15
        threshold = self._dynamic_stoch_threshold(50, vix)

        if stoch_k_5 <= threshold:
            price = float(today['close'])
            pnl = price - pos['entry_price']  # always LONG
            return {
                'signal_id': pos['signal_id'],
                'direction': pos['direction'],
                'price': price,
                'reason': f'signal_exit (stoch_k_5={stoch_k_5:.1f} <= {threshold:.1f}, VIX={vix:.1f})',
                'pnl': pnl,
            }
        return None

    def _check_exit_dry16(self, pos, today, yesterday):
        """KAUFMAN_DRY_16 exit: low < pivot (long) or high > r1 (short)"""
        price = float(today['close'])
        if pos['direction'] == 'LONG':
            if float(today['low']) < float(today['pivot']):
                return {
                    'signal_id': pos['signal_id'],
                    'direction': 'LONG',
                    'price': price,
                    'reason': f'signal_exit (low={today["low"]:.0f} < pivot={today["pivot"]:.0f})',
                    'pnl': price - pos['entry_price'],
                }
        else:
            if float(today['high']) > float(today['r1']):
                return {
                    'signal_id': pos['signal_id'],
                    'direction': 'SHORT',
                    'price': price,
                    'reason': f'signal_exit (high={today["high"]:.0f} > r1={today["r1"]:.0f})',
                    'pnl': pos['entry_price'] - price,
                }
        return None

    def _check_exit_dry12(self, pos, today, yesterday):
        """KAUFMAN_DRY_12 exit: close reversal vs prev_close."""
        price = float(today['close'])
        prev_close = float(yesterday['close'])

        if pos['direction'] == 'LONG' and price < prev_close:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'LONG',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} < prev_close={prev_close:.0f})',
                'pnl': price - pos['entry_price'],
            }
        elif pos['direction'] == 'SHORT' and price > prev_close:
            return {
                'signal_id': pos['signal_id'],
                'direction': 'SHORT',
                'price': price,
                'reason': f'signal_exit (close={price:.0f} > prev_close={prev_close:.0f})',
                'pnl': pos['entry_price'] - price,
            }
        return None

    # ================================================================
    # POSITION SIZING
    # ================================================================

    def _get_sizing_context(self) -> dict:
        """Load recent trade stats for sizing decisions."""
        try:
            cur = self.conn.cursor()
            # Get last 20 closed trades for streak/drawdown calculation
            cur.execute("""
                SELECT net_pnl, return_pct FROM trades
                WHERE trade_type = 'PAPER' AND exit_date IS NOT NULL
                ORDER BY exit_date DESC LIMIT 20
            """)
            rows = cur.fetchall()
        except Exception:
            rows = []

        if not rows:
            return {'consecutive_wins': 0, 'consecutive_losses': 0,
                    'drawdown_pct': 0.0}

        # Count consecutive wins/losses from most recent
        consec_wins = 0
        consec_losses = 0
        for pnl, _ in rows:
            if pnl is None:
                break
            if pnl > 0:
                if consec_losses > 0:
                    break
                consec_wins += 1
            else:
                if consec_wins > 0:
                    break
                consec_losses += 1

        # Compute drawdown from equity curve of last 20 trades
        cumulative = 0.0
        peak = 0.0
        drawdown = 0.0
        for pnl, _ in reversed(rows):
            if pnl is not None:
                cumulative += float(pnl)
                peak = max(peak, cumulative)
                dd = (peak - cumulative) / max(peak, 1.0) if peak > 0 else 0.0
                drawdown = max(drawdown, dd)

        return {
            'consecutive_wins': consec_wins,
            'consecutive_losses': consec_losses,
            'drawdown_pct': drawdown,
        }

    def _compute_size_multiplier(self, today, sizing_ctx: dict) -> float:
        """Compute position size multiplier using the active sizing rule."""
        hvol_20 = float(today.get('hvol_20', 0.0)) if pd.notna(today.get('hvol_20', None)) else None

        risk_pct = self.sizing_rule.effective_risk_pct(
            consecutive_wins=sizing_ctx.get('consecutive_wins', 0),
            consecutive_losses=sizing_ctx.get('consecutive_losses', 0),
            drawdown_pct=sizing_ctx.get('drawdown_pct', 0.0),
            rolling_volatility=hvol_20,
        )

        # Convert to multiplier: 1.0 = baseline (1% risk)
        baseline = 0.01
        return round(risk_pct / baseline, 2) if baseline > 0 else 1.0

    # ================================================================
    # DATABASE OPERATIONS
    # ================================================================

    def _load_market_data(self, as_of_date: date) -> Optional[pd.DataFrame]:
        """Load last 250 trading days of OHLCV data, enriched with
        options summary and FII data where available."""
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume, india_vix "
                "FROM nifty_daily WHERE date <= %s ORDER BY date DESC LIMIT 250",
                self.conn, params=(as_of_date,)
            )
            if df.empty:
                return None
            df = df.sort_values('date').reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])

            # Merge options daily summary
            try:
                opts = pd.read_sql(
                    "SELECT date, max_pain_strike, pcr_oi, "
                    "put_oi_max_strike, call_oi_max_strike, "
                    "atm_put_iv, atm_call_iv, iv_skew, "
                    "atm_straddle_premium, oi_concentration_ratio, "
                    "oi_concentration_center "
                    "FROM options_daily_summary WHERE date <= %s",
                    self.conn, params=(as_of_date,)
                )
                if not opts.empty:
                    opts['date'] = pd.to_datetime(opts['date'])
                    df = df.merge(opts, on='date', how='left')
            except Exception:
                pass  # Table may not exist

            # Merge FII data
            try:
                fii = pd.read_sql(
                    "SELECT trade_date AS date, "
                    "fii_long_contracts, fii_short_contracts, "
                    "(fii_long_contracts - fii_short_contracts) AS fii_net, "
                    "net_long_ratio AS fii_net_ratio "
                    "FROM fii_daily_metrics WHERE trade_date <= %s",
                    self.conn, params=(as_of_date,)
                )
                if not fii.empty:
                    fii['date'] = pd.to_datetime(fii['date'])
                    df = df.merge(fii, on='date', how='left')
            except Exception:
                pass  # Table may not exist

            return df
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return None

    def _load_open_positions(self, trade_type: str = 'PAPER') -> List[Dict]:
        """Load open positions by trade_type (no exit_date yet)."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT trade_id, signal_id, direction, entry_price, entry_date
                FROM trades
                WHERE trade_type = %s
                  AND exit_date IS NULL
                ORDER BY entry_date
            """, (trade_type,))
            rows = cur.fetchall()
            return [
                {
                    'trade_id': r[0],
                    'signal_id': r[1],
                    'direction': r[2],
                    'entry_price': r[3],
                    'entry_date': r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to load open positions: {e}")
            return []

    def _record_entry(self, signal_id: str, entry: dict,
                      today, as_of_date: date, indicators: dict,
                      trade_type: str = 'PAPER',
                      size_multiplier: float = 1.0,
                      confidence_label: str = None):
        """Record a new trade entry."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO trades (
                    signal_id, trade_type, entry_date, entry_time,
                    instrument, direction, lots, entry_price,
                    entry_regime, entry_vix,
                    intended_lots, fill_quality,
                    size_multiplier, confidence_label, created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    'FUTURES', %s, 1, %s,
                    %s, %s,
                    1, 'SIMULATED',
                    %s, %s, NOW()
                )
            """, (
                signal_id, trade_type, as_of_date, datetime.now().time(),
                entry['direction'], entry['price'],
                indicators.get('regime'),
                indicators.get('india_vix'),
                size_multiplier, confidence_label,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to record entry: {e}")
            self.conn.rollback()

    def _record_exit(self, pos: dict, exit_signal: dict, today):
        """Record exit for an open paper trade."""
        try:
            entry_price = pos['entry_price']
            exit_price = exit_signal['price']
            direction = pos['direction']

            if direction == 'LONG':
                gross_pnl = exit_price - entry_price
            else:
                gross_pnl = entry_price - exit_price

            return_pct = gross_pnl / entry_price

            cur = self.conn.cursor()
            cur.execute("""
                UPDATE trades SET
                    exit_date = CURRENT_DATE,
                    exit_time = CURRENT_TIME,
                    exit_price = %s,
                    exit_reason = %s,
                    gross_pnl = %s,
                    costs = 0,
                    net_pnl = %s,
                    return_pct = %s
                WHERE trade_id = %s
            """, (
                exit_price,
                exit_signal['reason'],
                gross_pnl,
                gross_pnl,  # no costs in paper trading
                return_pct,
                pos['trade_id'],
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to record exit: {e}")
            self.conn.rollback()

    # ================================================================
    # INTRADAY REGIME BRIDGE
    # ================================================================

    @staticmethod
    def _vix_to_intraday_regime(vix: float, daily_regime: str) -> str:
        """
        Map India VIX + daily regime label → intraday regime for filtering.

        Uses VIX thresholds aligned with regime_labeler.py:
          VIX ≥ 25 → CRISIS   (matches CRISIS in regime_labeler)
          VIX ≥ 18 → HIGH_VOL (matches HIGH_VOL in regime_labeler)
          VIX ≥ 14 → ELEVATED
          VIX ≥ 10 → NORMAL
          VIX < 10 → CALM

        Also cross-checks with daily regime label for consistency.
        """
        # VIX-based classification (primary)
        if vix >= 25:
            return 'CRISIS'
        if vix >= 18:
            return 'HIGH_VOL'
        if vix >= 14:
            return 'ELEVATED'
        if vix >= 10:
            return 'NORMAL'
        if vix > 0:
            return 'CALM'

        # VIX unavailable — fall back to daily regime label
        regime_map = {
            'CRISIS': 'CRISIS',
            'HIGH_VOL': 'HIGH_VOL',
            'TRENDING': 'ELEVATED',
            'RANGING': 'NORMAL',
        }
        return regime_map.get(daily_regime, 'NORMAL')

    # ================================================================
    # INTRADAY STUB ENTRY/EXIT (daily pipeline placeholders)
    # ================================================================

    def _check_entry_intraday_stub(self, signal_id, config, today, yesterday):
        """
        Stub for intraday signals in the daily pipeline.

        Real intraday signals fire from 5-min bars via intraday_runner.py.
        In the daily pipeline, we log a synthetic entry if:
          - The day was an expiry day (for gamma signals), OR
          - The daily bar had characteristics favorable for the signal

        This gives us daily-level shadow tracking of intraday strategies.
        """
        close = float(today['close'])
        atr_val = float(today.get('atr_14', 0)) if pd.notna(today.get('atr_14')) else 0

        # For gamma signals on expiry days — always log a shadow entry
        if 'GAMMA' in signal_id:
            return {
                'signal_id': signal_id,
                'direction': 'LONG',  # placeholder — real direction from 5-min
                'price': close,
                'reason': f'Expiry day shadow log (actual entry via intraday_runner)',
            }

        # For KAUFMAN_BB_MR / GUJRAL_RANGE: log if daily conditions suggest
        # mean-reversion setup (price within BBands, low ADX)
        bb_upper = float(today.get('bb_upper', 0)) if pd.notna(today.get('bb_upper')) else 0
        bb_lower = float(today.get('bb_lower', 0)) if pd.notna(today.get('bb_lower')) else 0
        adx = float(today.get('adx_14', 20)) if pd.notna(today.get('adx_14')) else 20

        if bb_upper > bb_lower > 0 and close > 0:
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)

            if signal_id == 'ID_KAUFMAN_BB_MR':
                # BB mean-reversion: price near band edges
                if bb_position > 0.85 or bb_position < 0.15:
                    direction = 'SHORT' if bb_position > 0.85 else 'LONG'
                    return {
                        'signal_id': signal_id,
                        'direction': direction,
                        'price': close,
                        'reason': f'Daily BB setup: pos={bb_position:.0%} ADX={adx:.0f}',
                    }

            elif signal_id == 'ID_GUJRAL_RANGE':
                # Range: low ADX + mid-band price
                if adx < 25 and 0.3 < bb_position < 0.7:
                    return {
                        'signal_id': signal_id,
                        'direction': 'LONG',
                        'price': close,
                        'reason': f'Daily range setup: ADX={adx:.0f} pos={bb_position:.0%}',
                    }

        return None

    def _check_exit_intraday_stub(self, pos, today, yesterday):
        """
        Stub exit for intraday signals — always exit same day.
        In daily pipeline, any open intraday shadow position closes at EOD.
        """
        entry_date = pos.get('entry_date')
        today_date = today.get('date', today.name) if hasattr(today, 'name') else today.get('date')

        # Intraday positions close same day
        if entry_date and str(entry_date) != str(today_date):
            return {
                'signal_id': pos['signal_id'],
                'reason': 'INTRADAY_EOD_CLOSE',
                'price': float(today['close']),
            }
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Daily signal computation')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--date', type=str, help='YYYY-MM-DD (default: today)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    as_of = date.fromisoformat(args.date) if args.date else date.today()

    computer = SignalComputer()
    result = computer.run(as_of_date=as_of, dry_run=args.dry_run)

    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
