"""
Unified configuration — single source of truth for ALL trading system parameters.

Consolidates settings from config/settings.py, paper_trading/signal_compute.py,
and scattered magic numbers into one canonical module.

Usage:
    from config.unified_config import (
        INITIAL_CAPITAL, MAX_CONCURRENT_POSITIONS, NIFTY_LOT_SIZE,
        ACTIVE_SCORING_SIGNALS, SIGNAL_MODULE_MAP,
    )
"""

import os
from datetime import time as dt_time

# ================================================================
# CAPITAL & POSITION LIMITS
# ================================================================
INITIAL_CAPITAL = 10_00_000  # ₹10 lakh
MAX_CONCURRENT_POSITIONS = 3
CAPITAL_RESERVE_FRACTION = 0.20
CAPITAL_RESERVE = int(INITIAL_CAPITAL * CAPITAL_RESERVE_FRACTION)
AVAILABLE_CAPITAL = INITIAL_CAPITAL - CAPITAL_RESERVE

# ================================================================
# INSTRUMENT SPECIFICATIONS
# ================================================================
NIFTY_LOT_SIZE = 25                # Post-Nov 2024 (NSE circular)
LOT_SIZE_PRE_JUL2023 = 75         # Before July 2023 reduction
BANKNIFTY_LOT_SIZE = 15
WEEKLY_EXPIRY_DAY = 1              # 0=Monday, 1=Tuesday (Nifty weekly)
BANKNIFTY_EXPIRY_DAY = 2           # Wednesday

# ================================================================
# RISK PARAMETERS
# ================================================================
BASE_RISK_PCT = 0.02               # 2% of capital per trade
KELLY_SAFETY_FACTOR = 0.5          # Half-Kelly (conservative)
MAX_SAME_DIRECTION = 2
RISK_FREE_RATE = 0.065             # RBI repo rate

# Loss limits (fractions of INITIAL_CAPITAL)
DAILY_LOSS_FRACTION = 0.05
WEEKLY_LOSS_FRACTION = 0.08
MONTHLY_DD_CRITICAL_FRACTION = 0.15
MONTHLY_DD_HALT_FRACTION = 0.25

# Drawdown-based position scaling tiers: (max_dd_pct, size_mult)
DRAWDOWN_TIERS = [
    (5.0,  1.00),
    (10.0, 0.75),
    (15.0, 0.50),
    (100.0, 0.25),
]

# ================================================================
# VIX REGIME THRESHOLDS
# ================================================================
VIX_LOW = 12
VIX_NORMAL = 16
VIX_HIGH = 20
VIX_EXTREME = 25

VIX_THRESHOLDS = {
    'LOW': VIX_LOW,
    'NORMAL': VIX_NORMAL,
    'HIGH': VIX_HIGH,
    'EXTREME': VIX_EXTREME,
}

# ================================================================
# TREND / INDICATOR THRESHOLDS
# ================================================================
ADX_TREND_THRESHOLD = 22           # ADX > 22 = trending market

# ================================================================
# MARKET HOURS (IST — Asia/Kolkata)
# ================================================================
MARKET_OPEN = dt_time(9, 15)       # Pre-open starts 9:00, continuous from 9:15
MARKET_CLOSE = dt_time(15, 30)
PRE_OPEN_START = dt_time(9, 0)
PRE_OPEN_END = dt_time(9, 8)
POST_CLOSE_TIME = dt_time(15, 40)  # Post-close session
EOD_PIPELINE_TIME = dt_time(15, 35)  # When daily pipeline runs
GLOBAL_PRE_MARKET_TIME = dt_time(8, 30)  # Cross-asset overlay cron

# ================================================================
# DATABASE & INFRASTRUCTURE
# ================================================================
DATABASE_DSN = os.environ.get(
    'DATABASE_DSN',
    'postgresql://trader:trader123@localhost:5450/trading',
)
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = 0

# ================================================================
# TRANSACTION COST CONSTANTS (NSE Nifty Futures)
# ================================================================
BROKERAGE_PER_ORDER = 20.0         # Flat ₹20 per order (discount broker)
GST_RATE = 0.18                    # 18% GST on brokerage
STT_FUTURES_SELL = 0.0125 / 100    # 0.0125% on sell side (futures)
STT_OPTIONS_SELL = 0.0625 / 100    # 0.0625% on sell side (options)
EXCHANGE_TXN_CHARGE = 0.00173 / 100  # NSE transaction charge
SEBI_TURNOVER_FEE = 0.0001 / 100  # SEBI turnover fee
STAMP_DUTY_BUY = 0.003 / 100      # Stamp duty on buy side

# ================================================================
# SCORING SIGNALS — fire real (paper) trades with full sizing
# ================================================================
ACTIVE_SCORING_SIGNALS = [
    # Daily signals (Kaufman + Gujral)
    'KAUFMAN_DRY_20',
    'KAUFMAN_DRY_16',
    'KAUFMAN_DRY_12',
    'GUJRAL_DRY_8',
    'GUJRAL_DRY_13',
    'BULKOWSKI_ADAM_AND_EVE_OR',
    'EXPIRY_PIN_FADE',
    # Structural signals (compute real modifiers)
    'MAX_OI_BARRIER',
    'MONDAY_STRADDLE',
    'GIFT_CONVERGENCE',
]

# Intraday scoring (promoted from shadow)
ACTIVE_INTRADAY_SCORING = [
    'EXPIRY_PIN_FADE',
    'ORR_REVERSION',
]

# ================================================================
# SHADOW SIGNALS — paper trade only, evaluate for promotion
# ================================================================
SHADOW_SCORING_SIGNALS = [
    'CHAN_AT_DRY_4',
    'CANDLESTICK_DRY_0',
    'KAUFMAN_DRY_7',
    'GUJRAL_DRY_9',
    'BULKOWSKI_CUP_HANDLE',
    'BULKOWSKI_ROUND_BOTTOM_RDB_PATTERN',
    'BULKOWSKI_EADT_BUSTED_PATTERN',
    'BULKOWSKI_FALLING_VOLUME_TREND_IN',
    'BULKOWSKI_EADB_EARLY_ATTEMPT_TO',
    'SSRN_WEEKLY_MOM',
    'COMBO_GRIMES32_DRY12_SEQ5',
    'COMBO_GRIMES60_DRY8_AND',
    'COMBO_GRIMES50_DRY16_SEQ3',
    'COMBO_GRIMES50_DRY16_SEQ5',
    'BN_ORR_REVERSION',
    'BN_EXPIRY_PIN_FADE',
    'STRIKE_CLUSTERING_GRAV',
    'WEEKLY_STRANGLE_SELL',
    'ID_GAMMA_BREAKOUT',
    'ID_GAMMA_REVERSAL',
    'ID_GAMMA_SQUEEZE',
    'L8_BEAR_CALL_RESISTANCE',
    # ── Mean Reversion (new) ──
    'NIFTY_BN_SPREAD',
    'RSI2_REVERSION',
    'BASIS_ZSCORE',
    'BOLLINGER_SQUEEZE',
    # ── Volatility Regime (new) ──
    'VIX_MEAN_REVERSION',
    'VOL_COMPRESSION',
    # ── Cross-Asset Macro (new) ──
    'USDINR_MOMENTUM',
    'US_YIELD_SHOCK',
    'CHINA_DECOUPLE',
    'GOLD_NIFTY_RATIO',
    'CRUDE_NIFTY_DIVERGENCE',
    # ── Seasonality/Calendar (new) ──
    'BUDGET_DAY',
    'RBI_POLICY_DAY',
    'SAMVAT_TRADING',
    'EXPIRY_WEEK_TUESDAY',
    'MONTH_END_FII',
    'QUARTER_END_DRESSING',
    # ── Options-Specific (new) ──
    'MAX_PAIN_GRAVITY',
    'OI_WALL_SHIFT',
    'IV_SKEW_MOMENTUM',
    'STRADDLE_DECAY',
    'OI_CONCENTRATION',
    # ── Microstructure (new) ──
    'LARGE_ORDER_DETECTION',
    'TRADE_AGGRESSOR',
]

# ================================================================
# OVERLAY SIGNALS — modify sizing/confidence, never standalone
# ================================================================
ACTIVE_OVERLAY_SIGNALS = [
    # These 12 compute real modifiers today
    'GUJRAL_DRY_7',
    'CRISIS_SHORT',
    'SSRN_JANUARY_EFFECT',
    'GLOBAL_OVERNIGHT_COMPOSITE',
    'PCR_AUTOTRENDER',
    'ROLLOVER_ANALYSIS',
    'FII_FUTURES_OI',
    'DELIVERY_PCT',
    'SENTIMENT_COMPOSITE',
    'BOND_YIELD_SPREAD',
    'GAMMA_EXPOSURE',
    'VOL_TERM_STRUCTURE',
    # ── New overlays (implemented with fallback logic) ──
    'VIX_TERM_STRUCTURE',
    'RV_IV_DIVERGENCE',
    'GOOGLE_TRENDS_FEAR',
    'TWITTER_SENTIMENT',
    'NEWS_EVENT_CLASSIFIER',
    'RETAIL_BROKER_SENTIMENT',
    'BID_ASK_REGIME',
]

DISABLED_OVERLAY_SIGNALS = [
    # ML overlays — fallback heuristics active, no trained model yet
    'RBI_MACRO_FILTER',
    'ORDER_FLOW_IMBALANCE',
    'XGBOOST_META_LEARNER',
    'MAMBA_REGIME',
    'TFT_FORECAST',
    'RL_POSITION_SIZER',
    'GNN_SECTOR_ROTATION',
    'NLP_SENTIMENT',
]

# ================================================================
# SIGNAL MODULE MAP — signal_id -> (module_path, class_name)
# ================================================================
SIGNAL_MODULE_MAP = {
    # Scoring signals
    'KAUFMAN_DRY_20':  ('paper_trading.signal_compute', 'SignalComputer'),
    'KAUFMAN_DRY_16':  ('paper_trading.signal_compute', 'SignalComputer'),
    'KAUFMAN_DRY_12':  ('paper_trading.signal_compute', 'SignalComputer'),
    'GUJRAL_DRY_8':    ('paper_trading.signal_compute', 'SignalComputer'),
    'GUJRAL_DRY_13':   ('paper_trading.signal_compute', 'SignalComputer'),
    'BULKOWSKI_ADAM_AND_EVE_OR': ('paper_trading.signal_compute', 'SignalComputer'),
    'EXPIRY_PIN_FADE': ('signals.expiry_day_detector', 'ExpiryDayDetector'),
    'ORR_REVERSION':   ('signals.india_intraday', 'ORRReversion'),

    # Structural signals
    'MAX_OI_BARRIER':     ('signals.structural.max_oi_barrier', 'MaxOIBarrier'),
    'MONDAY_STRADDLE':    ('signals.structural.monday_straddle', 'MondayStraddle'),
    'GIFT_CONVERGENCE':   ('signals.structural.gift_convergence', 'GiftConvergence'),

    # Overlays
    'GUJRAL_DRY_7':               ('paper_trading.regime_detector', 'GujralRegimeDetector'),
    'CRISIS_SHORT':               ('paper_trading.signal_compute', 'SignalComputer'),
    'GLOBAL_OVERNIGHT_COMPOSITE': ('signals.global_composite', 'GlobalComposite'),
    'PCR_AUTOTRENDER':            ('signals.pcr_signal', 'PCRSignal'),
    'ROLLOVER_ANALYSIS':          ('signals.rollover_signal', 'RolloverSignal'),
    'FII_FUTURES_OI':             ('signals.fii_futures_oi', 'FIIFuturesOI'),
    'DELIVERY_PCT':               ('signals.delivery_signal', 'DeliverySignal'),
    'SENTIMENT_COMPOSITE':        ('signals.sentiment_signal', 'SentimentComposite'),
    'BOND_YIELD_SPREAD':          ('signals.bond_yield_signal', 'BondYieldSpread'),
    'GAMMA_EXPOSURE':             ('signals.gamma_exposure', 'GammaExposure'),
    'VOL_TERM_STRUCTURE':         ('signals.vol_term_structure', 'VolTermStructure'),
    'SSRN_JANUARY_EFFECT':        ('paper_trading.signal_compute', 'SignalComputer'),

    # FII overlay (separate module)
    'FII_OVERLAY': ('signals.fii_overlay', 'FIIOverlay'),
    'CALENDAR_OVERLAY': ('signals.calendar_overlay', 'CalendarOverlay'),

    # ── Mean Reversion ──
    'NIFTY_BN_SPREAD':       ('signals.mean_reversion', 'NiftyBankNiftySpreadSignal'),
    'RSI2_REVERSION':        ('signals.mean_reversion', 'RSI2ReversionSignal'),
    'BASIS_ZSCORE':          ('signals.mean_reversion', 'BasisZScoreSignal'),
    'BOLLINGER_SQUEEZE':     ('signals.mean_reversion', 'BollingerSqueezeSignal'),

    # ── Volatility Regime ──
    'VIX_MEAN_REVERSION':    ('signals.volatility', 'VixMeanReversionSignal'),
    'VIX_TERM_STRUCTURE':    ('signals.volatility', 'VixTermStructureSignal'),
    'RV_IV_DIVERGENCE':      ('signals.volatility', 'RvIvDivergenceSignal'),
    'VOL_COMPRESSION':       ('signals.volatility', 'VolCompressionSignal'),

    # ── Cross-Asset Macro ──
    'USDINR_MOMENTUM':       ('signals.macro', 'UsdInrMomentumSignal'),
    'US_YIELD_SHOCK':        ('signals.macro', 'UsYieldShockSignal'),
    'CHINA_DECOUPLE':        ('signals.macro', 'ChinaDecoupleSignal'),
    'GOLD_NIFTY_RATIO':      ('signals.macro', 'GoldNiftyRatioSignal'),
    'CRUDE_NIFTY_DIVERGENCE':('signals.macro', 'CrudeNiftyDivergenceSignal'),

    # ── Seasonality / Calendar ──
    'BUDGET_DAY':            ('signals.seasonality', 'BudgetDaySignal'),
    'RBI_POLICY_DAY':        ('signals.seasonality', 'RBIPolicyDaySignal'),
    'SAMVAT_TRADING':        ('signals.seasonality', 'SamvatTradingSignal'),
    'EXPIRY_WEEK_TUESDAY':   ('signals.seasonality', 'ExpiryWeekTuesdaySignal'),
    'MONTH_END_FII':         ('signals.seasonality', 'MonthEndFIISignal'),
    'QUARTER_END_DRESSING':  ('signals.seasonality', 'QuarterEndDressingSignal'),

    # ── Options-Specific ──
    'MAX_PAIN_GRAVITY':      ('signals.options', 'MaxPainGravitySignal'),
    'OI_WALL_SHIFT':         ('signals.options', 'OIWallShiftSignal'),
    'IV_SKEW_MOMENTUM':      ('signals.options', 'IVSkewMomentumSignal'),
    'STRADDLE_DECAY':        ('signals.options', 'StraddleDecaySignal'),
    'OI_CONCENTRATION':      ('signals.options', 'OIConcentrationSignal'),

    # ── Sentiment ──
    'GOOGLE_TRENDS_FEAR':    ('signals.sentiment', 'GoogleTrendsFearSignal'),
    'TWITTER_SENTIMENT':     ('signals.sentiment', 'TwitterSentimentSignal'),
    'NEWS_EVENT_CLASSIFIER': ('signals.sentiment', 'NewsEventClassifierSignal'),
    'RETAIL_BROKER_SENTIMENT':('signals.sentiment', 'RetailBrokerSentimentSignal'),

    # ── Microstructure ──
    'BID_ASK_REGIME':        ('signals.microstructure', 'BidAskRegimeSignal'),
    'LARGE_ORDER_DETECTION': ('signals.microstructure', 'LargeOrderDetectionSignal'),
    'TRADE_AGGRESSOR':       ('signals.microstructure', 'TradeAggressorSignal'),

    # ── ML Overlays ──
    'XGBOOST_META_LEARNER':  ('signals.ml', 'XGBoostMetaLearner'),
    'MAMBA_REGIME':          ('signals.ml', 'MambaRegimeDetector'),
    'TFT_FORECAST':          ('signals.ml', 'TFTForecastSignal'),
    'RL_POSITION_SIZER':     ('signals.ml', 'RLPositionSizer'),
    'GNN_SECTOR_ROTATION':   ('signals.ml', 'GNNSectorRotation'),
    'NLP_SENTIMENT':         ('signals.ml', 'NLPSentimentSignal'),
}

# ================================================================
# CONVICTION TIERS — map scoring engine output to position sizing
# ================================================================
CONVICTION_TIERS = {
    'HIGH':   {'size_mult': 1.00, 'min_score': 3, 'regime': 'FAVORABLE'},
    'MEDIUM': {'size_mult': 0.50, 'min_score': 2, 'regime': 'ANY'},
    'LOW':    {'size_mult': 0.25, 'min_score': 1, 'regime': 'STANDARD'},
}

# ================================================================
# DEFENSIVE TIERS — drawdown-triggered risk reduction
# ================================================================
DEFENSIVE_TIERS = {
    'NORMAL':   {'max_dd_pct': 5.0,  'size_mult': 1.00, 'max_positions': 3},
    'CAUTION':  {'max_dd_pct': 10.0, 'size_mult': 0.75, 'max_positions': 2},
    'DEFENSE':  {'max_dd_pct': 15.0, 'size_mult': 0.50, 'max_positions': 1},
    'CRITICAL': {'max_dd_pct': 25.0, 'size_mult': 0.25, 'max_positions': 1},
    'HALT':     {'max_dd_pct': 100.0, 'size_mult': 0.00, 'max_positions': 0},
}

# ================================================================
# WALK-FORWARD ENGINE
# ================================================================
WF_TRAIN_MONTHS = 36
WF_TEST_MONTHS = 12
WF_STEP_MONTHS = 3
WF_PURGE_DAYS = 21
WF_EMBARGO_DAYS = 5
WF_MIN_PASS_RATE = 0.75

# ================================================================
# GREEK LIMITS (scale with capital)
# ================================================================
_CAPITAL_SCALE = INITIAL_CAPITAL / 1_000_000

GREEK_LIMITS = {
    'max_portfolio_delta': 0.50,
    'max_portfolio_vega': int(3000 * _CAPITAL_SCALE),
    'max_portfolio_gamma': int(30000 * _CAPITAL_SCALE),
    'max_portfolio_theta': int(-5000 * _CAPITAL_SCALE),
    'max_same_direction_positions': MAX_SAME_DIRECTION,
}
