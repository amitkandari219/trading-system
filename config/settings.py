"""
Central configuration for the trading system.
All thresholds expressed as fractions of total_capital for scalability.
Reads from environment variables when running in Docker.
"""

import os

# ================================================================
# CAPITAL CONFIGURATION
# ================================================================
TOTAL_CAPITAL = int(os.environ.get('TOTAL_CAPITAL', 1_000_000))
# Set via env var or here. All limits scale automatically.

# Derived limits (scale automatically with TOTAL_CAPITAL)
CAPITAL_RESERVE_FRACTION = 0.20
CAPITAL_RESERVE = int(TOTAL_CAPITAL * CAPITAL_RESERVE_FRACTION)
AVAILABLE_CAPITAL = TOTAL_CAPITAL - CAPITAL_RESERVE

# ================================================================
# POSITION LIMITS
# ================================================================
MAX_POSITIONS = 4
MAX_SAME_DIRECTION = 2

# ================================================================
# LOSS LIMITS (as fractions of TOTAL_CAPITAL)
# ================================================================
DAILY_LOSS_FRACTION = 0.05       # 5% of capital
WEEKLY_LOSS_FRACTION = 0.12      # 12% of capital
MONTHLY_DD_CRITICAL_FRACTION = 0.15   # 15% of capital
MONTHLY_DD_HALT_FRACTION = 0.25       # 25% of capital

DAILY_LOSS_LIMIT = int(TOTAL_CAPITAL * DAILY_LOSS_FRACTION)
WEEKLY_LOSS_LIMIT = int(TOTAL_CAPITAL * WEEKLY_LOSS_FRACTION)
MONTHLY_DD_CRITICAL = int(TOTAL_CAPITAL * MONTHLY_DD_CRITICAL_FRACTION)
MONTHLY_DD_HALT = int(TOTAL_CAPITAL * MONTHLY_DD_HALT_FRACTION)

# ================================================================
# GREEK LIMITS
# ================================================================
# Greek limits scale with capital (base values at ₹10L)
_CAPITAL_SCALE = TOTAL_CAPITAL / 1_000_000

GREEK_LIMITS = {
    'max_portfolio_delta': 0.50,                          # dimensionless, no scaling
    'max_portfolio_vega': int(3000 * _CAPITAL_SCALE),
    'max_portfolio_gamma': int(30000 * _CAPITAL_SCALE),
    'max_portfolio_theta': int(-5000 * _CAPITAL_SCALE),
    'max_same_direction_positions': MAX_SAME_DIRECTION,
}

# ================================================================
# DATABASE
# ================================================================
DATABASE_DSN = os.environ.get('DATABASE_DSN', 'postgresql://localhost/trading')

# ================================================================
# REDIS
# ================================================================
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = 0

# ================================================================
# NIFTY LOT SIZE (NSE defined — post 2024)
# ================================================================
NIFTY_LOT_SIZE = 25

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
# RISK FREE RATE (RBI repo rate)
# ================================================================
RISK_FREE_RATE = 0.065
