-- 005_theta_tracking.sql
-- Layer 8: Theta tracking + L8 shadow state tables

-- ================================================================
-- THETA DAILY LOG — tracks daily theta decay for options positions
-- ================================================================
CREATE TABLE IF NOT EXISTS theta_daily_log (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    signal_id       VARCHAR(64) NOT NULL,
    strategy_type   VARCHAR(32),
    -- Theta metrics
    theta_per_day   FLOAT NOT NULL DEFAULT 0,       -- theta in points
    theta_per_day_rs FLOAT NOT NULL DEFAULT 0,      -- theta in rupees
    cumulative_theta FLOAT NOT NULL DEFAULT 0,      -- total theta collected
    days_held       INT NOT NULL DEFAULT 0,
    -- P&L comparison
    actual_pnl      FLOAT DEFAULT 0,
    expected_theta_pnl FLOAT DEFAULT 0,
    pnl_divergence  FLOAT DEFAULT 0,                -- actual / expected
    -- Position state
    spot_price      FLOAT DEFAULT 0,
    position_delta  FLOAT DEFAULT 0,
    position_vega   FLOAT DEFAULT 0,
    iv_at_log       FLOAT DEFAULT 0,
    dte_remaining   INT DEFAULT 0,
    --
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_theta_log_date
    ON theta_daily_log (trade_date);
CREATE INDEX IF NOT EXISTS idx_theta_log_signal
    ON theta_daily_log (signal_id, trade_date);

-- ================================================================
-- L8 SHADOW STATE — shadow tracking for L8 options strategies
-- ================================================================
CREATE TABLE IF NOT EXISTS l8_shadow_state (
    id              SERIAL PRIMARY KEY,
    signal_id       VARCHAR(64) NOT NULL,
    status          VARCHAR(16) NOT NULL DEFAULT 'SHADOW',  -- SHADOW, MONITOR, PROMOTED
    -- Rolling metrics
    trade_count     INT DEFAULT 0,
    win_rate        FLOAT DEFAULT 0,
    rolling_sharpe  FLOAT DEFAULT 0,
    rolling_pf      FLOAT DEFAULT 0,
    max_drawdown    FLOAT DEFAULT 0,
    cumulative_pnl  FLOAT DEFAULT 0,
    -- WF validation
    wf_verdict      VARCHAR(8) DEFAULT 'PENDING',   -- PASS, FAIL, PENDING
    wf_sharpe       FLOAT DEFAULT 0,
    wf_pass_rate    FLOAT DEFAULT 0,
    -- Timestamps
    first_shadow_date DATE,
    last_updated    TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_l8_shadow_signal
    ON l8_shadow_state (signal_id);

-- ================================================================
-- L8 SHADOW TRADES — individual shadow trade records
-- ================================================================
CREATE TABLE IF NOT EXISTS l8_shadow_trades (
    id              SERIAL PRIMARY KEY,
    signal_id       VARCHAR(64) NOT NULL,
    strategy_type   VARCHAR(32),
    entry_date      DATE NOT NULL,
    exit_date       DATE,
    -- Legs
    legs_json       JSONB,
    -- P&L
    entry_credit    FLOAT DEFAULT 0,
    exit_cost       FLOAT DEFAULT 0,
    gross_pnl       FLOAT DEFAULT 0,
    net_pnl         FLOAT DEFAULT 0,
    costs           FLOAT DEFAULT 0,
    -- Exit
    exit_reason     VARCHAR(32),
    days_held       INT DEFAULT 0,
    -- Market context
    spot_at_entry   FLOAT DEFAULT 0,
    vix_at_entry    FLOAT DEFAULT 0,
    iv_rank_at_entry FLOAT DEFAULT 0,
    --
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_l8_shadow_trades_signal
    ON l8_shadow_trades (signal_id, entry_date);

-- ================================================================
-- PORTFOLIO THETA DAILY — aggregate portfolio-level theta
-- ================================================================
CREATE TABLE IF NOT EXISTS portfolio_theta_daily (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL UNIQUE,
    net_theta_per_day FLOAT DEFAULT 0,
    net_theta_rs    FLOAT DEFAULT 0,
    num_positions   INT DEFAULT 0,
    is_net_seller   BOOLEAN DEFAULT TRUE,
    spot_price      FLOAT DEFAULT 0,
    vix             FLOAT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW()
);
