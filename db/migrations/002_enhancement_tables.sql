-- Migration 002: Tables for trading system enhancements
-- Run after 001_base_schema.sql

-- ================================================================
-- VARIANT SWITCHES: Track adaptive signal variant changes
-- Used by: paper_trading/adaptive_variant.py
-- ================================================================
CREATE TABLE IF NOT EXISTS variant_switches (
    id              SERIAL PRIMARY KEY,
    signal_id       VARCHAR(50) NOT NULL,
    old_variant     VARCHAR(30) NOT NULL,
    new_variant     VARCHAR(30) NOT NULL,
    switch_date     DATE NOT NULL,
    reason          TEXT,
    rolling_sharpe  FLOAT,
    rolling_trades  INT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(signal_id, switch_date)
);

CREATE INDEX IF NOT EXISTS idx_variant_switches_signal
    ON variant_switches(signal_id, switch_date DESC);

-- ================================================================
-- FII DAILY METRICS: Foreign institutional investor positioning
-- Used by: signals/fii_overlay.py
-- ================================================================
CREATE TABLE IF NOT EXISTS fii_daily_metrics (
    trade_date          DATE PRIMARY KEY,
    fii_long_contracts  BIGINT,
    fii_short_contracts BIGINT,
    fii_long_oi         BIGINT,
    fii_short_oi        BIGINT,
    dii_long_contracts  BIGINT,
    dii_short_contracts BIGINT,
    net_long_ratio      FLOAT GENERATED ALWAYS AS (
        CASE WHEN (fii_long_contracts + fii_short_contracts) > 0
             THEN fii_long_contracts::FLOAT / (fii_long_contracts + fii_short_contracts)
             ELSE 0.5
        END
    ) STORED,
    source              VARCHAR(30) DEFAULT 'nse_archive',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fii_daily_date
    ON fii_daily_metrics(trade_date DESC);

-- ================================================================
-- CROSS ASSET DAILY: Cache for cross-asset overlay data
-- Used by: paper_trading/cross_asset_bridge.py
-- ================================================================
CREATE TABLE IF NOT EXISTS cross_asset_daily (
    trade_date          DATE PRIMARY KEY,
    usdinr_close        FLOAT,
    usdinr_daily_ret    FLOAT,
    crude_wti_close     FLOAT,
    crude_daily_ret     FLOAT,
    gold_close          FLOAT,
    gold_daily_ret      FLOAT,
    sp500_close         FLOAT,
    sp500_daily_ret     FLOAT,
    us_vix_close        FLOAT,
    composite_mult      FLOAT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ================================================================
-- REGIME PREDICTIONS: Log ML regime classifications
-- Used by: paper_trading/regime_bridge.py
-- ================================================================
CREATE TABLE IF NOT EXISTS regime_predictions (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    method          VARCHAR(20) NOT NULL,  -- 'ensemble', 'fixed_rules'
    regime          VARCHAR(30) NOT NULL,
    confidence      FLOAT,
    hmm_regime      VARCHAR(30),
    rf_regime       VARCHAR(30),
    models_agree    BOOLEAN,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date, method)
);

-- ================================================================
-- TRANSACTION COSTS LOG: Track actual vs estimated costs
-- Used by: backtest/transaction_costs.py
-- ================================================================
CREATE TABLE IF NOT EXISTS transaction_costs_log (
    id              SERIAL PRIMARY KEY,
    trade_id        INT REFERENCES trades(trade_id),
    entry_price     FLOAT,
    exit_price      FLOAT,
    lots            INT DEFAULT 1,
    brokerage       FLOAT,
    stt             FLOAT,
    exchange_charges FLOAT,
    gst             FLOAT,
    stamp_duty      FLOAT,
    estimated_slippage FLOAT,
    total_cost      FLOAT,
    cost_as_pct     FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
