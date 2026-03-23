-- Migration 010: Tier 3 ML/AI Model Tables
-- Tables for Mamba regime, TFT forecaster, RL sizer, GNN sector,
-- NLP sentiment, AMFI MF flows, Credit Card spending

BEGIN;

-- ================================================================
-- Mamba / S4 Regime Detection
-- ================================================================
CREATE TABLE IF NOT EXISTS regime_labels (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    regime          VARCHAR(30) NOT NULL,  -- CALM_BULL, VOLATILE_BULL, NEUTRAL, VOLATILE_BEAR, CALM_BEAR
    confidence      NUMERIC(6,4),
    nifty_return_5d NUMERIC(8,4),
    realized_vol_5d NUMERIC(8,4),
    method          VARCHAR(30) DEFAULT 'hmm_baseline',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date, method)
);

CREATE INDEX IF NOT EXISTS idx_regime_labels_date ON regime_labels(trade_date);

-- ================================================================
-- TFT Forecaster
-- ================================================================
CREATE TABLE IF NOT EXISTS tft_forecasts (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    horizon_days    INT NOT NULL,      -- 1, 3, 5, 10, 20
    p10             NUMERIC(10,6),
    p50             NUMERIC(10,6),
    p90             NUMERIC(10,6),
    direction       VARCHAR(10),
    confidence      NUMERIC(6,4),
    model_version   VARCHAR(30) DEFAULT 'v1_baseline',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date, horizon_days, model_version)
);

CREATE INDEX IF NOT EXISTS idx_tft_forecasts_date ON tft_forecasts(trade_date);

-- ================================================================
-- RL Position Sizer
-- ================================================================
CREATE TABLE IF NOT EXISTS rl_sizing_log (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    state_vector    JSONB,             -- 20-dim state as JSON
    action_size     NUMERIC(6,4),      -- chosen size_modifier
    q_value         NUMERIC(10,6),
    reward          NUMERIC(10,6),
    method          VARCHAR(30),       -- sac, sac_untrained
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_sizing_date ON rl_sizing_log(trade_date);

-- ================================================================
-- GNN Sector Rotation
-- ================================================================
CREATE TABLE IF NOT EXISTS gnn_sector_signals (
    id                  SERIAL PRIMARY KEY,
    trade_date          DATE NOT NULL,
    regime              VARCHAR(30),    -- NORMAL, HIGH_CORRELATION, FRAGMENTED, etc.
    rotation_signal     VARCHAR(20),    -- RISK_ON, RISK_OFF, MIXED
    clustering_coeff    NUMERIC(8,6),
    eigenvalue_ratio    NUMERIC(8,6),
    graph_density       NUMERIC(8,6),
    n_stocks            INT,
    n_edges             INT,
    top_sectors         JSONB,
    bottom_sectors      JSONB,
    direction           VARCHAR(10),
    confidence          NUMERIC(6,4),
    size_modifier       NUMERIC(6,4),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date)
);

CREATE INDEX IF NOT EXISTS idx_gnn_sector_date ON gnn_sector_signals(trade_date);

-- ================================================================
-- NLP Sentiment (Earnings + RBI)
-- ================================================================
CREATE TABLE IF NOT EXISTS earnings_data (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(20) NOT NULL,
    headline    TEXT,
    content     TEXT,
    date        DATE NOT NULL,
    source      VARCHAR(50),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_earnings_symbol_date ON earnings_data(symbol, date);

CREATE TABLE IF NOT EXISTS rbi_statements (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT,
    date        DATE NOT NULL,
    doc_type    VARCHAR(30),  -- MPC_STATEMENT, PRESS_RELEASE, SPEECH
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rbi_statements_date ON rbi_statements(date);

CREATE TABLE IF NOT EXISTS market_news (
    id          SERIAL PRIMARY KEY,
    headline    TEXT NOT NULL,
    source      VARCHAR(50),
    date        DATE NOT NULL,
    sentiment   NUMERIC(6,4),  -- Pre-computed if available
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_news_date ON market_news(date);

CREATE TABLE IF NOT EXISTS nlp_sentiment_log (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    earnings_score  NUMERIC(6,4),
    rbi_score       NUMERIC(6,4),
    combined_score  NUMERIC(6,4),
    category        VARCHAR(20),
    direction       VARCHAR(10),
    confidence      NUMERIC(6,4),
    method          VARCHAR(30),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date)
);

CREATE INDEX IF NOT EXISTS idx_nlp_sentiment_date ON nlp_sentiment_log(trade_date);

-- ================================================================
-- AMFI Mutual Fund Flows
-- ================================================================
CREATE TABLE IF NOT EXISTS mf_monthly_flows (
    id              SERIAL PRIMARY KEY,
    month_date      DATE NOT NULL,
    sip_amount      NUMERIC(12,2),     -- ₹ Crore
    equity_net_flow NUMERIC(12,2),     -- ₹ Crore
    debt_aum        NUMERIC(14,2),     -- ₹ Crore
    equity_aum      NUMERIC(14,2),     -- ₹ Crore
    hybrid_net_flow NUMERIC(12,2),
    total_aum       NUMERIC(14,2),
    source          VARCHAR(30) DEFAULT 'amfi',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(month_date)
);

CREATE INDEX IF NOT EXISTS idx_mf_flows_date ON mf_monthly_flows(month_date);

-- ================================================================
-- Credit Card Spending
-- ================================================================
CREATE TABLE IF NOT EXISTS credit_card_monthly (
    id              SERIAL PRIMARY KEY,
    month_date      DATE NOT NULL,
    spending_value  NUMERIC(14,2),     -- ₹ Crore
    txn_count       NUMERIC(14,2),     -- Millions
    cards_in_force  NUMERIC(10,2),     -- Millions
    online_share    NUMERIC(6,4),      -- Online % of total
    source          VARCHAR(30) DEFAULT 'rbi',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(month_date)
);

CREATE INDEX IF NOT EXISTS idx_cc_monthly_date ON credit_card_monthly(month_date);

-- ================================================================
-- Unified Tier 3 Signal Log
-- ================================================================
CREATE TABLE IF NOT EXISTS tier3_signal_log (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    signal_id       VARCHAR(40) NOT NULL,
    direction       VARCHAR(10),
    confidence      NUMERIC(6,4),
    size_modifier   NUMERIC(6,4),
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tier3_signal_date ON tier3_signal_log(trade_date, signal_id);

COMMIT;
