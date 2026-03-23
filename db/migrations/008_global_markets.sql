-- Migration 008: Global market data tables
-- Stores international market data for pre-market signals
-- Created: 2026-03-22

-- ================================================================
-- DAILY HISTORY FOR GLOBAL INSTRUMENTS
-- ================================================================
CREATE TABLE IF NOT EXISTS global_market_daily (
    id              SERIAL PRIMARY KEY,
    instrument      VARCHAR(32) NOT NULL,    -- sp500, us_vix, dxy, brent, etc.
    trade_date      DATE NOT NULL,
    open            NUMERIC(12, 4),
    high            NUMERIC(12, 4),
    low             NUMERIC(12, 4),
    close           NUMERIC(12, 4) NOT NULL,
    volume          BIGINT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(instrument, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_gmd_instrument_date
    ON global_market_daily(instrument, trade_date DESC);

-- ================================================================
-- PRE-MARKET SNAPSHOTS (one per trading day)
-- ================================================================
CREATE TABLE IF NOT EXISTS global_market_snapshots (
    id                      SERIAL PRIMARY KEY,
    snapshot_date           DATE NOT NULL UNIQUE,
    snapshot_time           TIMESTAMP NOT NULL,

    -- GIFT Nifty
    gift_nifty_last         NUMERIC(10, 2),
    gift_nifty_gap_pct      NUMERIC(8, 4),
    nifty_prev_close        NUMERIC(10, 2),

    -- US Equities
    sp500_close             NUMERIC(10, 2),
    sp500_change_pct        NUMERIC(8, 4),
    nasdaq_close            NUMERIC(10, 2),
    nasdaq_change_pct       NUMERIC(8, 4),

    -- US VIX
    us_vix_close            NUMERIC(8, 2),
    us_vix_change_pct       NUMERIC(8, 4),
    us_vix_spike            BOOLEAN DEFAULT FALSE,

    -- Dollar Index
    dxy_close               NUMERIC(8, 3),
    dxy_change_pct          NUMERIC(8, 4),
    dxy_strong_move         BOOLEAN DEFAULT FALSE,

    -- Crude Oil (Brent)
    brent_close             NUMERIC(8, 2),
    brent_change_pct        NUMERIC(8, 4),
    crude_shock             BOOLEAN DEFAULT FALSE,

    -- US Treasury
    us_10y_close            NUMERIC(6, 3),

    -- Derived Scores
    us_overnight_return     NUMERIC(8, 4),
    global_risk_score       NUMERIC(5, 3),

    -- Raw JSON backup
    raw_json                JSONB,
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gms_date
    ON global_market_snapshots(snapshot_date DESC);

-- ================================================================
-- SIGNAL EVALUATION LOG
-- ================================================================
CREATE TABLE IF NOT EXISTS global_signal_evaluations (
    id                  SERIAL PRIMARY KEY,
    eval_date           DATE NOT NULL UNIQUE,
    direction           VARCHAR(8),           -- LONG, SHORT, or NULL
    bias_strength       NUMERIC(5, 3),        -- -1 to +1
    size_modifier       NUMERIC(5, 3),        -- 0.3 to 1.3
    confidence          NUMERIC(5, 3),        -- 0 to 1
    risk_off            BOOLEAN DEFAULT FALSE,
    composite_score     NUMERIC(5, 3),
    gift_gap_score      NUMERIC(5, 3),
    us_overnight_score  NUMERIC(5, 3),
    crude_score         NUMERIC(5, 3),
    asian_score         NUMERIC(5, 3),
    regime_warning      JSONB,
    reason              TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gse_date
    ON global_signal_evaluations(eval_date DESC);

-- ================================================================
-- GRANT SELECT FOR READONLY USERS
-- ================================================================
-- GRANT SELECT ON global_market_daily TO readonly_user;
-- GRANT SELECT ON global_market_snapshots TO readonly_user;
-- GRANT SELECT ON global_signal_evaluations TO readonly_user;
