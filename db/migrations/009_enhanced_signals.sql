-- Migration 009: Enhanced signal tables for all new signal modules
-- PCR, Rollover, FII Futures OI, Delivery %, Sentiment, Bond Yield,
-- Gamma Exposure, Vol Term Structure, Order Flow, Macro Events, Meta-Learner

-- ================================================================
-- 1. Market Mood Index (Tickertape MMI)
-- ================================================================
CREATE TABLE IF NOT EXISTS market_mood_index (
    date          DATE PRIMARY KEY,
    mmi_value     REAL NOT NULL,         -- 0-100
    mmi_zone      VARCHAR(20),           -- EXTREME_FEAR, FEAR, NEUTRAL, GREED, EXTREME_GREED
    fetched_at    TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- 2. Google Trends Sentiment
-- ================================================================
CREATE TABLE IF NOT EXISTS google_trends_sentiment (
    date          DATE PRIMARY KEY,
    score         REAL NOT NULL,          -- Relative to baseline (1.0 = normal)
    is_spike      BOOLEAN DEFAULT FALSE,  -- True if >2x baseline
    keywords      TEXT,                   -- JSON array of keywords used
    fetched_at    TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- 3. Bond Yields (India + US)
-- ================================================================
CREATE TABLE IF NOT EXISTS bond_yields (
    date          DATE PRIMARY KEY,
    india_10y     REAL,                   -- India 10Y yield %
    us_10y        REAL,                   -- US 10Y yield %
    spread        REAL GENERATED ALWAYS AS (india_10y - us_10y) STORED,
    fetched_at    TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- 4. GST Monthly Collection
-- ================================================================
CREATE TABLE IF NOT EXISTS gst_monthly (
    month_date         DATE PRIMARY KEY,   -- First of month
    collection_amount  REAL NOT NULL,       -- In ₹ Crore
    yoy_change_pct     REAL,               -- Year-over-year change %
    regime             VARCHAR(20),         -- EXPANSION, STABLE, CONTRACTION
    fetched_at         TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- 5. USD/INR Daily (for RBI intervention detection)
-- ================================================================
CREATE TABLE IF NOT EXISTS usdinr_daily (
    date          DATE PRIMARY KEY,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL NOT NULL,
    prev_close    REAL,
    volume        BIGINT,
    fetched_at    TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- 6. Security-wise Delivery Data (NSE Bhavcopy)
-- ================================================================
CREATE TABLE IF NOT EXISTS security_bhav (
    date          DATE NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    series        VARCHAR(5) DEFAULT 'EQ',
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    prev_close    REAL,
    volume        BIGINT,
    delivery_qty  BIGINT,
    delivery_pct  REAL,                    -- delivery_qty / volume * 100
    PRIMARY KEY (date, symbol, series)
);
CREATE INDEX IF NOT EXISTS idx_security_bhav_date ON security_bhav(date);
CREATE INDEX IF NOT EXISTS idx_security_bhav_symbol ON security_bhav(symbol);

-- ================================================================
-- 7. FII Participant-wise OI (Index Futures specific)
-- ================================================================
CREATE TABLE IF NOT EXISTS fii_participant_oi (
    date              DATE NOT NULL,
    instrument_type   VARCHAR(30) NOT NULL,  -- INDEX_FUTURES, STOCK_FUTURES, etc.
    fii_long_oi       BIGINT DEFAULT 0,
    fii_short_oi      BIGINT DEFAULT 0,
    dii_long_oi       BIGINT DEFAULT 0,
    dii_short_oi      BIGINT DEFAULT 0,
    pro_long_oi       BIGINT DEFAULT 0,
    pro_short_oi      BIGINT DEFAULT 0,
    client_long_oi    BIGINT DEFAULT 0,
    client_short_oi   BIGINT DEFAULT 0,
    PRIMARY KEY (date, instrument_type)
);

-- ================================================================
-- 8. Signal Evaluations (unified table for all signals)
-- ================================================================
CREATE TABLE IF NOT EXISTS signal_evaluations (
    id            SERIAL PRIMARY KEY,
    eval_date     DATE NOT NULL,
    signal_id     VARCHAR(50) NOT NULL,
    result_json   JSONB NOT NULL,          -- Full signal output
    direction     VARCHAR(10),             -- BULLISH, BEARISH, NEUTRAL
    confidence    REAL,
    size_modifier REAL,
    created_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE (eval_date, signal_id)
);
CREATE INDEX IF NOT EXISTS idx_signal_eval_date ON signal_evaluations(eval_date);
CREATE INDEX IF NOT EXISTS idx_signal_eval_signal ON signal_evaluations(signal_id);

-- ================================================================
-- 9. Meta-Learner Training Log
-- ================================================================
CREATE TABLE IF NOT EXISTS meta_learner_log (
    id              SERIAL PRIMARY KEY,
    train_date      DATE NOT NULL,
    train_start     DATE,
    train_end       DATE,
    n_samples       INT,
    n_features      INT,
    train_rmse      REAL,
    test_rmse       REAL,
    feature_importance JSONB,
    model_version   VARCHAR(20),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- 10. Intraday Ticks (for order flow analysis)
-- ================================================================
CREATE TABLE IF NOT EXISTS intraday_ticks (
    id            BIGSERIAL PRIMARY KEY,
    instrument    VARCHAR(30) NOT NULL,
    timestamp     TIMESTAMP NOT NULL,
    ltp           REAL NOT NULL,
    bid_price     REAL,
    ask_price     REAL,
    bid_qty       INT,
    ask_qty       INT,
    volume        BIGINT,
    oi            BIGINT
);
CREATE INDEX IF NOT EXISTS idx_ticks_instrument_ts
    ON intraday_ticks(instrument, timestamp);

-- ================================================================
-- Done
-- ================================================================
