-- ================================================================
-- NIFTY F&O DYNAMIC SIGNAL SYSTEM — COMPLETE SCHEMA
-- Run: psql -d trading -f db/schema.sql
-- Requires: PostgreSQL 14+ with TimescaleDB extension
-- ================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ================================================================
-- TABLE 1: BOOKS
-- Master list of all ingested books
-- ================================================================
CREATE TABLE IF NOT EXISTS books (
    book_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    author          TEXT NOT NULL,
    edition         TEXT,
    year            INTEGER,
    abstraction_level TEXT NOT NULL
                    CHECK (abstraction_level IN (
                        'CONCRETE', 'PRINCIPLE',
                        'METHODOLOGY', 'PSYCHOLOGY'
                    )),
    pdf_path        TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunk_count     INTEGER,
    notes           TEXT
);

INSERT INTO books VALUES
('GUJRAL',         'How to Make Money in Intraday Trading',
 'Ashwani Gujral', '2nd', 2008,  'CONCRETE',    '/books/gujral.pdf',   NOW(), NULL, NULL),
('NATENBERG',      'Option Volatility and Pricing',
 'Natenberg',      '2nd', 2014,  'PRINCIPLE',   '/books/natenberg.pdf',NOW(), NULL, NULL),
('SINCLAIR',       'Options Trading',
 'Euan Sinclair',  '1st', 2010,  'PRINCIPLE',   '/books/sinclair.pdf', NOW(), NULL, NULL),
('GRIMES',         'The Art and Science of Technical Analysis',
 'Adam Grimes',    '1st', 2012,  'METHODOLOGY', '/books/grimes.pdf',   NOW(), NULL, NULL),
('KAUFMAN',        'Trading Systems and Methods',
 'Perry Kaufman',  '5th', 2013,  'CONCRETE',    '/books/kaufman.pdf',  NOW(), NULL, NULL),
('DOUGLAS',        'Trading in the Zone',
 'Mark Douglas',   '1st', 2000,  'PSYCHOLOGY',  '/books/douglas.pdf',  NOW(), NULL, NULL),
('LOPEZ',          'Advances in Financial Machine Learning',
 'Lopez de Prado', '1st', 2018,  'METHODOLOGY', '/books/lopez.pdf',    NOW(), NULL, NULL),
('HILPISCH',       'Python for Finance',
 'Yves Hilpisch',  '2nd', 2018,  'METHODOLOGY', '/books/hilpisch.pdf', NOW(), NULL, NULL),
('MCMILLAN',       'Options as a Strategic Investment',
 'Lawrence McMillan','5th', 2012, 'CONCRETE',    '/books/mcmillan.pdf', NOW(), NULL,
 'Butterflies, calendars, diagonals, ratio spreads. Direct rule extraction.'),
('AUGEN',          'The Volatility Edge in Options Trading',
 'Jeff Augen',     '1st', 2008,  'CONCRETE',    '/books/augen.pdf',    NOW(), NULL,
 'Weekly expiry dynamics, IV crush, pre-event IV expansion. Nifty weekly options.'),
('HARRIS',         'Trading and Exchanges',
 'Larry Harris',   '1st', 2003,  'METHODOLOGY', '/books/harris.pdf',   NOW(), NULL,
 'Market microstructure — liquidity filters, bid-ask regime detection. Filters only.'),
('NSE_EMPIRICAL',  'NSE India Empirical Patterns',
 'Internal',       'v1',  2026,  'CONCRETE',    NULL,                  NOW(), NULL,
 'FII derivatives positioning patterns. India-specific.')
ON CONFLICT (book_id) DO NOTHING;


-- ================================================================
-- TABLE 2: SIGNALS (current version — head record)
-- ================================================================
CREATE TABLE IF NOT EXISTS signals (
    -- Identity
    signal_id           TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    book_id             TEXT NOT NULL REFERENCES books(book_id),
    source_chapter      TEXT,
    source_page_start   INTEGER,
    source_page_end     INTEGER,
    raw_chunk_text      TEXT,
    version             INTEGER NOT NULL DEFAULT 1,

    -- Classification
    signal_category     TEXT NOT NULL
                        CHECK (signal_category IN (
                            'TREND', 'REVERSION', 'VOL',
                            'PATTERN', 'EVENT', 'SIZING',
                            'RISK', 'REGIME'
                        )),
    direction           TEXT NOT NULL
                        CHECK (direction IN (
                            'LONG', 'SHORT', 'NEUTRAL',
                            'CONTEXT_DEPENDENT'
                        )),
    instrument          TEXT NOT NULL
                        CHECK (instrument IN (
                            'FUTURES', 'OPTIONS_BUYING',
                            'OPTIONS_SELLING', 'SPREAD',
                            'COMBINED', 'ANY'
                        )),
    timeframe           TEXT NOT NULL
                        CHECK (timeframe IN (
                            'INTRADAY', 'POSITIONAL',
                            'SWING', 'ANY'
                        )),
    target_regimes      TEXT[] NOT NULL,

    -- Rule definition (structured)
    entry_conditions    JSONB NOT NULL,
    parameters          JSONB NOT NULL,
    exit_conditions     JSONB NOT NULL,
    sizing_rule         TEXT,

    -- India-specific fields
    expiry_week_behavior TEXT NOT NULL DEFAULT 'NORMAL'
                        CHECK (expiry_week_behavior IN (
                            'NORMAL', 'AVOID_MONDAY',
                            'REDUCE_SIZE_MONDAY',
                            'AVOID_EXPIRY_WEEK'
                        )),
    min_strike_distance INTEGER DEFAULT 0,
    max_strike_distance INTEGER DEFAULT 3,

    -- Lifecycle status
    status              TEXT NOT NULL DEFAULT 'CANDIDATE'
                        CHECK (status IN (
                            'CANDIDATE',
                            'BACKTESTING',
                            'ACTIVE',
                            'WATCH',
                            'INACTIVE',
                            'ARCHIVED',
                            'PAPER_ONLY'
                        )),
    classification      TEXT
                        CHECK (classification IS NULL OR classification IN (
                            'PRIMARY', 'SECONDARY'
                        )),
    avoid_rbi_day       BOOLEAN NOT NULL DEFAULT FALSE,

    -- Backtest results
    backtest_tier       INTEGER CHECK (backtest_tier IS NULL OR backtest_tier IN (1, 2, 3)),
    sharpe_ratio        FLOAT,
    calmar_ratio        FLOAT,
    max_drawdown        FLOAT,
    win_rate            FLOAT,
    profit_factor       FLOAT,
    avg_win_loss_ratio  FLOAT,
    total_trades        INTEGER,
    nifty_correlation   FLOAT,
    windows_passed      INTEGER,
    fragility_rating    TEXT
                        CHECK (fragility_rating IS NULL OR fragility_rating IN (
                            'ROBUST', 'MODERATE', 'FRAGILE'
                        )),

    -- Stress test results
    drawdown_mar2020    FLOAT,
    drawdown_mar2026    FLOAT,
    stress_test_passed  BOOLEAN,

    -- Live performance tracking
    rolling_sharpe_60d  FLOAT,
    rolling_sharpe_updated_at TIMESTAMPTZ,
    live_trades_count   INTEGER DEFAULT 0,
    live_win_rate       FLOAT,
    last_trade_date     DATE,

    -- Capital tracking
    required_margin     INTEGER,
    starvation_rate     FLOAT,

    -- Metadata
    human_reviewer      TEXT,
    review_notes        TEXT,
    conflict_group_id   TEXT,

    -- Change-context columns for auto-versioning trigger
    pending_change_by     TEXT NOT NULL DEFAULT 'SYSTEM',
    pending_change_reason TEXT NOT NULL DEFAULT '',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_category ON signals(signal_category);
CREATE INDEX IF NOT EXISTS idx_signals_book ON signals(book_id);
CREATE INDEX IF NOT EXISTS idx_signals_classification ON signals(classification);
CREATE INDEX IF NOT EXISTS idx_signals_regime ON signals USING GIN(target_regimes);


-- ================================================================
-- TABLE 3: SIGNAL_VERSIONS
-- Full audit trail — every change preserved
-- ================================================================
CREATE TABLE IF NOT EXISTS signal_versions (
    version_id          SERIAL PRIMARY KEY,
    signal_id           TEXT NOT NULL REFERENCES signals(signal_id),
    version_number      INTEGER NOT NULL,

    status              TEXT,
    classification      TEXT,
    entry_conditions    JSONB,
    parameters          JSONB,
    exit_conditions     JSONB,
    sharpe_ratio        FLOAT,
    rolling_sharpe_60d  FLOAT,
    expiry_week_behavior TEXT,

    changed_by          TEXT NOT NULL,
    change_reason       TEXT NOT NULL,
    changed_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (signal_id, version_number)
);

-- Auto-versioning trigger
CREATE OR REPLACE FUNCTION record_signal_version()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO signal_versions (
        signal_id, version_number, status, classification,
        entry_conditions, parameters, exit_conditions,
        sharpe_ratio, rolling_sharpe_60d, expiry_week_behavior,
        changed_by, change_reason
    ) VALUES (
        OLD.signal_id, OLD.version,
        OLD.status, OLD.classification,
        OLD.entry_conditions, OLD.parameters, OLD.exit_conditions,
        OLD.sharpe_ratio, OLD.rolling_sharpe_60d,
        OLD.expiry_week_behavior,
        NEW.pending_change_by,
        NEW.pending_change_reason
    );
    NEW.version              := OLD.version + 1;
    NEW.updated_at           := NOW();
    NEW.pending_change_by    := 'SYSTEM';
    NEW.pending_change_reason := '';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS signal_version_trigger ON signals;
CREATE TRIGGER signal_version_trigger
BEFORE UPDATE ON signals
FOR EACH ROW EXECUTE FUNCTION record_signal_version();


-- ================================================================
-- TABLE 4: CONFLICT_GROUPS
-- ================================================================
CREATE TABLE IF NOT EXISTS conflict_groups (
    conflict_group_id   TEXT PRIMARY KEY,
    conflict_type       TEXT NOT NULL
                        CHECK (conflict_type IN (
                            'DIRECT_CONTRADICTION',
                            'PARAMETER_DISAGREEMENT',
                            'CONDITION_DISAGREEMENT',
                            'EXIT_DISAGREEMENT',
                            'COMPLEMENTARY'
                        )),
    conflict_description TEXT NOT NULL,
    resolution_method   TEXT,
    winning_signal_id   TEXT REFERENCES signals(signal_id),
    resolved_at         TIMESTAMPTZ,
    resolution_notes    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conflict_group_members (
    conflict_group_id   TEXT REFERENCES conflict_groups(conflict_group_id),
    signal_id           TEXT REFERENCES signals(signal_id),
    PRIMARY KEY (conflict_group_id, signal_id)
);


-- ================================================================
-- TABLE 5: TRADES
-- ================================================================
CREATE TABLE IF NOT EXISTS trades (
    trade_id            BIGSERIAL,
    signal_id           TEXT NOT NULL,
    trade_type          TEXT NOT NULL
                        CHECK (trade_type IN ('LIVE', 'PAPER')),

    entry_date          DATE NOT NULL,
    entry_time          TIME NOT NULL,
    instrument          TEXT NOT NULL,
    direction           TEXT NOT NULL,
    lots                INTEGER NOT NULL,
    entry_price         FLOAT NOT NULL,
    entry_regime        TEXT,
    entry_vix           FLOAT,

    exit_date           DATE,
    exit_time           TIME,
    exit_price          FLOAT,
    exit_reason         TEXT,

    gross_pnl           FLOAT,
    costs               FLOAT,
    net_pnl             FLOAT,
    return_pct          FLOAT,

    intended_lots       INTEGER,
    fill_quality        TEXT,
    slippage_pts        FLOAT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (trade_id, entry_date)
);

SELECT create_hypertable('trades', 'entry_date',
                          if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_trades_signal ON trades(signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(trade_type);


-- ================================================================
-- TABLE 6: PORTFOLIO_STATE
-- ================================================================
CREATE TABLE IF NOT EXISTS portfolio_state (
    snapshot_id         BIGSERIAL,
    snapshot_time       TIMESTAMPTZ NOT NULL,
    snapshot_type       TEXT NOT NULL
                        CHECK (snapshot_type IN (
                            'DAILY_OPEN', 'DAILY_CLOSE',
                            'POST_TRADE', 'RECONCILIATION'
                        )),

    total_capital       FLOAT NOT NULL,
    deployed_capital    FLOAT NOT NULL,
    cash_reserve        FLOAT NOT NULL,
    open_positions      JSONB,
    portfolio_delta     FLOAT,
    portfolio_vega      FLOAT,
    portfolio_gamma     FLOAT,
    portfolio_theta     FLOAT,
    daily_pnl           FLOAT,
    mtd_pnl             FLOAT,
    ytd_pnl             FLOAT,
    PRIMARY KEY (snapshot_id, snapshot_time)
);

SELECT create_hypertable('portfolio_state', 'snapshot_time',
                          if_not_exists => TRUE);


-- ================================================================
-- TABLE 7: ALERTS_LOG
-- ================================================================
CREATE TABLE IF NOT EXISTS alerts_log (
    alert_id            BIGSERIAL PRIMARY KEY,
    alert_time          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_level         TEXT NOT NULL
                        CHECK (alert_level IN (
                            'INFO', 'WARNING',
                            'CRITICAL', 'EMERGENCY'
                        )),
    alert_type          TEXT NOT NULL,
    message             TEXT NOT NULL,
    signal_id           TEXT,
    acknowledged        BOOLEAN DEFAULT FALSE,
    acknowledged_at     TIMESTAMPTZ
);


-- ================================================================
-- TABLE 8: NIFTY_DAILY (OHLCV + India VIX)
-- ================================================================
CREATE TABLE IF NOT EXISTS nifty_daily (
    date         DATE        PRIMARY KEY,
    open         FLOAT       NOT NULL,
    high         FLOAT       NOT NULL,
    low          FLOAT       NOT NULL,
    close        FLOAT       NOT NULL,
    volume       BIGINT,
    india_vix    FLOAT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nifty_daily_date ON nifty_daily(date DESC);


-- ================================================================
-- TABLE 9: MARKET_CALENDAR
-- ================================================================
CREATE TABLE IF NOT EXISTS market_calendar (
    trading_date        DATE PRIMARY KEY,
    is_trading_day      BOOLEAN NOT NULL,
    holiday_name        TEXT
);


-- ================================================================
-- TABLE 10: REGIME_LABELS
-- ================================================================
CREATE TABLE IF NOT EXISTS regime_labels (
    label_date          DATE PRIMARY KEY,
    regime              TEXT NOT NULL
                        CHECK (regime IN ('TRENDING','RANGING','HIGH_VOL','CRISIS')),
    adx_value           FLOAT,
    ema_50              FLOAT,
    india_vix           FLOAT,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);


-- ================================================================
-- TABLE 11: ECONOMIC_CALENDAR
-- ================================================================
CREATE TABLE IF NOT EXISTS economic_calendar (
    event_id            SERIAL PRIMARY KEY,
    event_date          DATE NOT NULL,
    event_type          TEXT NOT NULL,
    event_name          TEXT,
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_economic_calendar_date ON economic_calendar(event_date);


-- ================================================================
-- TABLE 12: FII_DAILY_METRICS
-- ================================================================
CREATE TABLE IF NOT EXISTS fii_daily_metrics (
    date                DATE PRIMARY KEY,
    fut_long            FLOAT,
    fut_short           FLOAT,
    fut_net             FLOAT,
    put_long            FLOAT,
    put_short           FLOAT,
    put_net             FLOAT,
    put_ratio           FLOAT,
    call_long           FLOAT,
    call_short          FLOAT,
    call_net            FLOAT,
    pcr                 FLOAT,
    fut_net_change      FLOAT,
    nifty_close         FLOAT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);


-- ================================================================
-- TABLE 13: FII_SIGNAL_RESULTS
-- ================================================================
CREATE TABLE IF NOT EXISTS fii_signal_results (
    id                  SERIAL PRIMARY KEY,
    data_date           DATE NOT NULL,
    valid_for_date      DATE NOT NULL,
    signal_id           TEXT,
    direction           TEXT,
    confidence          FLOAT,
    pattern_name        TEXT NOT NULL,
    notes               TEXT,
    was_executed        BOOLEAN DEFAULT FALSE,
    execution_pnl       FLOAT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);


-- ================================================================
-- TABLE 14: VOL_ADJUSTMENT_FACTORS
-- ================================================================
CREATE TABLE IF NOT EXISTS vol_adjustment_factors (
    dte_bucket          INTEGER NOT NULL PRIMARY KEY,
    vol_multiplier      FLOAT   NOT NULL,
    sample_count        INTEGER,
    derived_at          TIMESTAMPTZ DEFAULT NOW()
);
