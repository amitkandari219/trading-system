-- 006_intraday_bars.sql
-- Layer 9: Intraday 5-min bar storage for WF validation

CREATE TABLE IF NOT EXISTS intraday_bars (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMP NOT NULL,
    instrument  VARCHAR(16) NOT NULL,   -- NIFTY, BANKNIFTY
    open        FLOAT NOT NULL,
    high        FLOAT NOT NULL,
    low         FLOAT NOT NULL,
    close       FLOAT NOT NULL,
    volume      BIGINT DEFAULT 0,
    oi          BIGINT DEFAULT 0,
    UNIQUE(timestamp, instrument)
);

CREATE INDEX IF NOT EXISTS idx_intraday_bars_inst_ts
    ON intraday_bars (instrument, timestamp);

CREATE INDEX IF NOT EXISTS idx_intraday_bars_date
    ON intraday_bars (DATE(timestamp), instrument);

-- L9 / BN signal WF results cache
CREATE TABLE IF NOT EXISTS intraday_wf_results (
    id              SERIAL PRIMARY KEY,
    signal_id       VARCHAR(64) NOT NULL,
    instrument      VARCHAR(16) NOT NULL,
    run_date        DATE NOT NULL,
    -- Metrics
    trades          INT DEFAULT 0,
    win_rate        FLOAT DEFAULT 0,
    profit_factor   FLOAT DEFAULT 0,
    sharpe          FLOAT DEFAULT 0,
    max_drawdown    FLOAT DEFAULT 0,
    wf_pass_rate    FLOAT DEFAULT 0,
    wf_windows_pass INT DEFAULT 0,
    wf_windows_total INT DEFAULT 0,
    verdict         VARCHAR(8) DEFAULT 'PENDING',
    -- Best params
    best_params     JSONB,
    --
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(signal_id, instrument, run_date)
);

-- Entry window optimization results
CREATE TABLE IF NOT EXISTS entry_window_results (
    id              SERIAL PRIMARY KEY,
    signal_id       VARCHAR(64) NOT NULL,
    instrument      VARCHAR(16) NOT NULL,
    -- Window
    optimal_start   TIME,
    optimal_end     TIME,
    blocked_hours   JSONB,      -- list of blocked hour ints
    -- Per-hour metrics
    pf_by_hour      JSONB,      -- {hour: pf}
    wr_by_hour      JSONB,      -- {hour: win_rate}
    trades_by_hour  JSONB,      -- {hour: count}
    --
    run_date        DATE NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(signal_id, instrument, run_date)
);
