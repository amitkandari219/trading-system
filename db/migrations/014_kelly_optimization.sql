CREATE TABLE IF NOT EXISTS kelly_grid_results (
    id              SERIAL PRIMARY KEY,
    kelly_fraction  NUMERIC(4,2) NOT NULL,
    cagr            NUMERIC(6,2),
    sharpe          NUMERIC(6,2),
    max_dd          NUMERIC(6,2),
    calmar          NUMERIC(6,2),
    final_equity    NUMERIC(14,0),
    wf_pass_rate    NUMERIC(4,2),
    run_date        DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS kelly_adaptive_log (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    base_kelly      NUMERIC(4,2),
    final_kelly     NUMERIC(4,2),
    adjustments     JSONB,
    gear            VARCHAR(20),
    reason          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_kelly_adaptive_date ON kelly_adaptive_log(trade_date);
