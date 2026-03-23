-- Migration 013: Conviction scoring log

CREATE TABLE IF NOT EXISTS conviction_log (
    id                  SERIAL PRIMARY KEY,
    trade_date          DATE NOT NULL,
    signal_id           VARCHAR(64),
    direction           VARCHAR(10),
    bull_score          INT,
    bear_score          INT,
    conviction_modifier NUMERIC(6,3),
    trend_confirmed     BOOLEAN,
    safeguards_active   TEXT[],
    modifier_breakdown  JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conviction_date ON conviction_log(trade_date);
