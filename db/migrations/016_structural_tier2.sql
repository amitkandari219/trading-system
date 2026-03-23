CREATE TABLE IF NOT EXISTS rollover_flow_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    expiry_date DATE,
    rollover_ratio NUMERIC(6,4),
    cost_of_carry NUMERIC(6,4),
    fii_long_short NUMERIC(6,4),
    signal_direction VARCHAR(10),
    confidence NUMERIC(6,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rollover_flow_date ON rollover_flow_log(trade_date);

CREATE TABLE IF NOT EXISTS index_rebalance_events (
    id SERIAL PRIMARY KEY,
    announcement_date DATE,
    effective_date DATE NOT NULL,
    index_name VARCHAR(30),
    action VARCHAR(10),
    stock_symbol VARCHAR(20),
    estimated_flow_cr NUMERIC(12,2),
    adtv_multiple NUMERIC(6,2),
    actual_return_pct NUMERIC(8,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rebalance_effective ON index_rebalance_events(effective_date);

CREATE TABLE IF NOT EXISTS quarter_window_log (
    id SERIAL PRIMARY KEY,
    quarter_end_date DATE NOT NULL,
    phase VARCHAR(20),
    nifty_return_pct NUMERIC(8,4),
    delivery_confirmation BOOLEAN,
    top_winners JSONB,
    bottom_losers JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dii_put_floor (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    strike INT,
    dii_put_oi BIGINT,
    dii_put_oi_change BIGINT,
    fii_put_oi BIGINT,
    nifty_distance_pts NUMERIC(8,2),
    floor_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dii_floor_date ON dii_put_floor(trade_date);
