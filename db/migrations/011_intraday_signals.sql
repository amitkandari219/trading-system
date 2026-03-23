-- Migration 011: Intraday signal logging and tracking tables
-- Captures ORB levels, VWAP snapshots, options flow, sector momentum,
-- expiry-day data, and intraday signal fire log.

-- ORB levels per day
CREATE TABLE IF NOT EXISTS intraday_orb_levels (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    instrument VARCHAR(16) DEFAULT 'NIFTY',
    or_high NUMERIC(10,2),
    or_low NUMERIC(10,2),
    or_range_pct NUMERIC(6,4),
    first_breakout_dir VARCHAR(10),
    first_breakout_time TIME,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date, instrument)
);

-- VWAP history (snapshots every 15 min)
CREATE TABLE IF NOT EXISTS intraday_vwap_history (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    snapshot_time TIME NOT NULL,
    vwap NUMERIC(10,2),
    upper_band_1s NUMERIC(10,2),
    lower_band_1s NUMERIC(10,2),
    price_vs_vwap_pct NUMERIC(6,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_vwap_hist_date ON intraday_vwap_history(trade_date);

-- Options flow snapshots
CREATE TABLE IF NOT EXISTS intraday_options_flow (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    snapshot_time TIME NOT NULL,
    pcr NUMERIC(6,4),
    pcr_change NUMERIC(6,4),
    max_call_oi_strike INT,
    max_put_oi_strike INT,
    iv_atm NUMERIC(6,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_options_flow_date ON intraday_options_flow(trade_date);

-- Sector momentum
CREATE TABLE IF NOT EXISTS intraday_sector_momentum (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    snapshot_time TIME NOT NULL,
    nifty_change_pct NUMERIC(6,4),
    banknifty_change_pct NUMERIC(6,4),
    it_change_pct NUMERIC(6,4),
    pharma_change_pct NUMERIC(6,4),
    momentum_score NUMERIC(6,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Expiry day specific
CREATE TABLE IF NOT EXISTS intraday_expiry_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    expiry_type VARCHAR(10),
    max_pain_strike INT,
    pin_distance_pct NUMERIC(6,4),
    gamma_acceleration NUMERIC(8,2),
    straddle_premium NUMERIC(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date)
);

-- Intraday signal fire log
CREATE TABLE IF NOT EXISTS intraday_signal_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    signal_time TIME NOT NULL,
    signal_id VARCHAR(64) NOT NULL,
    direction VARCHAR(10),
    confidence NUMERIC(4,2),
    entry_price NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    target NUMERIC(10,2),
    size_modifier NUMERIC(4,2),
    outcome VARCHAR(20),
    pnl NUMERIC(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_intraday_sig_date ON intraday_signal_log(trade_date);
CREATE INDEX IF NOT EXISTS idx_intraday_sig_id ON intraday_signal_log(signal_id, trade_date);
