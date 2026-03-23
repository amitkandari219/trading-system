-- ================================================================
-- 017: Options Chain Daily + Options Daily Summary
--
-- Raw intraday options chain data (from Kite Connect)
-- and derived daily summary (max pain, PCR, OI walls, IV skew).
-- ================================================================

CREATE TABLE IF NOT EXISTS options_chain_daily (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    strike NUMERIC NOT NULL,
    option_type VARCHAR(2) NOT NULL,  -- CE or PE
    open_interest BIGINT,
    change_in_oi BIGINT,
    volume BIGINT,
    iv NUMERIC,
    ltp NUMERIC,
    bid_price NUMERIC,
    ask_price NUMERIC,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, expiry_date, strike, option_type)
);
CREATE INDEX IF NOT EXISTS idx_options_chain_date ON options_chain_daily(date);
CREATE INDEX IF NOT EXISTS idx_options_chain_expiry ON options_chain_daily(expiry_date);

CREATE TABLE IF NOT EXISTS options_daily_summary (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    max_pain_strike NUMERIC,
    pcr_oi NUMERIC,              -- put/call OI ratio
    put_oi_max_strike NUMERIC,   -- put wall
    call_oi_max_strike NUMERIC,  -- call wall
    total_put_oi BIGINT,
    total_call_oi BIGINT,
    atm_put_iv NUMERIC,
    atm_call_iv NUMERIC,
    iv_skew NUMERIC,             -- put IV - call IV
    atm_straddle_premium NUMERIC,
    oi_concentration_ratio NUMERIC,
    oi_concentration_center NUMERIC,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_options_daily_summary_date ON options_daily_summary(date);
