CREATE TABLE IF NOT EXISTS gift_convergence_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    gift_price NUMERIC(10,2),
    nse_open NUMERIC(10,2),
    basis NUMERIC(8,2),
    direction VARCHAR(10),
    entry_price NUMERIC(10,2),
    exit_price NUMERIC(10,2),
    pnl_points NUMERIC(8,2),
    exit_reason VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gift_conv_date ON gift_convergence_log(trade_date);

CREATE TABLE IF NOT EXISTS max_oi_levels (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    snapshot_time TIME,
    max_put_strike INT,
    max_put_oi BIGINT,
    max_call_strike INT,
    max_call_oi BIGINT,
    max_pain_strike INT,
    nifty_price NUMERIC(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_max_oi_date ON max_oi_levels(trade_date);

CREATE TABLE IF NOT EXISTS options_trade_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    signal_id VARCHAR(40),
    strategy VARCHAR(30),
    atm_strike INT,
    entry_credit NUMERIC(10,2),
    exit_debit NUMERIC(10,2),
    pnl_points NUMERIC(8,2),
    iv_entry NUMERIC(6,2),
    iv_exit NUMERIC(6,2),
    iv_crush_pct NUMERIC(6,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_options_trade_date ON options_trade_log(trade_date);

CREATE TABLE IF NOT EXISTS structural_signal_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    signal_id VARCHAR(40) NOT NULL,
    direction VARCHAR(10),
    confidence NUMERIC(6,4),
    entry_price NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    target NUMERIC(10,2),
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_structural_sig_date ON structural_signal_log(trade_date);
