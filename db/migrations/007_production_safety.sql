-- 007_production_safety.sql
-- Crash recovery and session reconciliation tables.
-- Supports execution/crash_recovery.py for orphan order detection on startup.

CREATE TABLE IF NOT EXISTS order_intents (
    intent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_name TEXT NOT NULL,
    direction TEXT NOT NULL,
    instrument TEXT NOT NULL,
    qty INTEGER NOT NULL,
    kite_order_id TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    fill_price NUMERIC,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_order_intents_status ON order_intents(status);
CREATE INDEX IF NOT EXISTS idx_order_intents_created ON order_intents(created_at);

CREATE TABLE IF NOT EXISTS session_reconciliation_log (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL,
    reconcile_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    orphaned_count INT DEFAULT 0,
    phantom_count INT DEFAULT 0,
    drift_count INT DEFAULT 0,
    details JSONB,
    actions_taken JSONB
);
CREATE INDEX IF NOT EXISTS idx_session_recon_date ON session_reconciliation_log(session_date);
