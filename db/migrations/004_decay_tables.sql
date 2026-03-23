-- Signal Decay Auto-Detection Tables
-- Migration 004: signal_overrides + decay_history
-- Run: psql -d trading -f db/migrations/004_decay_tables.sql

BEGIN;

-- ════════════════════════════════════════════════════════════
-- signal_overrides: active size overrides from decay detection
-- ════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS signal_overrides (
    override_id   BIGSERIAL PRIMARY KEY,
    signal_id     TEXT        NOT NULL,
    override_type TEXT        NOT NULL,
        -- SIZE_REDUCE, BLOCK (factor=0), MANUAL
    factor        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        -- 0.0 = blocked, 0.5 = half size, 1.0 = normal
    reason        TEXT        NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at    TIMESTAMPTZ,
        -- NULL = no expiry (manual override or demotion)
    active        BOOLEAN     NOT NULL DEFAULT TRUE,
    created_by    TEXT        NOT NULL DEFAULT 'DECAY_AUTO_MANAGER'
);

CREATE INDEX IF NOT EXISTS idx_signal_overrides_active
    ON signal_overrides (signal_id, active)
    WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_signal_overrides_expires
    ON signal_overrides (expires_at)
    WHERE active = TRUE AND expires_at IS NOT NULL;

-- ════════════════════════════════════════════════════════════
-- decay_history: log of every decay scan result
-- ════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS decay_history (
    id              BIGSERIAL   PRIMARY KEY,
    signal_id       TEXT        NOT NULL,
    scan_date       DATE        NOT NULL,
    scan_type       TEXT        NOT NULL DEFAULT 'WEEKLY',
        -- WEEKLY, DAILY_QUICK, MANUAL
    status          TEXT        NOT NULL,
        -- HEALTHY, WATCH, WARNING, CRITICAL, STRUCTURAL_DECAY, IMPROVING
    severity_score  INTEGER     NOT NULL DEFAULT 0,
        -- count of flags raised
    sharpe_20       DOUBLE PRECISION,
    win_rate_20     DOUBLE PRECISION,
    max_drawdown_20 DOUBLE PRECISION,
    consec_losses   INTEGER,
    bocd_cp_prob    DOUBLE PRECISION,
    cusum_alert     BOOLEAN     DEFAULT FALSE,
    regime          TEXT,
    regime_sharpe   DOUBLE PRECISION,
    flags           TEXT[]      DEFAULT '{}',
        -- array of flag strings: LOW_SHARPE, HIGH_DD, LOSS_STREAK, etc.
    action_taken    TEXT,
        -- SIZE_REDUCE, AUTO_DEMOTE, CLEAR_OVERRIDE, RECOMMEND_PROMOTE, NONE
    notes           TEXT        DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decay_history_signal_date
    ON decay_history (signal_id, scan_date DESC);

CREATE INDEX IF NOT EXISTS idx_decay_history_status
    ON decay_history (status, scan_date DESC);

-- ════════════════════════════════════════════════════════════
-- Helper view: consecutive critical weeks per signal
-- ════════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW v_consecutive_critical AS
WITH ranked AS (
    SELECT
        signal_id,
        scan_date,
        status,
        ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY scan_date DESC) AS rn
    FROM decay_history
    WHERE scan_type = 'WEEKLY'
)
SELECT
    signal_id,
    COUNT(*) AS consecutive_critical_weeks
FROM ranked
WHERE rn <= 10  -- look at last 10 weeks max
  AND status IN ('CRITICAL', 'STRUCTURAL_DECAY')
GROUP BY signal_id;

COMMIT;
