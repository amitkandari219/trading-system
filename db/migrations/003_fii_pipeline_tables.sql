-- Migration 003: FII pipeline tables
-- Run after 002_enhancement_tables.sql
--
-- Ensures fii_daily_metrics and fii_signal_results exist with all
-- columns required by the FII overnight pipeline.
--
-- fii_daily_metrics: uses 'date' as PK (from schema.sql TABLE 12)
-- fii_signal_results: uses serial PK (from schema.sql TABLE 13)
--
-- This migration is idempotent — safe to run multiple times.

-- ================================================================
-- fii_daily_metrics: add columns for DII data and z-scores
-- (table already exists from schema.sql TABLE 12)
-- ================================================================

-- Add DII columns for divergence signal (NSE_005)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'fii_daily_metrics' AND column_name = 'dii_fut_long'
    ) THEN
        ALTER TABLE fii_daily_metrics ADD COLUMN dii_fut_long FLOAT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'fii_daily_metrics' AND column_name = 'dii_fut_short'
    ) THEN
        ALTER TABLE fii_daily_metrics ADD COLUMN dii_fut_short FLOAT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'fii_daily_metrics' AND column_name = 'dii_fut_net'
    ) THEN
        ALTER TABLE fii_daily_metrics ADD COLUMN dii_fut_net FLOAT;
    END IF;

    -- Z-score and percentile (cached from pipeline computation)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'fii_daily_metrics' AND column_name = 'z_score_20d'
    ) THEN
        ALTER TABLE fii_daily_metrics ADD COLUMN z_score_20d FLOAT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'fii_daily_metrics' AND column_name = 'percentile_252d'
    ) THEN
        ALTER TABLE fii_daily_metrics ADD COLUMN percentile_252d FLOAT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'fii_daily_metrics' AND column_name = 'flow_5d'
    ) THEN
        ALTER TABLE fii_daily_metrics ADD COLUMN flow_5d FLOAT;
    END IF;
END $$;

-- Index for date lookups
CREATE INDEX IF NOT EXISTS idx_fii_daily_metrics_date
    ON fii_daily_metrics(date DESC);


-- ================================================================
-- fii_signal_results: add index for valid_for_date lookups
-- (table already exists from schema.sql TABLE 13)
-- ================================================================

CREATE INDEX IF NOT EXISTS idx_fii_signal_results_valid_for
    ON fii_signal_results(valid_for_date);

CREATE INDEX IF NOT EXISTS idx_fii_signal_results_data_date
    ON fii_signal_results(data_date DESC);


-- ================================================================
-- fii_signals: active signal view for quick lookups
-- Used by fii_overlay.load_active_signals()
-- ================================================================

CREATE OR REPLACE VIEW fii_active_signals AS
SELECT
    signal_id,
    direction,
    confidence,
    pattern_name,
    notes,
    data_date,
    valid_for_date,
    was_executed
FROM fii_signal_results
WHERE valid_for_date >= CURRENT_DATE
  AND signal_id IS NOT NULL
ORDER BY valid_for_date, confidence DESC;
