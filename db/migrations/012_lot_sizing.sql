-- Migration 012: Lot-based sizing log + overlay attribution

CREATE TABLE IF NOT EXISTS lot_sizing_log (
    id                  SERIAL PRIMARY KEY,
    trade_date          DATE NOT NULL,
    signal_id           VARCHAR(64) NOT NULL,
    lots                INT NOT NULL,
    base_lots           INT,
    lot_size            INT,
    composite_modifier  NUMERIC(6,3),
    modifier_breakdown  JSONB,
    margin_used         NUMERIC(12,0),
    margin_available    NUMERIC(12,0),
    equity_at_trade     NUMERIC(12,0),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lot_sizing_date ON lot_sizing_log(trade_date);
CREATE INDEX IF NOT EXISTS idx_lot_sizing_signal ON lot_sizing_log(signal_id, trade_date);

CREATE TABLE IF NOT EXISTS overlay_attribution (
    id                  SERIAL PRIMARY KEY,
    trade_date          DATE NOT NULL,
    signal_id           VARCHAR(64) NOT NULL,
    overlay_id          VARCHAR(64) NOT NULL,
    modifier_value      NUMERIC(6,3),
    category            VARCHAR(30),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_overlay_attr_date ON overlay_attribution(trade_date);
