#!/usr/bin/env bash
set -euo pipefail

CONTAINER="trading-db"
DB_USER="trader"
DB_NAME="trading"
DB_PASS="trader123"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "  Structural Tier 2 Signal Setup"
echo "  4 institutional flow edges"
echo "========================================"

echo ""
echo "[1/3] Running DB migration: 016_structural_tier2.sql"
docker cp "$SCRIPT_DIR/db/migrations/016_structural_tier2.sql" "$CONTAINER:/tmp/016.sql"
docker exec -e PGPASSWORD="$DB_PASS" "$CONTAINER" \
    psql -U "$DB_USER" -d "$DB_NAME" -f /tmp/016.sql
echo "Done"

echo ""
echo "[2/3] Verifying imports"
cd "$SCRIPT_DIR"
venv/bin/python3 -c "
from signals.structural.rollover_flow import RolloverFlowSignal
from signals.structural.index_rebalance import IndexRebalanceSignal
from signals.structural.quarter_window import QuarterWindowSignal
from signals.structural.dii_put_floor import DIIPutFloorSignal
print('All 4 Tier 2 structural signals import OK')
"
echo "Done"

echo ""
echo "[3/3] Running tests"
venv/bin/python3 -m pytest tests/test_structural_tier2.py -v --tb=short 2>&1 | tail -40
echo "Done"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "  1. ROLLOVER_FLOW     (SCORING)"
echo "  2. INDEX_REBALANCE   (SCORING)"
echo "  3. QUARTER_WINDOW    (SCORING)"
echo "  4. DII_PUT_FLOOR     (SCORING)"
echo "========================================"
