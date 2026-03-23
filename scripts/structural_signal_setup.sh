#!/usr/bin/env bash
set -euo pipefail

CONTAINER="trading-db"
DB_USER="trader"
DB_NAME="trading"
DB_PASS="trader123"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "  Structural Signal Setup"
echo "  4 India-specific structural edges"
echo "========================================"

echo ""
echo "[1/3] Running DB migration: 015_structural_signals.sql"
docker cp "$SCRIPT_DIR/db/migrations/015_structural_signals.sql" "$CONTAINER:/tmp/015.sql"
docker exec -e PGPASSWORD="$DB_PASS" "$CONTAINER" \
    psql -U "$DB_USER" -d "$DB_NAME" -f /tmp/015.sql
echo "Done"

echo ""
echo "[2/3] Verifying imports"
cd "$SCRIPT_DIR"
venv/bin/python3 -c "
from signals.structural.gift_convergence import GiftConvergence
from signals.structural.max_oi_barrier import MaxOIBarrier
from signals.structural.monday_straddle import MondayStraddle
from signals.structural.event_iv_crush import EventIVCrush
print('All 4 structural signals import OK')
"
echo "Done"

echo ""
echo "[3/3] Running tests"
venv/bin/python3 -m pytest tests/test_structural_signals.py -v --tb=short 2>&1 | tail -40
echo "Done"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "  1. GIFT_CONVERGENCE  (SCORING)"
echo "  2. MAX_OI_BARRIER    (SCORING)"
echo "  3. MONDAY_STRADDLE   (SCORING)"
echo "  4. EVENT_IV_CRUSH    (SCORING)"
echo "========================================"
