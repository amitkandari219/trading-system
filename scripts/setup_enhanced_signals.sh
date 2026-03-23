#!/usr/bin/env bash
# ============================================================
# Setup script for Enhanced Signal modules (11 new signals)
# Runs inside Docker container: trading-db
# ============================================================
set -euo pipefail

CONTAINER="trading-db"
DB_USER="trader"
DB_NAME="trading"
DB_PASS="trader123"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MIGRATION_FILE="db/migrations/009_enhanced_signals.sql"

echo "========================================"
echo "  Enhanced Signals Setup"
echo "  11 new signal modules"
echo "========================================"

# Step 1: Run DB migration
echo ""
echo "[1/4] Running DB migration: 009_enhanced_signals.sql"
echo "----------------------------------------"
docker cp "$SCRIPT_DIR/$MIGRATION_FILE" "$CONTAINER:/tmp/009_enhanced_signals.sql"
docker exec -e PGPASSWORD="$DB_PASS" "$CONTAINER" \
    psql -U "$DB_USER" -d "$DB_NAME" -f /tmp/009_enhanced_signals.sql
echo "✅ Migration complete"

# Step 2: Install Python dependencies
echo ""
echo "[2/4] Installing Python dependencies"
echo "----------------------------------------"
pip install xgboost pytrends --break-system-packages 2>/dev/null || \
    pip install xgboost pytrends || \
    echo "⚠️  Could not install xgboost/pytrends — meta-learner will use heuristic fallback"
echo "✅ Dependencies checked"

# Step 3: Run tests
echo ""
echo "[3/4] Running signal tests"
echo "----------------------------------------"
export DATABASE_DSN="postgresql://${DB_USER}:${DB_PASS}@localhost:5450/${DB_NAME}"
cd "$SCRIPT_DIR"
python -m pytest tests/test_enhanced_signals.py -v --tb=short 2>&1 | head -80
echo "✅ Tests complete"

# Step 4: Create meta-learner model directory
echo ""
echo "[4/4] Setting up model directory"
echo "----------------------------------------"
mkdir -p "$SCRIPT_DIR/models/meta_learner"
echo "✅ Model directory created"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo ""
echo "  New signals registered as OVERLAY in"
echo "  paper_trading/signal_compute.py:"
echo ""
echo "  1.  PCR_AUTOTRENDER"
echo "  2.  ROLLOVER_ANALYSIS"
echo "  3.  FII_FUTURES_OI"
echo "  4.  DELIVERY_PCT"
echo "  5.  SENTIMENT_COMPOSITE"
echo "  6.  BOND_YIELD_SPREAD"
echo "  7.  GAMMA_EXPOSURE"
echo "  8.  VOL_TERM_STRUCTURE"
echo "  9.  RBI_MACRO_FILTER"
echo "  10. ORDER_FLOW_IMBALANCE"
echo "  11. XGBOOST_META_LEARNER"
echo ""
echo "  Run on your Mac:"
echo "    chmod +x scripts/setup_enhanced_signals.sh"
echo "    ./scripts/setup_enhanced_signals.sh"
echo "========================================"
