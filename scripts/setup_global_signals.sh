#!/bin/bash
# ================================================================
# Setup script for Global Market Signals
# Run this on your Mac from the trading-system directory:
#   chmod +x scripts/setup_global_signals.sh
#   ./scripts/setup_global_signals.sh
# ================================================================

set -e

TRADING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Global Market Signals Setup ==="
echo "Trading dir: $TRADING_DIR"
echo ""

# ── Docker DB settings (from docker-compose.yml) ─────────────
DB_CONTAINER="trading-db"
DB_USER="trader"
DB_NAME="trading"

# Verify container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${DB_CONTAINER}$"; then
    echo "ERROR: Docker container '${DB_CONTAINER}' is not running."
    echo "  Start it with: docker compose up -d"
    exit 1
fi
echo "Docker container '${DB_CONTAINER}' is running ✓"
echo ""

# ── Step 1: Run DB migration ─────────────────────────────────
echo "[1/4] Running DB migration 008_global_markets.sql..."
docker cp "$TRADING_DIR/db/migrations/008_global_markets.sql" "${DB_CONTAINER}:/tmp/008_global_markets.sql"
docker exec -i "${DB_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" -f /tmp/008_global_markets.sql
echo "  ✓ Migration complete"
echo ""

# ── Step 2: Install yfinance ─────────────────────────────────
echo "[2/4] Installing yfinance..."
pip install yfinance --quiet 2>/dev/null || pip3 install yfinance --quiet
echo "  ✓ yfinance installed"
echo ""

# ── Step 3: Backfill global data ─────────────────────────────
echo "[3/4] Backfilling 5 years of global market history..."
cd "$TRADING_DIR"
DATABASE_DSN="postgresql://${DB_USER}:trader123@localhost:5450/${DB_NAME}" python -m scripts.global_pre_market --backfill 1825
echo "  ✓ Backfill complete"
echo ""

# ── Step 4: Install crontab ──────────────────────────────────
echo "[4/4] Installing updated crontab..."
crontab "$TRADING_DIR/scripts/crontab.txt"
echo "  ✓ Crontab installed"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "New cron schedule:"
echo "  8:30 AM  Global pre-market (S&P, VIX, DXY, Brent, GIFT Nifty)"
echo "  8:45 AM  Daily signals + orders"
echo "  9:00 AM  Health check"
echo "  9:14 AM  Intraday runner"
echo "  3:45 PM  Post-market reconciliation"
echo "  4:00 PM  Data refresh"
echo "  6:30 PM  FII pipeline"
echo ""
echo "To run walk-forward validation:"
echo "  DATABASE_DSN='postgresql://trader:trader123@localhost:5450/trading' python -m backtest.global_walk_forward"
echo ""
echo "To test pre-market pipeline (dry run):"
echo "  DATABASE_DSN='postgresql://trader:trader123@localhost:5450/trading' python -m scripts.global_pre_market --dry-run"
