#!/usr/bin/env bash
set -euo pipefail

CONTAINER="trading-db"
DB_USER="trader"
DB_NAME="trading"
DB_PASS="trader123"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "  Intraday Signal Setup"
echo "  10 new intraday signals"
echo "========================================"

# Step 1: Run DB migration
echo ""
echo "[1/3] Running DB migration: 011_intraday_signals.sql"
docker cp "$SCRIPT_DIR/db/migrations/011_intraday_signals.sql" "$CONTAINER:/tmp/011.sql"
docker exec -e PGPASSWORD="$DB_PASS" "$CONTAINER" \
    psql -U "$DB_USER" -d "$DB_NAME" -f /tmp/011.sql
echo "Done"

# Step 2: Verify imports
echo ""
echo "[2/3] Verifying signal imports"
cd "$SCRIPT_DIR"
venv/bin/python3 -c "
from signals.intraday.orb_signal import ORBSignal
from signals.intraday.vwap_signal import VWAPSignal
from signals.intraday.momentum_candles import MomentumCandleSignal
from signals.intraday.gift_gap_signal import GiftGapSignal
from signals.intraday.rsi_divergence import RSIDivergenceSignal
from signals.intraday.options_flow import OptionsFlowOverlay
from signals.intraday.microstructure import MicrostructureOverlay
from signals.intraday.sector_momentum import SectorMomentumOverlay
from signals.intraday.time_seasonality import TimeSeasonalityOverlay
from signals.intraday.expiry_scalper import ExpiryScalperOverlay
print('All 10 signals import OK')
"
echo "Done"

# Step 3: Run tests
echo ""
echo "[3/3] Running tests"
venv/bin/python3 -m pytest tests/test_intraday_signals.py -v --tb=short 2>&1 | tail -60
echo "Done"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "  6 SCORING + 4 OVERLAY intraday signals"
echo "========================================"
