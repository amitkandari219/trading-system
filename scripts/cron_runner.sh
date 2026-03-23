#!/bin/bash
# ================================================================
# Trading System Cron Runner
#
# Wrapper for all cron jobs. Handles:
#   - Lock files (prevent overlapping runs)
#   - Logging with rotation
#   - Telegram alerting on failure
#   - Health check support
# ================================================================

set -euo pipefail

# ── Configuration ──
TRADING_DIR="/Users/amitkandari/Desktop/trading-system"
VENV_PYTHON="$TRADING_DIR/venv/bin/python3"
LOG_DIR="$TRADING_DIR/logs"
LOCK_DIR="$TRADING_DIR/logs/locks"
HEALTH_DIR="$TRADING_DIR/logs/health"

mkdir -p "$LOG_DIR" "$LOCK_DIR" "$HEALTH_DIR"

# ── Arguments ──
JOB_NAME="${1:-unknown}"
PYTHON_MODULE="${2:-}"
shift 2 || true
EXTRA_ARGS="$@"

# ── Lock file ──
LOCK_FILE="$LOCK_DIR/${JOB_NAME}.lock"

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "0")
        if kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] SKIP: $JOB_NAME already running (PID $LOCK_PID)" >> "$LOG_DIR/cron.log"
            exit 0
        else
            echo "[$(date '+%H:%M:%S')] STALE LOCK: $JOB_NAME (PID $LOCK_PID dead), removing" >> "$LOG_DIR/cron.log"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
}

release_lock() {
    rm -f "$LOCK_FILE"
}

# ── Logging ──
DATE_STR=$(date '+%Y-%m-%d')
LOG_FILE="$LOG_DIR/${JOB_NAME}_${DATE_STR}.log"

# Rotate: keep last 14 days of logs
find "$LOG_DIR" -name "${JOB_NAME}_*.log" -mtime +14 -delete 2>/dev/null || true

# ── Health check timestamp ──
touch_health() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$HEALTH_DIR/${JOB_NAME}.last_run"
}

# ── Telegram alert on failure ──
alert_failure() {
    local exit_code=$1
    local tail_log=$(tail -20 "$LOG_FILE" 2>/dev/null | head -15)

    # Try to send via Python (uses the trading system's alerter)
    "$VENV_PYTHON" -c "
import sys
sys.path.insert(0, '$TRADING_DIR')
try:
    from monitoring.telegram_alerter import TelegramAlerter
    import os
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    if token and chat_id:
        a = TelegramAlerter(token, chat_id)
        a.send('CRITICAL', 'CRON FAILED: $JOB_NAME (exit $exit_code)\n\`\`\`\n${tail_log}\n\`\`\`', priority='critical')
        import time; time.sleep(2)  # let queue drain
except Exception as e:
    print(f'Telegram alert failed: {e}', file=sys.stderr)
" 2>/dev/null || true
}

# ── Main execution ──
acquire_lock
trap release_lock EXIT

echo "════════════════════════════════════════════════════════" >> "$LOG_FILE"
echo "  $JOB_NAME started at $(date '+%Y-%m-%d %H:%M:%S IST')" >> "$LOG_FILE"
echo "  Module: $PYTHON_MODULE $EXTRA_ARGS" >> "$LOG_FILE"
echo "════════════════════════════════════════════════════════" >> "$LOG_FILE"

cd "$TRADING_DIR"

EXIT_CODE=0
"$VENV_PYTHON" -m "$PYTHON_MODULE" $EXTRA_ARGS >> "$LOG_FILE" 2>&1 || EXIT_CODE=$?

echo "" >> "$LOG_FILE"
echo "  Finished at $(date '+%Y-%m-%d %H:%M:%S IST') (exit code: $EXIT_CODE)" >> "$LOG_FILE"
echo "════════════════════════════════════════════════════════" >> "$LOG_FILE"

touch_health

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] FAILED: $JOB_NAME exit=$EXIT_CODE" >> "$LOG_DIR/cron.log"
    alert_failure $EXIT_CODE
    exit $EXIT_CODE
else
    echo "[$(date '+%H:%M:%S')] OK: $JOB_NAME" >> "$LOG_DIR/cron.log"
fi
