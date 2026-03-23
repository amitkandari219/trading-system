#!/bin/bash
# ================================================================
# Cron Watchdog — monitors and triggers jobs if cron misses them
# Runs continuously, checks every 60 seconds
# ================================================================

TRADING_DIR="/Users/amitkandari/Desktop/trading-system"
RUNNER="$TRADING_DIR/scripts/cron_runner.sh"
LOG_DIR="$TRADING_DIR/logs"
HEALTH_DIR="$LOG_DIR/health"
TODAY=$(date '+%Y-%m-%d')
DOW=$(date '+%u')  # 1=Monday, 7=Sunday

mkdir -p "$HEALTH_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] WATCHDOG: $1" >> "$LOG_DIR/cron.log"
    echo "[$(date '+%H:%M:%S')] $1"
}

check_and_run() {
    local job_name="$1"
    local module="$2"
    local scheduled_hour="$3"
    local scheduled_min="$4"

    local now_hour=$(date '+%-H')
    local now_min=$(date '+%-M')
    local now_total=$((now_hour * 60 + now_min))
    local sched_total=$((scheduled_hour * 60 + scheduled_min))
    local grace=5  # 5 min grace period

    # Only check if we're past scheduled time + grace
    if [ $now_total -lt $((sched_total + grace)) ]; then
        return
    fi

    # Don't re-trigger if already ran today
    local health_file="$HEALTH_DIR/${job_name}.last_run"
    if [ -f "$health_file" ]; then
        local last_date=$(cat "$health_file" | cut -d' ' -f1)
        if [ "$last_date" = "$TODAY" ]; then
            return
        fi
    fi

    # Don't trigger if too late (more than 60 min past schedule)
    if [ $now_total -gt $((sched_total + 60)) ]; then
        # Special case: intraday can start late until 14:00
        if [ "$job_name" = "intraday" ] && [ $now_total -lt 840 ]; then
            :  # Allow late start
        else
            return
        fi
    fi

    # Check if already running (lock file)
    if [ -f "$LOG_DIR/locks/${job_name}.lock" ]; then
        local lock_pid=$(cat "$LOG_DIR/locks/${job_name}.lock" 2>/dev/null)
        if kill -0 "$lock_pid" 2>/dev/null; then
            return  # Already running
        fi
    fi

    log "TRIGGERING $job_name (missed cron at ${scheduled_hour}:${scheduled_min})"
    nohup "$RUNNER" "$job_name" "$module" >> /dev/null 2>&1 &
}

log "Watchdog started (PID $$)"

while true; do
    TODAY=$(date '+%Y-%m-%d')
    DOW=$(date '+%u')

    # Only run on weekdays
    if [ "$DOW" -le 5 ]; then
        check_and_run "global_pre_market" "scripts.global_pre_market"    8 30
        check_and_run "pre_market"        "paper_trading.enhanced_daily_run" 8 45
        check_and_run "intraday"          "paper_trading.intraday_runner"    9 14
        check_and_run "post_market"       "paper_trading.eod_reconciler"    15 45
        check_and_run "data_refresh"      "data.kite_daily_refresh"         16 0
        check_and_run "fii_pipeline"      "fii.daily_pipeline"              18 30
    fi

    # Sunday weekly jobs
    if [ "$DOW" -eq 7 ]; then
        check_and_run "decay_scan"  "paper_trading.decay_weekly_scan"  10 0
        check_and_run "bn_review"   "paper_trading.bn_weekly_review"   10 15
        check_and_run "robustness"  "backtest.run_robustness_suite"    10 30
    fi

    sleep 60
done
