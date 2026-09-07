#!/bin/bash
# m03_fulltext_baseline - Run script
# Usage:
#   ./run.sh                    # Run all domains (foreground)
#   ./run.sh --bg               # Run in background
#   ./run.sh --domains insurance regulatory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Parse --bg flag
BG_MODE=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--bg" ]; then
        BG_MODE=true
    else
        ARGS+=("$arg")
    fi
done

if [ "$BG_MODE" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$SCRIPT_DIR/logs/bg_run_${TIMESTAMP}.log"
    mkdir -p "$SCRIPT_DIR/logs"
    echo "Starting in background..."
    echo "  Log: $LOG_FILE"
    nohup python3 "$SCRIPT_DIR/run.py" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "  PID: $PID"
    echo "$PID" > "$SCRIPT_DIR/logs/pid.txt"
    sleep 2
    tail -20 "$LOG_FILE"
else
    python3 "$SCRIPT_DIR/run.py" "${ARGS[@]}"
fi
