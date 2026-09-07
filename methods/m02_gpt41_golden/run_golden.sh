#!/bin/bash
# Golden Label Generation - Background Runner
# Usage: bash run_golden.sh [--domains regulatory insurance ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/golden_run_${TIMESTAMP}.log"

mkdir -p logs output embedding_cache

echo "============================================"
echo "  Golden Label Generation"
echo "  Log: $LOG_FILE"
echo "  Time: $(date)"
echo "============================================"

# Run in background with nohup
nohup python3 run.py "$@" > "$LOG_FILE" 2>&1 &
PID=$!

echo "  Started with PID: $PID"
echo "  Monitor: tail -f $LOG_FILE"
echo "  Check:   ps -p $PID"
echo "============================================"
echo "$PID" > logs/golden_pid.txt
