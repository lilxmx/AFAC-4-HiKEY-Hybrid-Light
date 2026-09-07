#!/bin/bash
# AFAC2026 Task4 Baseline - Full Pipeline Runner
# This script runs the complete pipeline in the background with logging.
#
# Usage:
#   bash run_full.sh              # Full pipeline (preprocess + inference)
#   bash run_full.sh --preprocess # Only preprocess
#   bash run_full.sh --inference  # Only inference (requires pre-built indices)
#
# Output:
#   - output/answer.csv          # Competition submission file
#   - output/evidence.json       # Evidence tracking file
#   - logs/pipeline.log          # Full pipeline log
#   - logs/question_details.jsonl # Per-question details for optimization
#   - logs/run_summary.json      # Run summary statistics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup environment. Copy .env.example to .env and fill in credentials locally,
# or export DASHSCOPE_API_KEY before running this script.
if [ -f "../../.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ../../.env
  set +a
fi

# Create log directory
mkdir -p logs

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/run_${TIMESTAMP}.log"

echo "============================================================"
echo "  AFAC2026 Task4 Baseline - Background Runner"
echo "============================================================"
echo "  Start time: $(date)"
echo "  Log file: $LOG_FILE"
echo "  PID file: logs/run.pid"
echo "============================================================"

# Determine mode
MODE="full"
ARGS=""
if [ "$1" == "--preprocess" ]; then
    MODE="preprocess"
    ARGS="--preprocess-only"
elif [ "$1" == "--inference" ]; then
    MODE="inference"
    ARGS="--inference-only"
fi

echo "  Mode: $MODE"
echo ""

# Run in background with nohup
nohup python run.py $ARGS > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > logs/run.pid

echo "  Started background process: PID=$PID"
echo ""
echo "  Monitor progress:"
echo "    tail -f $LOG_FILE"
echo ""
echo "  Check if running:"
echo "    ps -p $PID"
echo ""
echo "  Kill if needed:"
echo "    kill $PID"
echo ""
echo "  View results when done:"
echo "    cat output/answer.csv"
echo "    cat logs/run_summary.json"
echo "============================================================"
