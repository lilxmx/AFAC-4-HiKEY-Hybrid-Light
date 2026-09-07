#!/bin/bash
# ============================================================
# AFAC-4 Unified Run Launcher
# ============================================================
# Usage:
#   ./run.sh m01 --inference-only                    # foreground
#   ./run.sh m01 --inference-only --bg               # background (nohup)
#   ./run.sh m02 --domains financial_contracts --bg  # background, single domain
#   ./run.sh eval --pred runs/lgr/m01.../output/answer.csv --ref references/gpt_pro/fin-c/parsed.json --domain fc
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Load env
if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

# Proxy (internal network)
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"

METHOD="$1"
shift || true

case "$METHOD" in
    m01|m01_baseline_qwen)
        METHOD_DIR="${PROJECT_ROOT}/methods/m01_baseline_qwen"
        ;;
    m02|m02_gpt41_golden)
        METHOD_DIR="${PROJECT_ROOT}/methods/m02_gpt41_golden"
        ;;
    m04|m04_HiKEY|m04_hikey)
        METHOD_DIR="${PROJECT_ROOT}/methods/m04_HiKEY"
        ;;
    m05|m05_HiKEY_question_option|m05_hikey_question_option)
        METHOD_DIR="${PROJECT_ROOT}/methods/m05_HiKEY_question_option"
        ;;
    eval|compare)
        python "${PROJECT_ROOT}/methods/_shared/eval/compare.py" "$@"
        exit 0
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available: m01, m02, m04, m05, eval"
        exit 1
        ;;
esac

# Check for --bg flag
BG=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--bg" ] || [ "$arg" = "--background" ]; then
        BG=1
    else
        ARGS+=("$arg")
    fi
done

if [ $BG -eq 1 ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    LOG_DIR="${PROJECT_ROOT}/runs/lgr/$(basename ${METHOD_DIR})"
    mkdir -p "${LOG_DIR}"
    LOGFILE="${LOG_DIR}/nohup_${TIMESTAMP}.log"
    echo "Starting in background..."
    echo "  Method: ${METHOD_DIR}"
    echo "  Log: ${LOGFILE}"
    nohup python "${METHOD_DIR}/run.py" "${ARGS[@]}" > "${LOGFILE}" 2>&1 &
    PID=$!
    echo "  PID: ${PID}"
    echo "${PID}" > "${LOG_DIR}/latest.pid"
else
    python "${METHOD_DIR}/run.py" "${ARGS[@]}"
fi
