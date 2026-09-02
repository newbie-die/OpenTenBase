#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT=${BENCHMARK_ROOT:-/workspace/benchmark}
RUN_ROOT=${RUN_ROOT:-"${BENCHMARK_ROOT}/runs"}
WORKER=${WORKER:-"${BENCHMARK_ROOT}/scripts/run_ivfflat_experiments_worker.sh"}
PYTHON_BIN=${PYTHON_BIN:-python3}
PHASE=${1:-all}

if [[ "${PHASE}" != "all" && "${PHASE}" != "a" && "${PHASE}" != "b" && "${PHASE}" != "2a" ]]; then
    echo "usage: $0 [all|a|b|2a]" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"
PID_FILE="${RUN_ROOT}/ivfflat-profile.pid"
if [[ -s "${PID_FILE}" ]]; then
    EXISTING_PID=$(<"${PID_FILE}")
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "experiment is already running: pid=${EXISTING_PID}" >&2
        exit 1
    fi
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/experiment.log"
mkdir -p "${RUN_DIR}"

nohup env \
    BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    RUN_DIR="${RUN_DIR}" \
    PID_FILE="${PID_FILE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    "${WORKER}" "${PHASE}" \
    >"${LOG_FILE}" 2>&1 </dev/null &

WORKER_PID=$!
echo "${WORKER_PID}" >"${PID_FILE}"
echo "started IVFFlat experiment phase=${PHASE}"
echo "pid: ${WORKER_PID}"
echo "run directory: ${RUN_DIR}"
echo "log: ${LOG_FILE}"
echo "monitor: tail -f ${LOG_FILE}"
