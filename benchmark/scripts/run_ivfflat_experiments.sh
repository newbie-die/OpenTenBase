#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd -P
)
BENCHMARK_ROOT=${BENCHMARK_ROOT:-"$(dirname -- "${SCRIPT_DIR}")"}
REPO_ROOT=${REPO_ROOT:-"$(dirname -- "${BENCHMARK_ROOT}")"}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-"$(dirname -- "${REPO_ROOT}")"}
BENCHMARK_RUNTIME_ROOT=${BENCHMARK_RUNTIME_ROOT:-"${WORKSPACE_ROOT}/benchmark"}
RUN_ROOT=${RUN_ROOT:-"${BENCHMARK_RUNTIME_ROOT}/runs"}
WORKER=${WORKER:-"${BENCHMARK_ROOT}/scripts/run_ivfflat_experiments_worker.sh"}
PYTHON_BIN=${PYTHON_BIN:-python3}
PHASE=${1:-all}

if [[ "${PHASE}" != "all" && "${PHASE}" != "a" && "${PHASE}" != "b" && "${PHASE}" != "2a" && "${PHASE}" != "2a2" && "${PHASE}" != "2a34" ]]; then
    echo "usage: $0 [all|a|b|2a|2a2|2a34]" >&2
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
    BENCHMARK_RUNTIME_ROOT="${BENCHMARK_RUNTIME_ROOT}" \
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
