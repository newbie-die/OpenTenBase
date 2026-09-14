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
if [[ $# -gt 0 ]]; then
    shift
fi

# Final comparison extends the shared Python/Phase C framework. Its own lock,
# manifest and explicit --formal gate also support a single foreground nohup job.
if [[ "${PHASE}" == "final-multi" ]]; then
    exec "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/ivfflat_profile.py" final-multi "$@"
fi

if [[ "${PHASE}" != "all" && "${PHASE}" != "a" && "${PHASE}" != "b" && "${PHASE}" != "2a" && "${PHASE}" != "2a2" && "${PHASE}" != "2a34" && "${PHASE}" != "2b" && "${PHASE}" != "2b-correctness" && "${PHASE}" != "2b-production" && "${PHASE}" != "formal" && "${PHASE}" != "all-fomal-exp" ]]; then
    echo "usage: $0 [all|a|b|2a|2a2|2a34|2b|2b-correctness|2b-production|formal|all-fomal-exp] [phase options]" >&2
    exit 2
fi

RESUME_RUN_DIR=
RUN_DIR_OVERRIDE=${RUN_DIR_OVERRIDE:-}
FOREGROUND=0
WORKER_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--foreground" ]]; then
        FOREGROUND=1
        shift
    elif [[ "$1" == "--resume" ]]; then
        if [[ ( "${PHASE}" != "formal" && "${PHASE}" != "all-fomal-exp" ) || $# -lt 2 ]]; then
            echo "--resume requires: formal|all-fomal-exp --resume RUN_DIR" >&2
            exit 2
        fi
        RESUME_RUN_DIR=$2
        WORKER_ARGS+=(--resume)
        shift 2
    elif [[ "$1" == "--run-dir" ]]; then
        if [[ $# -lt 2 ]]; then
            echo "--run-dir requires a path" >&2
            exit 2
        fi
        RUN_DIR_OVERRIDE=$2
        shift 2
    else
        WORKER_ARGS+=("$1")
        shift
    fi
done
if [[ "${PHASE}" != "formal" && "${PHASE}" != "all-fomal-exp" && "${PHASE}" != "2b" && "${PHASE}" != "2b-correctness" && "${PHASE}" != "2b-production" && ${#WORKER_ARGS[@]} -gt 0 ]]; then
    echo "additional CLI options are supported only for formal and Phase 2B modes" >&2
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

if [[ -n "${RESUME_RUN_DIR}" ]]; then
    if [[ ! -d "${RESUME_RUN_DIR}" ]]; then
        echo "resume run directory does not exist: ${RESUME_RUN_DIR}" >&2
        exit 2
    fi
    RUN_DIR=$(cd -- "${RESUME_RUN_DIR}" && pwd -P)
elif [[ -n "${RUN_DIR_OVERRIDE}" ]]; then
    mkdir -p "${RUN_DIR_OVERRIDE}"
    RUN_DIR=$(cd -- "${RUN_DIR_OVERRIDE}" && pwd -P)
else
    RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
    if [[ "${PHASE}" == "all-fomal-exp" ]]; then
        RUN_DIR="${RUN_ROOT}/all_fomal_exp_${RUN_ID}"
    else
        RUN_DIR="${RUN_ROOT}/${RUN_ID}"
    fi
    mkdir -p "${RUN_DIR}"
fi
LOG_FILE="${RUN_DIR}/experiment.log"

# Foreground mode is useful for supervised sessions and reproducible interruption tests.
if [[ "${FOREGROUND}" == "1" ]]; then
    echo "run directory: ${RUN_DIR}"
    echo "log: ${LOG_FILE}"
    exec env BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
        BENCHMARK_RUNTIME_ROOT="${BENCHMARK_RUNTIME_ROOT}" RUN_ROOT="${RUN_ROOT}" \
        RUN_DIR="${RUN_DIR}" PID_FILE="${PID_FILE}" PYTHON_BIN="${PYTHON_BIN}" \
        "${WORKER}" "${PHASE}" "${WORKER_ARGS[@]}" >>"${LOG_FILE}" 2>&1
fi

nohup env \
    BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
    BENCHMARK_RUNTIME_ROOT="${BENCHMARK_RUNTIME_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    RUN_DIR="${RUN_DIR}" \
    PID_FILE="${PID_FILE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    "${WORKER}" "${PHASE}" "${WORKER_ARGS[@]}" \
    >>"${LOG_FILE}" 2>&1 </dev/null &

WORKER_PID=$!
echo "${WORKER_PID}" >"${PID_FILE}"
echo "started IVFFlat experiment phase=${PHASE}"
echo "pid: ${WORKER_PID}"
echo "run directory: ${RUN_DIR}"
echo "log: ${LOG_FILE}"
echo "monitor: tail -f ${LOG_FILE}"
