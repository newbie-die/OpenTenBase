#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-all}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-/workspace/benchmark}
RUN_ROOT=${RUN_ROOT:-"${BENCHMARK_ROOT}/runs"}
RUN_DIR=${RUN_DIR:?RUN_DIR must be set by the launcher}
PID_FILE=${PID_FILE:-"${RUN_ROOT}/ivfflat-profile.pid"}
PYTHON_BIN=${PYTHON_BIN:-python3}
PGVECTOR_ROOT=${PGVECTOR_ROOT:-/workspace/OpenTenBase/contrib/pgvector}
PG_CONFIG=${PG_CONFIG:-/workspace/install/bin/pg_config}
PG_CTL=${PG_CTL:-/workspace/install/bin/pg_ctl}
PGDATA=${PGDATA:-/workspace/data}
PG_OS_USER=${PG_OS_USER:-dev}
LISTS=${LISTS:-1000}
INSTALL_DEPS=${INSTALL_DEPS:-1}
VENV_DIR=${VENV_DIR:-"${BENCHMARK_ROOT}/.venv"}
WARMUP=${WARMUP:-100}
QUERIES=${QUERIES:-1000}
TOPK=${TOPK:-10}
PHASE2A_WARMUP=${PHASE2A_WARMUP:-500}
PHASE2A_QUERIES=${PHASE2A_QUERIES:-10000}
PHASE2A_PROBES=${PHASE2A_PROBES:-16,64,128}
PHASE2A_BOUNDS=${PHASE2A_BOUNDS:-0,10,20,40,100}
PHASE2A_ROUNDS=${PHASE2A_ROUNDS:-3}
PHASE2A2_WARMUP=${PHASE2A2_WARMUP:-500}
PHASE2A2_QUERIES=${PHASE2A2_QUERIES:-10000}
PHASE2A2_PROBES=${PHASE2A2_PROBES:-16,64,128}
PHASE2A34_WARMUP=${PHASE2A34_WARMUP:-5}
PHASE2A34_QUERIES=${PHASE2A34_QUERIES:-20}
PHASE2A34_PROBES=${PHASE2A34_PROBES:-64}
PHASE2A34_FILTER_DIVISORS=${PHASE2A34_FILTER_DIVISORS:-1,2,10,100,1000,10000}
DB_HOST=${DB_HOST:-127.0.0.1}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-taskdb}
DB_USER=${DB_USER:-dev}
PROFILE_SCRIPT="${BENCHMARK_ROOT}/scripts/ivfflat_profile.py"
STATUS_FILE="${RUN_DIR}/status"

finish() {
    status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "complete" >"${STATUS_FILE}"
    else
        echo "failed:${status}" >"${STATUS_FILE}"
    fi
    rm -f "${PID_FILE}"
}
trap finish EXIT

echo $$ >"${PID_FILE}"
echo "running" >"${STATUS_FILE}"

echo "[$(date -u --iso-8601=seconds)] preflight"
for executable in "${PYTHON_BIN}" "${PG_CONFIG}" "${PG_CTL}"; do
    if [[ ! -x "${executable}" ]] && ! command -v "${executable}" >/dev/null 2>&1; then
        echo "missing executable: ${executable}" >&2
        exit 1
    fi
done

if ! "${PYTHON_BIN}" -c 'import h5py, numpy, psycopg2' 2>/dev/null; then
    if [[ "${INSTALL_DEPS}" != "1" ]]; then
        echo "missing Python dependencies; install ${BENCHMARK_ROOT}/requirements.txt" >&2
        exit 1
    fi
    echo "[$(date -u --iso-8601=seconds)] install Python dependencies into ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    PYTHON_BIN="${VENV_DIR}/bin/python"
    env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy "${PYTHON_BIN}" -m pip install -r "${BENCHMARK_ROOT}/requirements.txt"
fi
test -f "${PROFILE_SCRIPT}"
test -d "${PGVECTOR_ROOT}"
test -d "${PGDATA}"

{
    echo "phase=${PHASE}"
    echo "lists=${LISTS}"
    echo "warmup=${WARMUP}"
    echo "queries=${QUERIES}"
    echo "topk=${TOPK}"
    echo "phase2a_warmup=${PHASE2A_WARMUP}"
    echo "phase2a_queries=${PHASE2A_QUERIES}"
    echo "phase2a_probes=${PHASE2A_PROBES}"
    echo "phase2a_bounds=${PHASE2A_BOUNDS}"
    echo "phase2a_rounds=${PHASE2A_ROUNDS}"
    echo "phase2a2_warmup=${PHASE2A2_WARMUP}"
    echo "phase2a2_queries=${PHASE2A2_QUERIES}"
    echo "phase2a2_probes=${PHASE2A2_PROBES}"
    echo "phase2a34_warmup=${PHASE2A34_WARMUP}"
    echo "phase2a34_queries=${PHASE2A34_QUERIES}"
    echo "phase2a34_probes=${PHASE2A34_PROBES}"
    echo "phase2a34_filter_divisors=${PHASE2A34_FILTER_DIVISORS}"
    echo "python=${PYTHON_BIN}"
    "${PG_CONFIG}" --version
    "${PYTHON_BIN}" --version
} >"${RUN_DIR}/environment.txt"

echo "[$(date -u --iso-8601=seconds)] build pgvector with IVFFLAT_BENCH"
make -C "${PGVECTOR_ROOT}" -B PG_CONFIG="${PG_CONFIG}" PG_CFLAGS="-DIVFFLAT_BENCH"
echo "[$(date -u --iso-8601=seconds)] install pgvector"
make -C "${PGVECTOR_ROOT}" install PG_CONFIG="${PG_CONFIG}" PG_CFLAGS="-DIVFFLAT_BENCH"
echo "[$(date -u --iso-8601=seconds)] restart PostgreSQL as ${PG_OS_USER}"
runuser -u "${PG_OS_USER}" -- "${PG_CTL}" -D "${PGDATA}" restart -m fast -w

COMMON_ARGS=(--host "${DB_HOST}" --port "${DB_PORT}" --dbname "${DB_NAME}" --user "${DB_USER}")
BUILD_DATASET=all
if [[ "${PHASE}" == "2a" ]]; then
    BUILD_DATASET=glove-l2
elif [[ "${PHASE}" == "2a2" || "${PHASE}" == "2a34" ]]; then
    BUILD_DATASET=glove-cosine
fi
echo "[$(date -u --iso-8601=seconds)] build indexes dataset=${BUILD_DATASET}"
"${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" build \
    --dataset "${BUILD_DATASET}" --lists "${LISTS}" --output "${RUN_DIR}/index_build.csv"

run_phase() {
    local phase=$1
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" run \
        --phase "${phase}" --warmup "${WARMUP}" --queries "${QUERIES}" \
        --topk "${TOPK}" --output "${RUN_DIR}/phase_${phase}_profile"
}

if [[ "${PHASE}" == "2a" ]]; then
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" run \
        --phase 2a --warmup "${PHASE2A_WARMUP}" --queries "${PHASE2A_QUERIES}" --topk "${TOPK}" \
        --probes-list "${PHASE2A_PROBES}" --sort-bounds "${PHASE2A_BOUNDS}" \
        --rounds "${PHASE2A_ROUNDS}" --lists "${LISTS}" --output "${RUN_DIR}/phase2a"
fi
if [[ "${PHASE}" == "2a2" ]]; then
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" --host "${DB_HOST}" --port "${DB_PORT}" --dbname "${DB_NAME}" --user "${DB_USER}" run \
        --phase 2a2 --warmup "${PHASE2A2_WARMUP}" --queries "${PHASE2A2_QUERIES}" --topk "${TOPK}" \
        --probes-list "${PHASE2A2_PROBES}" --lists "${LISTS}" \
        --output "${RUN_DIR}/phase_2a2_profile"
fi
if [[ "${PHASE}" == "2a34" ]]; then
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" --host "${DB_HOST}" --port "${DB_PORT}" --dbname "${DB_NAME}" --user "${DB_USER}" run \
        --phase 2a34 --warmup "${PHASE2A34_WARMUP}" --queries "${PHASE2A34_QUERIES}" --topk "${TOPK}" \
        --probes-list "${PHASE2A34_PROBES}" --filter-divisors "${PHASE2A34_FILTER_DIVISORS}" --lists "${LISTS}" \
        --output "${RUN_DIR}/phase_2a34_robustness"
fi
if [[ "${PHASE}" == "all" || "${PHASE}" == "a" ]]; then run_phase a; fi
if [[ "${PHASE}" == "all" || "${PHASE}" == "b" ]]; then run_phase b; fi

echo "[$(date -u --iso-8601=seconds)] experiments complete"
