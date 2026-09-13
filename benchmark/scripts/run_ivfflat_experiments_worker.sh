#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-all}
if [[ $# -gt 0 ]]; then
    shift
fi
PHASE_ARGS=("$@")
FORMAL_ARGS=()
if [[ "${PHASE}" == "formal" ]]; then FORMAL_ARGS=("$@"); fi
FORMAL_RESUME=0
for argument in "${FORMAL_ARGS[@]}"; do
    if [[ "${argument}" == "--resume" ]]; then
        FORMAL_RESUME=1
    fi
done
if [[ "${PHASE}" != "formal" && "${PHASE}" != "2b" && "${PHASE}" != "2b-correctness" && "${PHASE}" != "2b-production" && ${#PHASE_ARGS[@]} -gt 0 ]]; then
    echo "additional CLI options are supported only for formal and Phase 2B modes" >&2
    exit 2
fi
if [[ "${PHASE}" == "2b" || "${PHASE}" == "2b-correctness" || "${PHASE}" == "2b-production" ]]; then
    for argument in "${PHASE_ARGS[@]}"; do
        case "${argument}" in
            --phase|--phase=*|--output|--output=*|--resume|--validate-only)
                echo "worker owns phase/output/resume; unsupported option: ${argument}" >&2
                exit 2 ;;
        esac
    done
fi
SCRIPT_DIR=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd -P
)
BENCHMARK_ROOT=${BENCHMARK_ROOT:-"$(dirname -- "${SCRIPT_DIR}")"}
REPO_ROOT=${REPO_ROOT:-"$(dirname -- "${BENCHMARK_ROOT}")"}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-"$(dirname -- "${REPO_ROOT}")"}
BENCHMARK_RUNTIME_ROOT=${BENCHMARK_RUNTIME_ROOT:-"${WORKSPACE_ROOT}/benchmark"}
RUN_ROOT=${RUN_ROOT:-"${BENCHMARK_RUNTIME_ROOT}/runs"}
RUN_DIR=${RUN_DIR:?RUN_DIR must be set by the launcher}
PID_FILE=${PID_FILE:-"${RUN_ROOT}/ivfflat-profile.pid"}
PYTHON_BIN=${PYTHON_BIN:-python3}
PGVECTOR_ROOT=${PGVECTOR_ROOT:-"${REPO_ROOT}/contrib/pgvector"}
PG_CONFIG=${PG_CONFIG:-"${WORKSPACE_ROOT}/install/bin/pg_config"}
PG_CTL=${PG_CTL:-"${WORKSPACE_ROOT}/install/bin/pg_ctl"}
PGDATA=${PGDATA:-"${WORKSPACE_ROOT}/data"}
PG_OS_USER=${PG_OS_USER:-dev}
LISTS=${LISTS:-1000}
INSTALL_DEPS=${INSTALL_DEPS:-1}
VENV_DIR=${VENV_DIR:-"${BENCHMARK_RUNTIME_ROOT}/.venv"}
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
PHASE2B_WARMUP=${PHASE2B_WARMUP:-100}
PHASE2B_QUERIES=${PHASE2B_QUERIES:-100}
PHASE2B_PROBES=${PHASE2B_PROBES:-16,64,128}
PHASE2B_MODE=${PHASE2B_MODE:-full}
PHASE2B_DISTANCE_PATH=${PHASE2B_DISTANCE_PATH:-generic}
PHASE2B_ROUNDS=${PHASE2B_ROUNDS:-2}
PHASE2B_BASELINE_PATH=${PHASE2B_BASELINE_PATH:-generic}
PHASE2B_TEST_PATH=${PHASE2B_TEST_PATH:-direct}
PHASE2B_CHECK_FUSED2_EDGES=${PHASE2B_CHECK_FUSED2_EDGES:-0}
PHASE2B_OPTFLAGS=${PHASE2B_OPTFLAGS:-"-march=haswell -mtune=haswell -mavx2 -mfma"}
PHASE2B_CORRECTNESS_WARMUP=${PHASE2B_CORRECTNESS_WARMUP:-100}
PHASE2B_CORRECTNESS_QUERIES=${PHASE2B_CORRECTNESS_QUERIES:-100}
PHASE2B_UNSUPPORTED_QUERIES=${PHASE2B_UNSUPPORTED_QUERIES:-10}
PHASE2B_PRODUCTION_WARMUP=${PHASE2B_PRODUCTION_WARMUP:-100}
PHASE2B_PRODUCTION_QUERIES=${PHASE2B_PRODUCTION_QUERIES:-1000}
PHASE2B_PRODUCTION_PROBES=${PHASE2B_PRODUCTION_PROBES:-16,64,128}
PHASE2B_PRODUCTION_ROUNDS=${PHASE2B_PRODUCTION_ROUNDS:-4}
FORMAL_WARMUP=${FORMAL_WARMUP:-1}
FORMAL_QUERIES=${FORMAL_QUERIES:-3}
FORMAL_PROBES=${FORMAL_PROBES:-16}
FORMAL_DATASET=${FORMAL_DATASET:-gist-l2}
for formal_arg_index in "${!FORMAL_ARGS[@]}"; do
    argument=${FORMAL_ARGS[formal_arg_index]}
    if [[ "${argument}" == "--dataset" ]]; then
        formal_dataset_index=$((formal_arg_index + 1))
        if [[ ${formal_dataset_index} -ge ${#FORMAL_ARGS[@]} ]]; then
            echo "--dataset requires a value" >&2
            exit 2
        fi
        FORMAL_DATASET=${FORMAL_ARGS[formal_dataset_index]}
    elif [[ "${argument}" == --dataset=* ]]; then
        FORMAL_DATASET=${argument#--dataset=}
    fi
done
if [[ "${FORMAL_DATASET}" != "glove-cosine" && "${FORMAL_DATASET}" != "gist-l2" ]]; then
    echo "formal dataset must be glove-cosine or gist-l2" >&2
    exit 2
fi
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
    REQUIREMENTS_FILE="${BENCHMARK_RUNTIME_ROOT}/requirements.txt"
    if [[ "${INSTALL_DEPS}" != "1" ]]; then
        echo "missing Python dependencies; install ${REQUIREMENTS_FILE}" >&2
        exit 1
    fi
    echo "[$(date -u --iso-8601=seconds)] install Python dependencies into ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    PYTHON_BIN="${VENV_DIR}/bin/python"
    env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy "${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS_FILE}"
fi
export BENCHMARK_RUNTIME_ROOT
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
    echo "phase2b_warmup=${PHASE2B_WARMUP}"
    echo "phase2b_queries=${PHASE2B_QUERIES}"
    echo "phase2b_probes=${PHASE2B_PROBES}"
    echo "phase2b_mode=${PHASE2B_MODE}"
    echo "phase2b_distance_path=${PHASE2B_DISTANCE_PATH}"
    echo "phase2b_rounds=${PHASE2B_ROUNDS}"
    echo "phase2b_baseline_path=${PHASE2B_BASELINE_PATH}"
    echo "phase2b_test_path=${PHASE2B_TEST_PATH}"
    echo "phase2b_check_fused2_edges=${PHASE2B_CHECK_FUSED2_EDGES}"
    printf 'phase_args='
    printf ' %q' "${PHASE_ARGS[@]}"
    printf '\n'
    echo "phase2b_optflags=${PHASE2B_OPTFLAGS}"
    echo "phase2b_correctness_warmup=${PHASE2B_CORRECTNESS_WARMUP}"
    echo "phase2b_correctness_queries=${PHASE2B_CORRECTNESS_QUERIES}"
    echo "phase2b_unsupported_queries=${PHASE2B_UNSUPPORTED_QUERIES}"
    echo "phase2b_production_warmup=${PHASE2B_PRODUCTION_WARMUP}"
    echo "phase2b_production_queries=${PHASE2B_PRODUCTION_QUERIES}"
    echo "phase2b_production_probes=${PHASE2B_PRODUCTION_PROBES}"
    echo "phase2b_production_rounds=${PHASE2B_PRODUCTION_ROUNDS}"
    echo "formal_warmup=${FORMAL_WARMUP}"
    echo "formal_queries=${FORMAL_QUERIES}"
    echo "formal_probes=${FORMAL_PROBES}"
    echo "formal_dataset=${FORMAL_DATASET}"
    echo "formal_resume=${FORMAL_RESUME}"
    printf 'formal_args='
    printf ' %q' "${FORMAL_ARGS[@]}"
    printf '\n'
    echo "python=${PYTHON_BIN}"
    "${PG_CONFIG}" --version
    "${PYTHON_BIN}" --version
} >"${RUN_DIR}/environment.txt"

COMMON_ARGS=(--host "${DB_HOST}" --port "${DB_PORT}" --dbname "${DB_NAME}" --user "${DB_USER}")
PHASE2B_RUN_ARGS=()
if [[ "${PHASE}" == "2b" || "${PHASE}" == "2b-correctness" || "${PHASE}" == "2b-production" ]]; then
    PHASE2B_EFFECTIVE_BASELINE_PATH=${PHASE2B_BASELINE_PATH}
    PHASE2B_EFFECTIVE_TEST_PATH=${PHASE2B_TEST_PATH}
    for ((phase_arg_index = 0; phase_arg_index < ${#PHASE_ARGS[@]}; phase_arg_index++)); do
        argument=${PHASE_ARGS[phase_arg_index]}
        case "${argument}" in
            --baseline-path=*) PHASE2B_EFFECTIVE_BASELINE_PATH=${argument#*=} ;;
            --test-path=*) PHASE2B_EFFECTIVE_TEST_PATH=${argument#*=} ;;
            --baseline-path)
                if ((phase_arg_index + 1 < ${#PHASE_ARGS[@]})); then
                    PHASE2B_EFFECTIVE_BASELINE_PATH=${PHASE_ARGS[phase_arg_index + 1]}
                fi ;;
            --test-path)
                if ((phase_arg_index + 1 < ${#PHASE_ARGS[@]})); then
                    PHASE2B_EFFECTIVE_TEST_PATH=${PHASE_ARGS[phase_arg_index + 1]}
                fi ;;
        esac
    done
    PHASE2B_RUN_ARGS=(--phase "${PHASE}" --baseline-path "${PHASE2B_BASELINE_PATH}"
                     --test-path "${PHASE2B_TEST_PATH}" --topk "${TOPK}" --lists "${LISTS}")
    if [[ "${PHASE}" == "2b" ]]; then
        PHASE2B_OUTPUT_NAME=phase_2b_profile
        if [[ "${PHASE2B_EFFECTIVE_BASELINE_PATH}" == "direct" && "${PHASE2B_EFFECTIVE_TEST_PATH}" == "fused2" ]]; then
            PHASE2B_OUTPUT_NAME=phase_2b_fused2
        fi
        PHASE2B_RUN_ARGS+=(--warmup "${PHASE2B_WARMUP}" --queries "${PHASE2B_QUERIES}"
                          --probes-list "${PHASE2B_PROBES}" --mode "${PHASE2B_MODE}"
                          --distance-path "${PHASE2B_DISTANCE_PATH}" --rounds "${PHASE2B_ROUNDS}"
                          --output "${RUN_DIR}/${PHASE2B_OUTPUT_NAME}")
    elif [[ "${PHASE}" == "2b-production" ]]; then
        PHASE2B_RUN_ARGS=(--phase "${PHASE}" --baseline-path direct --test-path fused2
                          --topk "${TOPK}" --lists "${LISTS}"
                          --warmup "${PHASE2B_PRODUCTION_WARMUP}"
                          --queries "${PHASE2B_PRODUCTION_QUERIES}"
                          --probes-list "${PHASE2B_PRODUCTION_PROBES}" --mode both
                          --distance-path interleaved --rounds "${PHASE2B_PRODUCTION_ROUNDS}"
                          --output "${RUN_DIR}/phase_2b_fused2_production")
    else
        PHASE2B_RUN_ARGS+=(--warmup "${PHASE2B_CORRECTNESS_WARMUP}" --queries "${PHASE2B_CORRECTNESS_QUERIES}"
                          --probes-list 64 --unsupported-queries "${PHASE2B_UNSUPPORTED_QUERIES}"
                          --output "${RUN_DIR}/phase_2b_correctness")
        if [[ "${PHASE2B_CHECK_FUSED2_EDGES}" == "1" ]]; then
            PHASE2B_RUN_ARGS+=(--check-fused2-edges)
        fi
    fi
    PHASE2B_RUN_ARGS+=("${PHASE_ARGS[@]}")
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" run "${PHASE2B_RUN_ARGS[@]}" --validate-only
    printf '%q ' "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" run "${PHASE2B_RUN_ARGS[@]}" >"${RUN_DIR}/run-command.txt"
    printf '\n' >>"${RUN_DIR}/run-command.txt"
fi

# Both Phase C parts use the shared runner. Part 2 never rebuilds binaries/index.
if [[ "${PHASE}" == "formal" ]]; then
    FORMAL_PART2=0
    for argument in "${FORMAL_ARGS[@]}"; do
        if [[ "${argument}" == "--part2" ]]; then FORMAL_PART2=1; fi
    done
    if [[ "${FORMAL_RESUME}" == "1" && -f "${RUN_DIR}/phase_c_artifacts/phase_c_manifest.json" ]]; then
        if "${PYTHON_BIN}" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("part") != 2)' "${RUN_DIR}/phase_c_artifacts/phase_c_manifest.json"; then
            FORMAL_PART2=1
        fi
    fi
    if [[ "${FORMAL_PART2}" == "1" ]]; then
        PHASE_C_DEFAULTS=(--part2 --dataset gist-l2 --queries 1000 --warmup 100
                          --rounds 10 --probes-list 1,2,4,8,16,32,64,128,256)
    else
        PHASE_C_DEFAULTS=(--part1 --dataset gist-l2 --queries 3 --warmup 1
                          --rounds 1 --probes-list 16)
    fi
    PHASE_C_RUN_ARGS=("${COMMON_ARGS[@]}" run --phase formal "${PHASE_C_DEFAULTS[@]}"
                      --lists 1000 --topk 10 --output "${RUN_DIR}/phase_c_artifacts" "${FORMAL_ARGS[@]}")
    printf '%q ' "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${PHASE_C_RUN_ARGS[@]}" >"${RUN_DIR}/phase_c_run_command.txt"
    printf '\n' >>"${RUN_DIR}/phase_c_run_command.txt"
    if [[ "${FORMAL_PART2}" == "1" ]]; then
        sed -i 's/^formal_warmup=.*/formal_warmup=100/; s/^formal_queries=.*/formal_queries=1000/; s/^formal_probes=.*/formal_probes=1,2,4,8,16,32,64,128,256/' "${RUN_DIR}/environment.txt"
        echo 'phase_c_part=2' >>"${RUN_DIR}/environment.txt"
        echo 'formal_rounds=10' >>"${RUN_DIR}/environment.txt"
    fi
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${PHASE_C_RUN_ARGS[@]}"
    exit 0
fi

PROFILE_BUILD_ARGS=()
if [[ "${PHASE}" == "2b" || "${PHASE}" == "2b-correctness" ]]; then
    PROFILE_CFLAGS="-DIVFFLAT_BENCH -DIVFFLAT_PROFILE_2B"
    PROFILE_BUILD_ARGS=(OPTFLAGS="${PHASE2B_OPTFLAGS}")
elif [[ "${PHASE}" == "2b-production" ]]; then
    PROFILE_CFLAGS="-DIVFFLAT_FUSED2"
    PROFILE_BUILD_ARGS=(OPTFLAGS="${PHASE2B_OPTFLAGS}")
fi
if [[ "${PHASE}" == "formal" && "${FORMAL_RESUME}" == "1" ]]; then
    echo "[$(date -u --iso-8601=seconds)] resume formal run; keep installed build and index"
else
    if [[ "${PHASE}" == "formal" ]]; then
        echo "[$(date -u --iso-8601=seconds)] build production pgvector without IVFFLAT_BENCH"
        env -u PG_CFLAGS make -C "${PGVECTOR_ROOT}" -B PG_CONFIG="${PG_CONFIG}"
        echo "[$(date -u --iso-8601=seconds)] install production pgvector"
        env -u PG_CFLAGS make -C "${PGVECTOR_ROOT}" install PG_CONFIG="${PG_CONFIG}"
    else
        if [[ "${PHASE}" == "2b-production" ]]; then
            echo "[$(date -u --iso-8601=seconds)] build pgvector with FUSED2 path switch and no profiling"
            env -u PG_CFLAGS make -C "${PGVECTOR_ROOT}" -B PG_CONFIG="${PG_CONFIG}" IVFFLAT_PROFILE_CFLAGS="${PROFILE_CFLAGS}" "${PROFILE_BUILD_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/build.log"
        else
            echo "[$(date -u --iso-8601=seconds)] build pgvector with IVFFLAT_BENCH"
            env -u PG_CFLAGS make -C "${PGVECTOR_ROOT}" -B PG_CONFIG="${PG_CONFIG}" IVFFLAT_PROFILE_CFLAGS="${PROFILE_CFLAGS:--DIVFFLAT_BENCH}" "${PROFILE_BUILD_ARGS[@]}"
        fi
        echo "[$(date -u --iso-8601=seconds)] install pgvector"
        env -u PG_CFLAGS make -C "${PGVECTOR_ROOT}" install PG_CONFIG="${PG_CONFIG}" IVFFLAT_PROFILE_CFLAGS="${PROFILE_CFLAGS:--DIVFFLAT_BENCH}" "${PROFILE_BUILD_ARGS[@]}"
        if [[ "${PHASE}" == "2b-production" ]]; then
            grep -E "(^|[[:space:]])(gcc|cc|clang).*src/(vector|ivfscan)\\.c" "${RUN_DIR}/build.log" >"${RUN_DIR}/compile_command.txt"
            if grep -q -- "-DIVFFLAT_PROFILE_2B\|-DIVFFLAT_BENCH" "${RUN_DIR}/compile_command.txt"; then
                echo "production compile command contains profiling macro" >&2
                exit 1
            fi
            "${CC:-cc}" --version >"${RUN_DIR}/compiler_version.txt"
            INSTALLED_VECTOR_SO=${PGVECTOR_INSTALLED_SO:-"$("${PG_CONFIG}" --pkglibdir)/vector.so"}
            test -f "${INSTALLED_VECTOR_SO}"
            sha256sum "${INSTALLED_VECTOR_SO}" >"${RUN_DIR}/vector.so.sha256"
            mkdir -p "${RUN_DIR}/assembly"
            objdump -d -M intel "${INSTALLED_VECTOR_SO}" >"${RUN_DIR}/assembly/vector.so.asm"
            if strings "${INSTALLED_VECTOR_SO}" | grep -q "IVFFLAT_PROFILE"; then
                echo "production vector.so contains profiling NOTICE strings" >&2
                exit 1
            fi
        fi
    fi
    echo "[$(date -u --iso-8601=seconds)] restart PostgreSQL as ${PG_OS_USER}"
    if [[ "$(id -un)" == "${PG_OS_USER}" ]]; then
        "${PG_CTL}" -D "${PGDATA}" restart -m fast -w
    elif [[ "$(id -u)" -eq 0 ]]; then
        runuser -u "${PG_OS_USER}" -- "${PG_CTL}" -D "${PGDATA}" restart -m fast -w
    else
        echo "cannot restart PostgreSQL: current user=$(id -un), required user=${PG_OS_USER}" >&2
        exit 1
    fi
fi

BUILD_DATASET=all
if [[ "${PHASE}" == "2a" ]]; then
    BUILD_DATASET=glove-l2
elif [[ "${PHASE}" == "2a2" || "${PHASE}" == "2a34" ]]; then
    BUILD_DATASET=glove-cosine
elif [[ "${PHASE}" == "formal" ]]; then
    BUILD_DATASET=${FORMAL_DATASET}
fi
if [[ "${PHASE}" == "formal" && "${FORMAL_RESUME}" == "1" ]]; then
    echo "[$(date -u --iso-8601=seconds)] resume formal run; skip index rebuild"
elif [[ "${PHASE}" == "2b" || "${PHASE}" == "2b-correctness" || "${PHASE}" == "2b-production" ]]; then
    echo "[$(date -u --iso-8601=seconds)] phase ${PHASE} reuses existing IVFFlat indexes"
else
    echo "[$(date -u --iso-8601=seconds)] build indexes dataset=${BUILD_DATASET}"
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" build \
        --dataset "${BUILD_DATASET}" --lists "${LISTS}" --output "${RUN_DIR}/index_build.csv"
fi

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
if [[ "${PHASE}" == "2b" || "${PHASE}" == "2b-correctness" || "${PHASE}" == "2b-production" ]]; then
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" run "${PHASE2B_RUN_ARGS[@]}"
fi
if [[ "${PHASE}" == "formal" ]]; then
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${COMMON_ARGS[@]}" run \
        --phase formal --dataset "${FORMAL_DATASET}" \
        --warmup-queries "${FORMAL_WARMUP}" --queries "${FORMAL_QUERIES}" \
        --topk "${TOPK}" --probes-list "${FORMAL_PROBES}" --lists "${LISTS}" \
        --output "${RUN_DIR}/phase_formal" "${FORMAL_ARGS[@]}"
fi
if [[ "${PHASE}" == "all" || "${PHASE}" == "a" ]]; then run_phase a; fi
if [[ "${PHASE}" == "all" || "${PHASE}" == "b" ]]; then run_phase b; fi

echo "[$(date -u --iso-8601=seconds)] experiments complete"
