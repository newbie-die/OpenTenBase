#!/usr/bin/env python3
"""D1-1: exactly one 100-query x 5-probe shadow-potential round."""
import csv
import datetime
import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import psycopg2

REPO = Path("/workspace/OpenTenBase")
SRC = REPO / "benchmark/d1_1"
RUNS = Path("/workspace/benchmark/runs")
OLD = RUNS / "20260909T083926Z"
D0B = RUNS / "phase_d0b_l2_20260911T024158Z"
D10 = REPO / "benchmark/docs/phase_d1_0/phase_d1_0_design.json"
PRODUCTION = Path("/workspace/install/lib/postgresql/vector.so")
OBSERVER = Path("/workspace/benchmark/d1_1_observer.so")
DATASET = Path("/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5")
PROBES = [16, 32, 64, 128, 256]
BLOCKS = [32, 64, 128, 256]
CAPACITIES = [10, 40]
SEED = 20260911
DIM = 960
BINARY = "d56dffd4db4e08d075cad229da5b04a7dc722f1d2cd45f03fb5570b0a64f6480"
CONFIGS = [f"k{k}_b{b}" for k in CAPACITIES for b in BLOCKS]
BUCKET_NAMES = ["1_32", "33_64", "65_128", "129_256", "257_512", "513_768", "769_960"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csvwrite(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        if not rows:
            raise RuntimeError(f"no rows for {path}")
        fields = list(rows[0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def protect():
    provenance = json.loads((OLD / "freeze_evidence/production_provenance.json").read_text())
    assert sha(PRODUCTION) == BINARY
    for name, digest in provenance["source_files_sha256"].items():
        assert sha(REPO / name) == digest, name
    assert not git("diff", provenance["source_commit"], "--", "contrib/pgvector", "src")
    snapshot = Path(json.loads((OLD / "freeze_evidence/termination_receipt.json").read_text())["snapshot"])
    inventory = json.loads((snapshot / "inventory.json").read_text())
    for name, item in inventory.items():
        if item["type"] == "file":
            assert sha(OLD / name) == item["sha256"], name
    d0_hashes = json.loads((D0B / "artifact_hashes.json").read_text())
    for name, digest in d0_hashes.items():
        assert sha(D0B / name) == digest, name
    d10 = json.loads(D10.read_text())
    assert d10["status"] == "COMPLETE" and d10["rq1"]["current_k"] == 40
    assert d10["verification"]["d1_1_started"] is False
    assert not Path("/proc/516593").exists()
    return {
        "production_sha256": BINARY,
        "production_source_commit": provenance["source_commit"],
        "protected_sources_unchanged": True,
        "phase_c_files_unchanged": True,
        "phase_c_verified_files": sum(x["type"] == "file" for x in inventory.values()),
        "d0b_hash_verified_files": len(d0_hashes),
        "d1_0_status": "COMPLETE",
        "old_phase_c_runner_absent": True,
    }


def index_identity(cursor):
    cursor.execute(
        "SELECT c.oid,c.relfilenode,pg_relation_size(c.oid),pg_get_indexdef(c.oid),i.indisvalid "
        "FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid "
        "WHERE c.oid='gist_ivf_l2'::regclass"
    )
    return dict(zip(["oid", "relfilenode", "bytes", "definition", "valid"], cursor.fetchone()))


def raw_header():
    fields = [
        "round", "query_id", "probes", "candidate_ordinal", "list_rank", "list_half",
        "candidate_tid", "full_distance_squared", "block_sizes",
    ]
    suffixes = [
        "threshold_before_candidate", "heap_ready", "first_abort_dim", "evaluated_dims",
        "avoided_dims", "early_abandoned", "full_key_relation_to_threshold",
        "abort_partial_squared", "abort_certified_lower_key",
    ]
    for config in CONFIGS:
        fields.extend(f"{config}_{suffix}" for suffix in suffixes)
    return fields


def exact_quantile(histogram, quantile):
    total = sum(histogram.values())
    target = max(1, math.ceil(total * quantile))
    cumulative = 0
    for dims in sorted(histogram):
        cumulative += histogram[dims]
        if cumulative >= target:
            return dims
    return DIM


def gate(probe_rows):
    production = [row for row in probe_rows if row["threshold_capacity"] == 40]
    decisions = {}
    for block in BLOCKS:
        high = [
            row["dimension_avoid_ratio"] for row in production
            if row["block_size"] == block and row["probes"] in (64, 128, 256)
        ]
        if sum(value >= 0.40 for value in high) >= 2:
            decisions[block] = "Strong Go"
        elif sum(value >= 0.25 for value in high) >= 2:
            decisions[block] = "Go"
        elif max(high) < 0.10:
            decisions[block] = "No-Go"
        else:
            decisions[block] = "Marginal"
    rank = {"No-Go": 0, "Marginal": 1, "Go": 2, "Strong Go": 3}
    decision = max(decisions.values(), key=lambda value: rank[value])
    return decision, decisions


def analyze(root, query_rows, correctness):
    probe_rows = []
    for probes in PROBES:
        for capacity in CAPACITIES:
            for block in BLOCKS:
                rows = [
                    row for row in query_rows
                    if row["probes"] == probes and row["threshold_capacity"] == capacity
                    and row["block_size"] == block
                ]
                histogram = {}
                for row in rows:
                    for entry in json.loads(row["evaluated_dims_histogram"]):
                        dims = int(entry["dims"])
                        histogram[dims] = histogram.get(dims, 0) + int(entry["count"])
                candidates = sum(row["candidates_total"] for row in rows)
                possible = candidates * DIM
                avoided = sum(row["avoided_dimensions"] for row in rows)
                evaluated = sum(row["evaluated_dimensions"] for row in rows)
                checks = sum(row["threshold_checks"] for row in rows)
                first_total = sum(row["first_half_candidates"] for row in rows)
                second_total = sum(row["second_half_candidates"] for row in rows)
                first_possible = first_total * DIM
                second_possible = second_total * DIM
                result = {
                    "potential": "potential_k40" if capacity == 40 else "potential_k10",
                    "production_gate_eligible": int(capacity == 40),
                    "probes": probes,
                    "threshold_capacity": capacity,
                    "block_size": block,
                    "queries": len(rows),
                    "candidates_total": candidates,
                    "candidates_before_threshold_ready": sum(row["candidates_before_threshold_ready"] for row in rows),
                    "candidates_after_threshold_ready": sum(row["candidates_after_threshold_ready"] for row in rows),
                    "early_abandoned_candidates": sum(row["early_abandoned_candidates"] for row in rows),
                    "candidate_early_abandon_ratio": sum(row["early_abandoned_candidates"] for row in rows) / candidates,
                    "full_evaluation_candidates": sum(row["full_evaluation_candidates"] for row in rows),
                    "total_possible_dimensions": possible,
                    "evaluated_dimensions": evaluated,
                    "avoided_dimensions": avoided,
                    "dimension_avoid_ratio": avoided / possible,
                    "mean_dims_per_candidate": evaluated / candidates,
                    "p50_dims_evaluated": exact_quantile(histogram, 0.50),
                    "p75_dims_evaluated": exact_quantile(histogram, 0.75),
                    "p90_dims_evaluated": exact_quantile(histogram, 0.90),
                    "p95_dims_evaluated": exact_quantile(histogram, 0.95),
                    "threshold_checks": checks,
                    "threshold_checks_per_candidate": checks / candidates,
                    "first_half_dimension_avoid_ratio": sum(row["first_half_avoided_dimensions"] for row in rows) / first_possible,
                    "second_half_dimension_avoid_ratio": sum(row["second_half_avoided_dimensions"] for row in rows) / second_possible,
                    "shadow_mismatch_count": sum(row["shadow_mismatch"] for row in rows),
                }
                for index, name in enumerate(BUCKET_NAMES):
                    result[f"abort_interval_{name}"] = sum(json.loads(row["abort_interval_histogram"])[index] for row in rows)
                probe_rows.append(result)

    block_rows = []
    for capacity in CAPACITIES:
        for block in BLOCKS:
            rows = [row for row in probe_rows if row["threshold_capacity"] == capacity and row["block_size"] == block]
            possible = sum(row["total_possible_dimensions"] for row in rows)
            candidates = sum(row["candidates_total"] for row in rows)
            block_rows.append({
                "potential": "potential_k40" if capacity == 40 else "potential_k10",
                "production_gate_eligible": int(capacity == 40),
                "threshold_capacity": capacity,
                "block_size": block,
                "probes": "16;32;64;128;256",
                "candidates_total": candidates,
                "dimension_avoid_ratio": sum(row["avoided_dimensions"] for row in rows) / possible,
                "mean_dims_per_candidate": sum(row["evaluated_dimensions"] for row in rows) / candidates,
                "threshold_checks_per_candidate": sum(row["threshold_checks"] for row in rows) / candidates,
                "high_probe_min_dimension_avoid_ratio": min(row["dimension_avoid_ratio"] for row in rows if row["probes"] in (64, 128, 256)),
                "high_probe_max_dimension_avoid_ratio": max(row["dimension_avoid_ratio"] for row in rows if row["probes"] in (64, 128, 256)),
                "shadow_mismatch_count": sum(row["shadow_mismatch_count"] for row in rows),
            })

    decision, per_block = gate(probe_rows)
    production_blocks = [row for row in block_rows if row["threshold_capacity"] == 40]
    # Prefer the smallest-check Pareto points within 3 percentage points of the
    # best saved-dimension ratio; report at most two, without a latency claim.
    best = max(row["dimension_avoid_ratio"] for row in production_blocks)
    contenders = [row for row in production_blocks if row["dimension_avoid_ratio"] >= best - 0.03]
    contenders.sort(key=lambda row: (row["threshold_checks_per_candidate"], -row["dimension_avoid_ratio"]))
    recommended = [row["block_size"] for row in contenders[:2]]
    if len(recommended) == 1:
        alternatives = sorted(production_blocks, key=lambda row: -row["dimension_avoid_ratio"])
        for row in alternatives:
            if row["block_size"] not in recommended:
                recommended.append(row["block_size"])
                break

    csvwrite(root / "phase_d1_1_query_summary.csv", query_rows)
    csvwrite(root / "phase_d1_1_probe_summary.csv", probe_rows)
    csvwrite(root / "phase_d1_1_block_summary.csv", block_rows)
    csvwrite(root / "phase_d1_1_correctness.csv", correctness)
    analysis = {
        "phase": "D1-1",
        "status": "COMPLETE",
        "decision": decision,
        "production_gate_basis": "potential_k40 only",
        "per_block_production_decision": {str(key): value for key, value in per_block.items()},
        "recommended_block_sizes_for_d1_2": recommended,
        "recommendation_is_not_a_latency_claim": True,
        "rounds": [1],
        "measured_query_executions": 500,
        "warmup_query_executions": 100,
        "shadow_mismatch_count": sum(row["shadow_mismatch"] for row in correctness),
        "real_sql_vs_shadow_full_mismatch_count": sum(row["real_sql_vs_shadow_full_mismatch"] for row in correctness),
        "frozen_phase_c_mismatch_count": sum(row["frozen_phase_c_id_mismatch"] for row in correctness),
        "potential_k10": [row for row in probe_rows if row["threshold_capacity"] == 10],
        "potential_k40": [row for row in probe_rows if row["threshold_capacity"] == 40],
        "diagnostic_notes": {
            "threshold_ready": "Heap readiness is sampled before each candidate; first ready ordinal is K+1 while candidates_before_threshold_ready equals K.",
            "abort_intervals": BUCKET_NAMES,
            "list_order": "first/second halves use the actual selected-list order captured from the real scan.",
            "trajectory": "Per-query fixed ordinal threshold snapshots are retained in query_summary; null means heap not ready.",
            "numerics": "Decision uses a conservative GIST1M-only partial lower key, and every checkpoint is verified <= the already known unchanged full support-proc score.",
        },
        "gate_rule": "For each K40 block: >=40% at at least two of probes 64/128/256 => Strong Go; >=25% at at least two => Go; all high probes <10% => No-Go; otherwise Marginal.",
        "instrumentation_latency_is_optimization_latency": False,
        "real_execution_skipped_dimensions": 0,
        "production_distance_path_modified": False,
        "d1_2_started": False,
        "protection_after": protect(),
    }
    save(root / "phase_d1_1_analysis.json", analysis)
    return analysis


def main():
    assert len(sys.argv) == 2, "pass exactly one new run directory"
    root = Path(sys.argv[1])
    assert root.parent == RUNS and root.name.startswith("phase_d1_1_l2_") and not root.exists()
    assert not Path("/workspace/benchmark/phase_d1_1_l2_once.json").exists(), "D1-1 round already attempted"
    before = protect()
    prior = json.loads((OLD / "phase_c_artifacts/phase_c_manifest.json").read_text())
    assert sha(DATASET) == prior["dataset_sha256"]
    root.mkdir(mode=0o777)
    os.chmod(root, 0o777)
    for name in ["observer.c", "run_validation.py", "build_command.json", "build.log"]:
        shutil.copy2(SRC / name, root / name)
    shutil.copy2(OBSERVER, root / "d1_1_observer.so")

    permutation = list(range(1000))
    random.Random(SEED).shuffle(permutation)
    query_ids = permutation[:100]
    permutation_hash = hashlib.sha256(json.dumps(query_ids, separators=(",", ":")).encode()).hexdigest()
    save(root / "query_permutation.json", {
        "seed": SEED,
        "method": "Python random.Random(seed).shuffle(range(1000)); first 100; same order for all probes",
        "query_ids": query_ids,
        "warmup_query_ids": query_ids[:20],
        "sha256": permutation_hash,
    })
    (root / "query_ids.txt").write_text("".join(f"{query_id}\n" for query_id in query_ids))

    with h5py.File(DATASET, "r") as dataset:
        queries = {query_id: np.asarray(dataset["test"][query_id], dtype=np.float32) for query_id in query_ids}
    literals = {query_id: "[" + ",".join(str(float(value)) for value in vector) + "]" for query_id, vector in queries.items()}
    frozen = {}
    with (OLD / "phase_c_artifacts/phase_c_raw.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["probes"]), int(row["query_id"]))
            if row["system"] == "phase2a" and int(row["round"]) == 1 and key[0] in PROBES and key[1] in queries:
                frozen[key] = [int(value) for value in row["result_ids"].split(";")]
    assert len(frozen) == 500

    raw_path = root / "phase_d1_1_raw.csv"
    with raw_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(raw_header())
    os.chmod(raw_path, 0o666)

    manifest = {
        "phase": "D1-1",
        "name": "L2 Progressive Distance Shadow Potential Study",
        "status": "PREPARED",
        "dataset": "GIST1M",
        "dimension": DIM,
        "metric": "L2 squared internal score",
        "lists": 1000,
        "topk": 10,
        "rounds": 1,
        "queries": 100,
        "warmup_per_probes": 20,
        "probes": PROBES,
        "block_sizes": BLOCKS,
        "production_safe_threshold_capacity": 40,
        "ideal_threshold_capacity": 10,
        "total_measured_query_executions": 500,
        "total_warmup_query_executions": 100,
        "query_permutation_sha256": permutation_hash,
        "dataset_sha256": prior["dataset_sha256"],
        "production_binary": str(PRODUCTION),
        "production_sha256": BINARY,
        "production_source_commit": prior["optimized_commit"],
        "repository_head": git("rev-parse", "HEAD"),
        "observer_sha256": sha(OBSERVER),
        "observer_source_sha256": sha(SRC / "observer.c"),
        "runner_sha256": sha(SRC / "run_validation.py"),
        "raw_encoding": "one physical candidate per CSV row; eight prefixed field groups encode K10/K40 x block32/64/128/256 without duplicating candidates",
        "raw_columns": raw_header(),
        "shadow_method": "actual SQL always computes full distance; read-only replay first calls unchanged support proc, then simulates original-dimension-order block accumulation",
        "partial_decision": "strict certified_lower_key > tau_before_candidate; raw partial also recorded; full known score never sets tau before its candidate",
        "instrumentation_latency_claims": False,
        "second_round_allowed": False,
        "d1_2_allowed": False,
        "protection_before": before,
    }
    save(root / "phase_d1_1_manifest.json", manifest)
    (root / "environment.txt").write_text(
        subprocess.check_output(["uname", "-a"], text=True)
        + subprocess.check_output(["gcc", "--version"], text=True)
        + git("status", "--short", "--untracked-files=all") + "\n"
    )
    (root / "pid").write_text(f"{os.getpid()}\n")

    connection = psycopg2.connect(
        host="127.0.0.1", port=5432, dbname="taskdb", user="dev",
        application_name="phase_d1_1_l2_single_round",
    )
    connection.autocommit = True
    query_rows = []
    correctness = []
    measured = 0
    warmups = 0
    try:
        with connection.cursor() as cursor:
            cursor.execute("LOAD 'vector'")
            cursor.execute("CREATE FUNCTION pg_temp.d1_selftest() RETURNS text AS %s,'d1_selftest' LANGUAGE C STRICT", (str(OBSERVER),))
            cursor.execute("CREATE FUNCTION pg_temp.d1_observe(regclass,vector,int,text,int) RETURNS text AS %s,'d1_observe' LANGUAGE C STRICT", (str(OBSERVER),))
            cursor.execute("SELECT pg_temp.d1_selftest()")
            save(root / "selftest.json", {"result": cursor.fetchone()[0]})
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("SET LOCAL lock_timeout=10000")
            cursor.execute("LOCK TABLE gist_base IN SHARE MODE")
            manifest["index"] = index_identity(cursor)
            assert manifest["index"] == prior["index"]
            cursor.execute("SELECT pg_backend_pid(),version()")
            manifest["backend_pid"], manifest["postgres_version"] = cursor.fetchone()
            settings = {
                "ivfflat.iterative_scan": "off",
                "ivfflat.distance_path": "generic",
                "ivfflat.bounded_scan": "on",
                "ivfflat.experimental_sort_bound": "0",
                "ivfflat.bound_overfetch": "4",
                "ivfflat.bound_min": "40",
                "ivfflat.bound_fastpath_limit": "100",
            }
            for name, value in settings.items():
                cursor.execute("SELECT set_config(%s,%s,false)", (name, value))
                assert cursor.fetchone()[0] == value
            for statement in [
                "SET LOCAL enable_seqscan=off", "SET LOCAL enable_indexscan=on",
                "SET LOCAL max_parallel_workers_per_gather=0", "SET LOCAL statement_timeout=0",
            ]:
                cursor.execute(statement)
            manifest.update(settings=settings, status="RUNNING")
            save(root / "phase_d1_1_manifest.json", manifest)
            once = Path("/workspace/benchmark/phase_d1_1_l2_once.json")
            with once.open("x") as handle:
                json.dump({"run": str(root), "pid": os.getpid(), "rounds": 1, "measured_limit": 500, "status": "RUNNING"}, handle)
            active = Path("/workspace/benchmark/phase_d1_1_l2_active.json")
            save(active, {"run": str(root), "pid": os.getpid(), "status": "RUNNING"})

            sql = "SELECT id,ctid::text FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10"
            last_serial = 0
            for probes in PROBES:
                cursor.execute("SELECT set_config('ivfflat.probes',%s,false)", (str(probes),))
                assert cursor.fetchone()[0] == str(probes)
                cursor.execute("EXPLAIN (FORMAT JSON) " + sql, (literals[query_ids[0]],))
                plan = cursor.fetchone()[0]
                assert "gist_ivf_l2" in json.dumps(plan) and "Index Scan" in json.dumps(plan)
                save(root / f"plan_p{probes}.json", plan)
                for position, query_id in enumerate(query_ids[:20], 1):
                    cursor.execute(sql, (literals[query_id],))
                    cursor.fetchall()
                    warmups += 1
                    if position % 10 == 0:
                        print(f"probes={probes} warmup={position}/20", flush=True)
                for position, query_id in enumerate(query_ids, 1):
                    cursor.execute(sql, (literals[query_id],))
                    actual = cursor.fetchall()
                    measured += 1
                    cursor.execute(
                        "SELECT pg_temp.d1_observe('gist_ivf_l2'::regclass,%s::vector,%s,%s,%s)",
                        (literals[query_id], probes, str(raw_path), query_id),
                    )
                    observed = json.loads(cursor.fetchone()[0])
                    assert observed["scan_serial"] > last_serial
                    last_serial = observed["scan_serial"]
                    real_ids = [row[0] for row in actual]
                    real_tids = [row[1] for row in actual]
                    full = observed["shadow_full_top10"]
                    full_tids = [entry["tid"] for entry in full]
                    mapping = {tid: identifier for identifier, tid in actual}
                    full_ids = [mapping.get(tid) for tid in full_tids]
                    real_mismatch = int(real_tids != full_tids or real_ids != full_ids)
                    frozen_mismatch = int(real_ids != frozen[(probes, query_id)])
                    for config in observed["configs"]:
                        progressive = config["shadow_progressive_top10"]
                        progressive_tids = [entry["tid"] for entry in progressive]
                        distance_mismatch = int(
                            [struct.pack("!d", entry["distance_squared"]) for entry in full]
                            != [struct.pack("!d", entry["distance_squared"]) for entry in progressive]
                        )
                        mismatch = int(config["shadow_mismatch"] or full_tids != progressive_tids or distance_mismatch)
                        check = {
                            "round": 1, "probes": probes, "query_order_position": position,
                            "query_id": query_id, "potential": "potential_k40" if config["capacity"] == 40 else "potential_k10",
                            "threshold_capacity": config["capacity"], "block_size": config["block_size"],
                            "ordered_tid_mismatch": int(full_tids != progressive_tids),
                            "distance_value_mismatch": distance_mismatch,
                            "real_sql_vs_shadow_full_mismatch": real_mismatch,
                            "frozen_phase_c_id_mismatch": frozen_mismatch,
                            "shadow_mismatch": mismatch,
                        }
                        correctness.append(check)
                        if mismatch or real_mismatch or frozen_mismatch:
                            save(root / "correctness_failure.json", {"record": check, "actual": actual, "observer": observed})
                            raise RuntimeError("D1-1 FAIL: ordered Top10 mismatch; stop immediately")
                        candidates = config["candidates_total"]
                        possible = candidates * DIM
                        query_rows.append({
                            "round": 1, "probes": probes, "query_order_position": position, "query_id": query_id,
                            "potential": check["potential"], "production_gate_eligible": int(config["capacity"] == 40),
                            "threshold_capacity": config["capacity"], "block_size": config["block_size"],
                            "selected_lists": observed["selected_lists"], "candidates_total": candidates,
                            "candidates_before_threshold_ready": config["candidates_before_threshold_ready"],
                            "candidates_after_threshold_ready": config["candidates_after_threshold_ready"],
                            "threshold_ready_ordinal": config["threshold_ready_ordinal"],
                            "early_abandoned_candidates": config["early_abandoned_candidates"],
                            "candidate_early_abandon_ratio": config["early_abandoned_candidates"] / candidates,
                            "full_evaluation_candidates": config["full_evaluation_candidates"],
                            "total_possible_dimensions": possible,
                            "evaluated_dimensions": config["evaluated_dimensions"],
                            "avoided_dimensions": config["avoided_dimensions"],
                            "dimension_avoid_ratio": config["avoided_dimensions"] / possible,
                            "mean_dims_per_candidate": config["evaluated_dimensions"] / candidates,
                            "p50_dims_evaluated": config["p50_dims_evaluated"],
                            "p75_dims_evaluated": config["p75_dims_evaluated"],
                            "p90_dims_evaluated": config["p90_dims_evaluated"],
                            "p95_dims_evaluated": config["p95_dims_evaluated"],
                            "threshold_checks": config["threshold_checks"],
                            "threshold_checks_per_candidate": config["threshold_checks"] / candidates,
                            "first_half_candidates": config["first_half_total"],
                            "first_half_early_abandoned": config["first_half_early"],
                            "first_half_avoided_dimensions": config["first_half_avoided_dimensions"],
                            "second_half_candidates": config["second_half_total"],
                            "second_half_early_abandoned": config["second_half_early"],
                            "second_half_avoided_dimensions": config["second_half_avoided_dimensions"],
                            "abort_interval_histogram": json.dumps(config["abort_hist"], separators=(",", ":")),
                            "evaluated_dims_histogram": json.dumps(config["eval_hist"], separators=(",", ":")),
                            "threshold_trajectory": json.dumps(config["trajectory"], separators=(",", ":")),
                            "final_threshold_squared": config["final_threshold"],
                            "actual_ids": json.dumps(real_ids, separators=(",", ":")),
                            "actual_tids": json.dumps(real_tids, separators=(",", ":")),
                            "shadow_full_top10": json.dumps(full, separators=(",", ":")),
                            "shadow_progressive_top10": json.dumps(progressive, separators=(",", ":")),
                            "shadow_mismatch": mismatch,
                        })
                    save(root / "progress.json", {
                        "status": "RUNNING", "round": 1, "measured_query_executions": measured,
                        "validated_config_executions": len(query_rows), "expected_measured_query_executions": 500,
                        "warmup_query_executions": warmups, "last_probes": probes, "last_query_id": query_id,
                        "raw_bytes": raw_path.stat().st_size,
                    })
                    if position % 10 == 0:
                        print(f"probes={probes} measured={position}/100 total={measured}/500 mismatches=0 raw_gib={raw_path.stat().st_size / 2**30:.2f}", flush=True)

            assert measured == 500 and warmups == 100 and len(query_rows) == 4000 and len(correctness) == 4000
            assert len({(row["round"], row["probes"], row["query_id"]) for row in correctness}) == 500
            assert index_identity(cursor) == manifest["index"]
            cursor.execute("ROLLBACK")
        connection.close()

        analysis = analyze(root, query_rows, correctness)
        manifest.update(
            status="COMPLETE", completed_rounds=[1], measured_query_executions=500,
            warmup_query_executions=100, shadow_config_executions=4000,
            shadow_mismatch_count=0, decision=analysis["decision"],
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            raw_rows=sum(row["candidates_total"] for row in query_rows if row["threshold_capacity"] == 40 and row["block_size"] == 32),
            raw_bytes=raw_path.stat().st_size,
        )
        save(root / "phase_d1_1_manifest.json", manifest)
        save(root / "progress.json", {
            "status": "COMPLETE", "completed_rounds": [1], "measured_query_executions": 500,
            "validated_config_executions": 4000, "warmup_query_executions": 100,
            "shadow_mismatch_count": 0, "decision": analysis["decision"], "raw_bytes": raw_path.stat().st_size,
        })
        save(Path("/workspace/benchmark/phase_d1_1_l2_active.json"), {"run": str(root), "pid": os.getpid(), "status": "COMPLETE"})
        save(Path("/workspace/benchmark/phase_d1_1_l2_once.json"), {"run": str(root), "pid": os.getpid(), "rounds": 1, "measured_limit": 500, "status": "COMPLETE"})
        save(root / "artifact_hashes.json", {
            path.name: sha(path) for path in root.iterdir()
            if path.is_file() and path.name not in {"artifact_hashes.json", "experiment.log"}
        })
        print(json.dumps({"status": "COMPLETE", "run": str(root), "decision": analysis["decision"], "measured": 500, "mismatches": 0}), flush=True)
    except BaseException as error:
        connection.close()
        manifest.update(status="FAIL", measured_query_executions=measured, warmup_query_executions=warmups, error=str(error))
        save(root / "phase_d1_1_manifest.json", manifest)
        save(root / "failure.json", {"error": str(error), "traceback": traceback.format_exc(), "no_automatic_retry": True, "measured_query_executions": measured})
        save(Path("/workspace/benchmark/phase_d1_1_l2_active.json"), {"run": str(root), "pid": os.getpid(), "status": "FAIL", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
