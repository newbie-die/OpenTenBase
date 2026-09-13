#!/usr/bin/env python3
"""Run reproducible IVFFlat profiling experiments, including Phase 2A."""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import struct
import subprocess
import shutil
import fcntl
import math
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent.parent
ROOT = Path(
    os.environ.get("BENCHMARK_RUNTIME_ROOT", WORKSPACE_ROOT / "benchmark")
).resolve()
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE\s+(.*)")
PROFILE_2B_RE = re.compile(r"IVFFLAT_PROFILE_2B\s+(.*)")
FIELD_RE = re.compile(r"([a-z0-9_]+)=(-?[0-9.]+)")
PROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FORMAL_DATASET_LABELS = {
    "glove-cosine": "glove100",
    "gist-l2": "gist1m",
}
CONFIGS = {
    "glove-l2": {
        "dataset": ROOT / "data/glove100/glove-100-angular.hdf5",
        "table": "glove_base", "operator": "<->", "opclass": "vector_l2_ops",
        "index": "glove_ivf_l2", "dimension": 100, "normalize": False,
        "rows": 1183514, "metric": "l2",
    },
    "glove-cosine": {
        "dataset": ROOT / "data/glove100/glove-100-angular.hdf5",
        "table": "glove_base", "operator": "<=>", "opclass": "vector_cosine_ops",
        "index": "glove_ivf_cosine", "dimension": 100, "normalize": False,
        "rows": 1183514, "metric": "cosine",
    },
    "glove-ip": {
        "dataset": ROOT / "data/glove100/glove-100-angular.hdf5",
        "table": "glove_ip_base", "operator": "<#>", "opclass": "vector_ip_ops",
        "index": "glove_ivf_ip", "dimension": 100, "normalize": True,
        "rows": 1183514, "metric": "ip",
    },
    "sift-l2": {
        "dataset": ROOT / "data/sift1m/sift-128-euclidean.hdf5",
        "table": "sift_base", "operator": "<->", "opclass": "vector_l2_ops",
        "index": "sift_ivf_l2", "dimension": 128, "normalize": False,
        "rows": 1000000, "metric": "l2",
    },
    "gist-l2": {
        "dataset": ROOT / "data/gist1m/gist-960-euclidean.hdf5",
        "table": "gist_base", "operator": "<->", "opclass": "vector_l2_ops",
        "index": "gist_ivf_l2", "dimension": 960, "normalize": False,
        "rows": 1000000, "metric": "l2",
    },
}


def connect(args):
    import psycopg2
    return psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname, user=args.user)


def vector_literal(vector):
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def load_workload(config, count, topk, ground_truth=False):
    import h5py
    import numpy as np
    with h5py.File(config["dataset"], "r") as dataset:
        queries = np.asarray(dataset["test"][:count], dtype=np.float32)
        neighbors = np.asarray(dataset["neighbors"][:count, :topk]) if ground_truth else None
    if config["normalize"]:
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        queries = queries / norms
    return queries, neighbors


def parse_profile(notices):
    for notice in reversed(notices):
        match = PROFILE_RE.search(notice)
        if match:
            values = {key: float(value) for key, value in FIELD_RE.findall(match.group(1))}
            for key in ("probes", "dimensions", "sort_bound", "logical_bound", "physical_bound", "bounded_active", "bounded_exhausted", "fallback_triggered", "returned_from_bounded", "returned_after_fallback", "candidates", "pages", "getitems_calls"):
                if key in values:
                    values[key] = int(values[key])
            return values
    raise RuntimeError("No IVFFLAT_PROFILE notice; build with IVFFLAT_PROFILE_CFLAGS=-DIVFFLAT_BENCH")


def parse_profile_2b(notices):
    for notice in reversed(notices):
        match = PROFILE_2B_RE.search(notice)
        if match:
            return {key: int(float(value)) for key, value in FIELD_RE.findall(match.group(1))}
    raise RuntimeError("No IVFFLAT_PROFILE_2B notice; build with IVFFLAT_PROFILE_2B")


def ensure_index(cur, config):
    cur.execute("SELECT to_regclass(%s)", (config["index"],))
    if cur.fetchone()[0] is None:
        raise RuntimeError(f"missing index: {config['index']}")


def execute_query(conn, cur, config, query, topk, capture):
    conn.notices.clear()
    sql = f"SELECT id FROM {config['table']} ORDER BY embedding {config['operator']} %s::vector LIMIT %s"
    started = time.perf_counter_ns()
    cur.execute(sql, (vector_literal(query), topk))
    ids = [row[0] for row in cur.fetchall()]
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return ids, latency_ms, parse_profile(conn.notices) if capture else None


def execute_filtered_query(conn, cur, config, query, topk, divisor, capture):
    conn.notices.clear()
    sql = (f"SELECT id FROM {config['table']} WHERE id %% %s = 0 "
           f"ORDER BY embedding {config['operator']} %s::vector LIMIT %s")
    started = time.perf_counter_ns()
    cur.execute(sql, (divisor, vector_literal(query), topk))
    ids = [row[0] for row in cur.fetchall()]
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return ids, latency_ms, parse_profile(conn.notices) if capture else None


def percentile(values, q):
    import numpy as np
    return float(np.percentile(np.asarray(values), q))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_csv_atomic(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def summarize_profile(rows, experiment, name, config, probes):
    result = {
        "experiment": experiment, "dataset": name, "dimension": config["dimension"],
        "metric": config["operator"], "probes": probes, "queries": len(rows),
    }
    numeric = [key for key in rows[0] if key != "query_id"]
    for key in numeric:
        result[f"mean_{key}"] = statistics.fmean(float(row[key]) for row in rows)
    scan_us = result["mean_scan_us"]
    result["candidate_pct"] = 100 * result["mean_candidate_us"] / scan_us if scan_us else 0
    result["distance_pct"] = 100 * result["mean_distance_us"] / scan_us if scan_us else 0
    result["sort_pct"] = 100 * result["mean_sort_us"] / scan_us if scan_us else 0
    return result


def run_group(conn, args, experiment, name, probes, raw_rows, summaries):
    config = CONFIGS[name]
    queries, _ = load_workload(config, args.warmup + args.queries, args.topk)
    with conn.cursor() as cur:
        ensure_index(cur, config)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET ivfflat.iterative_scan = off")
        cur.execute("SELECT set_config('ivfflat.experimental_sort_bound', '0', false)")
        cur.execute("SELECT set_config('ivfflat.probes', %s, false)", (str(probes),))
        for query in queries[:args.warmup]:
            execute_query(conn, cur, config, query, args.topk, False)
        measured = []
        for query_id, query in enumerate(queries):
            _, latency_ms, profile = execute_query(conn, cur, config, query, args.topk, True)
            profile.update(query_id=query_id, latency_us=latency_ms * 1000)
            measured.append(profile)
            raw_rows.append({"experiment": experiment, "dataset": name,
                             "dimension": config["dimension"], "metric": config["operator"],
                             "probes": probes, **profile})
    summaries.append(summarize_profile(measured, experiment, name, config, probes))


def parse_int_list(value):
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def phase2a_config(conn, args, config, queries, neighbors, probes, bound, round_no):
    profiles, latencies, recalls, returned = [], [], [], []
    with conn.cursor() as cur:
        ensure_index(cur, config)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET ivfflat.iterative_scan = off")
        cur.execute("SELECT set_config('ivfflat.probes', %s, false)", (str(probes),))
        cur.execute("SELECT set_config('ivfflat.experimental_sort_bound', %s, false)", (str(bound),))
        for query in queries[:args.warmup]:
            execute_query(conn, cur, config, query, args.topk, False)
        started = time.perf_counter()
        for query_id, query in enumerate(queries):
            ids, latency_ms, profile = execute_query(conn, cur, config, query, args.topk, True)
            gt = set(int(value) for value in neighbors[query_id])
            recalls.append(len(gt.intersection(ids)) / args.topk)
            returned.append(len(ids))
            latencies.append(latency_ms)
            profiles.append(profile)
        elapsed = time.perf_counter() - started

    def avg(name):
        return statistics.fmean(float(row[name]) for row in profiles)

    row = {
        "dataset": "glove100", "metric": "l2", "rows": config["rows"],
        "dimension": config["dimension"], "lists": args.lists, "probes": probes,
        "sort_bound": bound, "round": round_no, "queries": args.queries, "topk": args.topk,
        "recall_at_10": statistics.fmean(recalls), "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95), "p99_ms": percentile(latencies, 99),
        "mean_ms": statistics.fmean(latencies), "qps": args.queries / elapsed,
        "avg_returned_rows": statistics.fmean(returned), "min_returned_rows": min(returned),
        "status": "correct" if min(returned) == args.topk else "unsafe/incorrect",
        "avg_candidates": avg("candidates"), "avg_getitems_us": avg("getitems_us"),
        "avg_candidate_us": avg("candidate_us"), "avg_distance_us": avg("distance_us"),
        "avg_sort_us": avg("sort_us"),
    }
    row["candidate_ratio"] = row["avg_candidate_us"] / row["avg_getitems_us"]
    row["distance_ratio"] = row["avg_distance_us"] / row["avg_getitems_us"]
    row["sort_ratio"] = row["avg_sort_us"] / row["avg_getitems_us"]
    return row


def summarize_phase2a(rows, topk):
    output = []
    for probes in sorted({row["probes"] for row in rows}):
        probe_rows = [row for row in rows if row["probes"] == probes]
        baseline_rows = [row for row in probe_rows if row["sort_bound"] == 0]
        baseline_p95 = statistics.median(row["p95_ms"] for row in baseline_rows)
        baseline_qps = statistics.median(row["qps"] for row in baseline_rows)
        for bound in sorted({row["sort_bound"] for row in probe_rows}):
            group = [row for row in probe_rows if row["sort_bound"] == bound]
            p95 = statistics.median(row["p95_ms"] for row in group)
            qps = statistics.median(row["qps"] for row in group)
            output.append({
                "probes": probes, "sort_bound": bound,
                "median_recall_at_10": statistics.median(row["recall_at_10"] for row in group),
                "median_p50_ms": statistics.median(row["p50_ms"] for row in group),
                "median_p95_ms": p95,
                "median_p99_ms": statistics.median(row["p99_ms"] for row in group),
                "median_qps": qps, "baseline_p95_ms": baseline_p95,
                "p95_improvement_pct": 100 * (baseline_p95 - p95) / baseline_p95,
                "baseline_qps": baseline_qps,
                "qps_improvement_pct": 100 * (qps - baseline_qps) / baseline_qps,
                "min_returned_rows": min(row["min_returned_rows"] for row in group),
                "status": "correct" if min(row["min_returned_rows"] for row in group) == topk else "unsafe/incorrect",
            })
    return output


def phase2a2_config(conn, args, config, queries, neighbors, probes, mode):
    profiles, latencies, recalls, returned, result_ids = [], [], [], [], []
    bounded_scan, oracle_bound = {
        "full": (False, 0),
        "oracle40": (False, 40),
        "auto": (True, 0),
    }[mode]
    with conn.cursor() as cur:
        ensure_index(cur, config)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET ivfflat.iterative_scan = off")
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.probes", str(probes)))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.experimental_sort_bound", str(oracle_bound)))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bounded_scan", "on" if bounded_scan else "off"))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_overfetch", "4"))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_min", "40"))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_fastpath_limit", "100"))
        for query in queries[:args.warmup]:
            execute_query(conn, cur, config, query, args.topk, False)
        started = time.perf_counter()
        for query_id, query in enumerate(queries):
            ids, latency_ms, profile = execute_query(conn, cur, config, query, args.topk, True)
            gt = set(int(value) for value in neighbors[query_id])
            recalls.append(len(gt.intersection(ids)) / args.topk)
            returned.append(len(ids))
            latencies.append(latency_ms)
            profiles.append(profile)
            result_ids.append(ids)
        elapsed = time.perf_counter() - started

    def avg(name):
        return statistics.fmean(float(row[name]) for row in profiles)

    row = {
        "mode": mode, "dataset": "glove100", "metric": "cosine",
        "operator": config["operator"], "rows": config["rows"],
        "dimension": config["dimension"], "lists": args.lists, "probes": probes,
        "sort_bound": oracle_bound, "bounded_scan": bounded_scan,
        "bound_overfetch": 4, "bound_min": 40, "bound_fastpath_limit": 100,
        "queries": args.queries, "topk": args.topk,
        "recall_at_10": statistics.fmean(recalls), "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95), "p99_ms": percentile(latencies, 99),
        "mean_ms": statistics.fmean(latencies), "qps": args.queries / elapsed,
        "avg_returned_rows": statistics.fmean(returned), "min_returned_rows": min(returned),
        "avg_logical_bound": avg("logical_bound"),
        "avg_physical_bound": avg("physical_bound"),
        "avg_bounded_active": avg("bounded_active"),
        "avg_candidates": avg("candidates"), "avg_getitems_us": avg("getitems_us"),
        "avg_candidate_us": avg("candidate_us"), "avg_distance_us": avg("distance_us"),
        "avg_sort_us": avg("sort_us"),
    }
    row["candidate_ratio"] = row["avg_candidate_us"] / row["avg_getitems_us"]
    row["distance_ratio"] = row["avg_distance_us"] / row["avg_getitems_us"]
    row["sort_ratio"] = row["avg_sort_us"] / row["avg_getitems_us"]
    return row, result_ids


def summarize_phase2a2(rows):
    output = []
    for probes in sorted({row["probes"] for row in rows}):
        probe_rows = [row for row in rows if row["probes"] == probes]
        baseline = next(row for row in probe_rows if row["mode"] == "full")
        for row in probe_rows:
            output.append({
                "probes": probes, "mode": row["mode"],
                "recall_at_10": row["recall_at_10"],
                "top10_match_full": row["top10_match_full"],
                "avg_logical_bound": row["avg_logical_bound"],
                "avg_physical_bound": row["avg_physical_bound"],
                "avg_bounded_active": row["avg_bounded_active"],
                "avg_candidates": row["avg_candidates"],
                "p50_ms": row["p50_ms"], "p95_ms": row["p95_ms"],
                "p99_ms": row["p99_ms"], "qps": row["qps"],
                "full_p95_ms": baseline["p95_ms"],
                "p95_improvement_pct": 100 * (baseline["p95_ms"] - row["p95_ms"]) / baseline["p95_ms"],
                "full_qps": baseline["qps"],
                "qps_improvement_pct": 100 * (row["qps"] - baseline["qps"]) / baseline["qps"],
                "min_returned_rows": row["min_returned_rows"],
                "status": row["status"],
            })
    return output


def phase2a34_config(conn, args, config, queries, probes, divisor, mode):
    bounded_scan = mode == "auto"
    rows = []
    with conn.cursor() as cur:
        ensure_index(cur, config)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET ivfflat.iterative_scan = off")
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.probes", str(probes)))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.experimental_sort_bound", "0"))
        cur.execute("SELECT set_config(%s, %s, false)",
                    ("ivfflat.bounded_scan", "on" if bounded_scan else "off"))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_overfetch", "4"))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_min", "40"))
        cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_fastpath_limit", "100"))
        for query in queries[:args.warmup]:
            execute_filtered_query(conn, cur, config, query, args.topk, divisor, False)
        for query_id, query in enumerate(queries):
            ids, latency_ms, profile = execute_filtered_query(
                conn, cur, config, query, args.topk, divisor, True)
            rows.append({
                "query_id": query_id, "mode": mode, "probes": probes,
                "filter_divisor": divisor, "selectivity_pct": 100.0 / divisor,
                "returned_rows": len(ids), "distinct_rows": len(set(ids)),
                "latency_ms": latency_ms, "result_ids": ";".join(str(value) for value in ids),
                **profile,
            })
    return rows


def summarize_phase2a34(rows):
    output = []
    keys = sorted({(row["filter_divisor"], row["mode"]) for row in rows})
    for divisor, mode in keys:
        group = [row for row in rows
                 if row["filter_divisor"] == divisor and row["mode"] == mode]
        output.append({
            "mode": mode, "probes": group[0]["probes"],
            "filter_divisor": divisor, "selectivity_pct": group[0]["selectivity_pct"],
            "queries": len(group),
            "all_results_match_full": all(row["topk_match_full"] for row in group),
            "all_returned_rows_match_full": all(row["returned_rows_match_full"] for row in group),
            "all_returned_tids_unique": all(row["returned_rows"] == row["distinct_rows"] for row in group),
            "fallback_queries": sum(row["fallback_triggered"] for row in group),
            "mean_returned_rows": statistics.fmean(row["returned_rows"] for row in group),
            "mean_returned_from_bounded": statistics.fmean(row["returned_from_bounded"] for row in group),
            "mean_returned_after_fallback": statistics.fmean(row["returned_after_fallback"] for row in group),
            "mean_candidates": statistics.fmean(row["candidates"] for row in group),
            "mean_getitems_calls": statistics.fmean(row["getitems_calls"] for row in group),
            "mean_list_us": statistics.fmean(row["list_us"] for row in group),
            "p95_ms": percentile([row["latency_ms"] for row in group], 95),
            "status": "correct" if all(row["status"] == "correct" for row in group) else "incorrect",
        })
    return output


def formal_query(cur, sql, query_literal, topk):
    started = time.perf_counter_ns()
    cur.execute(sql, (query_literal, topk))
    ids = [int(row[0]) for row in cur.fetchall()]
    return ids, (time.perf_counter_ns() - started) / 1000.0




PHASE2B_WORKLOAD_FIELDS = (
    "selected_lists", "scanned_pages", "scanned_candidates", "distance_calls",
    "tuplesort_input_calls", "physical_bound", "bounded_active",
    "bounded_exhausted", "fallback_triggered", "returned_rows",
    "page_candidates_0", "page_candidates_1", "page_candidates_2",
    "page_candidates_3", "page_candidates_4plus",
)
PHASE2B_RESULT_FIELDS = (
    "result_tids", "result_ids", "result_distances", "result_distance_bits",
)
FUSED2_COUNTERS = (
    "fused_pair_calls", "fused_candidates", "single_tail_candidates",
    "fused_fallback_candidates",
)
PRODUCTION_PATHS = ("direct", "fused2")
PRODUCTION_MODES = ("full", "auto")
PRODUCTION_PROBE_ORDERS = (
    (16, 64, 128),
    (128, 64, 16),
    (64, 16, 128),
    (64, 128, 16),
)
PRODUCTION_RAW_FIELDS = (
    "experiment", "dataset", "dimension", "metric", "mode", "lists",
    "probes", "topk", "round", "config_position", "order_position",
    "distance_path", "query_id", "latency_us", "returned_rows",
    "result_ids", "result_checksum",
)


def require_check(condition, message):
    # These checks must remain enabled under python -O.
    if not condition:
        raise RuntimeError(message)


def configure_phase2b_scan(cur, probes, mode, distance_path):
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET enable_seqscan = off")
    for key, value in {
        "iterative_scan": "off", "probes": str(probes),
        "experimental_sort_bound": "0",
        "bounded_scan": "on" if mode == "auto" else "off",
        "bound_overfetch": "4", "bound_min": "40", "bound_fastpath_limit": "100",
        "distance_path": distance_path,
    }.items():
        cur.execute("SELECT set_config(%s, %s, false)", (f"ivfflat.{key}", value))
    cur.execute("SHOW ivfflat.distance_path")
    require_check(cur.fetchone()[0] == distance_path, "distance_path was not applied")


def execute_correctness_query(conn, cur, config, query_literal, topk,
                              filter_divisor=None):
    conn.notices.clear()
    where = "WHERE id %% %s = 0 " if filter_divisor is not None else ""
    sql = (f"SELECT ctid::text, id, embedding {config['operator']} %s::vector "
           f"FROM {config['table']} {where}"
           f"ORDER BY embedding {config['operator']} %s::vector LIMIT %s")
    params = [query_literal]
    if filter_divisor is not None:
        params.append(filter_divisor)
    params.extend((query_literal, topk))
    cur.execute(sql, params)
    results = [(row[0], int(row[1]), float(row[2])) for row in cur.fetchall()]
    profile = parse_profile_2b(conn.notices)
    profile.update(parse_profile(conn.notices))
    return results, profile


def correctness_row(results, profile, config, mode, distance_path, query_id):
    return {
        "dataset": config["table"], "metric": config["metric"],
        "mode": mode, "distance_path": distance_path, "query_id": query_id,
        "result_tids": ";".join(r[0] for r in results),
        "result_ids": ";".join(str(r[1]) for r in results),
        "result_distances": ";".join(repr(r[2]) for r in results),
        "result_distance_bits": ";".join(struct.pack("!d", r[2]).hex() for r in results),
        **profile,
    }


def phase2b_correctness_config(conn, args, config, query_literals, mode,
                                distance_path, query_count):
    require_check(len(query_literals) >= max(query_count, args.warmup),
                  "insufficient correctness queries")
    rows = []
    with conn.cursor() as cur:
        ensure_index(cur, config)
        configure_phase2b_scan(cur, args.probes_list[0], mode, distance_path)
        for query_literal in query_literals[:args.warmup]:
            execute_correctness_query(conn, cur, config, query_literal, args.topk)
        for query_id, query_literal in enumerate(query_literals[:query_count]):
            results, profile = execute_correctness_query(
                conn, cur, config, query_literal, args.topk)
            rows.append(correctness_row(results, profile, config, mode, distance_path, query_id))
    return rows


def verify_fused2_coverage(rows):
    for row in rows:
        for key in FUSED2_COUNTERS + ("fused2_active", "fused_distance_ns"):
            require_check(key in row, f"missing FUSED2 instrumentation: {key}")
        require_check(row["fused_candidates"] == 2 * row["fused_pair_calls"],
                      "fused_candidates != 2 * fused_pair_calls")
        require_check(row["fused_candidates"] + row["single_tail_candidates"] +
                      row["fused_fallback_candidates"] == row["scanned_candidates"],
                      "FUSED2 logical candidate accounting mismatch")
        require_check(row["direct_distance_calls"] == row["single_tail_candidates"] +
                      row["fused_fallback_candidates"], "FUSED2 DIRECT tail accounting mismatch")
    totals = {key: sum(r[key] for r in rows) for key in
              ("scanned_pages", "scanned_candidates", "direct_distance_calls",
               "page_candidates_0", "page_candidates_1", "page_candidates_2",
               "page_candidates_3", "page_candidates_4plus") + FUSED2_COUNTERS}
    totals["fused_coverage_pct"] = (
        100 * totals["fused_candidates"] / totals["scanned_candidates"]
        if totals["scanned_candidates"] else 0.0)
    return totals


def verify_fused2_tail(rows):
    """For the observed <=2-candidate pages, check the per-query pair/tail counts."""
    for row in rows:
        if row["page_candidates_3"] == row["page_candidates_4plus"] == 0:
            require_check(row["fused_fallback_candidates"] == 0,
                          "unexpected unpairable candidate in GIST workload")
            require_check(row["single_tail_candidates"] == row["page_candidates_1"],
                          "single-candidate pages did not all use DIRECT")
            require_check(row["fused_pair_calls"] == row["page_candidates_2"],
                          "two-candidate page / pair count mismatch")


def validate_phase2b_dispatch(rows, distance_path, eligible=True):
    if distance_path == "fused2" and eligible:
        verify_fused2_coverage(rows)
    for row in rows:
        n = row["scanned_candidates"]
        require_check(n == row["distance_calls"] == row["tuplesort_input_calls"],
                      f"logical distance / tuplesort workload mismatch: {row}")
        active = eligible and distance_path != "generic"
        fused = active and distance_path == "fused2"
        require_check(row["direct_l2_eligible"] == int(eligible) and
                      row["direct_l2_active"] == int(active) and
                      row.get("fused2_active", 0) == int(fused),
                      f"invalid {distance_path} activation: {row}")
        require_check(sum(row[f"page_candidates_{k}"] for k in
                          ("0", "1", "2", "3", "4plus")) == row["scanned_pages"],
                      "page histogram does not sum to scanned_pages")
        require_check(row["distance_ns"] == row["generic_distance_ns"] +
                      row["direct_distance_ns"] + row.get("fused_distance_ns", 0),
                      "distance timer accounting mismatch")
        if not active:
            valid = (row["generic_distance_calls"] == n and
                     row["direct_distance_calls"] == 0 and
                     row["generic_distance_ns"] == row["distance_ns"])
        elif fused:
            valid = row["generic_distance_calls"] == row["generic_distance_ns"] == 0
        else:
            valid = (row["direct_distance_calls"] == n and
                     row["generic_distance_calls"] == 0 and
                     row["direct_distance_ns"] == row["distance_ns"])
        if not fused:
            valid = valid and all(row.get(k, 0) == 0 for k in FUSED2_COUNTERS)
        require_check(valid, f"invalid {distance_path} dispatch: {row}")


def compare_phase2b_paths(baseline_rows, test_rows, mode, baseline_path,
                         test_path, expected_active=1):
    baseline = {r["query_id"]: r for r in baseline_rows}
    tested = {r["query_id"]: r for r in test_rows}
    require_check(len(baseline) == len(baseline_rows) and
                  len(tested) == len(test_rows) and baseline.keys() == tested.keys(),
                  "baseline/test query IDs differ or contain duplicates")
    validate_phase2b_dispatch(baseline_rows, baseline_path, bool(expected_active))
    validate_phase2b_dispatch(test_rows, test_path, bool(expected_active))
    result = {"mode": mode, "baseline_path": baseline_path, "test_path": test_path,
              "queries": len(baseline)}
    for key in PHASE2B_WORKLOAD_FIELDS + PHASE2B_RESULT_FIELDS:
        result[f"{key}_mismatch"] = sum(baseline[q][key] != tested[q][key] for q in baseline)
    for prefix, rows in (("baseline", baseline_rows), ("test", test_rows)):
        for key in ("selected_lists", "scanned_pages", "scanned_candidates", "distance_calls",
                    "tuplesort_input_calls", "returned_rows", "fallback_triggered"):
            result[f"{prefix}_{key}"] = sum(row[key] for row in rows)
    changed = 0
    max_abs = max_rel = 0.0
    for q in baseline:
        a = [float(v) for v in baseline[q]["result_distances"].split(";") if v]
        b = [float(v) for v in tested[q]["result_distances"].split(";") if v]
        changed += abs(len(a) - len(b))
        for x, y in zip(a, b):
            changed += struct.pack("!d", x) != struct.pack("!d", y)
            max_abs = max(max_abs, abs(x - y))
            max_rel = max(max_rel, abs(x - y) / max(abs(x), abs(y)) if x or y else 0.0)
    result.update(differing_distances=changed, max_absolute_difference=max_abs,
                  max_relative_difference=max_rel,
                  changed_topk_orders=result["result_tids_mismatch"],
                  dispatch_counter_mismatch=0, direct_activation_mismatch=0)
    # Preserve existing 2B-1 summary column names for consumers.
    for old, new in {"selected_list": "selected_lists", "candidate": "scanned_candidates",
                     "page": "scanned_pages", "distance_call": "distance_calls",
                     "tuplesort_input": "tuplesort_input_calls", "returned_row": "returned_rows",
                     "tid": "result_tids", "id": "result_ids", "fallback": "fallback_triggered"}.items():
        result[f"{old}_mismatch"] = result[f"{new}_mismatch"]
    return result


def check_phase2b_comparison(summary):
    require_check(not any(v for k, v in summary.items() if k.endswith("_mismatch")) and
                  summary["differing_distances"] == 0,
                  f"{summary['baseline_path']}/{summary['test_path']} correctness failed: {summary}")


def verify_fused2_edges(args, query_literal, output):
    # A separate session contains all temporary objects and edge-specific GUCs.
    conn = connect(args)
    conn.autocommit = True
    reports = []
    try:
        with conn.cursor() as cur:
            cur.execute("LOAD 'vector'")
            def pair(label, config, mode, probes=64, filtered=False):
                rows = []
                for path in ("direct", "fused2"):
                    configure_phase2b_scan(cur, probes, mode, path)
                    results, profile = execute_correctness_query(
                        conn, cur, config, query_literal, 10, 10 if filtered else None)
                    row = correctness_row(results, profile, config, mode, path, 0)
                    require_check(len({r[0] for r in results}) == len(results), "duplicate result TID")
                    rows.append(row)
                summary = compare_phase2b_paths(rows[:1], rows[1:], mode, "direct", "fused2")
                reports.append({"label": label, "baseline": rows[0], "test": rows[1], "summary": summary})
                write_json_atomic(Path(f"{output}_edges.json"), reports)
                check_phase2b_comparison(summary)
                return rows[1]
            full = pair("filtered_full", CONFIGS["gist-l2"], "full", filtered=True)
            auto = pair("forced_fallback", CONFIGS["gist-l2"], "auto", filtered=True)
            require_check(auto["fallback_triggered"] == 1 and
                          auto["returned_from_bounded"] > 0 and auto["returned_after_fallback"] > 0,
                          "edge query did not exercise bounded returns and fallback")
            require_check(all(auto[k] == full[k] for k in PHASE2B_RESULT_FIELDS),
                          "filtered Full/Automatic results differ")
            cur.execute("CREATE TEMP TABLE fused2_edge (id int, embedding vector(960))")
            cur.execute("INSERT INTO fused2_edge SELECT i,array_fill((i::float4 / 10),ARRAY[960])::vector FROM generate_series(1,5) i")
            cur.execute("INSERT INTO fused2_edge VALUES (6,NULL)")
            cur.execute("CREATE INDEX fused2_edge_idx ON fused2_edge USING ivfflat(embedding vector_l2_ops) WITH (lists=1)")
            cur.execute("ANALYZE fused2_edge")
            config = dict(CONFIGS["gist-l2"], table="fused2_edge", index="fused2_edge_idx")
            row = pair("two_pairs_one_tail_null_excluded", config, "full", probes=1)
            require_check(row["fused_pair_calls"] == 2 and row["single_tail_candidates"] == 1 and
                          row["scanned_candidates"] == 5, "pair/tail/NULL edge failed")
            verify_fused2_tail([row])
            cur.execute("DELETE FROM fused2_edge")
            cur.execute("VACUUM fused2_edge")
            row = pair("empty_pages_after_vacuum", config, "full", probes=1)
            require_check(row["page_candidates_0"] > 0 and row["scanned_candidates"] == 0,
                          "vacuum empty-page edge failed")
    finally:
        conn.close()
    return reports


def run_phase2b_correctness(conn, args):
    if len(args.probes_list) != 1 or args.queries <= 0 or args.unsupported_queries <= 0 or args.warmup < 0:
        raise ValueError("correctness requires one probes value, positive query counts and nonnegative warmup")
    if args.baseline_path == args.test_path:
        raise ValueError("baseline-path and test-path must differ")
    if args.check_fused2_edges and "fused2" not in (args.baseline_path, args.test_path):
        raise ValueError("--check-fused2-edges requires a fused2 comparison")
    config = CONFIGS["gist-l2"]
    queries, _ = load_workload(config, max(args.warmup, args.queries), args.topk)
    literals = [vector_literal(q) for q in queries]
    output = Path(args.output or ROOT / "results/phase_2b_correctness")
    manifest = {"baseline_path": args.baseline_path, "test_path": args.test_path,
                "queries": args.queries, "query_sha256": hashlib.sha256("\n".join(literals[:args.queries]).encode()).hexdigest(),
                "probes": args.probes_list[0], "lists": args.lists, "topk": args.topk,
                "dimension": 960, "iterative_scan": "off", "warmup": args.warmup}
    with conn.cursor() as cur:
        cur.execute("LOAD 'vector'")
        cur.execute("SHOW ivfflat.distance_path")
        require_check(cur.fetchone()[0] == "generic", "default distance_path must be generic")
        cur.execute("SELECT oid, relfilenode, reloptions FROM pg_class WHERE oid=%s::regclass", (config["index"],))
        manifest["index_before"] = cur.fetchone()
        require_check(f"lists={args.lists}" in (manifest["index_before"][2] or []), "index lists differ from --lists")
        cur.execute("SELECT version()")
        manifest["version"] = cur.fetchone()[0]
        cur.execute("SHOW ALL")
        manifest["config_before"] = cur.fetchall()
    raw, summaries, coverage = [], [], []
    for mode in ("full", "auto"):
        paths = {}
        for path in (args.baseline_path, args.test_path):
            rows = phase2b_correctness_config(conn, args, config, literals, mode, path, args.queries)
            paths[path] = rows
            raw.extend(rows)
            write_csv(Path(f"{output}_raw.csv"), raw)
            active = mode == "auto" and 0 < args.topk <= 100
            for r in rows:
                require_check(r["physical_bound"] == (max(40, args.topk * 4) if active else 0) and
                              r["bounded_active"] == int(active), "unexpected tuple bound")
                require_check(r["returned_rows"] == args.topk and
                              len(r["result_ids"].split(";")) == args.topk, "incorrect result count")
            if path == "fused2":
                coverage.append({"mode": mode, "distance_path": path, **verify_fused2_coverage(rows)})
                verify_fused2_tail(rows)
            print(f"correctness mode={mode} path={path} queries={len(rows)} complete", flush=True)
        summary = compare_phase2b_paths(paths[args.baseline_path], paths[args.test_path],
                                        mode, args.baseline_path, args.test_path)
        summaries.append(summary)
        write_csv(Path(f"{output}_summary.csv"), summaries)
        check_phase2b_comparison(summary)
        with conn.cursor() as cur:
            cur.execute("EXPLAIN (FORMAT JSON) SELECT ctid::text,id,embedding <-> %s::vector FROM gist_base ORDER BY embedding <-> %s::vector LIMIT %s", (literals[0], literals[0], args.topk))
            plan = cur.fetchone()[0]
            require_check(config["index"] in json.dumps(plan), "expected GIST index not used")
            write_json_atomic(Path(f"{output}_plan_{mode}.json"), plan)
    if coverage:
        write_csv(Path(f"{output}_coverage.csv"), coverage)
    unsupported, unsupported_summaries = [], []
    for name in ("glove-ip", "glove-cosine"):
        cfg = CONFIGS[name]
        qs, _ = load_workload(cfg, max(args.warmup, args.unsupported_queries), args.topk)
        ls = [vector_literal(q) for q in qs]
        paths = {}
        for path in (args.baseline_path, args.test_path):
            paths[path] = phase2b_correctness_config(conn, args, cfg, ls, "full", path, args.unsupported_queries)
            unsupported.extend(paths[path])
        summary = compare_phase2b_paths(paths[args.baseline_path], paths[args.test_path], name,
                                       args.baseline_path, args.test_path, expected_active=0)
        unsupported_summaries.append(summary)
        write_csv(Path(f"{output}_unsupported_raw.csv"), unsupported)
        write_csv(Path(f"{output}_unsupported_summary.csv"), unsupported_summaries)
        check_phase2b_comparison(summary)
    if args.check_fused2_edges:
        verify_fused2_edges(args, literals[0], output)
    with conn.cursor() as cur:
        cur.execute("SELECT oid, relfilenode, reloptions FROM pg_class WHERE oid=%s::regclass", (config["index"],))
        manifest["index_after"] = cur.fetchone()
    require_check(manifest["index_before"] == manifest["index_after"], "index changed during correctness run")
    write_json_atomic(Path(f"{output}_manifest.json"), manifest)
    print(f"PASS: {args.baseline_path}/{args.test_path} correctness", flush=True)


def summarize_phase2b_path(rows, mode, probes, round_no, distance_path,
                           order_position):
    def total(key):
        return sum(row[key] for row in rows)

    pages = total("scanned_pages")
    candidates = total("scanned_candidates")
    scan_ns = total("scan_items_total_ns")
    # Include DIRECT tails when profiling FUSED2.
    path_ns_key = "distance_ns"
    result = {
        "mode": mode, "probes": probes, "round": round_no,
        "order_position": order_position, "distance_path": distance_path,
        "queries": len(rows), "selected_lists": total("selected_lists"),
        "scanned_pages": pages, "scanned_candidates": candidates,
        "distance_calls": total("distance_calls"),
        "tuplesort_input_calls": total("tuplesort_input_calls"),
        "fallback_queries": total("fallback_triggered"),
        "returned_rows": total("returned_rows"),
        "generic_distance_calls": total("generic_distance_calls"),
        "direct_distance_calls": total("direct_distance_calls"),
        "generic_distance_ns": total("generic_distance_ns"),
        "direct_distance_ns": total("direct_distance_ns"),
        "distance_ns": total("distance_ns"),
        "distance_ns_per_candidate": total(path_ns_key) / candidates,
        "scan_items_ns_per_query": scan_ns / len(rows),
        "candidate_extract_ns": total("candidate_extract_ns"),
        "tuple_materialization_ns": total("tuple_materialization_ns"),
        "sort_insert_ns": total("sort_insert_ns"),
        "sort_finalize_ns": total("sort_finalize_ns"),
        "scan_items_total_ns": scan_ns,
        "distance_pct": 100 * total(path_ns_key) / scan_ns,
        "candidate_extraction_pct":
            100 * total("candidate_extract_ns") / scan_ns,
        "sort_insertion_pct": 100 * total("sort_insert_ns") / scan_ns,
        "sort_finalize_pct": 100 * total("sort_finalize_ns") / scan_ns,
        "average_candidates_per_page": candidates / pages,
        "max_candidates_per_page":
            max(row["max_candidates_per_page"] for row in rows),
    }
    for key in FUSED2_COUNTERS + ("fused_distance_ns",):
        result[key] = sum(row.get(key, 0) for row in rows)
    result["fused_coverage_pct"] = 100 * result["fused_candidates"] / candidates
    result["direct_tail_distance_ns"] = (
        result["direct_distance_ns"] if distance_path == "fused2" else 0)
    result["fused_distance_ns_per_pair"] = (
        result["fused_distance_ns"] / result["fused_pair_calls"]
        if result["fused_pair_calls"] else 0.0)
    return result


def compare_phase2b_profile_pair(baseline, test):
    equality_fields = (
        "queries", "selected_lists", "scanned_pages", "scanned_candidates",
        "distance_calls", "tuplesort_input_calls", "fallback_queries", "returned_rows",
    )
    differences = {
        field: baseline[field] - test[field] for field in equality_fields
    }
    logic_equal = all(value == 0 for value in differences.values())
    baseline_ns = baseline["distance_ns_per_candidate"]
    test_ns = test["distance_ns_per_candidate"]
    saved_ns = baseline_ns - test_ns
    result = {
        "mode": baseline["mode"], "probes": baseline["probes"],
        "round": baseline["round"], "queries": baseline["queries"],
        "baseline_path": baseline["distance_path"], "test_path": test["distance_path"],
        "order": "-".join(row["distance_path"] for row in
                          sorted((baseline, test), key=lambda row: row["order_position"])),
        "baseline_pages": baseline["scanned_pages"],
        "test_pages": test["scanned_pages"],
        "page_difference": differences["scanned_pages"],
        "baseline_candidates": baseline["scanned_candidates"],
        "test_candidates": test["scanned_candidates"],
        "candidate_difference": differences["scanned_candidates"],
        "baseline_distance_calls": baseline["distance_calls"],
        "test_distance_calls": test["distance_calls"],
        "distance_call_difference": differences["distance_calls"],
        "selected_list_difference": differences["selected_lists"],
        "logic_workload_equal": logic_equal,
        "baseline_ns_per_candidate": baseline_ns,
        "test_ns_per_candidate": test_ns,
        "saved_ns_per_candidate": saved_ns,
        "distance_stage_speedup": baseline_ns / test_ns,
        "test_distance_improvement_pct": 100 * saved_ns / baseline_ns,
        "baseline_scan_items_total_ns": baseline["scan_items_total_ns"],
        "test_scan_items_total_ns": test["scan_items_total_ns"],
        "baseline_scan_items_ns_per_query": baseline["scan_items_ns_per_query"],
        "test_scan_items_ns_per_query": test["scan_items_ns_per_query"],
        "baseline_candidate_extract_ns": baseline["candidate_extract_ns"],
        "test_candidate_extract_ns": test["candidate_extract_ns"],
        "baseline_sort_insert_ns": baseline["sort_insert_ns"],
        "test_sort_insert_ns": test["sort_insert_ns"],
        "baseline_sort_finalize_ns": baseline["sort_finalize_ns"],
        "test_sort_finalize_ns": test["sort_finalize_ns"],
    }

    result["scan_items_speedup"] = baseline["scan_items_total_ns"] / test["scan_items_total_ns"]
    result["scan_items_improvement_pct"] = 100 * (1 - test["scan_items_total_ns"] / baseline["scan_items_total_ns"])
    result["fused_coverage_pct"] = test["fused_coverage_pct"]
    result["fused_pair_calls"] = test["fused_pair_calls"]
    result["fused_candidates"] = test["fused_candidates"]
    result["single_tail_candidates"] = test["single_tail_candidates"]
    result["fused_fallback_candidates"] = test["fused_fallback_candidates"]
    result["direct_tail_distance_ns"] = test["direct_tail_distance_ns"]
    result["fused_distance_ns"] = test["fused_distance_ns"]
    result["fused_distance_ns_per_pair"] = test["fused_distance_ns_per_pair"]
    # Keep historical G/D columns only for the original comparison.
    if (baseline["distance_path"], test["distance_path"]) == ("generic", "direct"):
        for key, value in list(result.items()):
            if key.startswith("baseline_") and key != "baseline_path":
                result[key.replace("baseline_", "generic_", 1)] = value
            elif key.startswith("test_") and key != "test_path":
                result[key.replace("test_", "direct_", 1)] = value
    return result


def summarize_phase2b_pairs(pairs):
    output = []
    for mode, probes in sorted({
            (row["mode"], row["probes"]) for row in pairs}):
        group = [
            row for row in pairs
            if row["mode"] == mode and row["probes"] == probes
        ]
        result = {
            "mode": mode, "probes": probes, "rounds": len(group),
            "baseline_path": group[0]["baseline_path"], "test_path": group[0]["test_path"],
            "queries_per_path_per_round": group[0]["queries"],
            "all_logic_workloads_equal":
                all(row["logic_workload_equal"] for row in group),
        }
        metrics = (
            "baseline_ns_per_candidate", "test_ns_per_candidate",
            "baseline_scan_items_ns_per_query", "test_scan_items_ns_per_query",
            "saved_ns_per_candidate", "distance_stage_speedup",
            "test_distance_improvement_pct", "scan_items_speedup",
            "scan_items_improvement_pct", "fused_coverage_pct",
        )
        for metric in metrics:
            values = [row[metric] for row in group]
            mean = statistics.fmean(values)
            result[f"{metric}_median"] = statistics.median(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
            result[f"{metric}_variance"] = statistics.pvariance(values)
            result[f"{metric}_run_range"] = max(values) - min(values)
            result[f"{metric}_cv_pct"] = (
                100 * statistics.pstdev(values) / abs(mean) if mean else 0.0
            )
        if (result["baseline_path"], result["test_path"]) == ("generic", "direct"):
            for key, value in list(result.items()):
                if key.startswith("baseline_") and key != "baseline_path":
                    result[key.replace("baseline_", "generic_", 1)] = value
                elif key.startswith("test_") and key != "test_path":
                    result[key.replace("test_", "direct_", 1)] = value
        output.append(result)
    return output


def run_phase2b(conn, args):
    config = CONFIGS["gist-l2"]
    queries, _ = load_workload(
        config, max(args.warmup, args.queries), args.topk)
    modes = ("full", "auto") if args.mode == "both" else (args.mode,)
    interleaved = args.distance_path == "interleaved"
    if args.queries <= 0 or args.warmup < 0:
        raise ValueError("queries must be positive and warmup nonnegative")
    if interleaved and args.baseline_path == args.test_path:
        raise ValueError("baseline-path and test-path must differ")
    require_check(len(queries) >= max(args.queries, args.warmup), "insufficient query vectors")
    round_count = args.rounds if interleaved else 1
    if round_count <= 0:
        raise ValueError("rounds must be positive")

    output = Path(args.output or ROOT / "results/phase_2b_profile")
    raw_path = Path(f"{output}_raw.csv")
    runs_path = Path(f"{output}_runs.csv")
    raw_rows = []
    run_summaries = []

    with conn.cursor() as cur:
        ensure_index(cur, config)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET ivfflat.iterative_scan = off")
        cur.execute("SELECT set_config(%s, %s, false)",
                    ("ivfflat.experimental_sort_bound", "0"))
        cur.execute("SELECT set_config(%s, %s, false)",
                    ("ivfflat.bound_overfetch", "4"))
        cur.execute("SELECT set_config(%s, %s, false)",
                    ("ivfflat.bound_min", "40"))
        cur.execute("SELECT set_config(%s, %s, false)",
                    ("ivfflat.bound_fastpath_limit", "100"))

        for mode in modes:
            cur.execute("SELECT set_config(%s, %s, false)",
                        ("ivfflat.bounded_scan",
                         "on" if mode == "auto" else "off"))
            for probes in args.probes_list:
                cur.execute("SELECT set_config(%s, %s, false)",
                            ("ivfflat.probes", str(probes)))
                for round_no in range(1, round_count + 1):
                    if interleaved:
                        path_order = (
                            (args.baseline_path, args.test_path) if round_no % 2
                            else (args.test_path, args.baseline_path)
                        )
                    else:
                        path_order = (args.distance_path,)
                    for order_position, distance_path in enumerate(
                            path_order, start=1):
                        print(
                            f"phase2b mode={mode} probes={probes} "
                            f"round={round_no} order={order_position} "
                            f"path={distance_path} warmup={args.warmup} "
                            f"queries={args.queries}",
                            flush=True)
                        cur.execute("SELECT set_config(%s, %s, false)",
                                    ("ivfflat.distance_path", distance_path))
                        for query in queries[:args.warmup]:
                            execute_query(
                                conn, cur, config, query, args.topk, False)

                        path_rows = []
                        for query_id, query in enumerate(
                                queries[:args.queries]):
                            conn.notices.clear()
                            ids, _, profile = execute_query(
                                conn, cur, config, query, args.topk, True)
                            row = parse_profile_2b(conn.notices)
                            row.update(profile)
                            row.update(
                                query_id=query_id, mode=mode, probes=probes,
                                round=round_no,
                                order_position=order_position,
                                distance_path=distance_path,
                                result_ids=";".join(str(value) for value in ids),
                                direct_tail_distance_ns=(
                                    row["direct_distance_ns"]
                                    if distance_path == "fused2" else 0))
                            path_rows.append(row)
                            raw_rows.append(row)

                        validate_phase2b_dispatch(path_rows, distance_path)
                        run_summaries.append(summarize_phase2b_path(
                            path_rows, mode, probes, round_no,
                            distance_path, order_position))
                        write_csv_atomic(
                            raw_path, raw_rows, tuple(raw_rows[0].keys()))
                        write_csv_atomic(
                            runs_path, run_summaries,
                            tuple(run_summaries[0].keys()))

    if not interleaved:
        write_csv(Path(f"{output}_summary.csv"), run_summaries)
        return

    pairs = []
    for mode in modes:
        for probes in args.probes_list:
            for round_no in range(1, round_count + 1):
                group = [
                    row for row in run_summaries
                    if row["mode"] == mode and row["probes"] == probes and
                    row["round"] == round_no
                ]
                baseline = next(row for row in group if row["distance_path"] == args.baseline_path)
                tested = next(row for row in group if row["distance_path"] == args.test_path)
                pair = compare_phase2b_profile_pair(baseline, tested)
                by_path = {
                    path: {r["query_id"]: r for r in raw_rows
                           if r["mode"] == mode and r["probes"] == probes and
                           r["round"] == round_no and r["distance_path"] == path}
                    for path in (args.baseline_path, args.test_path)
                }
                a, b = by_path[args.baseline_path], by_path[args.test_path]
                pair["workload_mismatch_queries"] = sum(
                    any(a[q][k] != b[q][k] for k in PHASE2B_WORKLOAD_FIELDS) for q in a)
                pair["result_mismatch_queries"] = sum(
                    a[q]["result_ids"] != b[q]["result_ids"] for q in a)
                pair["logic_workload_equal"] &= (
                    pair["workload_mismatch_queries"] == 0 and
                    pair["result_mismatch_queries"] == 0)
                pairs.append(pair)

    pair_path = Path(f"{output}_paired.csv")
    write_csv(pair_path, pairs)
    summaries = summarize_phase2b_pairs(pairs)
    write_csv(Path(f"{output}_summary.csv"), summaries)
    invalid = [row for row in pairs if not row["logic_workload_equal"]]
    if invalid:
        raise RuntimeError(
            f"{args.baseline_path}/{args.test_path} logical workload mismatch: {invalid}")



def production_metric_stats(values):
    mean = statistics.fmean(values)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "variance": statistics.pvariance(values),
        "cv_pct": 100 * statistics.pstdev(values) / abs(mean) if mean else 0.0,
        "run_range": max(values) - min(values),
    }


def production_group_stats(rows):
    latencies = [float(row["latency_us"]) for row in rows]
    total_seconds = sum(latencies) / 1_000_000.0
    return {
        "p50_us": percentile(latencies, 50),
        "p95_us": percentile(latencies, 95),
        "p99_us": percentile(latencies, 99),
        "mean_us": statistics.fmean(latencies),
        "qps": len(rows) / total_seconds if total_seconds else 0.0,
        "min_returned_rows": min(int(row["returned_rows"]) for row in rows),
    }


def phase2b_production_schedule(probes_list, rounds):
    require_check(tuple(sorted(probes_list)) == (16, 64, 128),
                  "2b-production requires probes=16,64,128")
    require_check(rounds == 4, "2b-production requires exactly four rounds")
    schedule = []
    sequence = 0
    for round_no in range(1, rounds + 1):
        probe_order = PRODUCTION_PROBE_ORDERS[round_no - 1]
        mode_order = PRODUCTION_MODES if round_no % 2 else tuple(reversed(PRODUCTION_MODES))
        path_order = PRODUCTION_PATHS if round_no % 2 else tuple(reversed(PRODUCTION_PATHS))
        config_position = 0
        for probes in probe_order:
            for mode in mode_order:
                config_position += 1
                for order_position, distance_path in enumerate(path_order, start=1):
                    sequence += 1
                    schedule.append({
                        "sequence": sequence, "round": round_no,
                        "config_position": config_position, "mode": mode,
                        "probes": probes, "order_position": order_position,
                        "distance_path": distance_path,
                    })
    return schedule


def summarize_phase2b_production_run(rows, mode, probes, round_no,
                                     distance_path, config_position,
                                     order_position):
    stats = production_group_stats(rows)
    result_digest = hashlib.sha256("\n".join(
        row["result_checksum"] for row in sorted(rows, key=lambda value: value["query_id"])
    ).encode()).hexdigest()
    return {
        "mode": mode, "probes": probes, "round": round_no,
        "config_position": config_position, "order_position": order_position,
        "distance_path": distance_path, "queries": len(rows), **stats,
        "result_checksum": result_digest,
    }


def compare_phase2b_production_pair(direct, fused, raw_rows):
    require_check((direct["mode"], direct["probes"], direct["round"]) ==
                  (fused["mode"], fused["probes"], fused["round"]),
                  "production pair configuration mismatch")
    matching = [
        row for row in raw_rows
        if row["mode"] == direct["mode"] and row["probes"] == direct["probes"] and
        row["round"] == direct["round"]
    ]
    by_path = {
        path: {row["query_id"]: row for row in matching if row["distance_path"] == path}
        for path in PRODUCTION_PATHS
    }
    require_check(set(by_path["direct"]) == set(by_path["fused2"]),
                  "production pair query IDs differ")
    query_ids = sorted(by_path["direct"])
    result_mismatches = sum(
        by_path["direct"][query_id]["result_checksum"] !=
        by_path["fused2"][query_id]["result_checksum"]
        for query_id in query_ids)
    returned_mismatches = sum(
        by_path["direct"][query_id]["returned_rows"] !=
        by_path["fused2"][query_id]["returned_rows"]
        for query_id in query_ids)
    pair = {
        "mode": direct["mode"], "probes": direct["probes"],
        "round": direct["round"], "queries": len(query_ids),
        "order": "-".join(row["distance_path"] for row in
                            sorted((direct, fused), key=lambda row: row["order_position"])),
        "result_mismatch_queries": result_mismatches,
        "returned_row_mismatch_queries": returned_mismatches,
        "all_result_checksums_equal": result_mismatches == 0,
        "all_returned_rows_equal": returned_mismatches == 0,
    }
    for metric in ("p50_us", "p95_us", "p99_us", "mean_us"):
        direct_value = direct[metric]
        fused_value = fused[metric]
        pair[f"direct_{metric}"] = direct_value
        pair[f"fused2_{metric}"] = fused_value
        pair[f"paired_{metric[:-3]}_improvement_pct"] = (
            100 * (direct_value - fused_value) / direct_value)
    pair["direct_qps"] = direct["qps"]
    pair["fused2_qps"] = fused["qps"]
    pair["paired_qps_improvement_pct"] = (
        100 * (fused["qps"] - direct["qps"]) / direct["qps"])
    pair["direct_min_returned_rows"] = direct["min_returned_rows"]
    pair["fused2_min_returned_rows"] = fused["min_returned_rows"]
    return pair


def summarize_phase2b_production(pairs):
    output = []
    for mode, probes in sorted({(row["mode"], row["probes"]) for row in pairs}):
        group = [row for row in pairs if row["mode"] == mode and row["probes"] == probes]
        result = {
            "mode": mode, "probes": probes, "rounds": len(group),
            "queries_per_path_per_round": group[0]["queries"],
            "result_mismatch_queries": sum(row["result_mismatch_queries"] for row in group),
            "returned_row_mismatch_queries": sum(
                row["returned_row_mismatch_queries"] for row in group),
            "all_result_checksums_equal": all(row["all_result_checksums_equal"] for row in group),
            "all_returned_rows_equal": all(row["all_returned_rows_equal"] for row in group),
        }
        for metric in ("p50", "p95", "p99", "mean"):
            for distance_path in PRODUCTION_PATHS:
                values = [row[f"{distance_path}_{metric}_us"] for row in group]
                for stat, value in production_metric_stats(values).items():
                    result[f"{distance_path}_{metric}_us_{stat}"] = value
        for distance_path in PRODUCTION_PATHS:
            values = [row[f"{distance_path}_qps"] for row in group]
            for stat, value in production_metric_stats(values).items():
                result[f"{distance_path}_qps_{stat}"] = value
        for metric in ("p50", "p95", "p99", "mean", "qps"):
            values = [row[f"paired_{metric}_improvement_pct"] for row in group]
            for stat, value in production_metric_stats(values).items():
                result[f"paired_{metric}_improvement_pct_{stat}"] = value
            result[f"paired_{metric}_positive_rounds"] = sum(value > 0 for value in values)
        output.append(result)
    return output


def verify_phase2b_production_build(conn, args, config, query_literal):
    sql = (f"SELECT id FROM {config['table']} "
           f"ORDER BY embedding {config['operator']} %s::vector LIMIT %s")
    with conn.cursor() as cur:
        configure_phase2b_scan(cur, args.probes_list[0], "full", "direct")
        cur.execute("SET client_min_messages = info")
        conn.notices.clear()
        try:
            formal_query(cur, sql, query_literal, args.topk)
            profiling = [notice for notice in conn.notices
                         if PROFILE_RE.search(notice) or PROFILE_2B_RE.search(notice)]
            require_check(not profiling,
                          "2b-production build contains IVFFlat profiling notices")
        finally:
            conn.notices.clear()
            cur.execute("RESET client_min_messages")
    print("2b-production build check: distance_path available; profiling notices off",
          flush=True)


def run_phase2b_production(conn, args):
    require_check(args.lists == 1000 and args.topk == 10,
                  "2b-production requires lists=1000 and topk=10")
    require_check(args.baseline_path == "direct" and args.test_path == "fused2",
                  "2b-production requires direct/fused2")
    require_check(args.mode == "both", "2b-production requires mode=both")
    require_check(args.warmup == 100 and args.queries == 1000,
                  "2b-production requires warmup=100 and queries=1000")
    schedule = phase2b_production_schedule(args.probes_list, args.rounds)
    config = CONFIGS["gist-l2"]
    queries, _ = load_workload(config, max(args.warmup, args.queries), args.topk)
    require_check(len(queries) >= args.queries, "insufficient GIST1M official queries")
    query_literals = [vector_literal(query) for query in queries]
    verify_phase2b_production_build(conn, args, config, query_literals[0])

    output = Path(args.output or ROOT / "results/phase_2b_fused2_production")
    raw_path = Path(f"{output}_raw.csv")
    runs_path = Path(f"{output}_runs.csv")
    paired_path = Path(f"{output}_paired.csv")
    summary_path = Path(f"{output}_summary.csv")
    run_order_path = output.parent / "run_order.txt"
    for path in (raw_path, runs_path, paired_path, summary_path, run_order_path):
        require_check(not path.exists(), f"production output already exists: {path}")
    run_order_path.write_text("")

    raw_rows = []
    run_rows = []
    pairs = []
    sql = (f"SELECT id FROM {config['table']} "
           f"ORDER BY embedding {config['operator']} %s::vector LIMIT %s")
    grouped_schedule = {}
    for entry in schedule:
        grouped_schedule.setdefault(
            (entry["round"], entry["config_position"], entry["mode"], entry["probes"]),
            []).append(entry)

    with conn.cursor() as cur:
        ensure_index(cur, config)
        for key, entries in grouped_schedule.items():
            round_no, config_position, mode, probes = key
            for entry in entries:
                distance_path = entry["distance_path"]
                with run_order_path.open("a") as run_order:
                    run_order.write(
                        f"{entry['sequence']}: round={round_no} config={config_position} "
                        f"mode={mode} probes={probes} order={entry['order_position']} "
                        f"path={distance_path}\n")
                print(
                    f"2b-production round={round_no} config={config_position} "
                    f"mode={mode} probes={probes} order={entry['order_position']} "
                    f"path={distance_path} warmup={args.warmup} queries={args.queries}",
                    flush=True)
                configure_phase2b_scan(cur, probes, mode, distance_path)
                for query_literal in query_literals[:args.warmup]:
                    formal_query(cur, sql, query_literal, args.topk)
                path_rows = []
                for query_id, query_literal in enumerate(query_literals[:args.queries]):
                    ids, latency_us = formal_query(cur, sql, query_literal, args.topk)
                    result_ids = ";".join(str(value) for value in ids)
                    row = {
                        "experiment": "phase_2b_fused2_production",
                        "dataset": "gist1m", "dimension": config["dimension"],
                        "metric": config["metric"], "mode": mode,
                        "lists": args.lists, "probes": probes, "topk": args.topk,
                        "round": round_no, "config_position": config_position,
                        "order_position": entry["order_position"],
                        "distance_path": distance_path, "query_id": query_id,
                        "latency_us": latency_us, "returned_rows": len(ids),
                        "result_ids": result_ids,
                        "result_checksum": hashlib.sha256(result_ids.encode()).hexdigest(),
                    }
                    path_rows.append(row)
                    raw_rows.append(row)
                run_rows.append(summarize_phase2b_production_run(
                    path_rows, mode, probes, round_no, distance_path,
                    config_position, entry["order_position"]))
                write_csv_atomic(raw_path, raw_rows, PRODUCTION_RAW_FIELDS)
                write_csv_atomic(runs_path, run_rows, tuple(run_rows[0].keys()))

            direct = next(row for row in run_rows if row["mode"] == mode and
                          row["probes"] == probes and row["round"] == round_no and
                          row["distance_path"] == "direct")
            fused = next(row for row in run_rows if row["mode"] == mode and
                         row["probes"] == probes and row["round"] == round_no and
                         row["distance_path"] == "fused2")
            pair = compare_phase2b_production_pair(direct, fused, raw_rows)
            pairs.append(pair)
            write_csv_atomic(paired_path, pairs, tuple(pairs[0].keys()))
            require_check(pair["result_mismatch_queries"] == 0 and
                          pair["returned_row_mismatch_queries"] == 0,
                          f"invalid production config: {pair}")

    summaries = summarize_phase2b_production(pairs)
    write_csv_atomic(summary_path, summaries, tuple(summaries[0].keys()))
    require_check(len(raw_rows) == 3 * 2 * 2 * 4 * args.queries,
                  "unexpected production raw row count")
    print(f"2b-production complete rows={len(raw_rows)} paired_rounds={len(pairs)}",
          flush=True)



def run_experiment(args):
    conn = connect(args)
    conn.autocommit = True
    if args.phase != "formal":
        with conn.cursor() as cur:
            cur.execute("SET client_min_messages = info")
    if args.phase == "all-fomal-exp":
        import sys
        d2p_path = str(Path(__file__).resolve().parents[1] / "d2p")
        if d2p_path not in sys.path:
            sys.path.insert(0, d2p_path)
        from formal_runner import run as run_d2p_formal
        run_d2p_formal(conn, args)
    elif args.phase == "formal":
        run_formal(conn, args)
    elif args.phase == "2b-production":
        run_phase2b_production(conn, args)
    elif args.phase == "2b-correctness":
        run_phase2b_correctness(conn, args)
    elif args.phase == "2b":
        run_phase2b(conn, args)
    elif args.phase == "2a34":
        config = CONFIGS["glove-cosine"]
        queries, _ = load_workload(config, args.queries, args.topk, False)
        rows = []
        output_prefix = Path(args.output or ROOT / "results/phase_2a34_robustness")
        for probes in args.probes_list:
            for divisor in args.filter_divisors:
                if divisor <= 0:
                    raise ValueError("filter divisors must be positive")
                print(f"phase2a34 probes={probes} filter_divisor={divisor}", flush=True)
                full_rows = phase2a34_config(conn, args, config, queries, probes, divisor, "full")
                auto_rows = phase2a34_config(conn, args, config, queries, probes, divisor, "auto")
                full_by_query = {row["query_id"]: row for row in full_rows}
                for row in full_rows + auto_rows:
                    baseline = full_by_query[row["query_id"]]
                    full_ids = baseline["result_ids"].split(";") if baseline["result_ids"] else []
                    result_ids = row["result_ids"].split(";") if row["result_ids"] else []
                    row["topk_match_full"] = result_ids == full_ids
                    row["returned_rows_match_full"] = row["returned_rows"] == baseline["returned_rows"]
                    row["recall_vs_full"] = (len(set(result_ids).intersection(full_ids)) / len(full_ids)
                                             if full_ids else 1.0)
                    row["status"] = ("correct" if row["topk_match_full"] and
                                     row["returned_rows_match_full"] and
                                     row["returned_rows"] == row["distinct_rows"] else "incorrect")
                    rows.append(row)
        write_csv(Path(f"{output_prefix}_raw.csv"), rows)
        write_csv(Path(f"{output_prefix}_summary.csv"), summarize_phase2a34(rows))
    elif args.phase == "2a2":
        config = CONFIGS["glove-cosine"]
        queries, neighbors = load_workload(config, args.queries, args.topk, True)
        rows = []
        output_prefix = Path(args.output or ROOT / "results/phase_2a2_profile")
        for probes in args.probes_list:
            full_results = None
            for mode in ("full", "oracle40", "auto"):
                print(f"phase2a2 probes={probes} mode={mode}", flush=True)
                row, result_ids = phase2a2_config(conn, args, config, queries, neighbors, probes, mode)
                if full_results is None:
                    full_results = result_ids
                row["top10_match_full"] = result_ids == full_results
                row["status"] = "correct" if row["min_returned_rows"] == args.topk and row["top10_match_full"] else "unsafe/incorrect"
                rows.append(row)
        write_csv(Path(f"{output_prefix}_raw.csv"), rows)
        write_csv(Path(f"{output_prefix}_summary.csv"), summarize_phase2a2(rows))
    elif args.phase == "2a":
        config = CONFIGS["glove-l2"]
        queries, neighbors = load_workload(config, args.queries, args.topk, True)
        rows = []
        output_prefix = Path(args.output or ROOT / "results/phase2a")
        for probes in args.probes_list:
            for bound in args.sort_bounds:
                for round_no in range(1, args.rounds + 1):
                    print(f"phase2a probes={probes} bound={bound} round={round_no}", flush=True)
                    row = phase2a_config(conn, args, config, queries, neighbors, probes, bound, round_no)
                    rows.append(row)
                    config_path = output_prefix.parent / f"glove_l2_p{probes}_b{bound}_r{round_no}.csv"
                    write_csv(config_path, [row])
        write_csv(output_prefix.parent / "phase2a_raw.csv", rows)
        write_csv(output_prefix.parent / "phase2a_summary.csv", summarize_phase2a(rows, args.topk))
    else:
        raw_rows, summaries = [], []
        if args.phase == "a":
            for name in ("glove-l2", "glove-cosine", "glove-ip"):
                for probes in PROBES:
                    run_group(conn, args, "A", name, probes, raw_rows, summaries)
        else:
            for name in ("glove-l2", "sift-l2", "gist-l2"):
                run_group(conn, args, "B", name, 64, raw_rows, summaries)
        prefix = args.output or ROOT / "results" / f"phase_{args.phase}_profile"
        write_csv(Path(f"{prefix}_raw.csv"), raw_rows)
        write_csv(Path(f"{prefix}_summary.csv"), summaries)
    conn.close()


def build_indexes(args):
    names = list(CONFIGS) if args.dataset == "all" else [args.dataset]
    rows = []
    conn = connect(args)
    conn.autocommit = True
    with conn.cursor() as cur:
        for name in names:
            config = CONFIGS[name]
            cur.execute(f"DROP INDEX IF EXISTS {config['index']}")
            started = time.perf_counter()
            cur.execute(f"CREATE INDEX {config['index']} ON {config['table']} USING ivfflat "
                        f"(embedding {config['opclass']}) WITH (lists = {args.lists})")
            cur.execute(f"ANALYZE {config['table']}")
            cur.execute("SELECT pg_relation_size(%s)", (config["index"],))
            rows.append({"dataset": name, "dimension": config["dimension"],
                         "metric": config["operator"], "lists": args.lists,
                         "build_seconds": time.perf_counter() - started,
                         "index_bytes": cur.fetchone()[0]})
    conn.close()
    write_csv(args.output or ROOT / "results/index_build.csv", rows)


def validate_phase2b_options(args):
    if args.phase not in ("2b", "2b-correctness", "2b-production"):
        if args.check_fused2_edges:
            raise ValueError("--check-fused2-edges is only supported by 2b-correctness")
        return
    if args.queries <= 0 or args.warmup < 0 or args.topk <= 0 or args.lists <= 0:
        raise ValueError("queries/topk/lists must be positive; warmup must be nonnegative")
    if not args.probes_list or any(p <= 0 or p > args.lists for p in args.probes_list):
        raise ValueError("probes must be between 1 and lists")
    paired = args.phase in ("2b-correctness", "2b-production") or args.distance_path == "interleaved"
    if paired and args.baseline_path == args.test_path:
        raise ValueError("baseline-path and test-path must differ")
    if args.phase == "2b-production":
        phase2b_production_schedule(args.probes_list, args.rounds)
        if (args.baseline_path, args.test_path, args.mode, args.distance_path) != (
                "direct", "fused2", "both", "interleaved"):
            raise ValueError(
                "2b-production requires direct/fused2, mode=both, distance-path=interleaved")
        if args.check_fused2_edges:
            raise ValueError("2b-production does not run fused2 edge checks")
        if args.queries != 1000 or args.warmup != 100 or args.topk != 10 or args.lists != 1000:
            raise ValueError("2b-production requires queries=1000, warmup=100, topk=10, lists=1000")
    elif args.phase == "2b-correctness":
        if len(args.probes_list) != 1 or args.unsupported_queries <= 0:
            raise ValueError("correctness requires one probes value and positive unsupported-queries")
    elif args.rounds <= 0 or args.check_fused2_edges:
        raise ValueError("profiling requires positive rounds; edge checks belong to 2b-correctness")
    if args.check_fused2_edges and "fused2" not in (args.baseline_path, args.test_path):
        raise ValueError("--check-fused2-edges requires a fused2 comparison")


# Phase C extends the shared runner with validated smoke and formal execution modes.
PHASE_C_SYSTEMS = ('pristine', 'vanilla_eq', 'phase2a', 'phase2b', 'final')
PHASE_C_FLAGS = '-march=haswell -mtune=haswell -mavx2 -mfma'


def phase_c_command(argv, **kwargs):
    return subprocess.check_output([str(x) for x in argv], text=True, **kwargs).strip()


def phase_c_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def phase_c_prepare(args, root):
    repo = SCRIPT_ROOT.parent
    pristine = Path(os.environ.get('PRISTINE_WORKTREE', '/workspace/OpenTenBase-pristine'))
    commit = os.environ.get('PRISTINE_COMMIT')
    if not commit:
        raise RuntimeError('PRISTINE_COMMIT must be explicitly supplied')
    git = lambda *a: phase_c_command(['git', '-C', repo, *a])
    commit = git('rev-parse', commit + '^{commit}')
    optimized = git('rev-parse', 'HEAD')
    # This reviewed allowlist is intentionally fail-closed for future source changes.
    reviewed_pristine = git('rev-parse', 'd65ea656^{commit}')
    if commit != reviewed_pristine:
        raise RuntimeError('Unreviewed pristine commit: inspect provenance and disk compatibility first')
    if git('rev-parse', 'HEAD:contrib/pgvector') != '5e22d78fca56dfd0c6abd126d8e459a2d4d7bd92':
        raise RuntimeError('Unreviewed optimized source tree; repeat compatibility review')
    changed = git('diff', '--name-only', commit, 'HEAD', '--', 'contrib/pgvector').splitlines()
    allowed = {'Makefile', 'src/ivfflat.c', 'src/ivfflat.h', 'src/ivfscan.c', 'src/vector.c', 'src/vector.h'}
    if any(p.removeprefix('contrib/pgvector/') not in allowed for p in changed):
        raise RuntimeError('Unreviewed pgvector storage/build changes; shared index forbidden')
    if git('status', '--porcelain', '--', 'contrib/pgvector'):
        raise RuntimeError('Optimized pgvector source must be clean')
    source_diff = git('diff', commit, 'HEAD', '--', 'contrib/pgvector')
    (root / 'phase_c_source_diff.patch').write_text(source_diff + '\n')
    # Disk structs and their page constants must remain byte-for-byte identical.
    before = git('show', commit + ':contrib/pgvector/src/ivfflat.h')
    after = (repo / 'contrib/pgvector/src/ivfflat.h').read_text()
    for name in ('IvfflatMetaPageData', 'IvfflatPageOpaqueData', 'IvfflatListData'):
        pattern = r'typedef struct ' + name + r'\b.*?\}\s*' + name + r';'
        a, b = re.search(pattern, before, re.S), re.search(pattern, after, re.S)
        if not a or not b or a.group() != b.group():
            raise RuntimeError('Disk format changed or unrecognized: ' + name)
    for token in ('bounded_scan', 'distance_path', 'IVFFLAT_FUSED2', 'IVFFLAT_PROFILE_2B', 'profile_candidates'):
        if token in git('show', commit + ':contrib/pgvector/src/ivfscan.c'):
            raise RuntimeError('Pristine contains research modifications: ' + token)
    if not pristine.exists():
        subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(pristine), commit], check=True)
    if phase_c_command(['git', '-C', pristine, 'rev-parse', 'HEAD']) != commit:
        raise RuntimeError('Wrong pristine worktree commit')
    if phase_c_command(['git', '-C', pristine, 'status', '--porcelain']):
        raise RuntimeError('Pristine worktree must be clean')
    compiler = phase_c_command(['gcc', '--version'])
    if phase_c_command(['gcc', '-dumpfullversion']) != '11.5.0':
        raise RuntimeError('GCC 11.5.0 is required')
    pg_config = os.environ.get('PG_CONFIG', '/workspace/install/bin/pg_config')
    metadata = {'pristine_commit': commit, 'optimized_commit': optimized,
                'optimized_pgvector_tree': git('rev-parse', 'HEAD:contrib/pgvector'),
                'pristine_pgvector_tree': git('rev-parse', commit + ':contrib/pgvector'),
                'compiler': compiler, 'compile_flags': '-O2 ' + PHASE_C_FLAGS + ' -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math',
                'profiling_timers_disabled': True, 'on_disk_format_changed': False,
                'shared_index_safe': True, 'postgres_sha256': phase_c_hash(Path(pg_config).with_name('postgres'))}
    for label, source in [('pristine', pristine), ('optimized', repo)]:
        status = phase_c_command(['git', '-C', source, 'status', '--short', '--untracked-files=all'])
        (root / f'phase_c_{label}_source_status.txt').write_text(status + '\n')
        metadata[label + '_clean'] = not bool(status)
        destination = root / 'binaries' / label
        destination.mkdir(parents=True, exist_ok=True)
        command = ['make', '-C', str(source / 'contrib/pgvector'), '-B', '-j8', 'CC=gcc', 'PG_CONFIG=' + pg_config, 'OPTFLAGS=' + PHASE_C_FLAGS]
        if label == 'optimized':
            command += ['IVFFLAT_PROFILE_CFLAGS=-DIVFFLAT_FUSED2']
        env = dict(os.environ)
        for name in ('PG_CFLAGS', 'CFLAGS', 'CPPFLAGS', 'IVFFLAT_PROFILE_CFLAGS', 'MAKEFLAGS'):
            env.pop(name, None)
        with (root / f'{label}_compile_command.txt').open('w') as log:
            log.write('argv=' + json.dumps(command) + '\n')
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (root / f'{label}_compile_command.txt').read_text()
        if '-DIVFFLAT_BENCH' in log or '-DIVFFLAT_PROFILE_2B' in log:
            raise RuntimeError('Profiling enabled in formal build')
        so = source / 'contrib/pgvector/vector.so'
        if 'IVFFLAT_PROFILE' in phase_c_command(['strings', so]):
            raise RuntimeError('Profiling NOTICE found')
        symbols = phase_c_command(['nm', '-u', so])
        if re.search(r'\b(clock_gettime|gettimeofday)\b', symbols):
            raise RuntimeError('Unexpected timer dependency in vector.so')
        shutil.copy2(so, destination / 'vector.so')
        metadata[label + '_binary_sha256'] = phase_c_hash(destination / 'vector.so')
        write_json_atomic(destination / 'phase_c_build.json', {'commit': commit if label == 'pristine' else optimized, 'compiler': compiler, 'flags': metadata['compile_flags'], 'sha256': metadata[label + '_binary_sha256'], 'profiling': False})
    metadata['installed_so'] = phase_c_command([pg_config, '--pkglibdir']) + '/vector.so'
    write_json_atomic(root / 'phase_c_build_manifest.json', metadata)
    return metadata


def phase_c_activate(args, root, metadata, binary):
    source = root / 'binaries' / binary / 'vector.so'
    expected = metadata[binary + '_binary_sha256']
    if phase_c_hash(source) != expected:
        raise RuntimeError('Archived binary hash mismatch')
    pg_ctl = os.environ.get('PG_CTL', '/workspace/install/bin/pg_ctl')
    data = os.environ.get('PGDATA', '/workspace/data')
    os_user = os.environ.get('PG_OS_USER', 'dev')
    command = (['runuser', '-u', os_user, '--'] if os.getuid() == 0 else []) + [pg_ctl, '-D', data]
    # Stop/start even if the on-disk hash matches: existing backends could have loaded another image.
    status = subprocess.run(command + ['status'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if status.returncode == 0:
        subprocess.run(command + ['stop', '-m', 'fast', '-w'], check=True)
    elif status.returncode != 3:
        raise RuntimeError('Cannot determine PostgreSQL status')
    shutil.copy2(source, metadata['installed_so'])
    if phase_c_hash(metadata['installed_so']) != expected:
        raise RuntimeError('Installed binary hash mismatch')
    subprocess.run(command + ['start', '-l', str(Path(data) / 'phase_c_server.log'), '-w'], check=True)
    conn = connect(args)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("LOAD 'vector'")
        cur.execute('SELECT 1')
        assert cur.fetchone()[0] == 1
    with (root / 'phase_c_activation.jsonl').open('a') as log:
        log.write(json.dumps({'binary': binary, 'sha256': expected, 'time': time.time(), 'readiness': True}) + '\n')
    return conn


def activate_pristine(args, root, metadata):
    return phase_c_activate(args, root, metadata, 'pristine')


def activate_optimized(args, root, metadata):
    return phase_c_activate(args, root, metadata, 'optimized')


def phase_c_index(cur):
    cur.execute("SELECT c.oid,c.relfilenode,pg_relation_size(c.oid),pg_get_indexdef(c.oid),i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid WHERE c.oid=to_regclass('gist_ivf_l2')")
    row = cur.fetchone()
    return dict(zip(('oid', 'relfilenode', 'bytes', 'definition', 'valid'), row)) if row else None


def phase_c_settings(cur, system, probes):
    settings = {'ivfflat.probes': str(probes), 'ivfflat.iterative_scan': 'off'}
    if system != 'pristine':
        settings.update({'ivfflat.distance_path': 'fused2' if system in ('phase2b', 'final') else 'generic',
                         'ivfflat.bounded_scan': 'on' if system in ('phase2a', 'final') else 'off',
                         'ivfflat.experimental_sort_bound': '0', 'ivfflat.bound_overfetch': '4',
                         'ivfflat.bound_min': '40', 'ivfflat.bound_fastpath_limit': '100'})
    # pg_settings verifies registration, unlike custom placeholder GUCs.
    for key, value in settings.items():
        cur.execute('SELECT setting FROM pg_settings WHERE name=%s', (key,))
        if cur.fetchone() is None:
            raise RuntimeError('Missing registered GUC: ' + key)
        cur.execute('SELECT set_config(%s,%s,false)', (key, value))
        if cur.fetchone()[0] != value:
            raise RuntimeError('GUC did not take effect: ' + key)
    cur.execute('SET enable_seqscan=off')
    cur.execute('SET enable_indexscan=on')
    cur.execute('SET max_parallel_workers_per_gather=0')
    return settings


def phase_c_audit(conn, root, queries, neighbors):
    import h5py
    import numpy as np
    result = {'queries_verified': 20, 'exact_matches': 0, 'status': 'FAIL', 'queries': []}
    sql = 'SELECT id FROM gist_base ORDER BY embedding <-> %s::vector LIMIT %s'
    with conn.cursor() as cur, h5py.File(CONFIGS['gist-l2']['dataset'], 'r') as dataset:
        if dataset['train'].shape != (1000000, 960) or dataset['test'].shape != (1000, 960):
            raise RuntimeError('Wrong GIST1M dimensions')
        cur.execute('SELECT count(*),count(DISTINCT id),min(id),max(id) FROM gist_base')
        result['table_counts'] = list(cur.fetchone())
        if result['table_counts'] != [1000000, 1000000, 0, 999999]:
            raise RuntimeError('Unsupported ID mapping; inspect data loader')
        sampled = sorted(random.Random(20260908).sample(range(1000000), 20))
        for rowid in sampled:
            cur.execute('SELECT embedding::text FROM gist_base WHERE id=%s', (rowid,))
            actual = np.asarray(json.loads(cur.fetchone()[0]), dtype=np.float32)
            if not np.array_equal(actual, dataset['train'][rowid]):
                raise RuntimeError('SQL id does not map to HDF5 train row')
        result['id_mapping'] = {'offset': 0, 'mapping': 'SQL id = HDF5 train row index', 'vector_content_checked_ids': sampled}
        cur.execute('SET enable_indexscan=off')
        cur.execute('SET enable_indexonlyscan=off')
        cur.execute('SET enable_bitmapscan=off')
        cur.execute('SET enable_seqscan=on')
        cur.execute('SET max_parallel_workers_per_gather=0')
        for qid in sorted(random.Random(20260908).sample(range(1000), 20)):
            literal = vector_literal(queries[qid])
            cur.execute('EXPLAIN (FORMAT JSON) ' + sql, (literal, 10))
            plan = cur.fetchone()[0]
            if 'Index' in json.dumps(plan) or 'Seq Scan' not in json.dumps(plan):
                raise RuntimeError('Exact audit did not use sequential scan')
            ids, latency = formal_query(cur, sql, literal, 10)
            gt = [int(i) for i in neighbors[qid]]
            match = set(ids) == set(gt)
            if match and ids != gt:
                cur.execute('SELECT id,embedding <-> %s::vector FROM gist_base WHERE id=ANY(%s)', (literal, ids))
                distances = dict(cur.fetchall())
                inversions = [(a, b) for pos, a in enumerate(ids) for b in ids[pos + 1:] if gt.index(a) > gt.index(b)]
                result.setdefault('ordering_differences', []).append({
                    'query_id': qid, 'inversions': inversions,
                    'sql_distances': distances,
                    'all_inversions_are_equal_distance': all(distances[a] == distances[b] for a, b in inversions)})
            result['exact_matches'] += int(match)
            result['queries'].append({'query_id': qid, 'ids': ids, 'official_ids': gt, 'set_equal': match, 'ordered_equal': ids == gt, 'latency_us': latency, 'plan': plan})
            write_json_atomic(root / 'ground_truth_verification.json', result)
            print(f'Phase C exact audit query={qid} match={match}', flush=True)
        for key in ('enable_indexscan', 'enable_indexonlyscan', 'enable_bitmapscan', 'enable_seqscan'):
            cur.execute('RESET ' + key)
    result['status'] = 'PASS' if result['exact_matches'] == 20 else 'FAIL'
    result['tie_policy'] = 'Require exact Top10 set equality; ordering differences within the same set are recorded. Boundary-set differences fail closed.'
    write_json_atomic(root / 'ground_truth_verification.json', result)
    if result['status'] != 'PASS':
        raise RuntimeError('Ground truth audit failed; STOP')
    return result


def phase_c_key(row):
    return (row['system'], row['probes'], row['round'], row['query_id'])


def phase_c_validate_checkpoint(rows, expected, manifest, neighbors):
    keys = [phase_c_key(row) for row in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise RuntimeError('Duplicate or unexpected execution key')
    fields = {'phase', 'experiment', 'smoke', 'system', 'probes', 'round', 'query_id',
              'result_ids', 'recall_at_10', 'latency_us', 'returned_rows', 'binary_sha256'}
    for row in rows:
        binary = 'pristine' if row['system'] == 'pristine' else 'optimized'
        ids = row['result_ids']
        if (set(row) != fields or row['phase'] != 'C' or row['experiment'] != 'phase_c'
                or row['smoke'] is not True or row['returned_rows'] != 10
                or len(ids) != 10 or len(set(ids)) != 10
                or any(type(i) is not int or not 0 <= i < 1000000 for i in ids)
                or row['binary_sha256'] != manifest[binary + '_binary_sha256']
                or not math.isfinite(row['latency_us']) or row['latency_us'] <= 0
                or row['recall_at_10'] != len(set(ids) & set(map(int, neighbors[row['query_id']]))) / 10):
            raise RuntimeError('Invalid checkpoint row')
    return keys


def validate_phase_c_options(args):
    if args.part2:
        if args.part1 or (args.dataset, args.queries, args.warmup, args.rounds, tuple(args.probes_list), args.lists, args.topk) != ('gist-l2', 1000, 100, 10, PROBES, 1000, 10):
            raise ValueError('Part 2 requires five systems, GIST1M, 1000 queries, 100 warmups, ten rounds and all nine probes')
        if args.stop_after or not 0 <= args.stop_after_configs < 450:
            raise ValueError('Part 2 interruption uses --stop-after-configs in 0..449')
        return
    if not args.part1 or (args.dataset, args.queries, args.warmup, args.rounds, tuple(args.probes_list), args.lists, args.topk) != ('gist-l2', 3, 1, 1, (16,), 1000, 10):
        raise ValueError('Phase C Part 1 only: --part1 --dataset gist-l2 --queries 3 --warmup 1 --rounds 1 --probes-list 16 --lists 1000 --topk 10')
    if args.stop_after < 0 or args.stop_after >= 15:
        raise ValueError('--stop-after must be 0 or between 1 and 14')


def run_formal(conn, args):
    conn.close()
    validate_phase_c_options(args)
    root = Path(args.output or ROOT / 'phase_c_artifacts').resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.phase_c.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The PostgreSQL installation is shared across output directories.
        with (ROOT / '.phase_c_database.lock').open('w') as db_lock:
            fcntl.flock(db_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.part2:
                phase_c_part2(args, root)
            else:
                phase_c_part1(args, root)


def phase_c_part1(args, root):
    manifest_path = root / 'phase_c_manifest.json'
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        metadata = json.loads((root / 'phase_c_build_manifest.json').read_text())
        supplied = os.environ.get('PRISTINE_COMMIT', '')
        if not supplied or phase_c_command(['git', '-C', SCRIPT_ROOT.parent, 'rev-parse', supplied + '^{commit}']) != manifest['pristine_commit']:
            raise RuntimeError('Resume pristine commit mismatch')
        if phase_c_command(['git', '-C', SCRIPT_ROOT.parent, 'rev-parse', 'HEAD']) != manifest['optimized_commit']:
            raise RuntimeError('Resume optimized commit mismatch')
        if phase_c_command(['git', '-C', SCRIPT_ROOT.parent, 'status', '--porcelain', '--', 'contrib/pgvector']):
            raise RuntimeError('Optimized source changed since build')
        if any(manifest.get(key) != value for key, value in metadata.items()):
            raise RuntimeError('Build metadata differs from manifest')
        if phase_c_hash(CONFIGS['gist-l2']['dataset']) != manifest['dataset_sha256']:
            raise RuntimeError('Dataset changed since audit')
        if phase_c_hash('/workspace/install/bin/postgres') != manifest['postgres_sha256']:
            raise RuntimeError('PostgreSQL binary changed since initial run')
    else:
        if manifest_path.exists():
            raise RuntimeError('Output exists; use --resume')
        metadata = phase_c_prepare(args, root)
        manifest = {'phase': 'C', 'experiment': 'phase_c', 'part': 1, 'smoke_only': True,
                    'dataset': 'GIST1M', 'dimension': 960, 'metric': 'L2', 'base_vectors': 1000000,
                    'official_queries': 1000, 'lists': 1000, 'topk': 10, 'queries': 3,
                    'warmup': 1, 'probes': [16], 'rounds': 1, 'shuffle_seeds': [20260908],
                    'systems': list(PHASE_C_SYSTEMS), **metadata}
        manifest['dataset_sha256'] = phase_c_hash(CONFIGS['gist-l2']['dataset'])
        write_json_atomic(manifest_path, manifest)
    for name in ('pristine', 'optimized'):
        if phase_c_hash(root / 'binaries' / name / 'vector.so') != manifest[name + '_binary_sha256']:
            raise RuntimeError('Resume archive hash mismatch')
    queries, neighbors = load_workload(CONFIGS['gist-l2'], 1000, 10, True)
    sql = 'SELECT id FROM gist_base ORDER BY embedding <-> %s::vector LIMIT %s'
    conn = activate_pristine(args, root, metadata)
    try:
        with conn.cursor() as cur:
            if 'index' not in manifest:
                print('Phase C building shared index once with pristine', flush=True)
                cur.execute('DROP INDEX IF EXISTS gist_ivf_l2')
                cur.execute('CREATE INDEX gist_ivf_l2 ON gist_base USING ivfflat (embedding vector_l2_ops) WITH (lists=1000)')
                cur.execute('ANALYZE gist_base')
                manifest['index'] = phase_c_index(cur)
                write_json_atomic(manifest_path, manifest)
            if phase_c_index(cur) != manifest['index']:
                raise RuntimeError('Shared index identity changed')
        audit_path = root / 'ground_truth_verification.json'
        if not audit_path.exists() or json.loads(audit_path.read_text()).get('status') != 'PASS':
            phase_c_audit(conn, root, queries, neighbors)
    finally:
        conn.close()
    checkpoint = root / 'phase_c_smoke_checkpoint.json'
    rows = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    expected = {(s, 16, 0, q) for s in PHASE_C_SYSTEMS for q in range(3)}
    keys = phase_c_validate_checkpoint(rows, expected, manifest, neighbors)
    resume_before = list(keys)
    executed = []
    mapping = {}
    plans = {}
    for system in PHASE_C_SYSTEMS:
        binary = 'pristine' if system == 'pristine' else 'optimized'
        conn = phase_c_activate(args, root, metadata, binary)
        try:
            with conn.cursor() as cur:
                if phase_c_hash(metadata['installed_so']) != manifest[binary + '_binary_sha256']:
                    raise RuntimeError('Active binary hash mismatch before config')
                if phase_c_index(cur) != manifest['index']:
                    raise RuntimeError('Shared index changed after switch')
                mapping[system] = {'binary': binary, 'sha256': manifest[binary + '_binary_sha256'], 'gucs': phase_c_settings(cur, system, 16)}
                cur.execute('EXPLAIN (FORMAT JSON) ' + sql, (vector_literal(queries[0]), 10))
                plans[system] = cur.fetchone()[0]
                if 'gist_ivf_l2' not in json.dumps(plans[system]) or 'Index Scan' not in json.dumps(plans[system]):
                    raise RuntimeError('Smoke did not select shared ANN index')
                missing = [q for q in range(3) if (system, 16, 0, q) not in keys]
                if missing:
                    formal_query(cur, sql, vector_literal(queries[0]), 10)
                random.Random(20260908).shuffle(missing)
                for qid in missing:
                    conn.notices.clear()
                    ids, latency = formal_query(cur, sql, vector_literal(queries[qid]), 10)
                    if len(ids) != 10 or len(set(ids)) != 10 or any('IVFFLAT_PROFILE' in n for n in conn.notices):
                        raise RuntimeError('Invalid result count or profiling NOTICE')
                    gt = {int(i) for i in neighbors[qid]}
                    row = {'phase': 'C', 'experiment': 'phase_c', 'smoke': True, 'system': system,
                           'probes': 16, 'round': 0, 'query_id': qid, 'result_ids': ids,
                           'recall_at_10': len(set(ids) & gt) / 10, 'latency_us': latency,
                           'returned_rows': len(ids), 'binary_sha256': manifest[binary + '_binary_sha256']}
                    rows.append(row)
                    keys.append(phase_c_key(row))
                    executed.append(phase_c_key(row))
                    write_json_atomic(checkpoint, rows)
                    print(f'Phase C smoke {system} query={qid} recall={row["recall_at_10"]}', flush=True)
                    if args.stop_after and len(executed) >= args.stop_after:
                        write_json_atomic(root / 'phase_c_interruption.json', {'checkpoint_keys': keys, 'stop_after': args.stop_after, 'status': 'interrupted'})
                        raise SystemExit(75)
        finally:
            conn.close()
    if set(keys) != expected or len(rows) != 15:
        raise RuntimeError('Missing checkpoint keys')
    comparisons = []
    lookup = {phase_c_key(r): r for r in rows}
    for left, right in [('pristine', 'vanilla_eq'), ('phase2b', 'vanilla_eq'), ('final', 'phase2a')]:
        mismatches, deltas = 0, []
        for qid in range(3):
            a, b = lookup[(left, 16, 0, qid)], lookup[(right, 16, 0, qid)]
            mismatches += int(a['result_ids'] != b['result_ids'])
            deltas.append(a['recall_at_10'] - b['recall_at_10'])
        comparisons.append({'left': left, 'right': right, 'top10_mismatch': mismatches, 'recall_deltas': deltas})
    interrupted = root / 'phase_c_interruption.json'
    resume_pass = args.resume and interrupted.exists() and bool(resume_before) and not (set(resume_before) & set(executed)) and set(resume_before) | set(executed) == expected
    report = {'decision': 'PASS' if resume_pass and all(c['top10_mismatch'] == 0 and all(d == 0 for d in c['recall_deltas']) for c in comparisons) else 'FAIL',
              'ground_truth': json.loads((root / 'ground_truth_verification.json').read_text()),
              'mapping': mapping, 'comparisons': comparisons, 'rows': len(rows),
              'resume': {'status': 'PASS' if resume_pass else 'FAIL', 'previous_keys': resume_before,
                         'executed_keys': executed, 'duplicates': 0, 'missing': 0, 'active_binary_revalidated': True},
              'index': manifest['index'], 'part2_executed': False}
    write_json_atomic(root / 'phase_c_smoke_plans.json', plans)
    write_json_atomic(root / 'phase_c_part1_report.json', report)
    write_csv(root / 'phase_c_smoke_raw.csv', [{**r, 'result_ids': json.dumps(r['result_ids'])} for r in rows])
    if report['decision'] != 'PASS':
        raise RuntimeError('Part 1 validation failed; formal benchmark forbidden')
    print('PASS: Phase C infrastructure validated; Part 2 was not run.', flush=True)


# Phase C formal execution uses the exact artifacts/configuration validated in Part 1.
PHASE_C_RAW_FIELDS = ('phase', 'experiment', 'system', 'binary_family', 'dataset',
    'dimension', 'metric', 'source_commit', 'binary_sha256', 'lists', 'probes',
    'topk', 'round', 'config_position', 'query_order_position', 'query_id',
    'latency_us', 'returned_rows', 'result_ids', 'result_checksum', 'recall_at_10')
PHASE_C_ROUND_FIELDS = ('system', 'binary_family', 'probes', 'round', 'config_position',
    'queries', 'warmup_queries', 'p50_us', 'p95_us', 'p99_us', 'mean_us', 'wall_seconds',
    'qps', 'mean_recall_at_10', 'minimum_returned_rows')


def phase_c_json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def phase_c_schedule():
    permutations, schedule = [], []
    optimized = list(PHASE_C_SYSTEMS[1:])
    for round_no in range(1, 11):
        seed = 20260908 + round_no
        queries = list(range(1000))
        random.Random(seed).shuffle(queries)
        probes = list(PROBES)
        random.Random(seed + 100000).shuffle(probes)
        rotation = (round_no - 1) % 4
        systems = optimized[rotation:] + optimized[:rotation]
        families = ['pristine', 'optimized'] if round_no % 2 else ['optimized', 'pristine']
        permutations.append(dict(round=round_no, seed=seed, permutation=queries,
            permutation_sha256=phase_c_json_hash(queries), probes_order=probes,
            family_order=families, optimized_system_order=systems))
        for family in families:
            for system in (['pristine'] if family == 'pristine' else systems):
                for probe in probes:
                    schedule.append(dict(system=system, binary_family=family, probes=probe,
                                         round=round_no, config_position=len(schedule) + 1))
    return permutations, schedule


def phase_c_formal_manifest(args, root):
    manifest_path = root / 'phase_c_manifest.json'
    previous = json.loads(manifest_path.read_text()) if args.resume else None
    if manifest_path.exists() and not args.resume:
        raise RuntimeError('Formal output exists; use --resume')
    source = args.part1_artifacts or (previous['part1_artifacts'] if previous else None)
    if not source:
        raise RuntimeError('--part1-artifacts must identify the validated Part 1 artifact directory')
    source = Path(source).resolve()
    part1 = json.loads((source / 'phase_c_manifest.json').read_text())
    report = json.loads((source / 'phase_c_part1_report.json').read_text())
    audit = json.loads((source / 'ground_truth_verification.json').read_text())
    if (part1['phase'], part1['part'], report['decision'], audit['status']) != ('C', 1, 'PASS', 'PASS') or audit['exact_matches'] != 20:
        raise RuntimeError('Part 1 PASS and successful ground-truth audit required')
    metadata = json.loads((source / 'phase_c_build_manifest.json').read_text())
    if any(part1.get(k) != v for k, v in metadata.items()):
        raise RuntimeError('Part 1 build provenance mismatch')
    supplied = os.environ.get('PRISTINE_COMMIT')
    if supplied and phase_c_command(['git', '-C', SCRIPT_ROOT.parent, 'rev-parse', supplied + '^{commit}']) != part1['pristine_commit']:
        raise RuntimeError('PRISTINE_COMMIT differs from validated binary')
    for binary in ('pristine', 'optimized'):
        if phase_c_hash(source / 'binaries' / binary / 'vector.so') != part1[binary + '_binary_sha256']:
            raise RuntimeError('Part 1 binary changed')
    # A framework-only commit after Part 1 must not relabel the binary source commit.
    if phase_c_command(['git', '-C', SCRIPT_ROOT.parent, 'diff', part1['optimized_commit'], '--', 'contrib/pgvector', 'src']):
        raise RuntimeError('Algorithm/server source differs from Part 1')
    pg_config = os.environ.get('PG_CONFIG', '/workspace/install/bin/pg_config')
    if phase_c_hash(Path(pg_config).with_name('postgres')) != part1['postgres_sha256']:
        raise RuntimeError('PostgreSQL executable changed since Part 1')
    if phase_c_hash(CONFIGS['gist-l2']['dataset']) != part1['dataset_sha256']:
        raise RuntimeError('Dataset changed since Part 1 audit')
    permutations, schedule = phase_c_schedule()
    manifest = {**metadata, 'phase': 'C', 'experiment': 'phase_c', 'part': 2,
        'smoke_only': False, 'dataset': 'GIST1M', 'dimension': 960, 'metric': 'L2',
        'lists': 1000, 'topk': 10, 'probes': list(PROBES), 'rounds': 10,
        'systems': list(PHASE_C_SYSTEMS), 'unique_queries': 1000, 'warmup_queries': 100,
        'expected_rows': 450000, 'index': part1['index'], 'part1_artifacts': str(source),
        'dataset_sha256': part1['dataset_sha256'], 'part1_report_sha256': phase_c_hash(source / 'phase_c_part1_report.json'),
        'ground_truth_sha256': phase_c_hash(source / 'ground_truth_verification.json'),
        'runner_sha256': phase_c_hash(__file__), 'query_permutations': permutations,
        'run_order': schedule, 'seed_base': 20260908,
        'sql': f"SELECT id FROM {CONFIGS['gist-l2']['table']} ORDER BY embedding <-> %s::vector LIMIT 10",
        'timing_boundary': 'execute + fetchall; literal, Recall, checksum and serialization excluded',
        'qps_definition': '1000 / measured sequential loop wall seconds; excludes warmup and checkpoint writes, includes per-query Python bookkeeping',
        'recall_sampling': '1000 unique queries repeated over 10 execution rounds',
        'database': {key: getattr(args, key) for key in ('host', 'port', 'dbname', 'user')}}
    if previous is not None and previous != manifest:
        differing = [k for k in set(previous) | set(manifest) if previous.get(k) != manifest.get(k)]
        raise RuntimeError('Reject resume: manifest mismatch in ' + ', '.join(differing))
    if previous is None:
        for binary in ('pristine', 'optimized'):
            target = root / 'binaries' / binary
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / 'binaries' / binary / 'vector.so', target / 'vector.so')
            shutil.copy2(source / f'{binary}_compile_command.txt', root / f'{binary}_compile_command.txt')
            (root / f'{binary}_commit.txt').write_text(part1[binary + '_commit'] + '\n')
            (root / f'{binary}_vector.so.sha256').write_text(part1[binary + '_binary_sha256'] + '  vector.so\n')
        shutil.copy2(source / 'ground_truth_verification.json', root / 'ground_truth_verification.json')
        write_json_atomic(root / 'query_permutations.json', permutations)
        (root / 'run_order.txt').write_text('\n'.join(json.dumps(item, sort_keys=True) for item in schedule) + '\n')
        (root / 'index_identity.txt').write_text(json.dumps(part1['index'], indent=2) + '\n')
        environment = root.parent / 'environment.txt'
        if environment.exists():
            shutil.copy2(environment, root / 'environment.txt')
        write_json_atomic(manifest_path, manifest)
    else:
        if json.loads((root / 'query_permutations.json').read_text()) != permutations:
            raise RuntimeError('Resume permutation evidence changed')
        for binary in ('pristine', 'optimized'):
            if phase_c_hash(root / 'binaries' / binary / 'vector.so') != manifest[binary + '_binary_sha256']:
                raise RuntimeError('Formal archived binary mismatch')
    return manifest


def phase_c_block_path(root, config):
    return root / 'checkpoints' / f"{config['config_position']:03d}_{config['system']}_p{config['probes']}_r{config['round']:02d}.json"


def phase_c_validate_block(block, config, manifest, gt):
    if block.get('complete') is not True or block.get('config') != config or block.get('manifest_sha256') != phase_c_json_hash(manifest):
        raise RuntimeError('Incomplete or mismatched formal checkpoint')
    rows = block['rows']
    permutation = manifest['query_permutations'][config['round'] - 1]['permutation']
    if len(rows) != 1000 or [r['query_id'] for r in rows] != permutation:
        raise RuntimeError('Missing, duplicate or out-of-order checkpoint query keys')
    family = config['binary_family']
    fixed = {'phase': 'C', 'experiment': 'phase_c', 'dataset': 'GIST1M', 'dimension': 960,
             'metric': 'L2', 'source_commit': manifest[family + '_commit'],
             'binary_sha256': manifest[family + '_binary_sha256'], 'lists': 1000, 'topk': 10, **config}
    for position, row in enumerate(rows):
        if set(row) != set(PHASE_C_RAW_FIELDS) or any(row[k] != v for k, v in fixed.items()) or row['query_order_position'] != position:
            raise RuntimeError('Formal row schema/config mismatch')
        ids = [int(i) for i in row['result_ids'].split(';')] if row['result_ids'] else []
        if (len(ids) > 10 or len(set(ids)) != len(ids) or row['returned_rows'] != len(ids)
                or any(i < 0 or i >= 1000000 for i in ids)
                or hashlib.sha256(row['result_ids'].encode()).hexdigest() != row['result_checksum']
                or not math.isfinite(row['latency_us']) or row['latency_us'] <= 0
                or row['recall_at_10'] != len(set(ids) & gt[row['query_id']]) / 10):
            raise RuntimeError('Invalid formal result, checksum, latency or Recall')
    summary = block['summary']
    if (set(summary) != set(PHASE_C_ROUND_FIELDS) or any(summary[k] != v for k, v in config.items())
            or summary['queries'] != 1000 or summary['warmup_queries'] != 100
            or summary['wall_seconds'] <= 0 or summary['qps'] != 1000 / summary['wall_seconds']):
        raise RuntimeError('Invalid formal summary metadata')
    if block['rows_sha256'] != phase_c_json_hash(rows):
        raise RuntimeError('Formal checkpoint content hash mismatch')
    return rows


def phase_c_append_csv(path, rows, fields):
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def phase_c_rebuild_csv(root, blocks):
    # Canonical atomic checkpoints recover derived CSVs if interruption occurred during append.
    def raw_rows():
        for path in blocks:
            yield from json.loads(path.read_text())['rows']
    def summaries():
        for path in blocks:
            yield json.loads(path.read_text())['summary']
    write_csv_atomic(root / 'phase_c_raw.csv', raw_rows(), PHASE_C_RAW_FIELDS)
    write_csv_atomic(root / 'phase_c_rounds.csv', summaries(), PHASE_C_ROUND_FIELDS)


def phase_c_part2(args, root):
    manifest = phase_c_formal_manifest(args, root)
    queries, neighbors = load_workload(CONFIGS['gist-l2'], 1000, 10, True)
    literals = [vector_literal(q) for q in queries]
    gt = [set(map(int, row)) for row in neighbors]
    schedule = manifest['run_order']
    expected_paths = {phase_c_block_path(root, c) for c in schedule}
    (root / 'checkpoints').mkdir(exist_ok=True)
    if set((root / 'checkpoints').glob('*.json')) - expected_paths:
        raise RuntimeError('Unexpected checkpoint files')
    completed, paths = set(), []
    for config in schedule:
        path = phase_c_block_path(root, config)
        if path.exists():
            phase_c_validate_block(json.loads(path.read_text()), config, manifest, gt)
            completed.add(config['config_position'])
            paths.append(path)
    phase_c_rebuild_csv(root, paths)
    if args.resume:
        with (root / 'phase_c_resume.jsonl').open('a') as log:
            log.write(json.dumps({'time': time.time(), 'validated_blocks': len(completed), 'validated_rows': len(completed) * 1000, 'manifest_sha256': phase_c_json_hash(manifest)}) + '\n')
    initial_completed = len(completed)
    conn, active_family = None, None
    try:
        for config in schedule:
            if config['config_position'] in completed:
                continue
            family = config['binary_family']
            if family != active_family:
                if conn is not None:
                    conn.close()
                conn = phase_c_activate(args, root, manifest, family)
                active_family = family
                with conn.cursor() as cur:
                    if phase_c_index(cur) != manifest['index']:
                        raise RuntimeError('STOP: index identity changed after binary switch')
            with conn.cursor() as cur:
                if phase_c_hash(manifest['installed_so']) != manifest[family + '_binary_sha256']:
                    raise RuntimeError('STOP: active binary hash changed before configuration')
                if phase_c_index(cur) != manifest['index']:
                    raise RuntimeError('STOP: index identity changed before configuration')
                gucs = phase_c_settings(cur, config['system'], config['probes'])
                permutation = manifest['query_permutations'][config['round'] - 1]['permutation']
                cur.execute('EXPLAIN (FORMAT JSON) ' + manifest['sql'], (literals[permutation[0]],))
                plan = cur.fetchone()[0]
                if CONFIGS['gist-l2']['index'] not in json.dumps(plan) or 'Index Scan' not in json.dumps(plan):
                    raise RuntimeError('Formal query did not select shared ANN index')
                print(f"Phase C config {config['config_position']}/450 {config} warmup=100", flush=True)
                for qid in permutation[:100]:
                    cur.execute(manifest['sql'], (literals[qid],))
                    cur.fetchall()
                conn.notices.clear()
                fixed = {'phase': 'C', 'experiment': 'phase_c', 'dataset': 'GIST1M',
                    'dimension': 960, 'metric': 'L2', 'source_commit': manifest[family + '_commit'],
                    'binary_sha256': manifest[family + '_binary_sha256'], 'lists': 1000, 'topk': 10, **config}
                rows = []
                loop_started = time.perf_counter_ns()
                for position, qid in enumerate(permutation):
                    literal = literals[qid]
                    started = time.perf_counter_ns()
                    cur.execute(manifest['sql'], (literal,))
                    fetched = cur.fetchall()
                    latency_us = (time.perf_counter_ns() - started) / 1000
                    ids = [int(row[0]) for row in fetched]
                    result_ids = ';'.join(map(str, ids))
                    rows.append({**fixed, 'query_order_position': position, 'query_id': qid,
                        'latency_us': latency_us, 'returned_rows': len(ids), 'result_ids': result_ids,
                        'result_checksum': hashlib.sha256(result_ids.encode()).hexdigest(),
                        'recall_at_10': len(set(ids) & gt[qid]) / 10})
                wall_seconds = (time.perf_counter_ns() - loop_started) / 1e9
                if any('IVFFLAT_PROFILE' in n for n in conn.notices):
                    raise RuntimeError('Profiling NOTICE during formal execution')
                latencies = [r['latency_us'] for r in rows]
                summary = {**config, 'queries': 1000, 'warmup_queries': 100,
                    'p50_us': percentile(latencies, 50), 'p95_us': percentile(latencies, 95),
                    'p99_us': percentile(latencies, 99), 'mean_us': statistics.fmean(latencies),
                    'wall_seconds': wall_seconds, 'qps': 1000 / wall_seconds,
                    'mean_recall_at_10': statistics.fmean(r['recall_at_10'] for r in rows),
                    'minimum_returned_rows': min(r['returned_rows'] for r in rows)}
                block = {'complete': True, 'config': config, 'manifest_sha256': phase_c_json_hash(manifest),
                    'rows': rows, 'rows_sha256': phase_c_json_hash(rows), 'summary': summary,
                    'gucs': gucs, 'plan': plan, 'index': manifest['index'], 'completed_at': time.time()}
                phase_c_validate_block(block, config, manifest, gt)
                write_json_atomic(phase_c_block_path(root, config), block)
                phase_c_append_csv(root / 'phase_c_raw.csv', rows, PHASE_C_RAW_FIELDS)
                phase_c_append_csv(root / 'phase_c_rounds.csv', [summary], PHASE_C_ROUND_FIELDS)
                completed.add(config['config_position'])
                write_json_atomic(root / 'phase_c_progress.json', {'status': 'RUNNING', 'completed_configs': len(completed), 'expected_configs': 450, 'measured_rows': len(completed) * 1000, 'last_config': config, 'updated_at': time.time()})
                print(f"Phase C checkpoint {len(completed)}/450 rows={len(completed)*1000} wall={wall_seconds:.3f}s", flush=True)
                if args.stop_after_configs and len(completed) - initial_completed >= args.stop_after_configs:
                    write_json_atomic(root / 'phase_c_interruption.json', {'status': 'INTERRUPTED', 'completed_configs': len(completed), 'measured_rows': len(completed) * 1000})
                    raise SystemExit(75)
    except Exception as exc:
        write_json_atomic(root / 'phase_c_failure.json', {'status': 'FAILED', 'error': str(exc), 'completed_configs': len(completed), 'time': time.time()})
        raise
    finally:
        if conn is not None:
            conn.close()
    # Validate every canonical row before publishing COMPLETE. No performance conclusions here.
    seen = set()
    minimum = 10
    paths = []
    for config in schedule:
        path = phase_c_block_path(root, config)
        block = json.loads(path.read_text())
        for row in phase_c_validate_block(block, config, manifest, gt):
            key = phase_c_key(row)
            if key in seen:
                raise RuntimeError('Duplicate formal execution key')
            seen.add(key)
            minimum = min(minimum, row['returned_rows'])
        paths.append(path)
    expected = {(s, p, r, q) for s in PHASE_C_SYSTEMS for p in PROBES for r in range(1, 11) for q in range(1000)}
    missing = len(expected - seen)
    if missing or len(seen) != 450000:
        raise RuntimeError('Formal execution incomplete')
    phase_c_rebuild_csv(root, paths)
    report = {'decision': 'COMPLETE', 'systems': 5, 'probes': 9, 'rounds': 10,
        'unique_queries': 1000, 'measured_per_system_probes': 10000,
        'expected_rows': 450000, 'actual_rows': len(seen), 'duplicate_keys': 0, 'missing_keys': missing,
        'minimum_returned_rows': minimum, 'invalid_result_checksums': 0, 'query_execution_errors': 0,
        'checkpoint_blocks': 450, 'resume_validation': 'PASS' if args.resume else 'available; not exercised',
        'pristine_binary_sha256': manifest['pristine_binary_sha256'],
        'optimized_binary_sha256': manifest['optimized_binary_sha256'],
        'family_balance': {'pristine_first_rounds': 5, 'optimized_first_rounds': 5},
        'part3_executed': False, 'completed_at': time.time()}
    write_json_atomic(root / 'phase_c_part2_report.json', report)
    write_json_atomic(root / 'phase_c_progress.json', {'status': 'COMPLETE', 'completed_configs': 450, 'measured_rows': 450000})
    print('COMPLETE: 450000 measured rows; no duplicates or missing keys. Part 3 was not run.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="taskdb")
    parser.add_argument("--user", default="dev")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--dataset", choices=["all", *CONFIGS], default="all")
    build.add_argument("--lists", type=int, default=1000)
    build.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--phase", choices=("a", "b", "2a", "2a2", "2a34", "2b", "2b-correctness", "2b-production", "formal", "all-fomal-exp"), required=True)
    run.add_argument("--dataset", choices=tuple(FORMAL_DATASET_LABELS), default="glove-cosine")
    run.add_argument("--warmup", "--warmup-queries", dest="warmup", type=int, default=100)
    run.add_argument("--queries", type=int, default=1000)
    run.add_argument("--topk", type=int, default=10)
    run.add_argument("--probes-list", type=parse_int_list, default=(16, 64, 128))
    run.add_argument("--filter-divisors", type=parse_int_list, default=(1, 2, 10, 100, 1000, 10000))
    run.add_argument("--sort-bounds", type=parse_int_list, default=(0, 10, 20, 40, 100))
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--lists", type=int, default=1000)
    run.add_argument("--mode", choices=("full", "auto", "both"),
                     default="full")
    run.add_argument("--distance-path",
                     choices=("generic", "direct", "fused2", "interleaved"),
                     default="generic")
    run.add_argument("--baseline-path", choices=("generic", "direct", "fused2"), default="generic")
    run.add_argument("--test-path", choices=("generic", "direct", "fused2"), default="direct")
    run.add_argument("--check-fused2-edges", action="store_true")
    run.add_argument("--validate-only", action="store_true", help="validate run options without connecting to PostgreSQL")
    run.add_argument("--unsupported-queries", type=int, default=10)
    run.add_argument("--output", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--part2", action="store_true", help="Execute the fixed 450000-query Phase C workload")
    run.add_argument("--part1-artifacts", type=Path)
    run.add_argument("--stop-after-configs", type=int, default=0, help="Interrupt after N complete formal configs, exit 75")
    run.add_argument("--part1", action="store_true", help="Run only Phase C infrastructure smoke")
    run.add_argument("--stop-after", type=int, default=0, help="Interrupt after N durable smoke rows, exit 75")
    run.add_argument("--verify-ground-truth", action="store_true")
    run.add_argument("--ground-truth-queries", type=int, default=20)
    args = parser.parse_args()
    if args.command == "build":
        build_indexes(args)
    else:
        validate_phase2b_options(args)
        if args.phase == "formal":
            validate_phase_c_options(args)
        if args.validate_only:
            if args.phase == "all-fomal-exp":
                import sys
                d2p_path = str(Path(__file__).resolve().parents[1] / "d2p")
                if d2p_path not in sys.path:
                    sys.path.insert(0, d2p_path)
                from formal_runner import validate_holdout
                validate_holdout(args.output)
            print("run options valid")
            return
        run_experiment(args)


if __name__ == "__main__":
    main()
