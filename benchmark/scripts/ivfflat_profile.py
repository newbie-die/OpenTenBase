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
FORMAL_MODES = ("full", "auto")
FORMAL_DATASET_LABELS = {
    "glove-cosine": "glove100",
    "gist-l2": "gist1m",
}
FORMAL_RAW_FIELDS = (
    "experiment", "dataset", "dimension", "metric", "mode", "lists",
    "probes", "topk", "query_id", "latency_us", "recall_at_10",
    "returned_rows",
)
FORMAL_SUMMARY_FIELDS = (
    "dataset", "metric", "mode", "lists", "probes", "queries", "topk",
    "mean_recall_at_10", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "qps",
    "min_returned_rows", "baseline_p50_ms", "p50_improvement_pct",
    "baseline_p95_ms", "p95_improvement_pct", "baseline_p99_ms",
    "p99_improvement_pct", "baseline_qps", "qps_improvement_pct",
)
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
    os.replace(temporary, path)


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


def formal_manifest_configuration(args):
    config = CONFIGS[args.dataset]
    return {
        "experiment": "phase_formal",
        "dataset": FORMAL_DATASET_LABELS[args.dataset],
        "dimension": config["dimension"],
        "metric": config["metric"],
        "modes": list(FORMAL_MODES),
        "lists": args.lists,
        "probes": list(args.probes_list),
        "topk": args.topk,
        "queries": args.queries,
        "warmup_queries": args.warmup,
    }


def prepare_formal_manifest(path, args):
    configuration = formal_manifest_configuration(args)
    if path.exists():
        with path.open() as source:
            manifest = json.load(source)
        if not args.resume:
            raise RuntimeError(f"formal output already exists; use --resume: {path.parent}")
        if manifest.get("configuration") != configuration:
            raise RuntimeError("resume configuration does not match phase_formal_manifest.json")
        return manifest
    if args.resume:
        raise RuntimeError(f"cannot resume without manifest: {path}")
    manifest = {"configuration": configuration, "ground_truth_verification": None}
    write_json_atomic(path, manifest)
    return manifest


def formal_checkpoint_path(checkpoint_dir, dataset, metric, mode, probes):
    return checkpoint_dir / f"{dataset}_{metric}_{mode}_p{probes}.csv"


def load_formal_checkpoint(path, expected, query_count):
    if not path.exists():
        return []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FORMAL_RAW_FIELDS:
            raise RuntimeError(f"unexpected formal checkpoint schema: {path}")
        parsed = list(reader)

    rows = []
    seen = set()
    for position, row in enumerate(parsed):
        if None in row or any(value is None or value == "" for value in row.values()):
            if position == len(parsed) - 1:
                print(f"discard incomplete trailing checkpoint row: {path}", flush=True)
                continue
            raise RuntimeError(f"invalid checkpoint row in {path}")
        query_id = int(row["query_id"])
        if query_id < 0 or query_id >= query_count:
            raise RuntimeError(f"query_id out of range in {path}: {query_id}")
        if query_id in seen:
            raise RuntimeError(f"duplicate query_id in {path}: {query_id}")
        for key, value in expected.items():
            if str(row[key]) != str(value):
                raise RuntimeError(f"checkpoint config mismatch in {path}: {key}")
        float(row["latency_us"])
        float(row["recall_at_10"])
        int(row["returned_rows"])
        seen.add(query_id)
        rows.append(row)
    return rows


def append_formal_checkpoint(path, row):
    with path.open("a", newline="") as output:
        csv.DictWriter(output, fieldnames=FORMAL_RAW_FIELDS).writerow(row)


def configure_formal_scan(cur, probes, mode):
    bounded_scan = "on" if mode == "auto" else "off"
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET ivfflat.iterative_scan = off")
    cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.probes", str(probes)))
    cur.execute("SELECT set_config(%s, %s, false)",
                ("ivfflat.experimental_sort_bound", "0"))
    cur.execute("SELECT set_config(%s, %s, false)",
                ("ivfflat.bounded_scan", bounded_scan))
    cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_overfetch", "4"))
    cur.execute("SELECT set_config(%s, %s, false)", ("ivfflat.bound_min", "40"))
    cur.execute("SELECT set_config(%s, %s, false)",
                ("ivfflat.bound_fastpath_limit", "100"))

    expected = {
        "ivfflat.probes": str(probes),
        "ivfflat.iterative_scan": "off",
        "ivfflat.experimental_sort_bound": "0",
        "ivfflat.bounded_scan": bounded_scan,
    }
    actual = {}
    for setting, expected_value in expected.items():
        cur.execute(f"SHOW {setting}")
        actual[setting] = cur.fetchone()[0]
        if actual[setting] != expected_value:
            raise RuntimeError(
                f"GUC verification failed for {setting}: "
                f"expected {expected_value}, got {actual[setting]}"
            )
    print(f"formal GUC mode={mode} probes={probes} values={actual}", flush=True)


def run_formal_config(conn, args, config, query_literals, neighbors,
                      checkpoint_dir, probes, mode):
    dataset_name = FORMAL_DATASET_LABELS[args.dataset]
    expected = {
        "experiment": "phase_formal",
        "dataset": dataset_name,
        "dimension": config["dimension"],
        "metric": config["metric"],
        "mode": mode,
        "lists": args.lists,
        "probes": probes,
        "topk": args.topk,
    }
    checkpoint = formal_checkpoint_path(
        checkpoint_dir, dataset_name, config["metric"], mode, probes)
    if checkpoint.exists() and not args.resume:
        raise RuntimeError(f"formal checkpoint already exists; use --resume: {checkpoint}")
    existing = load_formal_checkpoint(checkpoint, expected, args.queries)
    write_csv_atomic(checkpoint, sorted(existing, key=lambda row: int(row["query_id"])),
                     FORMAL_RAW_FIELDS)
    completed = {int(row["query_id"]) for row in existing}
    if len(completed) == args.queries:
        print(f"formal skip complete mode={mode} probes={probes}", flush=True)
        return

    print(f"formal run mode={mode} probes={probes} "
          f"remaining={args.queries - len(completed)}", flush=True)
    sql = (f"SELECT id FROM {config['table']} "
           f"ORDER BY embedding {config['operator']} %s::vector LIMIT %s")
    with conn.cursor() as cur:
        ensure_index(cur, config)
        configure_formal_scan(cur, probes, mode)
        for query_literal in query_literals[:args.warmup]:
            formal_query(cur, sql, query_literal, args.topk)
        for query_id in range(args.queries):
            if query_id in completed:
                continue
            ids, latency_us = formal_query(
                cur, sql, query_literals[query_id], args.topk)
            ground_truth = {int(value) for value in neighbors[query_id]}
            row = {
                **expected,
                "query_id": query_id,
                "latency_us": latency_us,
                "recall_at_10": len(ground_truth.intersection(ids)) / args.topk,
                "returned_rows": len(ids),
            }
            append_formal_checkpoint(checkpoint, row)
            existing.append(row)

    write_csv_atomic(checkpoint, sorted(existing, key=lambda row: int(row["query_id"])),
                     FORMAL_RAW_FIELDS)


def formal_group_stats(rows):
    latencies_us = [float(row["latency_us"]) for row in rows]
    total_seconds = sum(latencies_us) / 1_000_000.0
    return {
        "mean_recall_at_10": statistics.fmean(
            float(row["recall_at_10"]) for row in rows),
        "p50_ms": percentile(latencies_us, 50) / 1000.0,
        "p95_ms": percentile(latencies_us, 95) / 1000.0,
        "p99_ms": percentile(latencies_us, 99) / 1000.0,
        "mean_ms": statistics.fmean(latencies_us) / 1000.0,
        "qps": len(rows) / total_seconds if total_seconds else 0.0,
        "min_returned_rows": min(int(row["returned_rows"]) for row in rows),
    }


def summarize_formal(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["probes"]), row["mode"]), []).append(row)

    output = []
    for probes in sorted({key[0] for key in grouped}):
        full_rows = grouped.get((probes, "full"))
        if not full_rows:
            continue
        baseline = formal_group_stats(full_rows)
        for mode in FORMAL_MODES:
            group = grouped.get((probes, mode))
            if not group:
                continue
            values = formal_group_stats(group)
            output.append({
                "dataset": group[0]["dataset"],
                "metric": group[0]["metric"],
                "mode": mode,
                "lists": int(group[0]["lists"]),
                "probes": probes,
                "queries": len(group),
                "topk": int(group[0]["topk"]),
                **values,
                "baseline_p50_ms": baseline["p50_ms"],
                "p50_improvement_pct": (
                    100 * (baseline["p50_ms"] - values["p50_ms"])
                    / baseline["p50_ms"] if baseline["p50_ms"] else 0.0),
                "baseline_p95_ms": baseline["p95_ms"],
                "p95_improvement_pct": (
                    100 * (baseline["p95_ms"] - values["p95_ms"])
                    / baseline["p95_ms"] if baseline["p95_ms"] else 0.0),
                "baseline_p99_ms": baseline["p99_ms"],
                "p99_improvement_pct": (
                    100 * (baseline["p99_ms"] - values["p99_ms"])
                    / baseline["p99_ms"] if baseline["p99_ms"] else 0.0),
                "baseline_qps": baseline["qps"],
                "qps_improvement_pct": (
                    100 * (values["qps"] - baseline["qps"])
                    / baseline["qps"] if baseline["qps"] else 0.0),
            })
    return output


def merge_formal_outputs(args, checkpoint_dir, output_prefix):
    config = CONFIGS[args.dataset]
    dataset_name = FORMAL_DATASET_LABELS[args.dataset]
    rows = []
    seen = set()
    for probes in args.probes_list:
        for mode in FORMAL_MODES:
            expected = {
                "experiment": "phase_formal", "dataset": dataset_name,
                "dimension": config["dimension"], "metric": config["metric"],
                "mode": mode,
                "lists": args.lists, "probes": probes, "topk": args.topk,
            }
            checkpoint = formal_checkpoint_path(
                checkpoint_dir, dataset_name, config["metric"], mode, probes)
            config_rows = load_formal_checkpoint(checkpoint, expected, args.queries)
            if len(config_rows) != args.queries:
                continue
            for row in config_rows:
                key = (row["dataset"], row["metric"], row["mode"],
                       int(row["probes"]), int(row["query_id"]))
                if key in seen:
                    raise RuntimeError(f"duplicate formal key while merging: {key}")
                seen.add(key)
                rows.append(row)

    mode_order = {mode: position for position, mode in enumerate(FORMAL_MODES)}
    rows.sort(key=lambda row: (int(row["probes"]), mode_order[row["mode"]],
                               int(row["query_id"])))
    if rows:
        write_csv_atomic(Path(f"{output_prefix}_raw.csv"), rows, FORMAL_RAW_FIELDS)
        write_csv_atomic(Path(f"{output_prefix}_summary.csv"),
                         summarize_formal(rows), FORMAL_SUMMARY_FIELDS)
    return rows


def verify_formal_production_build(conn, args, config, query_literal):
    sql = (f"SELECT id FROM {config['table']} "
           f"ORDER BY embedding {config['operator']} %s::vector LIMIT %s")
    with conn.cursor() as cur:
        configure_formal_scan(cur, args.probes_list[0], "full")
        cur.execute("SET client_min_messages = info")
        conn.notices.clear()
        try:
            formal_query(cur, sql, query_literal, args.topk)
            if any(PROFILE_RE.search(notice) for notice in conn.notices):
                raise RuntimeError(
                    "formal phase requires a production build without IVFFLAT_BENCH")
        finally:
            conn.notices.clear()
            cur.execute("RESET client_min_messages")
    print("formal production-build check: IVFFLAT_BENCH profiling is off", flush=True)


def verify_formal_ground_truth(conn, args, config, query_literals, neighbors):
    sample_count = min(args.ground_truth_queries, args.queries)
    query_ids = sorted(random.Random(20260903).sample(range(args.queries), sample_count))
    sql = (f"SELECT id FROM {config['table']} "
           f"ORDER BY embedding {config['operator']} %s::vector LIMIT %s")
    recalls = []
    matches = 0
    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_indexonlyscan = off")
        cur.execute("SET enable_bitmapscan = off")
        cur.execute("SET enable_seqscan = on")
        try:
            for query_id in query_ids:
                ids, _ = formal_query(cur, sql, query_literals[query_id], args.topk)
                ground_truth = {int(value) for value in neighbors[query_id]}
                recall = len(ground_truth.intersection(ids)) / args.topk
                recalls.append(recall)
                matches += int(set(ids) == ground_truth)
        finally:
            cur.execute("RESET enable_indexscan")
            cur.execute("RESET enable_indexonlyscan")
            cur.execute("RESET enable_bitmapscan")
            cur.execute("RESET enable_seqscan")
    result = {
        "queries": sample_count,
        "exact_set_matches": matches,
        "mean_recall_at_10": statistics.fmean(recalls),
        "query_ids": query_ids,
    }
    print(f"formal ground-truth verification: {result}", flush=True)
    if matches != sample_count:
        raise RuntimeError("ANN-Benchmarks ground truth differs from PostgreSQL exact Top-10")
    return result


def run_formal(conn, args):
    if args.queries <= 0:
        raise ValueError("--queries must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup-queries must not be negative")
    if args.ground_truth_queries <= 0:
        raise ValueError("--ground-truth-queries must be positive")
    if args.lists != 1000 or args.topk != 10:
        raise ValueError("formal phase requires lists=1000 and topk=10")
    if len(set(args.probes_list)) != len(args.probes_list):
        raise ValueError("--probes-list must not contain duplicates")
    if any(probes <= 0 or probes > args.lists for probes in args.probes_list):
        raise ValueError("formal probes must be between 1 and lists")

    config = CONFIGS[args.dataset]
    load_count = max(args.queries, args.warmup)
    queries, neighbors = load_workload(config, load_count, args.topk, True)
    if len(queries) != load_count or len(neighbors) != load_count:
        raise RuntimeError(
            f"dataset contains fewer than requested {load_count} query vectors")
    query_literals = [vector_literal(query) for query in queries]

    verify_formal_production_build(conn, args, config, query_literals[0])
    output_prefix = Path(args.output or ROOT / "results/phase_formal")
    checkpoint_dir = output_prefix.parent / f".{output_prefix.name}_checkpoints"
    manifest_path = output_prefix.parent / f"{output_prefix.name}_manifest.json"
    manifest = prepare_formal_manifest(manifest_path, args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_ground_truth:
        verification = verify_formal_ground_truth(
            conn, args, config, query_literals, neighbors)
        manifest["ground_truth_verification"] = verification
        write_json_atomic(manifest_path, manifest)

    for probes in args.probes_list:
        for mode in FORMAL_MODES:
            run_formal_config(conn, args, config, query_literals, neighbors,
                              checkpoint_dir, probes, mode)
            merge_formal_outputs(args, checkpoint_dir, output_prefix)

    rows = merge_formal_outputs(args, checkpoint_dir, output_prefix)
    expected_rows = len(args.probes_list) * len(FORMAL_MODES) * args.queries
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"formal merge incomplete: expected {expected_rows}, got {len(rows)}")
    print(f"formal complete configs={len(args.probes_list) * len(FORMAL_MODES)} "
          f"queries_per_config={args.queries} rows={len(rows)}", flush=True)


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
    if args.phase == "formal":
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
    run.add_argument("--phase", choices=("a", "b", "2a", "2a2", "2a34", "2b", "2b-correctness", "2b-production", "formal"), required=True)
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
    run.add_argument("--verify-ground-truth", action="store_true")
    run.add_argument("--ground-truth-queries", type=int, default=20)
    args = parser.parse_args()
    if args.command == "build":
        build_indexes(args)
    else:
        validate_phase2b_options(args)
        if args.validate_only:
            print("run options valid")
            return
        run_experiment(args)


if __name__ == "__main__":
    main()
