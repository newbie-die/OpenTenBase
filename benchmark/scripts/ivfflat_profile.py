#!/usr/bin/env python3
"""Run reproducible IVFFlat profiling experiments, including Phase 2A."""

import argparse
import csv
import json
import os
import random
import re
import statistics
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
    raise RuntimeError("No IVFFLAT_PROFILE notice; build with PG_CFLAGS=-DIVFFLAT_BENCH")


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


def run_phase2b(conn, args):
    config = CONFIGS["gist-l2"]
    queries, _ = load_workload(config, max(args.warmup, args.queries), args.topk)
    raw_rows = []
    with conn.cursor() as cur:
        ensure_index(cur, config)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET ivfflat.iterative_scan = off")
        cur.execute("SELECT set_config('ivfflat.experimental_sort_bound', '0', false)")
        cur.execute("SELECT set_config('ivfflat.bounded_scan', %s, false)",
                    ("on" if args.mode == "auto" else "off",))
        cur.execute("SELECT set_config('ivfflat.bound_overfetch', '4', false)")
        cur.execute("SELECT set_config('ivfflat.bound_min', '40', false)")
        cur.execute("SELECT set_config('ivfflat.bound_fastpath_limit', '100', false)")
        for probes in args.probes_list:
            cur.execute("SELECT set_config('ivfflat.probes', %s, false)", (str(probes),))
            for query in queries[:args.warmup]:
                execute_query(conn, cur, config, query, args.topk, False)
            for query_id, query in enumerate(queries[:args.queries]):
                conn.notices.clear()
                _, _, profile = execute_query(conn, cur, config, query, args.topk, True)
                row = parse_profile_2b(conn.notices)
                row.update(profile)
                row.update(query_id=query_id, mode=args.mode)
                raw_rows.append(row)
    summaries = []
    for probes in args.probes_list:
        group = [row for row in raw_rows if row["probes"] == probes]
        total = lambda key: sum(row[key] for row in group)
        pages = total("scanned_pages")
        candidates = total("scanned_candidates")
        scan_ns = total("scan_items_total_ns")
        summary = {"mode": args.mode, "probes": probes, "queries": len(group),
                   "scanned_pages": pages, "scanned_candidates": candidates,
                   "distance_calls": total("distance_calls"),
                   "avg_pages_per_query": pages / len(group),
                   "avg_candidates_per_query": candidates / len(group),
                   "distance_ns_per_candidate": total("distance_ns") / candidates,
                   "candidate_extract_ns": total("candidate_extract_ns"),
                   "distance_ns": total("distance_ns"),
                   "tuple_materialization_ns": total("tuple_materialization_ns"),
                   "sort_insert_ns": total("sort_insert_ns"),
                   "sort_finalize_ns": total("sort_finalize_ns"),
                   "scan_items_total_ns": scan_ns,
                   "distance_pct": 100 * total("distance_ns") / scan_ns,
                   "candidate_extraction_pct": 100 * total("candidate_extract_ns") / scan_ns,
                   "tuple_materialization_pct": 100 * total("tuple_materialization_ns") / scan_ns,
                   "sort_insertion_pct": 100 * total("sort_insert_ns") / scan_ns,
                   "sort_finalize_pct": 100 * total("sort_finalize_ns") / scan_ns,
                   "average_candidates_per_page": candidates / pages,
                   "max_candidates_per_page": max(row["max_candidates_per_page"] for row in group)}
        for bucket in (0, 1, 2, 3):
            count = total(f"page_candidates_{bucket}")
            summary[f"page_candidates_{bucket}"] = count
            summary[f"page_candidates_{bucket}_pct"] = 100 * count / pages
        count = total("page_candidates_4plus")
        summary["page_candidates_4plus"] = count
        summary["page_candidates_4plus_pct"] = 100 * count / pages
        summaries.append(summary)
    output = Path(args.output or ROOT / "results/phase_2b_profile")
    write_csv(Path(f"{output}_raw.csv"), raw_rows)
    write_csv(Path(f"{output}_summary.csv"), summaries)


def run_experiment(args):
    conn = connect(args)
    conn.autocommit = True
    if args.phase != "formal":
        with conn.cursor() as cur:
            cur.execute("SET client_min_messages = info")
    if args.phase == "formal":
        run_formal(conn, args)
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
    run.add_argument("--phase", choices=("a", "b", "2a", "2a2", "2a34", "2b", "formal"), required=True)
    run.add_argument("--dataset", choices=tuple(FORMAL_DATASET_LABELS), default="glove-cosine")
    run.add_argument("--warmup", "--warmup-queries", dest="warmup", type=int, default=100)
    run.add_argument("--queries", type=int, default=1000)
    run.add_argument("--topk", type=int, default=10)
    run.add_argument("--probes-list", type=parse_int_list, default=(16, 64, 128))
    run.add_argument("--filter-divisors", type=parse_int_list, default=(1, 2, 10, 100, 1000, 10000))
    run.add_argument("--sort-bounds", type=parse_int_list, default=(0, 10, 20, 40, 100))
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--lists", type=int, default=1000)
    run.add_argument("--mode", choices=("full", "auto"), default="full")
    run.add_argument("--output", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--verify-ground-truth", action="store_true")
    run.add_argument("--ground-truth-queries", type=int, default=20)
    args = parser.parse_args()
    if args.command == "build":
        build_indexes(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
