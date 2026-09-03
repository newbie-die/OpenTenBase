#!/usr/bin/env python3
"""Run reproducible IVFFlat profiling experiments, including Phase 2A."""

import argparse
import csv
import re
import statistics
import time
from pathlib import Path

ROOT = Path("/workspace/benchmark")
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE\s+(.*)")
FIELD_RE = re.compile(r"([a-z_]+)=(-?[0-9.]+)")
PROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
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


def run_experiment(args):
    conn = connect(args)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET client_min_messages = info")
    if args.phase == "2a34":
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
    run.add_argument("--phase", choices=("a", "b", "2a", "2a2", "2a34"), required=True)
    run.add_argument("--warmup", type=int, default=100)
    run.add_argument("--queries", type=int, default=1000)
    run.add_argument("--topk", type=int, default=10)
    run.add_argument("--probes-list", type=parse_int_list, default=(16, 64, 128))
    run.add_argument("--filter-divisors", type=parse_int_list, default=(1, 2, 10, 100, 1000, 10000))
    run.add_argument("--sort-bounds", type=parse_int_list, default=(0, 10, 20, 40, 100))
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--lists", type=int, default=1000)
    run.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build_indexes(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
