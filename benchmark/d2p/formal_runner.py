"""Canonical D2-P one-round formal workload implementation."""
import csv
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

DEFAULT_DATASET = Path("/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5")
POLICY = Path("/workspace/OpenTenBase/benchmark/d2p/artifacts/d2p_policy.json")
STATIC = Path("/workspace/OpenTenBase/benchmark/d2p/artifacts/static_tuned.json")
CONFIGS = ("fixed16", "fixed32", "fixed64", "static_tuned", "d2p")
FIELDS = ("qid", "config", "latency_ms", "latency_us", "recall_at_10", "probes", "candidates",
          "pages", "distance_calls", "stop_stage", "result_ids")
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE_2B (.*)")
PAIR_RE = re.compile(r"([a-z0-9_]+)=([^ ]+)")


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_path():
    return Path(os.environ.get("D2P_FORMAL_HDF5", str(DEFAULT_DATASET)))


def validate_holdout(output=None):
    import h5py
    path = dataset_path()
    result = {"dataset": str(path), "required_qids": [1000, 9999], "status": "FAIL"}
    if not path.exists():
        result["reason"] = "dataset does not exist"
    else:
        with h5py.File(path, "r") as source:
            result["test_rows"] = int(source["test"].shape[0])
            result["neighbor_rows"] = int(source["neighbors"].shape[0])
            result["dimensions"] = int(source["test"].shape[1])
        if result["test_rows"] < 10000 or result["neighbor_rows"] < 10000:
            result["reason"] = "GIST holdout qids 1000..9999 and matching GT are absent"
        elif result["dimensions"] != 960:
            result["reason"] = "holdout dimension does not match the existing GIST1M index"
        else:
            result["status"] = "PASS"
    if output:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "holdout_preflight.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(result["reason"] + f"; checked {path}")
    return result


def vector_literal(vector):
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise RuntimeError(f"checkpoint schema mismatch: {path}")
        rows = list(reader)
    qids = [int(row["qid"]) for row in rows]
    if qids != list(range(1000, 1000 + len(qids))):
        raise RuntimeError(f"checkpoint qids are not a durable prefix: {path}")
    return rows


def summarize(rows, baseline_recall=None):
    recalls = [float(row["recall_at_10"]) for row in rows]
    latencies = [float(row["latency_us"]) for row in rows]
    mean_recall = statistics.fmean(recalls)
    stops = {str(stage): sum(int(row["stop_stage"]) == stage for row in rows)
             for stage in (16, 32, 64)}
    return {
        "queries": len(rows), "mean_recall_at_10": mean_recall,
        "recall_delta_vs_fixed64": None if baseline_recall is None else mean_recall - baseline_recall,
        "latency_mean_us": statistics.fmean(latencies), "latency_p50_us": percentile(latencies, .5),
        "latency_p95_us": percentile(latencies, .95), "latency_p99_us": percentile(latencies, .99),
        "qps": 1e6 / statistics.fmean(latencies),
        "average_probes": statistics.fmean(float(row["probes"]) for row in rows),
        "average_candidates": statistics.fmean(float(row["candidates"]) for row in rows),
        "average_pages": statistics.fmean(float(row["pages"]) for row in rows),
        "average_distance_calls": statistics.fmean(float(row["distance_calls"]) for row in rows),
        "stop_distribution": stops,
        "stop16_ratio": stops["16"] / len(rows), "stop32_ratio": stops["32"] / len(rows),
        "fallback64_ratio": stops["64"] / len(rows),
    }


def run(conn, args):
    import h5py
    output = args.output
    if args.queries != 9000 or args.topk != 10 or args.lists != 1000 or args.warmup < 0:
        raise ValueError("all-fomal-exp requires queries=9000, topk=10, lists=1000 and nonnegative warmup")
    output.mkdir(parents=True, exist_ok=True)
    preflight = validate_holdout(output)
    if not POLICY.exists() or not STATIC.exists():
        raise RuntimeError("frozen d2p_policy.json/static_tuned.json are missing")
    policy = json.loads(POLICY.read_text())
    static = json.loads(STATIC.read_text())
    p_static = int(static["p_static"])
    manifest = {
        "status": "RUNNING", "phase": "all-fomal-exp", "rounds": 1,
        "qid_range": [1000, 9999], "query_order": "ascending", "configs": list(CONFIGS),
        "warmup_per_config": args.warmup, "limit": 10, "lists": 1000,
        "dataset": preflight, "policy_sha256": sha256(POLICY),
        "static_sha256": sha256(STATIC), "p_static": p_static,
        "phase2a": False, "fused2": False,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "plan.json", {
        "config_order": list(CONFIGS), "qid_order": list(range(1000, 10000)),
        "checkpoint": "one append-flushed CSV per config; resume at next qid",
    })
    with h5py.File(dataset_path(), "r") as source:
        queries = source["test"][1000:10000]
        truths = [set(map(int, row[:10])) for row in source["neighbors"][1000:10000]]

    started = time.perf_counter()
    completed_total = 0
    summaries = {}
    sql = "SELECT id FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10"
    with conn.cursor() as cursor:
        cursor.execute("LOAD 'vector'")
        cursor.execute("SET enable_seqscan=off")
        cursor.execute("SET max_parallel_workers_per_gather=0")
        cursor.execute("SELECT reloptions FROM pg_class WHERE oid='gist_ivf_l2'::regclass")
        identity = cursor.fetchone()
        if identity is None or "lists=1000" not in (identity[0] or []):
            raise RuntimeError("existing gist_ivf_l2 lists=1000 index is required")
        for config_index, config in enumerate(CONFIGS, 1):
            config_dir = output / config
            config_dir.mkdir(parents=True, exist_ok=True)
            raw_path = config_dir / "raw.csv"
            rows = read_rows(raw_path)
            completed_total += len(rows)
            summary_path = config_dir / "summary.json"
            if len(rows) == 9000:
                if not summary_path.exists():
                    raise RuntimeError(f"complete raw checkpoint lacks summary: {config}")
                saved_summary = json.loads(summary_path.read_text())
                if (saved_summary.get("queries") != 9000 or
                        saved_summary.get("raw_sha256") != sha256(raw_path)):
                    raise RuntimeError(f"complete checkpoint validation failed: {config}")
                summaries[config] = saved_summary
                continue
            probes = p_static if config == "static_tuned" else (
                int(config.removeprefix("fixed")) if config.startswith("fixed") else 64)
            progressive = "on" if config == "d2p" else "off"
            settings = {"ivfflat.probes": probes, "ivfflat.progressive_scan": progressive,
                        "ivfflat.progressive_scan_debug": "off", "ivfflat.iterative_scan": "off",
                        "ivfflat.adaptive_probes": "off", "ivfflat.adaptive_probes_trace": "off",
                        "ivfflat.bounded_scan": "off", "ivfflat.experimental_sort_bound": 0,
                        "ivfflat.distance_path": "generic"}
            for name, value in settings.items():
                cursor.execute("SELECT set_config(%s,%s,false)", (name, str(value)))
            for warm in range(args.warmup):
                cursor.execute(sql, (vector_literal(queries[warm % len(queries)]),))
                cursor.fetchall()
            mode = "a" if raw_path.exists() else "w"
            with raw_path.open(mode, newline="", buffering=1) as target:
                writer = csv.DictWriter(target, fieldnames=FIELDS)
                if mode == "w": writer.writeheader()
                for offset in range(len(rows), 9000):
                    qid = offset + 1000
                    literal = vector_literal(queries[offset])
                    conn.notices.clear()
                    before = time.perf_counter_ns()
                    cursor.execute(sql, (literal,))
                    ids = [int(row[0]) for row in cursor.fetchall()]
                    latency = (time.perf_counter_ns() - before) / 1000.0
                    matches = [PROFILE_RE.search(notice) for notice in conn.notices]
                    matches = [match for match in matches if match]
                    if len(matches) != 1:
                        raise RuntimeError(f"qid={qid}: missing unique IVFFLAT_PROFILE_2B")
                    profile = dict(PAIR_RE.findall(matches[0].group(1)))
                    actual_probes = int(profile["probes"])
                    stop_stage = int(profile.get("progressive_stop_stage", actual_probes))
                    row = {"qid": qid, "config": config,
                           "latency_ms": format(latency / 1000.0, ".17g"),
                           "latency_us": format(latency, ".17g"),
                           "recall_at_10": len(set(ids) & truths[offset]) / 10.0,
                           "probes": actual_probes, "candidates": profile["scanned_candidates"],
                           "pages": profile["scanned_pages"], "distance_calls": profile["distance_calls"],
                           "stop_stage": stop_stage, "result_ids": ";".join(map(str, ids))}
                    writer.writerow(row); target.flush(); rows.append(row); completed_total += 1
                    elapsed = time.perf_counter() - started
                    remaining = len(CONFIGS) * 9000 - completed_total
                    progress = {"status": "RUNNING", "current_config": f"{config_index}/{len(CONFIGS)} {config}",
                                "completed_queries": completed_total,
                                "total_queries": len(CONFIGS) * 9000, "current_qid": qid,
                                "elapsed_seconds": elapsed,
                                "ETA_seconds": elapsed * remaining / completed_total if completed_total else None}
                    atomic_json(output / "progress.json", progress)
                    if len(rows) % 10 == 0:
                        print(" ".join(f"{key}={value}" for key, value in progress.items()
                                       if key != "status"), flush=True)
            summaries[config] = summarize(rows)
            summaries[config].update({"config": config, "raw_sha256": sha256(raw_path)})
            atomic_json(summary_path, summaries[config])

    fixed64_recall = summaries["fixed64"]["mean_recall_at_10"]
    comparison = []
    for config in CONFIGS:
        raw_path = output / config / "raw.csv"
        summary = summarize(read_rows(raw_path), fixed64_recall)
        summaries[config] = summary
        atomic_json(output / config / "summary.json",
                    {**summary, "config": config, "raw_sha256": sha256(raw_path)})
        comparison.append({"config": config, **summary})
    fixed64 = summaries["fixed64"]
    for row in comparison:
        row["candidate_ratio_vs_fixed64"] = row["average_candidates"] / fixed64["average_candidates"]
        row["page_ratio_vs_fixed64"] = row["average_pages"] / fixed64["average_pages"]
        row["distance_call_ratio_vs_fixed64"] = row["average_distance_calls"] / fixed64["average_distance_calls"]
    with (output / "comparison.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=[key for key in comparison[0] if key != "stop_distribution"])
        writer.writeheader()
        for row in comparison:
            writer.writerow({key: value for key, value in row.items() if key != "stop_distribution"})
    d2p, tuned = summaries["d2p"], summaries["static_tuned"]
    recall_pass = d2p["recall_delta_vs_fixed64"] >= -0.005
    mechanism_go = recall_pass and (d2p["latency_mean_us"] < tuned["latency_mean_us"] or
                                    d2p["latency_p50_us"] < tuned["latency_p50_us"])
    gate = ("D2P_RECALL_FAIL_STOP" if not recall_pass else
            ("D2P_MECHANISM_GO" if mechanism_go else "D2P_ADAPTIVITY_NO_GO"))
    result = {"summaries": summaries, "gate": gate,
              "recall_pass": recall_pass, "mechanism_go_vs_static": mechanism_go,
              "adaptivity_no_go": recall_pass and not mechanism_go,
              "d2p_vs_static_tuned": {
                  "mean_latency_ratio": d2p["latency_mean_us"] / tuned["latency_mean_us"],
                  "p50_latency_ratio": d2p["latency_p50_us"] / tuned["latency_p50_us"],
                  "candidate_ratio": d2p["average_candidates"] / tuned["average_candidates"]}}
    atomic_json(output / "comparison.json", result)
    manifest["status"] = "COMPLETE"; manifest["gate"] = gate
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "progress.json", {"status": "COMPLETE", "current_config": "5/5 d2p",
                                            "completed_queries": 45000, "total_queries": 45000,
                                            "elapsed_seconds": time.perf_counter() - started,
                                            "ETA_seconds": 0.0})
