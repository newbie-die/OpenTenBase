#!/usr/bin/env python3
"""Collect one fixed-probe GIST1M calibration configuration."""
import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(os.environ.get("BENCHMARK_RUNTIME_ROOT", WORKSPACE_ROOT / "benchmark")).resolve()
DATASET = Path(os.environ.get("D2P_DATASET",
                              RUNTIME_ROOT / "data/gist1m/gist-960-euclidean.hdf5"))
# Dataset parameters share the original collector and feature definitions.
TABLE = os.environ.get("D2P_TABLE", "gist_base")
INDEX = os.environ.get("D2P_INDEX", "gist_ivf_l2")
if (TABLE, INDEX) not in (("gist_base", "gist_ivf_l2"), ("sift_base", "sift_ivf_l2")):
    raise ValueError("Only audited L2 calibration workloads are allowed")

PROFILE_RE = re.compile(r"IVFFLAT_PROFILE_2B (.*)")
FIELD_RE = re.compile(r"([a-z0-9_]+)=([^ ]+)")
FIELDS = ("query_id", "probes", "latency_us", "recall_at_10", "result_ids",
          "candidates", "pages", "distance_calls")


def vector_literal(vector):
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=int, choices=(16, 32, 40, 48, 56, 64), required=True)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="taskdb")
    parser.add_argument("--user", default="dev")
    args = parser.parse_args()
    if not 1 <= args.queries <= 1000:
        raise ValueError("--queries must be 1..1000")

    import h5py
    import psycopg2

    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / f"fixed{args.probes}.csv"
    completed = {}
    if raw.exists():
        with raw.open(newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"unexpected CSV columns in {raw}")
            completed = {int(row["query_id"]): row for row in reader}
    with h5py.File(DATASET, "r") as source:
        queries = source["test"][:args.queries]
        truth = [set(map(int, row[:10])) for row in source["neighbors"][:args.queries]]

    connection = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname,
                                  user=args.user, application_name="d2p_static_calibration")
    connection.autocommit = False
    recalls = [float(completed[qid]["recall_at_10"]) for qid in sorted(completed)]
    started = time.perf_counter()
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("SET LOCAL lock_timeout=10000")
            cursor.execute(f"LOCK TABLE {TABLE} IN SHARE MODE")
            cursor.execute("LOAD 'vector'")
            settings = {
                "ivfflat.probes": args.probes,
                "ivfflat.iterative_scan": "off",
                "ivfflat.adaptive_probes": "off",
                "ivfflat.adaptive_probes_trace": "off",
                "ivfflat.bounded_scan": "off",
                "ivfflat.experimental_sort_bound": 0,
                "ivfflat.progressive_scan": "off",
                "ivfflat.progressive_scan_debug": "off",
            }
            for name, value in settings.items():
                cursor.execute("SELECT set_config(%s,%s,true)", (name, str(value)))
            cursor.execute("SELECT set_config('ivfflat.distance_path','generic',true)")
            cursor.execute("SET LOCAL enable_seqscan=off")
            cursor.execute("SET LOCAL max_parallel_workers_per_gather=0")
            sql = (f"SELECT id FROM {TABLE} ORDER BY embedding <-> %s::vector LIMIT 10")
            mode = "a" if raw.exists() else "w"
            with raw.open(mode, newline="", buffering=1) as target:
                writer = csv.DictWriter(target, fieldnames=FIELDS)
                if mode == "w":
                    writer.writeheader()
                for qid, query in enumerate(queries):
                    if qid in completed:
                        continue
                    literal = vector_literal(query)
                    connection.notices.clear()
                    before = time.perf_counter_ns()
                    cursor.execute(sql, (literal,))
                    ids = [int(row[0]) for row in cursor.fetchall()]
                    latency_us = (time.perf_counter_ns() - before) / 1000.0
                    profiles = [PROFILE_RE.search(notice) for notice in connection.notices]
                    profiles = [match for match in profiles if match]
                    if len(profiles) != 1:
                        raise RuntimeError("expected exactly one IVFFLAT_PROFILE_2B notice")
                    profile = dict(FIELD_RE.findall(profiles[0].group(1)))
                    recall = len(set(ids) & truth[qid]) / 10.0
                    row = {
                        "query_id": qid, "probes": args.probes,
                        "latency_us": format(latency_us, ".17g"),
                        "recall_at_10": recall, "result_ids": ";".join(map(str, ids)),
                        "candidates": profile["scanned_candidates"],
                        "pages": profile["scanned_pages"],
                        "distance_calls": profile["distance_calls"],
                    }
                    writer.writerow(row)
                    target.flush()
                    completed[qid] = row
                    recalls.append(recall)
                    done = len(completed)
                    elapsed = time.perf_counter() - started
                    progress = {
                        "status": "RUNNING", "config": "1/1",
                        "queries": f"{done}/{args.queries}",
                        "average_probes_so_far": float(args.probes),
                        "Recall_so_far": statistics.fmean(recalls),
                        "elapsed_seconds": elapsed,
                        "ETA_seconds": elapsed * (args.queries - done) / done,
                    }
                    atomic_json(args.output / f"fixed{args.probes}_progress.json", progress)
                    if done == 1 or done % 10 == 0:
                        print(" ".join(f"{key}={value}" for key, value in progress.items()
                                       if key != "status"), flush=True)
            connection.commit()
            summary = {
                "status": "COMPLETE", "probes": args.probes, "queries": args.queries,
                "mean_recall_at_10": statistics.fmean(recalls),
            }
            atomic_json(args.output / f"fixed{args.probes}_summary.json", summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
