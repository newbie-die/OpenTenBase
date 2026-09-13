#!/usr/bin/env python3
"""Collect D2-P 16/32/64 progressive-shadow snapshots without early stopping."""
import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

DATASET = Path("/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5")
STAGES = (16, 32, 64)
FIELD_RE = re.compile(r"([a-z0-9_]+)=([^ ]+)")
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE_2B (.*)")
TRACE_RE = re.compile(r"IVFFLAT_ADAPTIVE (.*)")
TOPK_RE = re.compile(r"IVFFLAT_PROGRESSIVE_TOPK stage=(16|32|64) tids_distances=(.*)")
CENTROID_FIELDS = ("d1", "d2", "d4", "d8", "d16", "d32", "d64", "d2_d1", "d4_d1",
                   "d8_d1", "d16_d1", "d32_d1", "d64_d1", "d32_d16", "d64_d32",
                   "gap", "gap32_16_d1", "gap64_32_d1")
BASE_FIELDS = ("query_id", "recall_at_10", "result_ids", "result_tids", "result_distances")
STAGE_FIELDS = tuple(f"stage{stage}_{name}" for stage in STAGES for name in
                     ("candidates", "pages", "distance_calls", "replacements", "kth_distance",
                      "topk_count", "topk_tids", "topk_distances"))
CSV_FIELDS = BASE_FIELDS + CENTROID_FIELDS + STAGE_FIELDS


def parse_fields(text):
    return {key: value.strip() for key, value in FIELD_RE.findall(text)}


def vector_literal(vector):
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_notices(notices):
    stages, topks, trace, profile = {}, {}, None, None
    for notice in notices:
        if "IVFFLAT_PROGRESSIVE stage=" in notice:
            record = parse_fields(notice.split("IVFFLAT_PROGRESSIVE ", 1)[1])
            stages[int(record["stage"])] = record
        match = TOPK_RE.search(notice)
        if match:
            entries = [] if not match.group(2).strip() else match.group(2).strip().split(",")
            tids, distances = [], []
            for entry in entries:
                tid, distance = entry.rsplit(":", 1)
                tids.append(tid)
                distances.append(float(distance))
            topks[int(match.group(1))] = (tids, distances)
        match = TRACE_RE.search(notice)
        if match:
            trace = parse_fields(match.group(1))
        match = PROFILE_RE.search(notice)
        if match:
            profile = parse_fields(match.group(1))
    if set(stages) != set(STAGES) or set(topks) != set(STAGES) or trace is None or profile is None:
        raise RuntimeError("incomplete shadow notices; install the profiled D2-P build and enable debug")
    return stages, topks, trace, profile


def canonical_tid(tid):
    return tid.strip("()").replace(",", "/")


def load_existing(path):
    if not path.exists():
        return {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"unexpected CSV columns in {path}")
        return {int(row["query_id"]): row for row in reader}


def write_progress(path, done, total, recalls, started, status="RUNNING"):
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - done) / done if done else None
    value = {
        "status": status,
        "config": "1/1",
        "queries": f"{done}/{total}",
        "average_probes_so_far": 64.0 if done else 0.0,
        "Recall_so_far": statistics.fmean(recalls) if recalls else 0.0,
        "elapsed_seconds": elapsed,
        "ETA_seconds": 0.0 if status == "COMPLETE" else eta,
    }
    atomic_json(path, value)
    print(" ".join(f"{key}={value[key]}" for key in
                   ("config", "queries", "average_probes_so_far", "Recall_so_far",
                    "elapsed_seconds", "ETA_seconds")), flush=True)


def set_guc(cursor, name, value):
    cursor.execute("SELECT setting FROM pg_settings WHERE name=%s", (name,))
    if cursor.fetchone() is None:
        raise RuntimeError(f"required GUC is missing: {name}")
    cursor.execute("SELECT set_config(%s,%s,true)", (name, str(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
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
    raw_path = args.output / "shadow_snapshots.csv"
    completed = load_existing(raw_path)
    with h5py.File(DATASET, "r") as source:
        queries = source["test"][:args.queries]
        truth = [set(int(value) for value in row) for row in source["neighbors"][:args.queries, :10]]
    if any(qid >= args.queries for qid in completed):
        raise ValueError("output contains qids outside requested range")

    connection = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname,
                                  user=args.user, application_name="d2p_shadow_collection")
    connection.autocommit = False
    started = time.perf_counter()
    recalls = [float(completed[qid]["recall_at_10"]) for qid in sorted(completed)]
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("SET LOCAL lock_timeout=10000")
            cursor.execute("LOCK TABLE gist_base IN SHARE MODE")
            cursor.execute("LOAD 'vector'")
            settings = {
                "ivfflat.probes": 64,
                "ivfflat.iterative_scan": "off",
                "ivfflat.adaptive_probes": "off",
                "ivfflat.adaptive_probes_trace": "on",
                "ivfflat.bounded_scan": "off",
                "ivfflat.experimental_sort_bound": 0,
                "ivfflat.progressive_scan": "shadow",
                "ivfflat.progressive_scan_debug": "on",
            }
            for name, value in settings.items():
                set_guc(cursor, name, value)
            cursor.execute("SELECT setting FROM pg_settings WHERE name='ivfflat.distance_path'")
            if cursor.fetchone() is not None:
                set_guc(cursor, "ivfflat.distance_path", "generic")
            cursor.execute("SET LOCAL enable_seqscan=off")
            cursor.execute("SET LOCAL enable_indexscan=on")
            cursor.execute("SET LOCAL max_parallel_workers_per_gather=0")
            cursor.execute("SET LOCAL statement_timeout=0")
            cursor.execute("SELECT oid::text, relfilenode::text, reloptions FROM pg_class WHERE oid='gist_ivf_l2'::regclass")
            index_identity = cursor.fetchone()
            if index_identity is None or "lists=1000" not in (index_identity[2] or []):
                raise RuntimeError("gist_ivf_l2 with lists=1000 is required")

            sql = ("SELECT id, ctid::text, embedding <-> %s::vector AS distance "
                   "FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10")
            cursor.execute("EXPLAIN (FORMAT JSON) " + sql,
                           (vector_literal(queries[0]), vector_literal(queries[0])))
            plan = cursor.fetchone()[0]
            if "gist_ivf_l2" not in json.dumps(plan) or "Index Scan" not in json.dumps(plan):
                raise RuntimeError("expected gist_ivf_l2 Index Scan")
            atomic_json(args.output / "plan.json", plan)
            atomic_json(args.output / "manifest.json", {
                "status": "RUNNING", "mode": "shadow", "stages": list(STAGES),
                "queries": args.queries, "dataset": str(DATASET), "index_identity": index_identity,
                "topk": "max(40, 4 * executor LIMIT+OFFSET)", "early_stop": False,
            })

            mode = "a" if raw_path.exists() else "w"
            with raw_path.open(mode, newline="", buffering=1) as target:
                writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
                if mode == "w":
                    writer.writeheader()
                for qid, query in enumerate(queries):
                    if qid in completed:
                        continue
                    value = vector_literal(query)
                    connection.notices.clear()
                    cursor.execute(sql, (value, value))
                    result = cursor.fetchall()
                    stages, topks, trace, profile = parse_notices(list(connection.notices))
                    ids = [int(row[0]) for row in result]
                    tids = [canonical_tid(row[1]) for row in result]
                    distances = [float(row[2]) for row in result]
                    candidates = [int(stages[stage]["candidates"]) for stage in STAGES]
                    if len(ids) != 10 or len(set(ids)) != 10 or distances != sorted(distances):
                        raise RuntimeError(f"qid={qid}: invalid final Top10")
                    if not candidates[0] < candidates[1] < candidates[2]:
                        raise RuntimeError(f"qid={qid}: candidate counts are not cumulative")
                    if [int(stages[s]["new_lists"]) for s in STAGES] != [16, 16, 32]:
                        raise RuntimeError(f"qid={qid}: incorrect incremental list ranges")
                    if stages[64]["list_mask"] != "ffffffffffffffff":
                        raise RuntimeError(f"qid={qid}: missing or repeated list")
                    if int(profile["distance_calls"]) != int(stages[64]["distance_calls"]):
                        raise RuntimeError(f"qid={qid}: final distance call mismatch")
                    if topks[64][0][:10] != tids:
                        raise RuntimeError(f"qid={qid}: shadow Top-K TIDs disagree with final tuplesort")

                    recall = len(set(ids) & truth[qid]) / 10.0
                    row = {
                        "query_id": qid, "recall_at_10": recall,
                        "result_ids": ";".join(map(str, ids)), "result_tids": ";".join(tids),
                        "result_distances": ";".join(format(value, ".17g") for value in distances),
                    }
                    for name in CENTROID_FIELDS:
                        row[name] = trace[name]
                    for stage in STAGES:
                        for name in ("candidates", "pages", "distance_calls", "replacements",
                                     "kth_distance", "topk_count"):
                            row[f"stage{stage}_{name}"] = stages[stage][name]
                        row[f"stage{stage}_topk_tids"] = ";".join(topks[stage][0])
                        row[f"stage{stage}_topk_distances"] = ";".join(
                            format(value, ".17g") for value in topks[stage][1])
                    writer.writerow(row)
                    target.flush()
                    completed[qid] = row
                    recalls.append(recall)
                    write_progress(args.output / "progress.json", len(completed), args.queries,
                                   recalls, started)
            connection.commit()
            atomic_json(args.output / "manifest.json", {
                "status": "COMPLETE", "mode": "shadow", "stages": list(STAGES),
                "queries": args.queries, "dataset": str(DATASET), "index_identity": index_identity,
                "topk": "max(40, 4 * executor LIMIT+OFFSET)", "early_stop": False,
                "mean_recall_at_10": statistics.fmean(recalls),
            })
            write_progress(args.output / "progress.json", args.queries, args.queries,
                           recalls, started, "COMPLETE")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
