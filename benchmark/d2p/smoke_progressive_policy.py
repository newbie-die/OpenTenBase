#!/usr/bin/env python3
"""Verify frozen offline D2-P actions against the PostgreSQL C policy."""
import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from calibrate_policy import feature_rows, read_csv

DATASET = Path("/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5")
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE_2B (.*)")
PAIR_RE = re.compile(r"([a-z0-9_]+)=([^ ]+)")


def evaluate(node, features):
    while "safe_probability" not in node:
        node = node["if_le"] if features[node["feature"]] <= node["threshold"] else node["if_gt"]
    return float(node["safe_probability"])


def vector_literal(vector):
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def run_query(cursor, connection, literal, mode):
    cursor.execute("SELECT set_config('ivfflat.progressive_scan',%s,false)", (mode,))
    cursor.execute("SELECT set_config('ivfflat.probes','64',false)")
    connection.notices.clear()
    before = time.perf_counter_ns()
    cursor.execute("SELECT id FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10", (literal,))
    ids = [int(row[0]) for row in cursor.fetchall()]
    latency_us = (time.perf_counter_ns() - before) / 1000.0
    matches = [PROFILE_RE.search(notice) for notice in connection.notices]
    matches = [match for match in matches if match]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one IVFFLAT_PROFILE_2B notice")
    return ids, latency_us, dict(PAIR_RE.findall(matches[0].group(1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="taskdb")
    parser.add_argument("--user", default="dev")
    args = parser.parse_args()
    import h5py
    import psycopg2

    args.output.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.policy.read_text())
    selected = read_csv(args.shadow)
    selected = [row for row in selected if int(row["query_id"]) < 20]
    features = feature_rows(selected)
    with h5py.File(DATASET, "r") as source:
        queries = source["test"][:20]
        truth = [set(map(int, row[:10])) for row in source["neighbors"][:20]]
    connection = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname, user=args.user,
                                  application_name="d2p_policy_smoke")
    connection.autocommit = True
    rows = []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET client_min_messages=info")
            cursor.execute("LOAD 'vector'")
            settings = {"ivfflat.iterative_scan": "off", "ivfflat.adaptive_probes": "off",
                        "ivfflat.adaptive_probes_trace": "off", "ivfflat.bounded_scan": "off",
                        "ivfflat.experimental_sort_bound": 0, "ivfflat.progressive_scan_debug": "off",
                        "ivfflat.distance_path": "generic"}
            for name, value in settings.items():
                cursor.execute("SELECT set_config(%s,%s,false)", (name, str(value)))
            cursor.execute("SET enable_seqscan=off")
            cursor.execute("SET max_parallel_workers_per_gather=0")
            for qid, query in enumerate(queries):
                f = features[qid]
                p16 = evaluate(policy["stage16_tree"], f)
                p32 = evaluate(policy["stage32_tree"], f)
                offline = 16 if p16 >= policy["stage16_threshold"] else (
                    32 if p32 >= policy["stage32_threshold"] else 64)
                literal = vector_literal(query)
                fixed_ids, _, fixed_profile = run_query(cursor, connection, literal, "off")
                d2p_ids, latency, d2p_profile = run_query(cursor, connection, literal, "on")
                online = int(d2p_profile["progressive_stop_stage"])
                rows.append({"query_id": qid, "offline_action": offline, "online_action": online,
                             "action_equal": offline == online, "latency_us": latency,
                             "fixed64_recall": len(set(fixed_ids) & truth[qid]) / 10.0,
                             "d2p_recall": len(set(d2p_ids) & truth[qid]) / 10.0,
                             "probes": d2p_profile["probes"],
                             "candidates": d2p_profile["scanned_candidates"],
                             "pages": d2p_profile["scanned_pages"],
                             "distance_calls": d2p_profile["distance_calls"],
                             "fixed64_candidates": fixed_profile["scanned_candidates"]})
    finally:
        connection.close()
    with (args.output / "raw.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    delta = statistics.fmean(row["d2p_recall"] - row["fixed64_recall"] for row in rows)
    distribution = {str(stage): sum(row["online_action"] == stage for row in rows) for stage in (16, 32, 64)}
    mismatch = sum(not row["action_equal"] for row in rows)
    summary = {"queries": 20, "action_mismatch": mismatch, "distribution": distribution,
               "recall_delta": delta,
               "average_probes": statistics.fmean(float(row["probes"]) for row in rows),
               "average_candidates": statistics.fmean(float(row["candidates"]) for row in rows),
               "fixed64_average_candidates": statistics.fmean(float(row["fixed64_candidates"]) for row in rows),
               "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
               "status": ("IMPLEMENTATION_FAIL" if mismatch else
                          ("D2P_ONLINE_NO_GO" if delta < -0.005 else "D2P_READY_FOR_FORMAL"))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "D2P_READY_FOR_FORMAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
