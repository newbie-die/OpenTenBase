#!/usr/bin/env python3
"""D2-P Part1.5 fixed-prefix versus progressive-shadow equivalence audit."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(os.environ.get("BENCHMARK_RUNTIME_ROOT", WORKSPACE_ROOT / "benchmark")).resolve()
DATASET = Path(os.environ.get("D2P_DATASET", RUNTIME_ROOT / "data/gist1m/gist-960-euclidean.hdf5"))
PROBES = (16, 32, 64)
FIELD_RE = re.compile(r"([a-z0-9_]+)=([^ ]+)")
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE_2B (.*)")
LIST_RE = re.compile(
    r"IVFFLAT_PROGRESSIVE_LISTS mode=(fixed|shadow) count=(\d+) pages_distances=(.*)"
)
TOPK_RE = re.compile(
    r"IVFFLAT_PROGRESSIVE_TOPK stage=(16|32|64) tids_distances=(.*)"
)


def atomic_json(path: Path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def vector_literal(vector):
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def parse_fields(text):
    return {key: value.strip() for key, value in FIELD_RE.findall(text)}


def canonical_tid(value):
    return value.strip("()").replace(",", "/")


def tid_key(value):
    block, offset = value.split("/")
    return int(block), int(offset)


def close_sequence(left, right, tolerance=2e-12):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def parse_list_notice(notices):
    matches = [LIST_RE.search(notice) for notice in notices]
    matches = [match for match in matches if match]
    if len(matches) != 1:
        raise RuntimeError(f"expected one ordered-list notice, found {len(matches)}")
    match = matches[0]
    entries = []
    if match.group(3).strip():
        for entry in match.group(3).strip().split(","):
            page, distance = entry.rsplit(":", 1)
            entries.append((int(page), float(distance)))
    if len(entries) != int(match.group(2)):
        raise RuntimeError("ordered-list notice count mismatch")
    return match.group(1), entries


def parse_profile(notices):
    matches = [PROFILE_RE.search(notice) for notice in notices]
    matches = [match for match in matches if match]
    if len(matches) != 1:
        raise RuntimeError(f"expected one PROFILE_2B notice, found {len(matches)}")
    return parse_fields(matches[0].group(1))


def parse_shadow(notices):
    stages = {}
    topks = {}
    for notice in notices:
        if "IVFFLAT_PROGRESSIVE stage=" in notice:
            row = parse_fields(notice.split("IVFFLAT_PROGRESSIVE ", 1)[1])
            stages[int(row["stage"])] = row
        match = TOPK_RE.search(notice)
        if match:
            items = []
            if match.group(2).strip():
                for entry in match.group(2).strip().split(","):
                    tid, distance = entry.rsplit(":", 1)
                    items.append((tid, float(distance)))
            topks[int(match.group(1))] = items
    if set(stages) != set(PROBES) or set(topks) != set(PROBES):
        raise RuntimeError("incomplete progressive stage snapshots")
    return stages, topks


def configure(cursor):
    cursor.execute("LOAD 'vector'")
    settings = {
        "ivfflat.iterative_scan": "off",
        "ivfflat.adaptive_probes": "off",
        "ivfflat.adaptive_probes_trace": "off",
        "ivfflat.bounded_scan": "off",
        "ivfflat.experimental_sort_bound": "0",
        "ivfflat.progressive_scan_debug": "on",
    }
    for name, value in settings.items():
        cursor.execute("SELECT set_config(%s,%s,false)", (name, value))
    cursor.execute("SELECT setting FROM pg_settings WHERE name='ivfflat.distance_path'")
    if cursor.fetchone() is not None:
        cursor.execute("SELECT set_config('ivfflat.distance_path','generic',false)")
    cursor.execute("SET enable_seqscan=off")
    cursor.execute("SET enable_indexscan=on")
    cursor.execute("SET max_parallel_workers_per_gather=0")


def execute(cursor, connection, sql, literal, probes, shadow):
    cursor.execute("SELECT set_config('ivfflat.probes',%s,false)", (str(probes),))
    cursor.execute(
        "SELECT set_config('ivfflat.progressive_scan',%s,false)",
        ("shadow" if shadow else "off",),
    )
    connection.notices.clear()
    cursor.execute(sql, (literal, literal))
    result = cursor.fetchall()
    notices = list(connection.notices)
    mode, lists = parse_list_notice(notices)
    if mode != ("shadow" if shadow else "fixed"):
        raise RuntimeError("ordered-list notice mode mismatch")
    return {
        "tids": [canonical_tid(row[0]) for row in result],
        "distances": [float(row[1]) for row in result],
        "lists": lists,
        "profile": parse_profile(notices),
        "notices": notices,
    }


def list_equivalence(fixed, shadow):
    fixed_pages = [page for page, _ in fixed]
    shadow_pages = [page for page, _ in shadow]
    fixed_distances = [distance for _, distance in fixed]
    shadow_distances = [distance for _, distance in shadow]
    exact = fixed_pages == shadow_pages
    set_equal = collections.Counter(fixed_pages) == collections.Counter(shadow_pages)
    distance_multiset_equal = sorted(fixed_distances) == sorted(shadow_distances)
    paired_equal = collections.Counter(fixed) == collections.Counter(shadow)
    tie_permutation = (
        not exact
        and paired_equal
        and all(fp == sp or fd == sd for (fp, fd), (sp, sd) in zip(fixed, shadow))
    )
    return {
        "exact_sequence": exact,
        "list_set_equal": set_equal,
        "centroid_distance_multiset_equal": distance_multiset_equal,
        "tie_permutation": tie_permutation,
        "true_mismatch": not exact and not tie_permutation,
    }


def result_equivalence(fixed, shadow_items):
    shadow_tids = [tid for tid, _ in shadow_items[:10]]
    shadow_distances = [distance for _, distance in shadow_items[:10]]
    # IVFFlat stores squared L2 internally; SQL's <-> result is sqrt(distance).
    fixed_internal_distances = [distance * distance for distance in fixed["distances"]]
    id_equal = fixed["tids"] == shadow_tids
    distance_equal = close_sequence(fixed_internal_distances, shadow_distances)
    if id_equal and distance_equal:
        classification = "exact_id_equal"
    elif distance_equal:
        classification = "distance_equivalent"
    else:
        classification = "true_mismatch"
    return {
        "classification": classification,
        "tid_sequence_equal": id_equal,
        "distance_sequence_equal": distance_equal,
        "fixed_tids": fixed["tids"],
        "shadow_tids": shadow_tids,
        "fixed_internal_distances": fixed_internal_distances,
        "shadow_distances": shadow_distances,
    }


def counter_equivalence(fixed, stage, list_result):
    pairs = {
        "candidates": ("scanned_candidates", "candidates"),
        "pages": ("scanned_pages", "pages"),
        "distance_calls": ("distance_calls", "distance_calls"),
    }
    values = {
        name: {"fixed": int(fixed["profile"][fixed_name]), "shadow": int(stage[stage_name])}
        for name, (fixed_name, stage_name) in pairs.items()
    }
    exact = all(value["fixed"] == value["shadow"] for value in values.values())
    tie_explained = not exact and list_result["tie_permutation"]
    return {
        "values": values,
        "exact": exact,
        "tie_explained": tie_explained,
        "true_mismatch": not exact and not tie_explained,
    }


def validate_topk(items, stage):
    ordered = all(
        (items[index - 1][1], tid_key(items[index - 1][0]))
        <= (items[index][1], tid_key(items[index][0]))
        for index in range(1, len(items))
    )
    kth_equal = len(items) == 40 and math.isclose(
        items[-1][1], float(stage["kth_distance"]), rel_tol=2e-12, abs_tol=2e-12
    )
    return {
        "topk_count": len(items),
        "unique_tids": len({tid for tid, _ in items}) == len(items),
        "ordered_by_distance_tid": ordered,
        "kth_distance_equal": kth_equal,
        "pass": len(items) == 40 and ordered and kth_equal
                and len({tid for tid, _ in items}) == 40,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="taskdb")
    parser.add_argument("--user", default="dev")
    args = parser.parse_args()
    if not 20 <= args.queries <= 50:
        raise ValueError("Part1.5 --queries must be 20..50")
    args.output.mkdir(parents=True, exist_ok=False)

    import h5py
    import psycopg2

    with h5py.File(DATASET, "r") as source:
        queries = source["test"][:args.queries]
    connection = psycopg2.connect(
        host=args.host, port=args.port, dbname=args.dbname, user=args.user,
        application_name="d2p_part15_prefix_audit",
    )
    connection.autocommit = True
    sql = (
        "SELECT ctid::text, embedding <-> %s::vector AS distance "
        "FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10"
    )
    details = []
    counts = {
        "list_exact_count": 0,
        "list_tie_count": 0,
        "list_true_mismatch_count": 0,
        "exact_id_equal_count": 0,
        "distance_equivalent_count": 0,
        "true_mismatch_count": 0,
        "counter_exact_count": 0,
        "counter_tie_explained_count": 0,
        "counter_true_mismatch_count": 0,
        "topk_pass_count": 0,
        "topk_failure_count": 0,
        "final_result_mismatch_count": 0,
    }
    started = time.perf_counter()
    try:
        with connection.cursor() as cursor:
            configure(cursor)
            for qid, vector in enumerate(queries):
                literal = vector_literal(vector)
                fixed = {
                    probes: execute(cursor, connection, sql, literal, probes, False)
                    for probes in PROBES
                }
                shadow = execute(cursor, connection, sql, literal, 64, True)
                stages, topks = parse_shadow(shadow["notices"])
                query_detail = {"query_id": qid, "stages": {}}

                final_distance_equal = close_sequence(
                    fixed[64]["distances"], shadow["distances"]
                )
                query_detail["fixed64_vs_shadow_final"] = {
                    "tid_sequence_equal": fixed[64]["tids"] == shadow["tids"],
                    "distance_sequence_equal": final_distance_equal,
                }
                if fixed[64]["tids"] != shadow["tids"] or not final_distance_equal:
                    counts["final_result_mismatch_count"] += 1

                for probes in PROBES:
                    prefix = shadow["lists"][:probes]
                    list_result = list_equivalence(fixed[probes]["lists"], prefix)
                    result = result_equivalence(fixed[probes], topks[probes])
                    counter = counter_equivalence(fixed[probes], stages[probes], list_result)
                    topk = validate_topk(topks[probes], stages[probes])
                    query_detail["stages"][str(probes)] = {
                        "list": list_result,
                        "result": result,
                        "counter": counter,
                        "topk": topk,
                    }
                    if list_result["exact_sequence"]:
                        counts["list_exact_count"] += 1
                    elif list_result["tie_permutation"]:
                        counts["list_tie_count"] += 1
                    else:
                        counts["list_true_mismatch_count"] += 1
                    counts[result["classification"] + "_count"] += 1
                    if counter["exact"]:
                        counts["counter_exact_count"] += 1
                    elif counter["tie_explained"]:
                        counts["counter_tie_explained_count"] += 1
                    else:
                        counts["counter_true_mismatch_count"] += 1
                    if topk["pass"]:
                        counts["topk_pass_count"] += 1
                    else:
                        counts["topk_failure_count"] += 1
                details.append(query_detail)
                elapsed = time.perf_counter() - started
                done = qid + 1
                eta = elapsed * (args.queries - done) / done
                print(
                    f"query={done}/{args.queries} exact_ids={counts['exact_id_equal_count']} "
                    f"distance_equivalent={counts['distance_equivalent_count']} "
                    f"true_mismatch={counts['true_mismatch_count']} "
                    f"elapsed={elapsed:.3f}s ETA={eta:.3f}s",
                    flush=True,
                )
    finally:
        connection.close()

    total_stage_cases = args.queries * len(PROBES)
    per_stage = {}
    for probes in PROBES:
        cases = [detail["stages"][str(probes)] for detail in details]
        per_stage[str(probes)] = {
            "queries": len(cases),
            "list_exact_count": sum(case["list"]["exact_sequence"] for case in cases),
            "list_tie_count": sum(case["list"]["tie_permutation"] for case in cases),
            "list_true_mismatch_count": sum(case["list"]["true_mismatch"] for case in cases),
            "exact_id_equal_count": sum(
                case["result"]["classification"] == "exact_id_equal" for case in cases
            ),
            "distance_equivalent_count": sum(
                case["result"]["classification"] == "distance_equivalent" for case in cases
            ),
            "true_mismatch_count": sum(
                case["result"]["classification"] == "true_mismatch" for case in cases
            ),
            "counter_exact_count": sum(case["counter"]["exact"] for case in cases),
            "counter_tie_explained_count": sum(
                case["counter"]["tie_explained"] for case in cases
            ),
            "counter_true_mismatch_count": sum(
                case["counter"]["true_mismatch"] for case in cases
            ),
            "topk_pass_count": sum(case["topk"]["pass"] for case in cases),
        }
    passed = (
        counts["true_mismatch_count"] == 0
        and counts["list_true_mismatch_count"] == 0
        and counts["counter_true_mismatch_count"] == 0
        and counts["final_result_mismatch_count"] == 0
        and counts["topk_failure_count"] == 0
        and counts["topk_pass_count"] == total_stage_cases
    )
    summary = {
        "status": "PREFIX_EQUIVALENCE_PASS" if passed else "PREFIX_EQUIVALENCE_FAIL",
        "queries": args.queries,
        "limit": 10,
        "stage_cases": total_stage_cases,
        "per_stage": per_stage,
        **counts,
        "shadow_k": 40,
        "shadow_topk_semantics_pass": counts["topk_pass_count"] == total_stage_cases,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output / "details.json", details)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
