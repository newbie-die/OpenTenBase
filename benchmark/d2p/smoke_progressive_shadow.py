#!/usr/bin/env python3
"""Small fixed-probes=64 versus progressive-shadow correctness smoke."""
import argparse
import re

DATASET = "/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5"
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE_2B (.*)")


def fields(text):
    return {key: value.strip() for key, value in re.findall(r"([a-z0-9_]+)=([^ ]+)", text)}


def literal(vector):
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def profile(notices):
    for notice in notices:
        match = PROFILE_RE.search(notice)
        if match:
            return fields(match.group(1))
    raise RuntimeError("missing IVFFLAT_PROFILE_2B; compile with -DIVFFLAT_BENCH -DIVFFLAT_PROFILE_2B")


def configure(cursor):
    cursor.execute("LOAD 'vector'")
    settings = {
        "ivfflat.probes": "64",
        "ivfflat.iterative_scan": "off",
        "ivfflat.adaptive_probes": "off",
        "ivfflat.adaptive_probes_trace": "off",
        "ivfflat.bounded_scan": "off",
        "ivfflat.experimental_sort_bound": "0",
        "ivfflat.progressive_scan_debug": "off",
    }
    for name, value in settings.items():
        cursor.execute("SELECT set_config(%s,%s,false)", (name, value))
    cursor.execute("SET enable_seqscan=off")
    cursor.execute("SET enable_indexscan=on")
    cursor.execute("SET max_parallel_workers_per_gather=0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="taskdb")
    parser.add_argument("--user", default="dev")
    parser.add_argument("--queries", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.queries <= 10:
        raise ValueError("smoke --queries must be 1..10")

    import h5py
    import psycopg2

    with h5py.File(DATASET, "r") as source:
        queries = source["test"][:args.queries]
    connection = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname,
                                  user=args.user, application_name="d2p_part1_smoke")
    connection.autocommit = True
    sql = ("SELECT ctid::text, embedding <-> %s::vector AS distance "
           "FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10")
    try:
        with connection.cursor() as cursor:
            configure(cursor)
            for qid, query in enumerate(queries):
                value = literal(query)
                cursor.execute("SELECT set_config('ivfflat.progressive_scan','off',false)")
                cursor.execute("SELECT set_config('ivfflat.progressive_scan_debug','off',false)")
                connection.notices.clear()
                cursor.execute(sql, (value, value))
                fixed = cursor.fetchall()
                fixed_notices = list(connection.notices)
                fixed_profile = profile(fixed_notices)
                if any("IVFFLAT_PROGRESSIVE" in notice for notice in fixed_notices):
                    raise AssertionError(f"qid={qid}: progressive debug leaked into off mode")

                cursor.execute("SELECT set_config('ivfflat.progressive_scan','shadow',false)")
                cursor.execute("SELECT set_config('ivfflat.progressive_scan_debug','on',false)")
                connection.notices.clear()
                cursor.execute(sql, (value, value))
                shadow = cursor.fetchall()
                notices = list(connection.notices)
                shadow_profile = profile(notices)
                stages = [fields(n.split("IVFFLAT_PROGRESSIVE ", 1)[1])
                          for n in notices if "IVFFLAT_PROGRESSIVE stage=" in n]

                if [int(row["stage"]) for row in stages] != [16, 32, 64]:
                    raise AssertionError(f"qid={qid}: missing stage records")
                candidates = [int(row["candidates"]) for row in stages]
                if not candidates[0] < candidates[1] < candidates[2]:
                    raise AssertionError(f"qid={qid}: candidates are not strictly cumulative: {candidates}")
                if [int(row["new_lists"]) for row in stages] != [16, 16, 32]:
                    raise AssertionError(f"qid={qid}: incorrect list ranges")
                if stages[-1]["list_mask"] != "ffffffffffffffff":
                    raise AssertionError(f"qid={qid}: duplicate/missing list scan mask")
                sequence_equal = fixed == shadow
                if not sequence_equal:
                    fixed_distances = [row[1] for row in fixed]
                    shadow_distances = [row[1] for row in shadow]
                    if fixed_distances != shadow_distances:
                        raise AssertionError(f"qid={qid}: distance sequence mismatch")
                    print(f"qid={qid} tie_audit=distance_equivalent tid_sequence_equal=0", flush=True)
                calls = int(shadow_profile["distance_calls"])
                if int(fixed_profile["distance_calls"]) != calls or int(stages[-1]["distance_calls"]) != calls:
                    raise AssertionError(f"qid={qid}: final distance_calls differ")
                if int(fixed_profile["scanned_candidates"]) != int(shadow_profile["scanned_candidates"]):
                    raise AssertionError(f"qid={qid}: final candidate counts differ")
                print(f"qid={qid} tid_distance_sequence_equal={int(sequence_equal)} "
                      f"cumulative_candidates={candidates} distance_calls={calls} "
                      f"list_mask={stages[-1]['list_mask']}", flush=True)
    finally:
        connection.close()
    print(f"SMOKE PASS queries={args.queries}")


if __name__ == "__main__":
    main()
