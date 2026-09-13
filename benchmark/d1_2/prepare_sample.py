#!/usr/bin/env python3
"""Build a deterministic standalone sample from D1-1 candidate metadata."""
import csv
import io
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg2

DIM = 960
DATASET = Path("/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5")
TID_RE = re.compile(r"^\((\d+),(\d+)\)$")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare_sample.py sample_metadata.csv sample.bin run_directory")
    metadata_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    run_directory = Path(sys.argv[3])
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    metadata = list(csv.DictReader(metadata_path.open()))
    if len(metadata) != 3 * 16384:
        raise RuntimeError(f"unexpected sample count: {len(metadata)}")
    if [int(row["sample_id"]) for row in metadata] != list(range(len(metadata))):
        raise RuntimeError("sample IDs are not contiguous")
    query_ids = sorted({int(row["query_id"]) for row in metadata})
    query_slots = {query_id: slot for slot, query_id in enumerate(query_ids)}
    with h5py.File(DATASET, "r") as dataset:
        queries = {
            query_id: np.asarray(dataset["test"][query_id], dtype="<f4")
            for query_id in query_ids
        }
    for query_id, vector in queries.items():
        if vector.shape != (DIM,) or not np.isfinite(vector).all():
            raise RuntimeError(f"invalid query {query_id}")

    connection = psycopg2.connect(
        host="127.0.0.1", port=5432, dbname="taskdb", user="dev",
        application_name="phase_d1_2_sample_preparation",
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    started = time.monotonic()
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("LOCK TABLE gist_base IN SHARE MODE")
            cursor.execute(
                "CREATE TEMP TABLE d1_sample_meta("
                "sample_id int PRIMARY KEY, source_row bigint, query_id int, probes int,"
                "candidate_ordinal bigint, candidate_tid tid,"
                "full_distance_squared double precision,"
                "threshold_before_candidate double precision) ON COMMIT DROP"
            )
            payload = io.StringIO()
            for row in metadata:
                payload.write(
                    "\t".join([
                        row["sample_id"], row["source_row"], row["query_id"], row["probes"],
                        row["candidate_ordinal"], row["candidate_tid"],
                        row["full_distance_squared"], row["threshold_before_candidate"],
                    ]) + "\n"
                )
            payload.seek(0)
            cursor.copy_from(
                payload, "d1_sample_meta",
                columns=(
                    "sample_id", "source_row", "query_id", "probes",
                    "candidate_ordinal", "candidate_tid",
                    "full_distance_squared", "threshold_before_candidate",
                ),
            )
            cursor.execute("ANALYZE d1_sample_meta")
            query = (
                "SELECT s.sample_id,s.source_row,s.query_id,s.probes,"
                "s.candidate_ordinal,s.candidate_tid::text,"
                "s.full_distance_squared,s.threshold_before_candidate,"
                "vector_send(b.embedding) "
                "FROM d1_sample_meta s JOIN gist_base b ON b.ctid=s.candidate_tid "
                "ORDER BY s.sample_id"
            )
            cursor.execute("EXPLAIN (FORMAT JSON) " + query)
            save(run_directory / "sample_fetch_plan.json", cursor.fetchone()[0])

        with temporary.open("xb") as output:
            output.write(b"D1SAMP2\0")
            output.write(struct.pack("<IIII", 1, DIM, len(query_ids), len(metadata)))
            for query_id in query_ids:
                output.write(struct.pack("<i", query_id))
                output.write(queries[query_id].tobytes(order="C"))

            stream = connection.cursor(name="d1_sample_vectors")
            stream.itersize = 1024
            stream.execute(query)
            completed = 0
            for record in stream:
                (
                    sample_id, source_row, query_id, probes, candidate_ordinal,
                    tid, full_distance, threshold, wire_vector,
                ) = record
                if sample_id != completed:
                    raise RuntimeError(f"out-of-order sample {sample_id} != {completed}")
                match = TID_RE.match(tid)
                if not match:
                    raise RuntimeError(f"invalid TID: {tid}")
                tid_block, tid_offset = map(int, match.groups())
                wire = bytes(wire_vector)
                if len(wire) != 4 + DIM * 4:
                    raise RuntimeError(f"unexpected vector_send length {len(wire)}")
                dimension, unused = struct.unpack("!hh", wire[:4])
                if dimension != DIM or unused != 0:
                    raise RuntimeError("unexpected vector_send header")
                candidate = np.frombuffer(wire, dtype=">f4", offset=4).astype("<f4")
                if not np.isfinite(candidate).all():
                    raise RuntimeError(f"non-finite candidate at sample {sample_id}")
                output.write(struct.pack(
                    "<iiiIHHQdd",
                    query_slots[query_id], query_id, probes, tid_block, tid_offset, 0,
                    source_row, threshold, full_distance,
                ))
                output.write(candidate.tobytes(order="C"))
                completed += 1
                if completed % 4096 == 0 or completed == len(metadata):
                    elapsed = time.monotonic() - started
                    eta = elapsed * (len(metadata) - completed) / completed
                    print(
                        f"current workload/config=sample_fetch/vector_send "
                        f"completed={completed}/{len(metadata)} elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                        flush=True,
                    )
            stream.close()
            if completed != len(metadata):
                raise RuntimeError(f"missing candidate vectors: {completed}")
        connection.rollback()
        connection.close()
        temporary.replace(output_path)
        save(run_directory / "sample_preparation.json", {
            "status": "COMPLETE",
            "records": len(metadata),
            "records_per_probes": {
                str(probes): sum(int(row["probes"]) == probes for row in metadata)
                for probes in (64, 128, 256)
            },
            "queries": len(query_ids),
            "dimension": DIM,
            "binary_bytes": output_path.stat().st_size,
            "format": "D1SAMP2 little-endian; deduplicated queries; packed metadata + float32 candidate",
            "threshold_source": "D1-1 K40 tau_before_candidate; no synthetic threshold",
            "elapsed_seconds": time.monotonic() - started,
        })
    except BaseException:
        connection.rollback()
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise


if __name__ == "__main__":
    main()
