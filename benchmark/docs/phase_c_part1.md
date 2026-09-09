# Phase C / Part 1

The shared runner now validates five systems (`pristine`, `vanilla_eq`, `phase2a`,
`phase2b`, `final`) using separate archived pristine and optimized `vector.so` files.
The formal entry point currently accepts only the Part 1 smoke configuration:
GIST1M, L2, lists=1000, Top10, probes=16, three queries, one warmup and one round.
It refuses larger runs; Phase C / Part 2 is not enabled by this change.

## Run and resume

```bash
PRISTINE_COMMIT=d65ea656a91b38c84dea234a83524b43e080e427 \
PYTHON_BIN=/workspace/benchmark/.venv/bin/python \
benchmark/scripts/run_ivfflat_experiments.sh formal --foreground --stop-after 4

# First command intentionally exits 75 after four measured smoke rows.
PRISTINE_COMMIT=d65ea656a91b38c84dea234a83524b43e080e427 \
PYTHON_BIN=/workspace/benchmark/.venv/bin/python \
benchmark/scripts/run_ivfflat_experiments.sh formal --foreground --resume RUN_DIR
```

The existing worker builds both extension binaries with GCC 11.5.0 and identical
Haswell AVX2/FMA flags, without overriding PG_CFLAGS. Only optimized adds
IVFFLAT_FUSED2. Each activation stops PostgreSQL, installs the archived binary,
checks SHA256, starts PostgreSQL, and opens a readiness connection. This operates
on the configured benchmark PostgreSQL instance, including rebuilding its
`gist_ivf_l2` index once with pristine. Each switch verifies index OID,
relfilenode, size, definition and validity.

Pristine provenance and compatibility are reviewed for the imported pgvector
commit and optimized pgvector tree `5e22d78fca56dfd0c6abd126d8e459a2d4d7bd92`.
Other source trees fail closed and need a fresh compatibility review. Both
extension binaries run on the same installed PostgreSQL server, whose hash is
recorded; pristine refers to the extension source, not an unmodified server.

Before smoke, twenty deterministic exact sequential scans must match official
HDF5 Top10 sets. Twenty sampled SQL vectors must match the corresponding HDF5
train vectors exactly as float32, and SQL IDs must be unique and cover 0–999999.
Boundary-set mismatches fail; result ordering differences are recorded.

The canonical checkpoint is `phase_c_smoke_checkpoint.json`. Each row has the
key `(system, probes, round, query_id)`, result IDs, Recall@10, latency, returned
count, binary hash and smoke designation. Writes use atomic replacement and
fsync. Resume validates schema, uniqueness, hashes and recomputed Recall before
skipping recorded keys; interrupted unrecorded queries may be retried. Warmups
are repeated for incomplete configurations and never enter measured rows.
Smoke results are stored separately from any future formal results.

Evidence is under `RUN_DIR/phase_c_artifacts`: manifest, source diff/status,
compile commands, both binaries and their metadata, activation log, exact audit,
query plans, raw CSV, checkpoint, interruption record and Part 1 report.
The mean Recall/latencies from three smoke queries are validation data only.

Validated run: `/workspace/benchmark/runs/20260909T065629Z`.
