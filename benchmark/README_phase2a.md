# Phase 2A: Oracle bounded tuplesort

This experiment measures the performance ceiling of PostgreSQL bounded tuplesort inside low-dimensional IVFFlat scans. It is an Oracle experiment: the user explicitly sets ivfflat.experimental_sort_bound; the value is not inferred from SQL LIMIT.

The bound is not production-safe LIMIT pushdown. Candidates can still be removed by MVCC visibility, heap fetches, rechecks, filters, OFFSET, and upper executor nodes. Every configuration recomputes Recall@10 and returned row counts; fewer than 10 rows is marked unsafe/incorrect.

## Workload

- GloVe ANN-Benchmarks, 100 dimensions, L2 (<->), 1,183,514 rows
- one IVFFlat index with lists=1000
- probes 16, 64, 128; bounds 0, 10, 20, 40, 100
- bound 0 is the full-tuplesort baseline
- Top-K 10, OFFSET 0, no WHERE filter, iterative_scan=off
- single-client sequential warm-cache execution, no cache dropping
- 10,000 measured queries and 3 rounds for the formal run

Candidate counts and distance time should remain stable while bounded sorting changes sort/materialization costs.

## Run

    WARMUP=500 QUERIES=10000 TOPK=10 \
      /workspace/OpenTenBase/benchmark/scripts/run_ivfflat_experiments.sh 2a

Optional smoke matrix:

    PHASE2A_PROBES=64 PHASE2A_BOUNDS=0,10 PHASE2A_ROUNDS=1 \
    WARMUP=10 QUERIES=100 \
      /workspace/OpenTenBase/benchmark/scripts/run_ivfflat_experiments.sh 2a

The worker builds and installs pgvector with IVFFLAT_BENCH, restarts the server, rebuilds only the GloVe L2 index, and runs serially. Outputs include per-configuration CSVs, phase2a_raw.csv, and phase2a_summary.csv. Rebuild summaries with:

    python3 /workspace/OpenTenBase/benchmark/scripts/summarize_phase2a.py RUN_DIRECTORY

## Correctness and decision criteria

A sanity check compares one query at bounds 0 and 10: both must return 10 rows with identical IDs and Recall. Formal runs require min_returned_rows=10 and no Recall regression.

If P95 and QPS both improve by less than 5%, stop (No-Go). A 5-15% improvement is Weak-Go. P95 or QPS improvement of at least 15%, unchanged Recall, and 10 rows throughout is Strong-Go; only then expand to IP, cosine, or SIFT.
