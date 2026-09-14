# Final Multi-Dataset Comparison

This extends `scripts/ivfflat_profile.py` and the Phase C v2 activation, GUC,
seeded permutation, atomic checkpoint, CSV and hashing helpers through a single
`scripts/final_multi.py` module. Use `scripts/benchmark.py` as the common entry
point. No dataset has a separate experiment runner. The dataset catalog is
`configs/final_multi_datasets.json`; relative data paths resolve below the
runtime root.

Preparation defaults to **no formal experiments**. Use the installed benchmark
Python environment, which includes NumPy, h5py, PyArrow, psycopg2 and scikit-learn.

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_prepare --resume
```

The database must already contain the catalog's tables and indexes. Existing
indexes are never rebuilt by the experiment runner. Preparation validates base
ID/vector mapping, dimensions, GT, index opclass and lists, EXPLAIN plans, file
hashes and query splits. One sequential-scan exact GT query per workload is also
checked. A missing index is an error, not grounds for silently using a seqscan.
Cohere's initial table/index preparation is recorded separately in the preparation
artifact directory; shuffled Parquet row positions are never used as IDs.

## Frozen execution matrix

| Dataset | Baseline | 2A | 2A+B | D | 2A+D |
|---|---|---|---|---|---|
| GIST | RUN | RUN | RUN | RUN: frozen GIST D2-P | UNSUPPORTED |
| SIFT | RUN | RUN | RUN | conditional on own clean calibration and deployment parity | UNSUPPORTED |
| GloVe-Cosine | RUN | RUN | N/A: L2-only | N/A: no existing clean split; preparation skipped this round | UNSUPPORTED |
| Cohere-Cosine | RUN | RUN | N/A: L2-only | N/A: no clean split | UNSUPPORTED |
| GloVe-IP | RUN | RUN | N/A: L2-only | N/A: feature redesign required | UNSUPPORTED |

GIST uses its 1,000 official test queries, verified equal to the combined formal
file's qids 1000..1999. Its frozen tree was trained on separate `gist_learn` rows.
SIFT's first 1,000 queries are reserved for calibration **before observing any
calibration outcomes**, leaving 9,000 formal queries. There are zero cross-split
exact duplicate vectors. The formal subset has 8,999 unique vectors: its internal
duplicate remains part of the original workload. GloVe-Cosine and GloVe-IP each
use 10,000 queries; Cohere uses 1,000. GloVe-IP normalizes both base and queries,
so the original angular GT is valid under negative inner product ordering.

## Methods and calibration

All timed runs use LIMIT 10, lists=1000, probes=64, the same index, 100 warmup
queries, and a shared seeded permutation within each workload and round.
Baseline is the production optimized-source binary with 2A, FUSED2 and D disabled
(the prior Phase C `vanilla_eq` configuration), not a different pristine binary.
2A uses bounded scan with the frozen 4x overfetch / min 40 / fastpath limit 100
settings. 2A+B changes only the distance path to FUSED2, and is restricted to L2.

GIST D is never retrained. SIFT retains 16 -> 32 -> 64, the original feature
names and formulas, DecisionTreeClassifier, depths {2,3}, min leaves {20,40},
seed 20260913, `qid % 5` OOF folds, static calibration sequence, and recall-loss
budget 0.005. The existing shadow/fixed collectors now accept an audited L2
table/index through environment parameters. The calibration program handles
single-class folds without changing their probability semantics.

A private source copy gets SIFT's generated tree/threshold and the full existing
training feature dictionary. This is necessary because the frozen GIST C policy
only materializes features selected by its own tree. It does not change the
probing algorithm or feature definitions. Main-tree C files and GIST artifacts
are untouched. Offline C/Python feature parity checks the adapter on archived
calibration snapshots; after SIFT calibration, 1,000 calibration-only executions
must match Python stop decisions and fixed-probe Top10 results before formal D.
SIFT Baseline vs D is remeasured in the same SIFT policy binary. Its earlier
Baseline/2A/2A+B results retain their shared generic production binary.

## Formal schedule and costs

The final schedule contains two effective rounds of 14 unique configurations.
Final Round A and B retain their original round IDs; historical reconnaissance
and duplicate SIFT baseline checkpoints remain on disk but are excluded from
the final statistics. Round B uses a deterministic rotation of Round A's
configuration order. Query order within a workload is unchanged.

| Category | Query executions |
|---|---:|
| Measured: 2 × (62,000 Baseline/2A + 10,000 2A+B + 10,000 D) | 164,000 |
| Warmup: 28 blocks × 100 | 2,800 |
| SIFT shadow + fixed calibration | 4,000–7,000 |
| SIFT deployment decision/Top10 parity | 1,000 |
| Untimed D stage replay, once per formal query | 10,000 |
| Formal measured total | **164,000** |

Counts exclude this preparation's smoke, SQL metadata queries, compilation and
index preparation. An interruption inside a block reruns that unfinished block
and its warmup; a finished checkpoint is verified and skipped. Thus retry work
can exceed the uninterrupted upper bound. The independent D stage replay checks
result checksums; debug is never enabled inside a timing interval.

## Launch and resume

After preparation produces `smoke_gate.json` with PASS, the following is the
single formal command. **Preparation does not execute this command.**

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_formal --formal \
  --prepared final_multi_prepare --resume --background
```

Use the same command to resume after interruption. An exclusive Phase C database
lock prevents concurrent experiment runners. Binary activation uses the existing
Phase C stop/install/start procedure. Checkpoint identity includes data/index,
policy and binary provenance; altered inputs are rejected. Calibration snapshots
and fixed collections resume using their existing durable CSV prefix behavior.
Preparation is required again after runner, algorithm source, policy, index or
dataset changes.

## Artifacts

```text
final_multi_prepare/ or final_multi_formal/
  plan.json                     config catalog, schedule, execution counts
  validation.json               paths, dimensions, GT/query/data hashes, indexes, split
  exact_gt_validation.json      preparation's five independent exact checks
  manifest.json                 immutable run identity, commit, flags, packages, server
  progress.json                 current block/query or terminal state
  binaries/{production,calibration,sift}/
    vector.so, build.json, compile.log
  build/                        isolated build sources, never the working-tree sources
  calibration/sift/
    shadow/, fixed/, policy/    calibration-only snapshots and frozen learned artifacts
    policy_parity.json          deployment vs Python/fixed-probe validation
    complete.json               calibrated policy and binary identity
  checkpoints/                  fsync + atomic rename, one completed config/round each
  diagnostics/{gist,sift}.json   independent untimed stage replays with result checksums
  raw.csv, summary.csv            rebuilt from validated checkpoints
  final_comparison.csv/.json      A/B round values, mean and sample std
  table_A.csv, table_B.csv, table_C.csv, comparison.json
  smoke_gate.json                preparation only: PASS certificate
```

Each result includes Recall@10 and mean/p50/p95/p99 latency with serial QPS as
1e6/mean_us. Final tables include both effective round values plus their mean
and sample standard deviation. Cross-dataset summaries use geometric means of
per-workload speedups, never averages of absolute latency or recall.
The 2A geomean requires all five workloads and the incremental B geomean both L2
workloads. Table C retains explicit N/A reasons. Production binaries are checked
for profiling strings and timer symbols; build commands, source hashes, git
commit/dirty source identity, vector.so SHA256, policy SHA256, data hashes, Python
package versions and relevant server settings are recorded.
