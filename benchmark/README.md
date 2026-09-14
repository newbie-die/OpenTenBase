# OpenTenBase benchmark layout

Benchmark source code is versioned here. Runtime data is kept outside the Git
working tree so that a run cannot turn logs, checkpoints, databases, or large
datasets into source files.

| Location | Responsibility |
| --- | --- |
| `benchmark/scripts/` | Command entry points, schedulers, aggregation and shared path handling |
| `benchmark/configs/` | Versioned workload definitions |
| `benchmark/d2p/` | D2-P algorithm-specific library code and frozen policy artifacts |
| `benchmark/tests/` | Runner, resume and aggregation regression tests |
| `$BENCHMARK_RUNTIME_ROOT` | Datasets, virtual environment, logs, checkpoints and run outputs |

The runtime root defaults to `<workspace>/benchmark`, which is
`/workspace/benchmark` in the standard development environment. Set
`BENCHMARK_RUNTIME_ROOT` or pass `--runtime-root` when using another layout.

## Unified command

Use `benchmark/scripts/benchmark.py` for new runs. The phase and parameters are
the only parts that change between workloads:

```bash
# Phase 2A smoke, foreground
benchmark/scripts/benchmark.py run 2a \
  --run-dir phase2a_smoke --warmup 10 --queries 100 --probes 64

# Final multi-dataset preparation
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_prepare

# Formal run or safe resume in the background
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_formal --formal \
  --prepared final_multi_prepare --resume --background

# Read state without touching the run
benchmark/scripts/benchmark.py status --run-dir final_multi_formal

# Preview the resolved command without creating files or executing queries
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_formal --formal \
  --prepared final_multi_prepare --resume --background --dry-run
```

Every run directory is under `$BENCHMARK_RUNTIME_ROOT/runs` unless an absolute
path inside the runtime root is supplied. Logs are written to
`<run-dir>/experiment.log`; launcher metadata and background PID files are run
artifacts in that same directory. The launcher rejects outputs outside the
runtime root.

The existing `run_ivfflat_experiments.sh` remains as a compatible lower-level
entry point. It now accepts `--run-dir`, while the Python entry point handles
path resolution and presents one interface for the legacy phases and
`final-multi`.

## Source/runtime duplication

Run `benchmark.py audit` to list Python, shell, C and header files found below
the runtime root, including exact SHA256 matches in the source tree:

```bash
benchmark/scripts/benchmark.py audit \
  --output audits/benchmark-code-layout.json
```

Historical run directories may contain copied source, one-time repair scripts,
or “before” snapshots. They are retained as immutable evidence, but are not
execution sources. New reusable logic belongs in this repository; new run
directories contain only inputs, logs, manifests, checkpoints and results.
