以下命令均使用 canonical runner：

```bash
cd /workspace/OpenTenBase
```

### 1. GIST：1,000 queries，probes=128

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --dataset gist-l2 \
  --queries 1000 \
  --warmup-queries 100 \
  --probes-list 128 \
  --lists 1000 \
  --topk 10
```

该命令自动运行：

```text
full × probes=128
auto × probes=128
```

### 2. GIST：快速 10-query 验证

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --dataset gist-l2 \
  --queries 10 \
  --warmup-queries 2 \
  --probes-list 128 \
  --lists 1000 \
  --topk 10
```

### 3. GIST：多个 probes

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --dataset gist-l2 \
  --queries 1000 \
  --warmup-queries 100 \
  --probes-list 16,64,128,256 \
  --lists 1000 \
  --topk 10
```

### 4. GIST：完整 probes sweep

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --dataset gist-l2 \
  --queries 1000 \
  --warmup-queries 100 \
  --probes-list 1,2,4,8,16,32,64,128,256 \
  --lists 1000 \
  --topk 10
```

GIST 官方测试集只有 1,000 queries，因此 GIST 不应指定 `--queries 10000`。

### 5. GloVe：100-query smoke test

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --dataset glove-cosine \
  --queries 100 \
  --warmup-queries 100 \
  --probes-list 16,64 \
  --lists 1000 \
  --topk 10
```

### 6. GloVe：完整 10,000-query formal

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --dataset glove-cosine \
  --queries 10000 \
  --warmup-queries 100 \
  --probes-list 1,2,4,8,16,32,64,128,256 \
  --lists 1000 \
  --topk 10
```

### 7. 启用 Ground Truth sanity check

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --dataset gist-l2 \
  --queries 100 \
  --warmup-queries 10 \
  --probes-list 128 \
  --verify-ground-truth \
  --ground-truth-queries 20
```

### 8. 恢复中断实验

必须传入原实验相同的配置：

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --resume /workspace/benchmark/runs/<RUN_ID> \
  --dataset gist-l2 \
  --queries 1000 \
  --warmup-queries 100 \
  --probes-list 128 \
  --lists 1000 \
  --topk 10
```

例如恢复本次 GIST 配置：

```bash
./benchmark/scripts/run_ivfflat_experiments.sh formal \
  --resume /workspace/benchmark/runs/gist_formal_1000_p128_20260904 \
  --dataset gist-l2 \
  --queries 1000 \
  --warmup-queries 100 \
  --probes-list 128 \
  --lists 1000 \
  --topk 10
```

### 9. 监控运行状态

启动后 runner 会输出实际 `RUN_DIR` 和日志路径。可使用：

```bash
tail -f /workspace/benchmark/runs/<RUN_ID>/experiment.log
```

查看状态：

```bash
cat /workspace/benchmark/runs/<RUN_ID>/status
```

检查进程：

```bash
ps -ef | grep '[r]un_ivfflat_experiments_worker'
```

formal phase 当前固定展开 `full` 和 `auto` 两种模式，不能通过 CLI 只选择其中一种。每个新运行目录会统一生成：

```text
phase_formal_raw.csv
phase_formal_summary.csv
```
### Phase 2B correctness：统一比较入口

数据库实验继续使用三级结构：launcher 管理 run-dir/PID/nohup，worker 管理构建、安装、重启和 phase dispatch，`ivfflat_profile.py` 管理查询、比较与结果输出。

FUSED2 Part 2（DIRECT → FUSED2，含 tail、NULL、vacuum 空页及 forced fallback 检查）：

```bash
PYTHON_BIN=/workspace/benchmark/.venv/bin/python \
./benchmark/scripts/run_ivfflat_experiments.sh 2b-correctness \
  --baseline-path direct --test-path fused2 \
  --probes-list 64 --queries 100 --warmup 0 \
  --topk 10 --lists 1000 --unsupported-queries 3 \
  --check-fused2-edges
```

原 Phase 2B-1（generic → DIRECT）仍可运行，且这是 path 参数的默认组合：

```bash
PYTHON_BIN=/workspace/benchmark/.venv/bin/python \
./benchmark/scripts/run_ivfflat_experiments.sh 2b-correctness \
  --baseline-path generic --test-path direct \
  --probes-list 64 --queries 100 --warmup 0
```

correctness 总是运行 Full/Automatic，复用已有 GIST1M、960D、L2 索引；不重建索引。额外使用已有 IP/cosine 索引验证不支持的路径落回 generic。`--check-fused2-edges` 默认关闭，只适用于含 fused2 的 correctness 比较；它用独立数据库连接创建临时表/索引，退出时清理，不修改 GIST 数据。其 forced fallback 子测试固定 query 0、probes=64、LIMIT 10，临时页子测试 probes=1。

worker 使用既有 Haswell flags 和 `IVFFLAT_PROFILE_CFLAGS`，不覆盖 `PG_CFLAGS`。参数在构建前校验。连接参数仍使用 `DB_HOST/DB_PORT/DB_NAME/DB_USER`，path 也可通过环境变量配置：

```text
PHASE2B_BASELINE_PATH=generic
PHASE2B_TEST_PATH=direct
PHASE2B_CHECK_FUSED2_EDGES=0
```

CLI 参数覆盖对应环境变量默认值；worker 管理 phase 和 output，不能通过额外参数替换。实际 Python 命令保存为 `run-command.txt`，环境和原始 CLI 保存在 `environment.txt`。

统一输出前缀为 `phase_2b_correctness`（替代旧 worker 的 `phase_2b_direct_correctness`）：

- `_raw.csv`：逐 query 的路径、结果 TID/ID、距离 bits、workload 和 counters。
- `_summary.csv`：Full/Automatic 的逐 query mismatch 汇总，标注 baseline/test；保留旧 2B-1 mismatch 列名。
- `_coverage.csv`：使用 fused2 时输出实际 pairs/candidates/tails/fallback 与 coverage。
- `_unsupported_raw.csv`、`_unsupported_summary.csv`：IP/cosine 安全 dispatch 检查。
- `_manifest.json`、`_plan_full.json`、`_plan_auto.json`：query hash、索引身份、配置、实际查询计划。
- `_edges.json`：启用 edge checks 时输出四组补充验证记录。

任意 workload、结果、计数或 dispatch 违规都会失败退出；检查不依赖 Python `assert`。距离 bits 是 SQL 返回 float8 的位表示，不是额外导出的 tuplesort 内部 squared-distance bits。page/pair counters 用于核对观察到的页分布，不声称仅靠聚合计数就能证明任意页面的 lifetime 安全。

如果当前扩展已构建，可绕过 build/install/restart，直接使用相同 Python 入口：

```bash
/workspace/benchmark/.venv/bin/python benchmark/scripts/ivfflat_profile.py \
  --host 127.0.0.1 --port 5432 --dbname taskdb --user dev \
  run --phase 2b-correctness --baseline-path direct --test-path fused2 \
  --queries 100 --warmup 0 --probes-list 64 --check-fused2-edges \
  --output /workspace/benchmark/results/fused2_correctness
```

两个原独立 Python 脚本已合并，不再作为入口。Raw C benchmark 保持独立，迁至 [microbench/fused2_raw_bench.c](microbench/fused2_raw_bench.c)，运行说明见 [microbench/README.md](microbench/README.md)。

### Phase 2B profiling：参数化比较对象

`--distance-path` 支持 `generic/direct/fused2/interleaved`。单路径模式按指定 path 运行；`interleaved` 根据 `--baseline-path/--test-path` 在相邻 round 交换顺序。默认仍为 generic/direct。

后续需要 profiling 时可使用以下入口；correctness 完成后不会自动执行：

```bash
PYTHON_BIN=/workspace/benchmark/.venv/bin/python \
./benchmark/scripts/run_ivfflat_experiments.sh 2b \
  --distance-path interleaved --baseline-path direct --test-path fused2 \
  --mode both --probes-list 64 --queries 100 --warmup 10 --rounds 2
```

`_runs.csv` 保存每路径的 logical count、fused counters、coverage 与计时；`_paired.csv` 和 `_summary.csv` 使用 `baseline_* / test_*`，并记录具体 path。generic/direct 组合额外保留历史 `generic_* / direct_*` 数值列。`order` 使用完整路径名。逐 query 检查 workload，防止聚合总数相等掩盖差异。

FUSED2 的 `distance_ns_per_candidate` 使用整个 distance stage（含 DIRECT tails）；`fused_distance_ns` 含 pairing/lookahead 与第二结果消费开销，不能当作纯 Raw kernel 时间。该入口重构不代表已完成 Part 3 性能判断。

离线回归检查（不连接数据库）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmark/tests -v
bash -n benchmark/scripts/run_ivfflat_experiments.sh
bash -n benchmark/scripts/run_ivfflat_experiments_worker.sh
```
