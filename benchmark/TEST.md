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