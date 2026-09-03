# Phase 2A 当前代码更改总结

## 1. 目标与实验边界

本次修改实现了 **Phase 2A: Low-Dim Prototype — Oracle Bounded Tuplesort**，用于验证低维 IVFFlat 查询中，如果扫描阶段提前获得候选排序上限，PostgreSQL 原生 bounded tuplesort 能否降低 candidate materialization 和 sorting 开销。

这是性能上限实验，不是生产级 SQL `LIMIT` 下推。实验 bound 由用户显式设置，代码没有修改 executor、`IndexScanDesc`、Index AM API、IVFFlat 磁盘格式或候选选择算法。

## 2. pgvector C 代码更改

### 2.1 新增实验 GUC

文件：`contrib/pgvector/src/ivfflat.c`

新增全局变量：

```c
int ivfflat_experimental_sort_bound;
```

注册 GUC：

```text
ivfflat.experimental_sort_bound
```

配置语义：

- 默认值：`0`
- 范围：`0 .. INT_MAX`
- 级别：`PGC_USERSET`
- `0`：保持原始 full tuplesort
- `N > 0`：启用实验性 bounded tuplesort，bound 为 N

示例：

```sql
SHOW ivfflat.experimental_sort_bound;
SET ivfflat.experimental_sort_bound = 10;
```

### 2.2 扩展 IVFFlat scan state

文件：`contrib/pgvector/src/ivfflat.h`

新增 GUC extern 声明，并在 `IvfflatScanOpaqueData` 中增加：

```c
int sortBound;
```

扫描开始时将当前 session GUC 值保存到 scan state，使一次扫描使用稳定的 bound。

### 2.3 集成 PostgreSQL bounded tuplesort

文件：`contrib/pgvector/src/ivfscan.c`

`InitScanSortState()` 已改为接收 bound：

```c
InitScanSortState(TupleDesc tupdesc, int bound)
```

行为如下：

```text
bound == 0  -> TUPLESORT_NONE
bound > 0   -> TUPLESORT_ALLOWBOUNDED + tuplesort_set_bound()
```

当前 OpenTenBase 的真实 API 最后一个参数为 `int sortopt`：

```c
Tuplesortstate *tuplesort_begin_heap(
    TupleDesc tupDesc,
    int nkeys,
    AttrNumber *attNums,
    Oid *sortOperators,
    Oid *sortCollations,
    bool *nullsFirstFlags,
    int workMem,
    SortCoordinate coordinate,
    int sortopt
);
```

### 2.4 处理 tuplesort_reset()

当前 OpenTenBase 的 `tuplesort_reset()` 会调用 `tuplesort_begin_batch()`，从而清除：

```c
state->bounded = false;
state->boundUsed = false;
```

因此 `GetScanItems()` 在每次 reset 后都会重新执行：

```c
if (so->sortBound > 0)
    tuplesort_set_bound(so->sortstate, so->sortBound);
```

这也覆盖 iterative scan 导致的重复 `GetScanItems()` 调用；Phase 2A 正式实验仍默认关闭 iterative scan。

### 2.5 扩展 profiling 输出

原有 `IVFFLAT_BENCH` instrumentation 被继续复用，`IVFFLAT_PROFILE` 新增：

- `probes`
- `dimensions`
- `sort_bound`

完整关键指标包括：

- candidates
- getitems_us
- candidate_us
- distance_us
- sort_us
- return_us
- scan_us

`sort_us` 只测量 `tuplesort_performsort()`。

## 3. Benchmark 代码更改

所有正式脚本位于 `benchmark/scripts/`，运行时默认根目录由脚本自身位置推导，也可通过 `BENCHMARK_ROOT` 覆盖。

### 3.1 Launcher

文件：`benchmark/scripts/run_ivfflat_experiments.sh`

- 保留原有 `all`、`a`、`b` phase。
- 新增 `2a` phase。
- 继续使用 PID 文件防止多个实验并发运行。
- 为每次实验创建独立时间戳目录和日志文件。

启动命令：

```bash
/workspace/OpenTenBase/benchmark/scripts/run_ivfflat_experiments.sh 2a
```

### 3.2 Worker

文件：`benchmark/scripts/run_ivfflat_experiments_worker.sh`

Phase 2A 默认参数：

```text
PHASE2A_WARMUP=500
PHASE2A_QUERIES=10000
PHASE2A_PROBES=16,64,128
PHASE2A_BOUNDS=0,10,20,40,100
PHASE2A_ROUNDS=3
TOPK=10
LISTS=1000
```

Worker 会：

1. 检查 Python、`pg_config`、`pg_ctl` 和数据目录。
2. 检查或准备 Python 依赖。
3. 使用 `-DIVFFLAT_BENCH` 构建并安装 pgvector。
4. 以数据目录所有者重启 PostgreSQL。
5. Phase 2A 只构建一次 GloVe L2 IVFFlat index。
6. 串行执行 3 probes × 5 bounds × 3 rounds。

#### Python 依赖安装修复

环境中的大小写代理变量都指向不可用的 `127.0.0.1:7890`，会导致 PyPI TLS 连接被关闭。pip 子进程现在显式清除：

```text
HTTPS_PROXY
HTTP_PROXY
https_proxy
http_proxy
```

依赖已安装并验证：

```text
h5py 3.14.0
numpy 2.0.2
psycopg2-binary 2.9.12
```

#### PostgreSQL 运行用户修复

`make install` 保持以 root 执行，但 PostgreSQL 不能由 root 启动。worker 新增：

```text
PG_OS_USER=dev
```

重启命令为：

```bash
runuser -u "${PG_OS_USER}" -- "${PG_CTL}" \
  -D "${PGDATA}" restart -m fast -w
```

可通过环境变量覆盖 `PG_OS_USER`。

### 3.3 Phase 2A Python 驱动

文件：`benchmark/scripts/ivfflat_profile.py`

Phase 2A 只运行：

- 数据集：GloVe-100
- metric：L2
- operator：`<->`
- table：`glove_base`
- index：`glove_ivf_l2`

每个配置显式执行：

```sql
SET enable_seqscan = off;
SET ivfflat.iterative_scan = off;
SET ivfflat.probes = ...;
SET ivfflat.experimental_sort_bound = ...;
```

正式阶段会测量完整 10,000 条 HDF5 test query；warmup 复用前 500 条 query，不会减少正式 query 数。

每个 `(probes, sort_bound, round)` 记录：

- dataset、metric、rows、dimension、lists
- probes、sort_bound、round、queries、topk
- recall_at_10
- p50_ms、p95_ms、p99_ms、mean_ms
- qps
- avg_returned_rows、min_returned_rows
- status
- avg_candidates
- avg_getitems_us、avg_candidate_us、avg_distance_us、avg_sort_us
- candidate_ratio、distance_ratio、sort_ratio

当 `min_returned_rows != topk` 时，配置被标记为 `unsafe/incorrect`。

输出文件不会互相覆盖，例如：

```text
glove_l2_p64_b0_r1.csv
glove_l2_p64_b10_r1.csv
glove_l2_p64_b20_r1.csv
phase2a_raw.csv
phase2a_summary.csv
```

### 3.4 独立汇总脚本

文件：`benchmark/scripts/summarize_phase2a.py`

从所有 `glove_l2_p*_b*_r*.csv` 重新生成：

```text
phase2a_raw.csv
phase2a_summary.csv
```

summary 以同 probes 下的 `sort_bound=0` 为 baseline，计算：

- median Recall、P50、P95、P99、QPS
- baseline P95/QPS
- P95 improvement percentage
- QPS improvement percentage
- 最小返回行数和正确性状态

## 4. 当前验证结果

### 4.1 构建与安装

以下路径均已验证成功：

- 普通 pgvector 构建
- `PG_CFLAGS=-DIVFFLAT_BENCH` 构建
- `make install`
- PostgreSQL 以 `dev` 用户重启
- `pg_isready`

### 4.2 GUC 验证

实际结果：

```text
SHOW -> 0
SET ivfflat.experimental_sort_bound = 10
SHOW -> 10
```

### 4.3 单查询正确性 sanity

同一 GloVe 查询、`probes=64`、`iterative_scan=off`：

```text
bound=0  -> 10 rows
bound=10 -> 10 rows
same_order=true
same_set=true
```

相同 ID 集合意味着该查询两种 bound 的 Recall 相同；正式脚本仍会对全部查询重新计算数值 Recall@10。

### 4.4 bounded sort 机制 sanity

一次实测 profiling：

```text
bound=0:
  candidates=268959
  sort_us=50427
  returned_rows=10

bound=10:
  candidates=268959
  sort_us=1
  returned_rows=10
```

该结果证明 bounded tuplesort 已实际启用，candidate 数量没有变化。它只是单查询机制检查，不能替代正式 3-round benchmark，也不能直接作为最终性能结论。

## 5. 尚未解决的语义问题

本次 Prototype 未解决：

- SQL LIMIT 到 Index AM 的安全传播
- MVCC 不可见 tuple
- heap fetch、recheck 和 WHERE filter 淘汰
- OFFSET
- join 或其他上层 executor 节点淘汰
- bound 过小时结果不足
- iterative scan 的跨 batch 全局排序语义

因此：

```text
SQL LIMIT 10 != IVFFlat 内部安全 bound 10
```

`ivfflat.experimental_sort_bound` 只能用于受控 benchmark，不应作为生产级正确性优化。

## 6. 正式实验运行方式

默认正式实验：

```bash
/workspace/OpenTenBase/benchmark/scripts/run_ivfflat_experiments.sh 2a
```

缩小矩阵进行 smoke test：

```bash
PHASE2A_PROBES=64 \
PHASE2A_BOUNDS=0,10 \
PHASE2A_ROUNDS=1 \
PHASE2A_WARMUP=10 \
PHASE2A_QUERIES=100 \
/workspace/OpenTenBase/benchmark/scripts/run_ivfflat_experiments.sh 2a
```

查看日志：

```bash
tail -f /workspace/benchmark/runs/<run-id>/experiment.log
```

## 7. 当前 Git 修改范围

已修改：

```text
contrib/pgvector/src/ivfflat.c
contrib/pgvector/src/ivfflat.h
contrib/pgvector/src/ivfscan.c
```

已新增：

```text
benchmark/README_phase2a.md
benchmark/PHASE2A_CHANGES.md
benchmark/scripts/ivfflat_profile.py
benchmark/scripts/run_ivfflat_experiments.sh
benchmark/scripts/run_ivfflat_experiments_worker.sh
benchmark/scripts/summarize_phase2a.py
```

没有修改：

- `/workspace/upstream-pgvector`
- executor 和 Index AM API
- IVFFlat index build/磁盘格式
- benchmark HDF5、TSV 或数据库文件

## 8. Go / No-Go 标准

- P95 和 QPS 改善均小于 5%：No-Go，停止 Phase 2A。
- 改善为 5% 到 15%：Weak-Go。
- P95 或 QPS 改善至少 15%，且 Recall 不下降、所有 query 始终返回 10 行：Strong-Go。
- 只有 Strong-Go 后才扩展到 GloVe IP、Cosine 或 SIFT。
