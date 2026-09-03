# Phase 2A Stage 1–4 核心改动与设计目的

## 1. 总体目标

Phase 2A 将 PostgreSQL Executor 已知的 SQL 行数需求传入 IVFFlat，并在不改变 list 选择和距离计算的前提下，用 PostgreSQL bounded tuplesort 降低低维、大候选集查询的排序开销。

完整链路为：

```text
SQL LIMIT / OFFSET
    ↓
LimitState.compute_tuples_needed()
    ↓
ExecSetTupleBound()
    ↓
IndexScanState.iss_TupleBound
    ↓
IndexScanDescData.xs_tuple_bound
    ↓
IvfflatScanOpaqueData.logicalBound
    ↓
conservative physicalBound
    ↓
PostgreSQL bounded tuplesort
    ↓
候选不足时重扫既有 listPages，执行安全 full-sort fallback
```

各阶段职责：

| 阶段 | 核心职责 | 是否改变结果 |
|---|---|---|
| Stage 1 | 传播 Executor logical bound | 否 |
| Stage 2 | 根据 logical bound 启用 conservative bounded sort | 正常无过滤场景不变，但单独使用仍可能因上层过滤而不足 |
| Stage 3 | 检测 bounded candidate exhaustion | 否，只增加状态检测 |
| Stage 4 | 重建 full sort 并跳过已经返回的 TID | 恢复与 full-sort baseline 一致的语义 |

## 2. Stage 1：Logical Bound Propagation

### 2.1 改动目的

复用 OpenTenBase 已有 `ExecSetTupleBound()` 机制，把 Executor 对子节点的最大 tuple 需求传递给普通 `IndexScan` 和 IVFFlat。Stage 1 只记录 logical bound，不启用 bounded sort，不改变候选选择、排序方式或查询结果。

### 2.2 LIMIT 与 OFFSET 语义

`nodeLimit.c:compute_tuples_needed()` 使用当前 OpenTenBase 语义：

```c
return node->count + node->offset;
```

因此：

```text
LIMIT 10             → logical_bound = 10
LIMIT 20             → logical_bound = 20
LIMIT 10 OFFSET 20   → logical_bound = 30
无 LIMIT / LIMIT ALL → logical_bound = -1
WITH TIES            → logical_bound = -1
溢出                 → 负数，按 unknown / unbounded 处理
```

### 2.3 Executor 和 Index AM 状态

新增字段：

```c
/* IndexScanState */
int64 iss_TupleBound;

/* IndexScanDescData */
int64 xs_tuple_bound;

/* IvfflatScanOpaqueData */
int64 logicalBound;
```

`ExecSetTupleBound()` 新增 `IndexScanState` 分支，将 bound 写入 `iss_TupleBound`。`IndexNext()`、`IndexNextWithReorder()`、并行 scan 初始化和 `ExecReScanIndexScan()` 在调用 `index_rescan()` 前将其复制到 `xs_tuple_bound`。

`index_beginscan_internal()` 将 `xs_tuple_bound` 初始化为 `-1`。`ivfflatrescan()` 再从 scan descriptor 复制到 `logicalBound`。

### 2.4 Rescan 目的

Limit 的表达式可能在不同 rescan 中得到不同结果，因此每次 `ExecReScanIndexScan()` 都必须刷新 `xs_tuple_bound`。无 LIMIT 时也必须传播 `-1`，避免复用上一次有限 bound。

### 2.5 修改文件

Stage 1 已提交为 `f9c4d329`，涉及：

```text
src/backend/executor/execProcnode.c
src/backend/executor/nodeIndexscan.c
src/backend/access/index/indexam.c
src/include/nodes/execnodes.h
src/include/access/relscan.h
contrib/pgvector/src/ivfflat.h
contrib/pgvector/src/ivfscan.c
```

## 3. Stage 2：Limit-Aware Bounded Fast Path

### 3.1 改动目的

IVFFlat 仍扫描同样的 lists、生成同样的 candidates 并计算全部候选距离，但只让 tuplesort 保留 conservative Top-N，避免把数万到数十万个候选全部 materialize 后再 full sort。

### 3.2 Logical Bound 到 Physical Bound

默认公式：

```text
physical_bound = max(40, logical_bound × 4)
```

对应 GUC：

| GUC | 默认值 | 目的 |
|---|---:|---|
| `ivfflat.bounded_scan` | `off` | 显式控制 automatic fast path |
| `ivfflat.bound_overfetch` | `4` | logical bound 的保守放大倍数 |
| `ivfflat.bound_min` | `40` | physical bound 下限 |
| `ivfflat.bound_fastpath_limit` | `100` | automatic path 可接受的最大 logical bound |

### 3.3 启用条件与优先级

```text
experimental_sort_bound > 0
    → Oracle bounded path

否则 bounded_scan = on
     且 logical_bound > 0
     且 logical_bound <= bound_fastpath_limit
     且 iterative_scan = off
    → automatic bounded path

否则
    → original full tuplesort
```

`ivfflat.experimental_sort_bound` 继续保留为 Oracle 对照，优先级高于 automatic path。

### 3.4 Tuplesort 生命周期

`InitScanSortState()` 根据当前 scan 是否 bounded，决定是否传入 `TUPLESORT_ALLOWBOUNDED` 和调用 `tuplesort_set_bound()`。

当前 OpenTenBase 的 `tuplesort_reset()` 会清除 batch 的 bounded 状态，因此 `GetScanItems()` 每次 reset 后，在仍处于 bounded phase 时重新设置 `physicalBound`。

排序键为：

```text
(distance ASC, heaptid ASC)
```

第二排序键只用于稳定 equal-distance candidate 的顺序，使 full、Oracle 和 automatic 模式可以逐 ID 对比。

### 3.5 Stage 2 状态与 Profiling

新增：

```c
int64 physicalBound;
bool boundedActive;
```

`IVFFLAT_PROFILE` 增加：

```text
logical_bound
physical_bound
bounded_active
```

GloVe-100 Cosine、`LIMIT 10` 的 automatic path 应得到：

```text
logical_bound=10
physical_bound=40
bounded_active=1
```

## 4. Stage 3：Bounded Candidate Exhaustion Detection

### 4.1 改动目的

SQL 的 logical bound 表示 Executor 最终需要的行数，不代表 Index AM 只需返回相同数量的 TID。MVCC、heap fetch、recheck、`WHERE` filter 和 `OFFSET` 都可能让 Executor 继续向 Index AM 请求 tuple。

如果 bounded phase 已经返回全部保留候选，而 Executor 再次调用 `ivfflatgettuple()`，说明 conservative physical bound 不足，必须进入 fallback。

### 4.2 PostgreSQL bounded tuplesort 的边界

bounded tuplesort 返回满 `physicalBound` 个 tuple 后，再调用一次 `tuplesort_gettupleslot()` 会报：

```text
retrieved too many tuples in a bounded sort
```

因此代码在调用 tuplesort 之前检查：

```c
returnedTidsCount >= physicalBound
```

若候选总数本来就少于 `physicalBound`，则由 `tuplesort_gettupleslot()` 的正常 `false` 返回路径检测 exhaustion。

### 4.3 新增状态

```c
bool boundedExhausted;
bool fallbackTriggered;
```

状态含义：

- `boundedExhausted`：当前 scan 的 bounded phase 已耗尽。
- `fallbackTriggered`：当前 scan 已切换到 full-sort fallback，防止重复触发和重新设置 bound。

## 5. Stage 4：Safe Full-Sort Fallback

### 5.1 改动目的

当 physical bound 不足时，用明确但保守的重扫恢复 full-sort baseline 语义。第一版不实现多级 heap、自适应 selectivity model 或增量扩容。

### 5.2 Fallback 流程

`StartFullSortFallback()` 执行：

```text
boundedExhausted = true
fallbackTriggered = true
    ↓
tuplesort_end(bounded sort)
    ↓
InitScanSortState(..., bounded=false)
    ↓
listIndex = 0
    ↓
GetScanItems(scan, existing query value)
```

该流程只重扫已经选择好的 `listPages`：

- 不调用 `GetScanLists()`。
- 不改变 probes 或 list selection。
- 不扫描额外 lists。
- full fallback 会重新计算这些 listPages 中的 candidate distance 并执行 full tuplesort。

### 5.3 Returned TID 去重

新增：

```c
ItemPointerData *returnedTids;
Size returnedTidsCount;
Size returnedTidsCapacity;
```

记录的是 bounded phase 已经返回给 Index AM caller 的 TID，而不是 Executor 最终接受的行。即使某 TID 被 MVCC、recheck 或 filter 淘汰，它仍然属于“AM 已返回”。

fallback full sort 包含 bounded Top-N，因此返回 candidate 前使用小数组线性检查：

```text
TID 已在 returnedTids → skip
否则                  → return
```

automatic fast path 的默认 bound 很小，第一版使用数组比 hash table 更简单。数组位于 IVFFlat scan temporary memory context，scan 结束时统一释放。

### 5.4 Rescan

每次 `ivfflatrescan()` 重置：

```c
boundedExhausted = false;
fallbackTriggered = false;
returnedTidsCount = 0;
```

数组容量可以复用，但旧 scan 的 TID 不会参与新 scan 的去重。logical/physical bound 和 sortstate 也按新一轮 rescan 重新计算和创建。

### 5.5 Iterative Scan

`ivfflat.iterative_scan != off` 时 automatic bounded path 仍然禁用，因此不会进入新的 exhaustion/fallback 状态机。当前阶段没有设计跨 batch 的 bounded 全局排序语义。

### 5.6 Profiling

新增：

```text
bounded_exhausted
fallback_triggered
returned_from_bounded
returned_after_fallback
```

`returned_from_bounded` 和 `returned_after_fallback` 统计的是 Index AM 实际返回次数。

## 6. 正确性与鲁棒性验证

### 6.1 Normal Case

GloVe-100 Cosine、`probes=64`、无 filter、`LIMIT 10`：

```text
logical_bound=10
physical_bound=40
bounded_active=1
bounded_exhausted=0
fallback_triggered=0
returned_from_bounded=10
returned_after_fallback=0
```

### 6.2 WHERE Filter Fallback

约 1% 选择率：

```sql
WHERE id % 100 = 0
ORDER BY embedding <=> query
LIMIT 10
```

验证结果：

```text
full candidates=79,334
auto candidates=158,668
auto returned_from_bounded=40
auto returned_after_fallback=1,062
auto fallback_triggered=1
```

auto 重扫一次相同 listPages，因此 candidate 计数约为 full 的两倍。full 与 auto 返回的有序 Top-10 ID 完全一致，均返回 10 个不重复 ID。相对精确顺序扫描，两者 Recall@10 都为 `0.8`，fallback 没有引入额外 recall 回归。

### 6.3 LIMIT + OFFSET

`LIMIT 10 OFFSET 20` 加 1% filter：

```text
logical_bound=30
physical_bound=120
fallback_triggered=1
returned_from_bounded=120
returned_after_fallback=3,141
```

full 与 auto 的最终 10 个有序 ID 完全一致。

### 6.4 参数化 Rescan

参数化 LATERAL IndexScan 连续执行两轮：第一轮触发 fallback，第二轮走 normal bounded path。两轮结果均与 full baseline 一致，每轮 `returned_rows=10`、`distinct_rows=10`。第二轮没有携带第一轮 returned TID 或 exhaustion 状态。

### 6.5 Robustness Matrix

现有 runner 增加 `phase=2a34`，小规模正确性矩阵使用：

```text
GloVe-100 Cosine
lists=1000
probes=64
LIMIT=10
20 queries
selectivity=100%, 50%, 10%, 1%, 0.1%, 0.01%
modes=full, auto
```

结果：

```text
240/240 records correct
0 duplicate result IDs
minimum recall_vs_full=1.0
79 automatic fallback queries
```

0.01% 选择率时 IVFFlat 所选 lists 平均只能提供 7.45 个符合 filter 的行；auto 与 full 的返回行数和顺序仍完全一致。

结果文件：

```text
/workspace/benchmark/runs/20260903T062000Z/phase_2a34_robustness_raw.csv
/workspace/benchmark/runs/20260903T062000Z/phase_2a34_robustness_summary.csv
```

## 7. 修改文件汇总

### 7.1 Stage 1 已提交文件

```text
src/backend/executor/execProcnode.c
src/backend/executor/nodeIndexscan.c
src/backend/access/index/indexam.c
src/include/nodes/execnodes.h
src/include/access/relscan.h
contrib/pgvector/src/ivfflat.h
contrib/pgvector/src/ivfscan.c
```

### 7.2 当前 Stage 2–4 核心源码

```text
contrib/pgvector/src/ivfflat.c
contrib/pgvector/src/ivfflat.h
contrib/pgvector/src/ivfscan.c
```

### 7.3 Benchmark runner

实际运行脚本位于当前 OpenTenBase Git worktree：

```text
/workspace/OpenTenBase/benchmark/scripts/ivfflat_profile.py
/workspace/OpenTenBase/benchmark/scripts/run_ivfflat_experiments.sh
/workspace/OpenTenBase/benchmark/scripts/run_ivfflat_experiments_worker.sh
```

这些脚本新增 `phase=2a2` 和 `phase=2a34`，并复用同一 launcher/worker，没有创建完整平行 runner。

## 8. 当前明确不做的内容

- 不实现 selectivity-aware overfetch。
- 不实现自适应 candidate model。
- 不实现 multi-tier heap。
- 不扩展 iterative scan 的 bounded/fallback 语义。
- fallback 不选择新 lists，只重扫现有 `listPages`。
- 当前 robustness matrix 是正确性验证，不是正式性能 sweep。

高选择率 normal case 保留 Stage 2 的性能收益；低选择率触发 fallback 时会重复 candidate distance 计算和 full sort，性能可能低于直接 full sort，这是第一版安全性优先设计的预期代价。

## 9. 构建与测试状态

- `IVFFLAT_BENCH` 构建成功。
- 非 profiling production 构建成功。
- 扩展安装成功。
- PostgreSQL 已使用 `dev` 用户重启并通过 `pg_isready`。
- Python 和 Shell benchmark 脚本语法检查通过。
- `git diff --check` 通过。
- pgvector 14 组 regression SQL 已执行；排除 `IVFFLAT_BENCH` 专有 `INFO` 行后，实际输出与 expected 完全一致。