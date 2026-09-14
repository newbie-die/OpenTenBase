# OpenTenBase IVFFlat 查询性能优化

## 1. 项目目标

本项目针对 pgvector IVFFlat 查询中的候选排序、L2 距离计算和固定 `probes` 带来的额外开销进行优化。

主要目标是：

- 在不改变 SQL Top-K 语义的前提下减少排序工作；
- 加快高维 L2 candidate 的 exact distance 计算；
- 让简单查询可以少扫描一部分 lists 和 candidates；
- 用可恢复、可审计的 benchmark 验证延迟、QPS 和 Recall@10；
- 所有核心代码保存在 OpenTenBase 仓库，数据、日志和结果保存在 `/workspace/benchmark`。

## 2. 完成的优化

### 2.1 LIMIT 感知的 Top-K 排序优化

原始 IVFFlat 扫描会把候选持续送入 tuplesort。候选数量较多时，排序和候选保留会占用明显的查询时间。

优化后将 SQL `LIMIT` 通过 executor 传入 IVFFlat 扫描，根据逻辑 Top-K 设置 bounded tuplesort。当前正式参数为：

```text
topk=10
bound_overfetch=4
bound_min=40
bound_fastpath_limit=100
```

扫描过程通常只需要在排序结构中保留 40 个候选。如果过滤条件导致候选不足，runner 会验证 fallback 能继续返回正确结果。

这项优化具有以下特点：

- 支持 L2、Cosine 和 Inner Product；
- 不改变选中的 lists 和扫描到的 candidates；
- 不改变 exact distance 结果；
- 不改变同一 workload、同一 `probes` 下的 Recall；
- 已在无过滤、Cosine、Inner Product 和不同过滤选择率下验证。

代码主要位于 PostgreSQL executor 的 LIMIT/IndexScan 传播路径以及 `contrib/pgvector/src/ivfscan.c`。

### 2.2 L2 距离计算优化

针对 L2 exact distance，新增 direct 和 FUSED2 路径。FUSED2 的主要做法是：

- 使用 AVX2/FMA；
- 一次处理两个 candidate vectors；
- 两个 candidate 共享 query vector 的加载；
- 使用两个独立 accumulator，增加指令级并行；
- 对单数尾项和不支持的距离类型保留原路径 fallback。

独立 IVFFlat-like candidate pool 微基准中，FUSED2 相对两次 direct kernel 的 median 时间降低 `30.55%`，即 `1.440×`。这个数字只表示距离内核，不等同于完整 SQL 查询的提升。

PostgreSQL 内的 correctness 和 production 测试还核对了 selected lists、scanned pages、candidate 数量、distance calls、tuplesort inputs、返回 TID/ID 和 distance bits。FUSED2 当前只用于 L2；Cosine 和 Inner Product 会回退，不进入正式 `2A+B` 矩阵。

核心实现位于 `contrib/pgvector/src/vector.c` 和 `contrib/pgvector/src/ivfscan.c`，独立微基准位于 `benchmark/microbench/`。

### 2.3 动态 probes

固定方案对每条查询都使用 `probes=64`。动态方案将一次扫描拆成：

```text
16 -> 32 -> 64
```

扫描到 16 和 32 个 lists 时，D2-P 根据 centroid distance 和当前扫描状态生成低成本特征。冻结的浅层决策树判断当前 Top-K 是否已经足够稳定：简单查询可以提前结束，困难查询继续扫描到 32 或 64。

为控制风险，动态 probes 包含以下保护：

- GIST 策略使用与正式 query 分离的 `gist_learn` 数据校准；
- SIFT 预留前 1000 条 query 校准，后 9000 条用于正式测试；
- 校准阶段核对固定 probes、progressive shadow 和正式部署路径；
- C 与 Python 特征、stop decision 和 Top10 结果必须一致；
- 不具备独立校准集或距离语义尚未适配的数据集不启用 D。

目前正式验证覆盖 GIST-L2 和 SIFT-L2。GloVe-Cosine、Cohere-Cosine 和 GloVe-IP 仍标记为 N/A。

核心实现位于 `contrib/pgvector/src/ivfscan.c`、`contrib/pgvector/src/ivfd2ppolicy.h` 和 `benchmark/d2p/`。

## 3. Benchmark

### 3.1 环境与统计口径

- CPU：2 × Intel Xeon E5-2678 v3，24 cores / 48 threads；
- 指令集：AVX2、FMA；
- OpenTenBase：PostgreSQL 18.6；
- pgvector：0.8.6；
- GCC：11.5.0；
- `lists=1000`、`topk=10`、基准 `probes=64`；
- 每个配置执行 100 条 warmup；
- Final Round A 和 B 共两个有效 rounds；
- 共 28 config-rounds、164000 次正式 query executions；
- 表格延迟是两轮 `mean_us` 的算术平均，单位换算为 ms/query；
- speedup 是两轮各自 speedup 的算术平均，未选择最好的一轮；
- Recall 是 Recall@10。

正式原始结果位于：

```text
/workspace/benchmark/runs/final_multi_formal/final_comparison.csv
/workspace/benchmark/runs/final_multi_formal/final_comparison.json
```

### 3.2 LIMIT Top-K 优化

| Dataset | Metric | Baseline mean (ms) | Top-K mean (ms) | Speedup | Baseline Recall | Top-K Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GIST-960 | L2 | 289.849 | 178.397 | 1.625× | 0.979500 | 0.979500 |
| SIFT-128 | L2 | 36.906 | 19.569 | 1.886× | 0.995178 | 0.995178 |
| GloVe-100 | Cosine | 44.381 | 20.425 | 2.176× | 0.928230 | 0.928230 |
| Cohere-768 | Cosine | 233.103 | 152.396 | 1.534× | 0.986100 | 0.986100 |
| GloVe-100 | Inner Product | 39.577 | 19.656 | 2.018× | 0.926090 | 0.926090 |

五个 workload 的 speedup 几何平均为 `1.831×`。各 workload 的 Recall 与 Baseline 相同。

### 3.3 L2 优化

`Top-K + L2 Kernel` 对应正式矩阵中的 `2A+B`。增量 speedup 以同一轮的 Top-K 延迟为基准。

| Dataset | Baseline (ms) | Top-K (ms) | Top-K + L2 Kernel (ms) | Kernel 增量 speedup | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| GIST-960 | 289.849 | 178.397 | 199.753 | 0.940× | 0.979500 |
| SIFT-128 | 36.906 | 19.569 | 17.918 | 1.093× | 0.995178 |

SIFT 的完整 SQL 延迟在 Top-K 优化基础上进一步降低约 `8.4%`。GIST 两轮的 FUSED2 增量 speedup 分别为 `0.727×` 和 `1.152×`，波动较大，两轮均值没有形成稳定增益。因此当前结论是：FUSED2 内核优化成立，SIFT 端到端收益明确；GIST 的组合收益还需要更多轮次或更稳定的运行环境确认。

### 3.4 动态 probes

| Dataset | Fixed 64 (ms) | Dynamic (ms) | Avg probes | Speedup | 延迟降低 | Fixed Recall | Dynamic Recall | Recall 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GIST-960 | 289.849 | 164.933 | 48.768 | 1.761× | 43.1% | 0.979500 | 0.973500 | -0.006000 |
| SIFT-128 | 36.906 | 26.150 | 45.140 | 1.418× | 29.1% | 0.995178 | 0.992722 | -0.002456 |

动态 probes 在两个 L2 workload 上都减少了平均扫描 lists，并获得端到端延迟收益。它会改变实际 `probes`，因此 Recall 不会像 Top-K/FUSED2 一样严格保持不变；表中给出了实际绝对变化。

## 4. 如何复现

### 4.1 准备环境与数据

完整依赖、数据路径、数据库表和索引要求见 [`../README.md`](../README.md)。默认布局为：

```text
源码       /workspace/OpenTenBase
运行目录   /workspace/benchmark
Python     /workspace/benchmark/.venv/bin/python
OpenTenBase /workspace/install
PGDATA     /workspace/data
```

进入仓库：

```bash
cd /workspace/OpenTenBase
```

### 4.2 执行 preparation

preparation 会检查五个 workload 的数据、ground truth、表、索引、opclass、lists、query split 和 EXPLAIN，同时完成 smoke 与 SIFT D 校准准备。它不会自动启动正式长实验。

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_prepare_repro
```

确认准备状态：

```bash
cat /workspace/benchmark/runs/final_multi_prepare_repro/smoke_gate.json
```

只有 `smoke_gate.json` 为 PASS 才能进入正式执行。

### 4.3 执行正式 benchmark

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_formal_repro \
  --formal \
  --prepared final_multi_prepare_repro \
  --background
```

查看进度和日志：

```bash
benchmark/scripts/benchmark.py status \
  --run-dir final_multi_formal_repro

tail -f /workspace/benchmark/runs/final_multi_formal_repro/experiment.log
```

如果进程中断，使用同一目录恢复。已完成且通过 checkpoint 校验的 config 不会重跑：

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_formal_repro \
  --formal \
  --prepared final_multi_prepare_repro \
  --resume --background
```

完成后查看：

```text
/workspace/benchmark/runs/final_multi_formal_repro/final_comparison.csv
/workspace/benchmark/runs/final_multi_formal_repro/final_comparison.json
```

### 4.4 单条复杂查询演示

仓库提供 `benchmark/scripts/demo_ivfflat_query.py`，默认演示 GloVe-Cosine 的 qid 7097。这是两个正式 rounds 中 Baseline/2A 结果和 Recall 均一致的查询里，提升最高的一条：延迟分别为 `63.204 -> 12.763 ms` 和 `54.245 -> 13.692 ms`，两轮 speedup 为 `4.952×` 和 `3.962×`，几何平均为 `4.429×`，Recall@10 均为 `1.0`。

交互演示命令：

```bash
benchmark/scripts/demo_ivfflat_query.py --pause
```

脚本先预热两个路径，再展示 SQL、query vector 和 ground truth。按两次 Enter 后依次运行无优化 Baseline 和最优 2A，每次都会打印查询结果、Recall@10 和查询时间，最后检查结果一致性并显示现场 speedup。默认查询会完整显示 100 维向量；自定义高维查询默认只显示头尾，需要完整向量时添加 `--show-full-vector`。也可通过 `--dataset` 和 `--qid` 选择其他正式查询。

## 5. 结果总结

- 通用 LIMIT Top-K 优化在五个 workload 上为 `1.534× ~ 2.176×`，speedup 几何平均为 `1.831×`，Recall 不变。
- 独立 IVFFlat-like L2 kernel 微基准最高降低 `30.55%` 距离计算时间。
- SIFT 的 L2 kernel 在 Top-K 优化基础上获得 `1.093×` 增量 speedup，完整 SQL 延迟进一步降低约 `8.4%`。
- GIST 的 FUSED2 端到端结果在两个正式 rounds 间波动较大，当前不将其描述为稳定收益。
- GIST 动态 probes 平均使用 `48.768` probes，延迟降低 `43.1%`，Recall@10 下降 `0.006000`。
- SIFT 动态 probes 平均使用 `45.140` probes，延迟降低 `29.1%`，Recall@10 下降 `0.002456`。
- 当前最稳定、覆盖范围最广的优化是 LIMIT Top-K；FUSED2 仅支持 L2；动态 probes 只在有独立校准和部署校验的 GIST/SIFT 上启用。

## 6. 尝试但未采用的方案

### Page-level distance pruning

D-0A 推导了 metric-aware 安全边界，D-0B 在 GIST-L2 上完成逐页 shadow replay。结果正确，但 radial-shell 下界能安全跳过的页面比例过低，无法抵消实现成本，因此没有进入 production。

### Partial-distance early abandon

D1 对 threshold-aware partial distance 和 sparse-page kernel 进行了单独分析。该方向在部分微基准上有潜力，但真实扫描中的阈值时机、页布局和额外分支使收益不够稳定，因此没有并入最终方法。

### 更复杂的 probes 预测模型

D2 测试了静态阈值、风险排序、loss-budgeted、随机森林、GBDT 和多种校准方式。一些模型离线指标较好，但存在特征计算成本、尾部风险或跨 workload 泛化问题。最终只部署了特征较少、决策成本可控并经过 C/Python parity 验证的浅层 D2-P 策略。

相关设计、验证和 No-Go 证据见：

- [`phase_d0a/phase_d0a_semantic_audit.md`](phase_d0a/phase_d0a_semantic_audit.md)
- [`phase_d0b/phase_d0b_l2_report.md`](phase_d0b/phase_d0b_l2_report.md)
- [`phase_d1_0/phase_d1_0_semantic_audit.md`](phase_d1_0/phase_d1_0_semantic_audit.md)
- [`phase_d2/phase_d2_oracle_report.json`](phase_d2/phase_d2_oracle_report.json)
- [`final_multi_comparison.md`](final_multi_comparison.md)
