# Phase C / Part 2 正式执行

复用 Part 1 已通过验证的源码 binary 与共享索引。Part 2 不重新编译、不重建索引，
也不改变算法、kernel、排序或 GUC 映射。binary 来源提交保留 Part 1 编译时的提交，
后续实验框架提交不会被误记为 binary 来源。

```bash
PYTHONDONTWRITEBYTECODE=1 PRISTINE_COMMIT=d65ea656 \
PYTHON_BIN=/workspace/benchmark/.venv/bin/python \
benchmark/scripts/run_ivfflat_experiments.sh formal --foreground --part2 \
  --part1-artifacts /workspace/benchmark/runs/20260909T065629Z/phase_c_artifacts \
  --stop-after-configs 1

# 首个完整配置写入后，预期以 75 退出。恢复时自动识别 Part 2。
PYTHONDONTWRITEBYTECODE=1 PRISTINE_COMMIT=d65ea656 \
PYTHON_BIN=/workspace/benchmark/.venv/bin/python \
benchmark/scripts/run_ivfflat_experiments.sh formal --foreground --resume RUN_DIR
```

固定 workload：五系统 × 九 probes × 十轮 × 1000 官方 unique queries，
共 450000 measured executions。每配置额外执行当前 permutation 前 100 条 warmup。
轮次从 1 开始，query seed 为 20260908 + round，probes shuffle seed 再加 100000。
奇数轮 pristine family 在前，偶数轮 optimized family 在前；优化系统每轮循环轮换。
所有系统在同一轮共用 query permutation 和 probes order。

每条 latency 只包含 execute 与 fetchall，向量字符串预先构造；ID 转换、Recall、
checksum、序列化均在单条计时外。QPS = 1000 / 完整 measured loop 的真实 wall time，
包含循环内 Python bookkeeping，但排除 warmup、统计计算与 checkpoint 写入。

低 probes 允许合法的不足十行结果（包括零行），始终以十为 Recall 分母。
例如 pristine/probes=1 的 query 654 返回六行；不得因此删除查询或 probes。
初次校验拒绝此短结果的尝试位于 20260909T083532Z，无已完成正式 checkpoint；
修正的是框架校验，不是算法。正式目录为 20260909T083926Z。

每配置在一个原子 JSON checkpoint 中保存 1000 raw rows、完成标记、SHA256、
轮次 summary、GUC、EXPLAIN 和索引身份。中断中的未完成配置整体重新 warmup/测量；
完成配置不可重复。CSV 是可重建的派生产物：每 checkpoint 后追加并 fsync，
resume 和最终完成时从已验证 checkpoint 原子重建，消除中断导致的半行/重复。

resume 必须通过 manifest、全套 permutation、数据集、源码/binary、已完成 raw
schema/ID/checksum/Recall/顺序和实际 QPS 校验。运行中只在 family 变化时安全停库
切换，并在每配置前核对 binary SHA256 与共享索引身份。

主要文件位于 RUN_DIR/phase_c_artifacts：
- phase_c_manifest.json、query_permutations.json、run_order.txt
- checkpoints/、phase_c_raw.csv、phase_c_rounds.csv
- phase_c_progress.json、phase_c_resume.jsonl、phase_c_activation.jsonl
- 双 binary、源码提交/编译/hash 证据、index_identity.txt、ground_truth_verification.json
- 完成后 phase_c_part2_report.json

本阶段仅采集与完整性核对。跨轮 determinism、unique-query Recall 重算及科研结论
留给 Part 3，不把十轮重复视为 10000 个独立 Recall 样本。
