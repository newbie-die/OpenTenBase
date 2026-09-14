# 重点实验结果 CSV

本目录保存体积较小、适合随源码评审的实验汇总 CSV。文件均从 `/workspace/benchmark/runs/` 中的对应产物逐字节复制，没有重新计算、筛选行或改写数值。完整的 raw CSV、日志、checkpoint、manifest 和中间产物仍保存在仓库外的原始运行目录。

## 文件索引

| 文件 | 原始运行产物 | 状态与用途 | 数据范围 | SHA256 |
| --- | --- | --- | --- | --- |
| [`final_multi_comparison.csv`](final_multi_comparison.csv) | `/workspace/benchmark/runs/final_multi_formal/final_comparison.csv` | **最终正式结果**；运行状态为 `COMPLETE`，是当前结论和文档表格的统计来源 | Final Round A/B，14 个方法与 workload 组合，共 28 config-rounds、164000 次正式查询 | `798d357ab42e87242cd1b27aab2fe1bd79540084ba62944ad5514ff87871d7b3` |
| [`fused2_production_summary.csv`](fused2_production_summary.csv) | `/workspace/benchmark/runs/phase2b2_part4_20260908/phase_2b_fused2_production_summary.csv` | 已完成的 FUSED2 独立生产实验；用于检查 direct/FUSED2 的正确性、各分位延迟和多轮稳定性 | GIST-L2，`probes=16/64/128`，full/automatic 两种模式，各 4 轮 | `db32daf0eb4995374f486eff9db976d5899faacbcb11dd1ebcaaafb0c8140b53` |
| [`d2p_gist_comparison.csv`](d2p_gist_comparison.csv) | `/workspace/benchmark/runs/all_fomal_exp_20260913T214530Z/comparison.csv` | 已完成的 D2-P 单数据集正式对照；用于展示动态 probes 与固定 probes 的取舍 | GIST-L2，fixed 16/32/64、static tuned 和 D2-P，共 1 轮 | `9fc699e92c08e6d907b6e71335e1186ce331a8c794f94f35bb5a6dd7d6059da1` |
| [`page_pruning_probe_summary.csv`](page_pruning_probe_summary.csv) | `/workspace/benchmark/runs/phase_d0b_l2_20260911T024158Z/phase_d0b_l2_probe_summary.csv` | 已完成的 D-0B 机制实验，结论为 `No-Go`；用于保留 page-level distance pruning 未采用的量化证据 | L2，6 个 probes 档位的页面、候选、距离调用、Recall 和 pruning 比例 | `dbec6f17872cd001c607dbe5c3cfae33fc70364fa85ef4f4568d1668394c3e74` |
| [`partial_distance_kernel_summary.csv`](partial_distance_kernel_summary.csv) | `/workspace/benchmark/runs/phase_d1_2b_l2_20260911T114738Z/phase_d1_2b_summary.csv` | 已完成的 D1.2b 内核实验，生产结论为 `No-Go`；用于保留 partial-distance early abandon 的收益与开销证据 | 不同 workload、probes、kernel 和 block size 的微基准与正确性汇总 | `2423e8bbb659e9a449558d0016bbb61a63d0864a8686a7f750b60b83e88f70e3` |

## 使用口径

`final_multi_comparison.csv` 是当前多数据集最终结果的唯一正式汇总。它同时保存 Round A、Round B、mean 和 std，不应把其余 CSV 的历史轮次或单独实验数据并入这两轮统计。

`fused2_production_summary.csv` 和 `d2p_gist_comparison.csv` 是已完成的独立实验，可用于追溯单项优化的行为。它们的运行矩阵、轮数和时间环境与 Final Multi 不同，不能取代 Final Multi 对 GIST、SIFT 及其他 workload 的最终结论。

`page_pruning_probe_summary.csv` 和 `partial_distance_kernel_summary.csv` 记录未采用方案的机制证据。它们用于解释为什么没有将相应方案放入最终生产矩阵，不应作为最终端到端性能声明。

## 完整性校验

在仓库根目录执行：

```bash
sha256sum benchmark/docs/results/*.csv
```

预期输出中的哈希应与上表一致。若本机还保留原始运行目录，可以用 `cmp` 验证提交副本与原始产物逐字节相同。
