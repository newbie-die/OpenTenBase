# Phase D-0B — L2 Safe Page-Pruning Potential Validation

**Decision：No-Go；Correctness：PASS。** 仅完成一轮，600 次 measured Top10 SQL，另有每档20次 warmup（共120，未计入统计）。runner与专属数据库backend均已退出；未进入 D-0C。

原始 artifacts：`/workspace/benchmark/runs/phase_d0b_l2_20260911T024158Z`。固定 seed=20260911，query permutation SHA256：`cff253037198d22367d935c26f3a98a51cf56a35ff0341953b9fe8f9fc7a1daf`。

## 结果

| probes | queries | scanned page visits | 可剪 page visits | 可避免 candidate/distance | 距离可避免比例 | 零剪枝 query 比例 | mean Recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 100 | 876,991 | 682 | 1,364 | 0.078% | 97.000% | 0.691 |
| 16 | 100 | 1,728,482 | 1,505 | 3,008 | 0.087% | 96.000% | 0.811 |
| 32 | 100 | 3,406,544 | 5,366 | 10,723 | 0.157% | 94.000% | 0.925 |
| 64 | 100 | 6,554,279 | 24,477 | 48,926 | 0.373% | 92.000% | 0.972 |
| 128 | 100 | 12,547,197 | 101,558 | 203,038 | 0.809% | 80.000% | 0.995 |
| 256 | 100 | 22,938,114 | 353,821 | 707,398 | 1.542% | 58.000% | 1.000 |

Gate 比例为 `sum(avoidable candidate distance calls) / sum(all index candidate distance calls)`，不是 instrumentation latency，也不是三个高 probes 的平均值。三个高 probes 均低于5%；无一达到15%。宏平均与per-query p25/p50/p75另见probe summary。所有probes的per-query中位剪枝率均为0；因此不能把少量有收益query描述成普遍收益。

## 正确性与方法边界

- 两个 shadow Top10 的 ordered TID、ID、distance values 均一致：600/600，无 mismatch；每页更新后立即检查两堆一致。真实SQL的ordered IDs/TIDs一致，且600组ordered IDs与冻结Phase C phase2a基线一致，官方Recall ground truth未改。
- 每个真实SQL完成后，只读hook复制其query、selected-list顺序和有效bound；同一锁定索引上的辅助observer完整重放全部entry pages/candidates。计数是该真实selected-list遍历所对应的工作量，不是直接生产timer/counter采样。辅助重放不是第二轮正式benchmark。
- Metadata使用非平方L2 residual半径的binary64外包区间；先保存page-entry τ，再读取/计算metadata，最后用保守squared LB与生产squared τ作严格大于比较。真实AM不跳页、不少算distance、不减少tuplesort输入。数值域与误差推导见run中的numerical_certificate.md。
- 下界逐页对本页所有实际generic distance做额外校验；该检查仅用于安全验证，不影响would-prune决定。所有页均通过。
- query/list/page occupancy维度的page_stats精确汇总全部page visits，不是抽样，也不是distinct physical pages。metadata首次读取candidate后计算并缓存，未写入索引。
- candidate Top10潜力不等于任意MVCC/quals场景的SQL安全剪枝证明；本次实际SQL一致性通过，后续生产协议仍需保留physical40/fallback。当前nextblkno在entry page中，可剪page visit不直接等于省掉物理I/O。

## 六项低潜力诊断

| 诊断量 | p64 | p128 | p256 |
|---|---:|---:|---:|
| Top10未ready的page比例 | 0.008% | 0.004% | 0.002% |
| LB=0 / d落在residual区间内 | 19.592% | 15.980% | 12.508% |
| ready后区间宽度≥τ的page比例 | 0.143% | 0.140% | 0.152% |
| 逐candidate radial bound可避免比例 | 0.551% | 1.088% | 1.988% |
| 事后exact page-min诊断可避免candidate比例 | 99.894% | 99.944% | 99.970% |

1. **阈值建立不晚。** 每query恰好前5页未ready（每页2个candidate）；高probes相应比例约0.0076%、0.0040%、0.0022%，不是主要损失。
2. **LB=0有贡献，但不是全部原因。** 高probes的19.592%、15.980%、12.508%页下界为0，数量与d落在[Rmin,Rmax]区间内的页一致；大多数剩余页虽然LB>0，仍不足以超过τ。
3. **区间宽不能独立解释结果。** 高probes平均区间宽度约0.1448、0.1460、0.1488；宽度≥τ的ready页仅约0.14%–0.15%。将每个candidate分别作为singleton residual bound，候选可避免率仍仅0.551%、1.088%、1.988%。
4. **d落在区间内使bound归零。** 该项已由逐页数据直接统计，不是由宽度均值推测；其比例随probes增大下降。
5. **page粒度不是主要瓶颈。** 99.94%以上page visits含2个candidate，3+为0；逐candidate反事实去掉页级区间合并后仍不到2%。p64/128/256分别只有27/83/160个页出现“所有singleton可剪、整页却未剪”。这区分了页合并损失与radial bound本身的信息不足。
6. **后半list确实更容易剪。** 下表使用原始selected-list顺序，前后各一半；每档分别统计自己的page-entry τ。

| probes | 前半list candidate可避免率 | 后半list candidate可避免率 |
|---:|---:|---:|
| 64 | 0.157% | 0.607% |
| 128 | 0.373% | 1.286% |
| 256 | 0.809% | 2.428% |

补充证据：高probes有约80.0%/83.2%/85.9%的page visits属于“阈值已ready、LB>0、但不够大”；事后真正page-min distance诊断对应超过99.89%的candidate都在page-min>τ的页上。后者是读取所有候选才知道的query-dependent上限，不能当作静态metadata可实现收益。这些数据支持：当前radial shell下界区分力不足，而非没有较差candidate可供排除。

浮点guard相对未加guard的radial判断仅额外保留7/15/78页，不能解释低potential。

## 停止状态

- 七类必需 artifacts 已生成；原始CSV 22a0b1ace5138c3538c47625c935caf1404d5b5fea0dadc6c13823a0abbce2e0。
- 600 unique execution keys；duplicates/missing/unexpected均为0；rounds=[1]。
- production vector.so SHA256：`d56dffd4db4e08d075cad229da5b04a7dc722f1d2cd45f03fb5570b0a64f6480`；四个受保护源码文件与Phase C冻结的206个文件均未变。
- 未启用FUSED2，未实现新distance kernel或production pruning，未修改page format，未执行SIFT/GLOVE/Cohere/IP workload；没有latency优化结论。
- No-Go仅针对本次GIST L2 residual-shell候选方案和该100-query子集，不外推为全部metric或其他metadata方案均无潜力。
- 本轮结束后停止，D-0C未启动。
