# Phase D-0A — Metric-Aware Safe Page Pruning Semantic Audit

本阶段完成当前源码审计和数学接口设计。**L2、IP，以及 Cosine 的实际 normalized-storage negative-IP 排序键均有实数域安全 bound；尚无 production 浮点剪枝实现获得安全认证。** 本报告不提供工作量减少比例或性能结论，不进入 D-0B。

## 范围与证据基线

审计对象为当前 `vector` 类型的三个 **IVFFlat** opclass，不把 HNSW、halfvec、bit 或自定义 support function 自动纳入结论。仓库 HEAD 为 `47e2b9b035a4ac0c9dcbd228272e1eca941c5f5e`；生产来源 commit 为 `6dab3886aa18a194f3cb955ad6c8c6addd8d7a8c`。当前 `contrib/pgvector` 和 PostgreSQL `src` 相对生产来源无 diff。

生产路径：`/workspace/install/lib/postgresql/vector.so`；SHA256：`d56dffd4db4e08d075cad229da5b04a7dc722f1d2cd45f03fb5570b0a64f6480`。既有 benchmark 框架改动保留，工作区整体非 clean。

新 D-0A 执行期间数据库 query 数为 **0**，benchmark 数为 **0**。之前中断的 D-0 只留下独立 observer 构建和一次合成 SQL selftest，无 GIST measured query、无 D-0 run。旧准备文件已可逆归档停用，路径见 `preparation_archive.txt`；旧 observer 的经验 epsilon 不作为本报告的浮点安全证明。

## 1. Build 与存储路径

| 环节 | 当前源码事实 | 证据 |
|---|---|---|
| 构建入口 | 初始化→计算中心→meta page→list pages→entry pages；entry 阶段分配、排序、写入 | [S13 ivfbuild.c:1018–1058](../../../contrib/pgvector/src/ivfbuild.c#L1018) |
| 样本 | 目标 `lists*50`，至少10000，受估计可用 tuple 数约束；block/reservoir 采样 | [S04 ivfbuild.c:435–480](../../../contrib/pgvector/src/ivfbuild.c#L435)、[S05 ivfbuild.c:57–155](../../../contrib/pgvector/src/ivfbuild.c#L57) |
| 中心初始化 | kmeans++，使用 support proc3；无样本时随机中心 | [S06 ivfkmeans.c:24–91](../../../contrib/pgvector/src/ivfkmeans.c#L24)、[S09 ivfkmeans.c:111–133](../../../contrib/pgvector/src/ivfkmeans.c#L111) |
| 中心更新 | float32 样本和累计均值；非空簇除以样本数，空簇随机值；有 proc4 时归一化新中心 | [S07 ivfkmeans.c:180–235](../../../contrib/pgvector/src/ivfkmeans.c#L180)、[S20 ivfutils.c:301–389](../../../contrib/pgvector/src/ivfutils.c#L301) |
| 聚类 metric | L2 使用非平方 L2；IP/Cosine 使用 angular `acos(clamp(dot))/pi`，输入样本已按 proc4 归一化 | [S08 ivfkmeans.c:239–480](../../../contrib/pgvector/src/ivfkmeans.c#L239)、[S24 vector.c:733–750](../../../contrib/pgvector/src/vector.c#L733) |
| 实际 list assignment | 实际入索引值按 **proc2** 决定是否归一化；调用 **proc1** 选最小距离的中心，不是 proc3 | [S10 ivfbuild.c:162–218](../../../contrib/pgvector/src/ivfbuild.c#L162) |
| entry tuple | `index_form_tuple` 保存实际 indexed vector，`t_tid` 指向 heap；不存 query distance、application ID、residual 或半径 | [S11 ivfbuild.c:250–331](../../../contrib/pgvector/src/ivfbuild.c#L250)、[S16 itup.h:23–51](../../../src/include/access/itup.h#L23) |
| list page | 每个 list item 保存 `startPage`、`insertPage`、实际 float32 `center`；一个 list 的 entry pages 由链连接 | [S12 ivfbuild.c:486–556](../../../contrib/pgvector/src/ivfbuild.c#L486)、[S14 ivfflat.h:272–299](../../../contrib/pgvector/src/ivfflat.h#L272) |
| page layout | PostgreSQL slotted page：header、line pointers、tuple 区、special space；special 保存 `nextblkno/unused/page_id` | [S17 bufpage.h:25–78](../../../src/include/storage/bufpage.h#L25)、[S18 ivfutils.c:147–202](../../../contrib/pgvector/src/ivfutils.c#L147) |
| meta page | magic、version、dimension、lists；当前没有 page-bound metadata | [S12 ivfbuild.c:486–556](../../../contrib/pgvector/src/ivfbuild.c#L486)、[S14 ivfflat.h:272–299](../../../contrib/pgvector/src/ivfflat.h#L272) |
| vector 表示 | varlena header、dimension、保留字段、float32 坐标数组 | [S15 vector.h:13–24](../../../contrib/pgvector/src/vector.h#L13) |
| 增量维护 | InsertTuple 重复 proc2 normalization 与 proc1 assignment；追加/链接页面走 WAL；vacuum 删除和复用空间 | [S40 ivfinsert.c:20–180](../../../contrib/pgvector/src/ivfinsert.c#L20)、[S41 ivfvacuum.c:85–135](../../../contrib/pgvector/src/ivfvacuum.c#L85) |

这里的“中心”不跨 opclass 等价。L2 中心近似原坐标空间均值；IP 和 Cosine 是归一化样本做 spherical kmeans 后的方向中心。IP candidate 虽然未经归一化，仍与方向中心处于同维坐标空间，因此 `x-c` 合法；其统计含义和尺度不同于 L2 residual。

## 2. Scan、distance 与 Top-K

`GetScanValue` 获取 ORDER BY 参数，只有 proc2 存在才 normalize。NULL ORDER BY key 使用 `ZeroDistance`。`GetScanLists` 完整读取 list pages，使用 proc1 的内部 score 和 pairing heap 选定 list；只在严格更小 score 时替换队首，同分 list 保持现有堆语义，不能在 D 阶段另加 list tie rule。（[S26 ivfscan.c:430–492](../../../contrib/pgvector/src/ivfscan.c#L430)；[S27 ivfscan.c:70–155](../../../contrib/pgvector/src/ivfscan.c#L70)）

`GetScanItems` 按已选 `listPages` 顺序逐页读取，并按 offset 取出 entry vector/TID。generic exact candidate distance 在 328 行调用 proc1，结果放在 float8 sort column；每个候选原样输入 tuplesort，整个 batch 完成后排序。当前没有 page bound 或提前剪枝。（[S28 ivfscan.c:161–424](../../../contrib/pgvector/src/ivfscan.c#L161)）

`InitScanSortState` 在 bounded 与 full 两种模式中均按 `(distance ASC, heap TID ASC)` 排序。TID 比较是 block number 后 offset，不是 application ID。不能将第二键换成 ID，也不能因 lower bound 等于第 k 名 distance 而跳页。（[S29 ivfscan.c:498–512](../../../contrib/pgvector/src/ivfscan.c#L498)；[S39 itemptr.c:51–72](../../../src/backend/storage/page/itemptr.c#L51)）

LIMIT 传播值是 `count+offset`；WITH TIES/无限 LIMIT 不产生有限 bound。IndexScanState 把它交给 `xs_tuple_bound`。`ivfflatrescan` 在自动模式且 iterative off 等条件满足时设置 `physicalBound=max(bound_min,logicalBound*overfetch)`；Phase2A 的参数得到 logical=10、physical=40。bounded tuplesort 减少保留/排序开销，**当前仍完整计算 candidates**。（[S34 nodeLimit.c:417–437](../../../src/backend/executor/nodeLimit.c#L417)；[S35 execProcnode.c:903–908](../../../src/backend/executor/execProcnode.c#L903)；[S36 nodeIndexscan.c:103–152](../../../src/backend/executor/nodeIndexscan.c#L103)；[S31 ivfscan.c:707–761](../../../contrib/pgvector/src/ivfscan.c#L707)）

调用者继续索取、physical bound 不够时，`StartFullSortFallback` 对同一 selected lists 重建完整排序并跳过已返回 TID。iterative scan 会以 batch 重置排序；page pruning 不能自动沿用另一 batch 的阈值。（[S32 ivfscan.c:518–568](../../../contrib/pgvector/src/ivfscan.c#L518)；[S33 ivfscan.c:767–884](../../../contrib/pgvector/src/ivfscan.c#L767)；[S28 ivfscan.c:161–424](../../../contrib/pgvector/src/ivfscan.c#L161)）

**SQL 安全边界：** index candidate 尚未通过 heap MVCC 和执行器过滤。一个 candidate shadow Top10 并不一定是 SQL 可见合格 Top10。后续实现至少选择一种已证明的协议：维护足够多的可见合格结果；或在 bounded 阶段保全 physical Top40，并在 fallback 时对全部 selected pages 关闭剪枝重新扫描。仅用 candidate 第10名永久剪掉剩余页面不符合当前通用 SQL 语义。offset、WITH TIES、过滤、HOT/MVCC 和 iterative batch 都不能被一个固定10阈值覆盖。（[S37 indexam.c:680–706](../../../src/backend/access/index/indexam.c#L680)；[S38 nodeIndexscan.c:523–541](../../../src/backend/executor/nodeIndexscan.c#L523)；[S32 ivfscan.c:518–568](../../../contrib/pgvector/src/ivfscan.c#L518)）

## 3. Metric semantics 对照

| 项 | L2 | Inner Product | Cosine |
|---|---|---|---|
| IVFFlat opclass | vector_l2_ops | vector_ip_ops | vector_cosine_ops |
| SQL operator | `<->`: sqrt(float32 squared-L2) | `<#>`: negative float32 dot | `<=>`: raw dot/norms、clamp 后 1-similarity |
| proc1：list 与 candidate score | vector_l2_squared_distance | vector_negative_inner_product | vector_negative_inner_product |
| proc2：indexed/query normalization | 无 | 无 | vector_norm，normalize 实际走 l2_normalize |
| proc3：kmeans distance | l2_distance，非平方 | vector_spherical_distance，angular | vector_spherical_distance，angular |
| proc4：kmeans normalization | 无 | vector_norm | vector_norm |
| stored candidate | raw float32 x | raw float32 x | l2_normalize(x) 的 float32 输出 x̃ |
| scan query | raw float32 q | raw float32 q | l2_normalize(q) 的 float32 输出 q̃ |
| stored center | raw-space mean-like c | normalized spherical center c | normalized spherical center c̃ |
| 内部排序键，越小越好 | fl32(sum((q-x)^2))，提升 float8 | -fl32(sum(q*x))，提升 float8 | -fl32(sum(q̃*x̃))，提升 float8 |

注册和 support 证据：[S01 vector.sql:292–311](../../../contrib/pgvector/sql/vector.sql#L292)、[S02 vector.sql:174–186](../../../contrib/pgvector/sql/vector.sql#L174)、[S03 ivfflat.h:45–49](../../../contrib/pgvector/src/ivfflat.h#L45)、[S19 ivfutils.c:71–115](../../../contrib/pgvector/src/ivfutils.c#L71)、[S20 ivfutils.c:301–389](../../../contrib/pgvector/src/ivfutils.c#L301)、[S30 ivfscan.c:574–665](../../../contrib/pgvector/src/ivfscan.c#L574)、[S45 ivfbuild.c:368–372](../../../contrib/pgvector/src/ivfbuild.c#L368)。kernel 证据：[S21 vector.c:561–633](../../../contrib/pgvector/src/vector.c#L561)、[S22 vector.c:636–675](../../../contrib/pgvector/src/vector.c#L636)、[S23 vector.c:678–724](../../../contrib/pgvector/src/vector.c#L678)、[S25 vector.c:797–847](../../../contrib/pgvector/src/vector.c#L797)。完整机器可读表见 `phase_d0a_metric_table.csv`。

**IP 注意：** proc4 只处理训练样本/中心，不意味着索引存储 candidate 或 query 被归一化。非零 raw candidate 可以用任意 norm；raw zero candidate 也不会因不存在的 proc2 被剔除。训练零样本被 norm 检查跳过。中心 normalization 遇到零均值仍可能返回零：`CheckNorms` 检查的是 proc2，所以 Cosine 会拒绝零中心，IP 不在此处拒绝；所有 bound 都不能依赖中心范数恰好为1。（[S42 ivfkmeans.c:554–570](../../../contrib/pgvector/src/ivfkmeans.c#L554)；[S43 ivfkmeans.c:540–547](../../../contrib/pgvector/src/ivfkmeans.c#L540)；[S44 ivfkmeans.c:518–534](../../../contrib/pgvector/src/ivfkmeans.c#L518)）

**Cosine 注意：** proc2 使 build/insert 跳过零范数 candidate；scan normalization 没有相同的非零检查，零 query 会 normalize 为零向量。`l2_normalize` 用 double 求 norm，但输出为 float32，因此 x̃、q̃、c̃ 不能按精确 unit vector 使用。center 与实际 indexed vector 在同一 rounded-normalized space，residual 必须是 `x̃-c̃`，不能使用原始 heap `x-c̃`。

Cosine 的内部 scan 排序并非直接调用 `cosine_distance`：后者对原始参数重算 float32 dot/norms 并 clamp，内部则对 rounded-normalized 参数算 negative dot，且 `xs_recheckorderby=false`。数学理想单位球面中两者单调等价，当前浮点值/并列关系不能据此认定 bitwise 相同。D 的 correctness 目标应先固定为**同一 query、同一 selected lists、当前 optimized AM 的结果顺序**，再保持现有 SQL 输出和 Recall；不能顺便重新定义“精确 cosine”或调整 ground truth。（[S23 vector.c:678–724](../../../contrib/pgvector/src/vector.c#L678)；[S25 vector.c:797–847](../../../contrib/pgvector/src/vector.c#L797)；[S33 ivfscan.c:767–884](../../../contrib/pgvector/src/ivfscan.c#L767)）

## 4. Residual 能共用到哪一层

在实际 indexed 表示中，三种 metric 都能定义 `Rmax ≥ max ||x-c||₂`，L2 可增加 `Rmin ≤ min ||x-c||₂`。Cauchy–Schwarz 的 `qᵀx=qᵀc+qᵀ(x-c)` 不要求 c 是 raw mean 或精确 unit vector，所以 IP 和 Cosine internal-key 都适用。

可以共用 **metadata 字段形状和生命周期框架**，包括 outward residual enclosure、metric/representation 标识、index/list/page 与 centroid epoch 绑定、有效性/数值证书。不能共用一个跨 opclass 的 residual 数值集合，也不能把现有 GIST L2 index 的半径直接用于重新解释成 IP/Cosine 的索引。

最小数学 payload：L2 单边球 bound 只需 Rmax，所要求的双边 shell bound 需 Rmin+Rmax；IP 和当前 Cosine internal-key 只需 Rmax。q 的真实表示 norm 上界和 q·c 的区间属于 query/list 临时状态，不是每页重复 metadata。浮点 IP 误差所需候选 norm 上界可由 `||c||+Rmax` 导出，单独 Nmax 是可选收紧字段。

Cosine 推荐 **A+B**：在实际 normalized-storage space 构建 residual，用 negative-IP bound 保留现有内部排序；不把 norm 简写成1，不把 dot 上界裁成1。**C 是额外目标的条件要求**：如要直接对 SQL `<=>` 的真实浮点值证明 bound，至少需要范数区间和 normalization/dot/norm 计算误差证书，或针对原始方向表示另建 certified enclosure。只增加 norm 数字还不能消除 float32 normalization 的方向误差。该 operator-space 路线标记 `NEEDS_ADDITIONAL_METADATA_AND_NUMERICAL_PROOF`，本阶段不输出可直接实施的 cosine-operator pruning 公式。

## 5. D0A-RQ 回答

1. **D0A-RQ1：L2 存在严格实数域 safe bound。** shell reverse-triangle 下界成立；落地要保守映射到当前 float32 squared-L2 key，不能拿 double 数学值直接和生产 τ 比较。
2. **D0A-RQ2：IP 存在严格实数域 safe bound。** similarity 上界 `q·c+||q||Rmax` 转为 negative-IP 下界；IP 本身不满足三角不等式并不阻止这个 residual/Cauchy–Schwarz 证明。
3. **D0A-RQ3：Cosine 当前内部排序键存在严格实数域 safe bound。** 在 x̃/q̃/c̃ 上使用 IP 推导，保留实际 norm；精确单位向量假设不成立。独立 SQL `<=>` 浮点 bound 仍需额外 metadata/误差证明，不宣称已完成。
4. **D0A-RQ4：能共享 framework 和 residual metadata schema，不能共享跨 opclass 的 metadata 数值。** bound computation、score unit、representation 和数值误差适配必须分 metric。
5. **D0A-RQ5：L2 Rmax（双边加 Rmin）；IP Rmax；Cosine internal-key Rmax。** 均还需正确的对象/版本绑定和 outward rounding validity。直接 cosine operator 路线另需 norm/表示误差证书。
6. **D0A-RQ6：已由源码支持的是 metric 路径、表示和排序键映射；已数学推导的是实数 bound 与 strict tie rule。** 具体编译产物的浮点误差上界、metadata 并发维护、SQL 阈值资格/fallback 集成和剪枝潜力均未验证。不得把 semantic audit 完成解释为 production pruning 已获准。

本阶段未修改 production 源码、binary、page format 或 Phase C artifacts，未执行正式实验或新 smoke query。D-0B 未开始。
