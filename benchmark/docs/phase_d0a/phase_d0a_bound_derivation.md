# Phase D-0A — Safe Bound Derivation 与统一接口设计

本文只给出数学设计与接口契约；没有可执行 pruning 实现。符号始终对应**当前实际 indexed/query float32 坐标所表示的实数**；`fl` 表示生产 kernel 计算的浮点结果，`τ_key` 是它的第 k 名内部排序键。实数定理成立不等于 float32 比较自动安全。

## 1. 共同前提

固定 query、snapshot、selected lists 及其顺序。页 P 属于固定中心 c，metadata 必须覆盖 P 中每个原本会进入候选处理的 entry vector，包含可能不可见的 index tuples；覆盖更大集合只会更保守。阈值使用独立 Top-K 状态或未来公开接口，不能读取/修改 tuplesort 私有堆。

当前内部排序统一为 `(D_key ascending, TID ascending)`。Top-K 未满时，阈值 ready=false（数学上 +∞），不剪枝。每页进入前取阈值；metadata 在 shadow 阶段可事后计算，但不能用“处理完本页后的阈值”代替 page-entry 阈值。

要证明 SQL 输出不变，阈值必须具备 eligibility 证书：已知至少 k 个可见且满足 quals 的不同结果；或保全当前 physical bounded prefix，fallback 全量重扫。candidate-only Top10 的数学证明只覆盖 candidate Top10，不覆盖任意 SQL。详见审计文档的 MVCC/OFFSET/fallback 源码引用。

## 2. L2

生产内部 score 是 `D_key=fl32(Σ(q_i-x_i)²)`，以 float8 容器存储；SQL `<->` 才取 sqrt。ranking 越小越好。实数非平方阈值记 τ_L2，实际实现阈值记 τ_sq_key，不能混为一谈。

令 `d=||q-c||₂`，`Rmin ≤ ||x-c||₂ ≤ Rmax`。reverse triangle 给出：

```
||q-x|| ≥ | ||x-c|| - ||q-c|| |
LB_L2 = max(0, Rmin-d, d-Rmax)
```

因此实数模式仅在 `LB_L2 > τ_L2` 时 prune；平方模式为 `LB_L2² > τ_sq`。Rmin=0 退化为只用 Rmax 的球界。

**平方放在最终的非负 lower bound 上。** 不可先把 residual 和 d 平方再相减：`Rmin²-d²`、`d²-Rmax²` 均不是这里的正确 squared-L2 lower bound。不要把生产 float32 list score 当成 exact d²。

metadata 的半径可在建立 metadata 时取 sqrt；每个 selected list 的 d 或其区间可按 query 计算一次，最后比较 `LB²` 和 stored τ_sq_key，不需要对每个 candidate 或每页 kth key 取 sqrt。这避免 candidate 层 sqrt，并非声称全流程不用 sqrt。若 metadata 存平方半径，转换必须保持区间方向。

**保守数值设计：** 存 `rmin_lo ≤ 所有 residual norms`、`rmax_hi ≥ 所有 residual norms`；计算 `d∈[d_lo,d_hi]`。用向下舍入得到

```
ell_lo = max(0, rmin_lo-d_hi, d_lo-rmax_hi)
```

还需对**实际生产 squared-L2 kernel**证明统一误差界，例如在已验证域内

```
D_key(q,x) ≥ (1-eps_L)*||q-x||² - eta_L,  0 ≤ eps_L < 1
L_key = round_down((1-eps_L)*ell_lo² - eta_L)
prune iff threshold_ready && L_key > τ_sq_key
```

`eps_L/eta_L` 在 D-0A 不指定经验值。必须覆盖 float32 subtraction、squaring、SIMD/FMA/reassociation、累加、underflow/FTZ，以及 compiler flags；溢出、NaN、未知 kernel/维度域、无证书时返回 UNKNOWN 并扫描。double/long double 计算再简单转 float、或随意减去 `1e-6`，不构成安全证明。单次 nextafter 只覆盖最后舍入，不能覆盖整个累加误差。

## 3. Inner Product

实际 indexed x/query q 未归一化，c 为已存方向中心，但都是同维实坐标；无需 c 为精确 unit vector。

```
x = c+r, ||r||₂ ≤ Rmax
qᵀx = qᵀc + qᵀr ≤ qᵀc + ||q||₂ Rmax = UB_IP
```

相似度越大越好。若使用实数 kth similarity `s_k`，只在 `UB_IP < s_k` 时 prune。

当前生产排序使用 **negative IP，越小越好**：`D=-qᵀx`。对应

```
LB_NIP = -UB_IP
τ_NIP = kth negative-inner-product key
prune iff LB_NIP > τ_NIP
```

两套符号只做完整转换，不把正 similarity bound 与负 distance τ 混用。`UB_IP == s_k` / `LB_NIP == τ_NIP` 都必须保留页面。页 Rmin 不是该 IP 上界的必需项。

浮点版本：取 certified `A_hi ≥ qᵀc`、`Q_hi ≥ ||q||`、`R_hi ≥ Rmax`，向上舍入

```
U_math = round_up(A_hi + Q_hi*R_hi)
X_hi = round_up(C_hi + R_hi)   // C_hi >= ||c||，不用强设 c norm=1
```

若生产 dot 的 page-uniform **绝对误差**上界为 `E_dot≥|fl32(qᵀx)-qᵀx|`，则

```
U_key_similarity = round_up(U_math + E_dot)
L_key = round_down(-U_key_similarity)
prune iff L_key > τ_NIP_key
```

IP 有正负项抵消，不能用仅依赖 `|qᵀx|` 的相对误差估计。候选设计为 `E_dot ≤ gamma_m * Q_hi * X_hi + eta_dot`，因为 `Σ|q_i x_i| ≤ ||q|| ||x||`。`gamma_m`、m、eta 和支持的浮点环境需按当前 kernel/编译产物建立证书，D-0A 未完成此证明。Nmax 单独存储可收紧 X_hi，但 Rmax+C_hi 已提供一个数学上足够的 norm 上界。

## 4. Cosine：内部排序与 SQL operator 分开证明

源码已证实：index build/insert 对 candidate 归一化，scan 对 query 归一化，spherical kmeans 归一化中心。这三者存储/传递为 float32，记 x̃、q̃、c̃。残差定义固定为 `r̃=x̃-c̃`。

**当前 AM 实际 score** 是 `D_cos_internal=-fl32(q̃ᵀx̃)`。因此直接使用上一节推导：

```
UB_stored_dot = q̃ᵀc̃ + ||q̃||₂ * Rmax_stored
LB_cos_internal = -UB_stored_dot
prune iff certified_L_key > τ_cos_internal_key
```

这里 ||q̃|| 和 ||c̃|| 均需实际 norm 区间；不得替成1。该界不需要对 candidate 再归一化，也不需重新计算 SQL `<=>`，保持当前 index order。浮点处理采用上一节的 outward interval 与 absolute dot error。

在**精确实数 normalization 的理想模型**下，`qhat=q/||q||`、`xhat=x/||x||`，可以写 `cosine_distance=1-qhatᵀxhat`；这只解释为什么使用 normalized IP 排序，不证明 `1 + 当前内部 key` 等于 SQL operator 的浮点值。后者另外对原始向量执行 float32 dot/norm reductions、double 除法、clamp。即使范数仅差几 ulp，tie 集合也可能变化。

不要将当前 internal dot 上界 clamp 到1：rounded-normalized dot 可略超1，而内部 negative-IP kernel 并不执行 cosine_distance 的 clamp。

**SQL `<=>` 浮点 key 直接 pruning 路线：NEEDS_ADDITIONAL_METADATA。** 当前 Rmax-only 元数据不足以直接采用 unit-sphere 公式。最少的额外信息类别包括正的 candidate norm 下界/上界、query norm 区间，以及把原始方向、rounded-normalized 表示和 kernel reductions 联系起来的保守误差证书。范数区间本身不恢复 normalization 丢失的方向信息。可选替代方案是对原始向量精确方向建立带误差 enclosure 的 residual metadata；本阶段不选择持久化方案、不输出已经安全的 operator-space 实现公式。

零范数 indexed candidate 当前跳过；零 query 返回零 normalize 值，internal dot 为零，所有有效 TID 都可能 tie。NULL query 使用 ZeroDistance。统一 framework 对这些情况禁用 pruning；不改变原有 SQL 特殊值行为。

## 5. Strict ties 与归纳安全性

假设 `L_key ≤ 每个本页 candidate 的实际生产 score`。只有 `L_key > τ_key` 时，全页 score 都严格差于当前第 k 名，无论页面内 TID 如何，都不可能击败当前 Top-K。未来加入更好结果只会降低 kth score，已剪页面仍不可能进入最终集合。这给出按 page-entry 顺序进行 simulated pruning 的归纳证明。

相等或 uncertainty interval 接触阈值时，必须遍历全部 candidates，仍用原来的 `(distance,TID)` comparator 插入。无需 page minimum TID metadata。不同向量同距离、完全相同向量但不同 TID 都要保留；完全相同 `(distance,TID)` 不增加新的 SQL 结果，原有 fallback 的 TID 去重语义保持。浮点 signed zero / NaN 的生产排序也不能由新的 epsilon tie comparator 改写；UNKNOWN 情况按原路径处理。

此证明的前提包括 kth 集合具有正确资格、score domain 相同、metadata 覆盖当前 page、选 list 与遍历顺序不变。数学 candidate heap 的证明不能替代 MVCC/过滤/fallback 证明。

## 6. 统一接口（纯设计，不写入 C 头文件）

设计拆为三个契约：

- `PageMetaProvider`：按 index/list/entry-page/centroid epoch 提供有效 enclosure，保证版本、快照或并发覆盖关系；负责 metadata 是否在读 entry page 前可用。
- `MetricBoundAdapter`：验证 opclass/support proc/type/representation；将 query、centroid 和 metadata 转为**内部实际排序键**的 certified lower bound。
- `PruningController`：维护阈值资格、strict comparison、page 顺序、计数、fallback；不关心 L2/IP 公式。

建议的逻辑类型如下，仅表达契约，不定义磁盘布局或 C ABI：

```text
PageBoundMeta:
    identity: index_generation, list_identity, centroid_generation, entry_block
    representation: metric, vector_type, dimension, normalization_version
    coverage: valid, page_generation_or_coverage_proof, empty_page_known
    residual: rmax_hi, optional rmin_lo
    numeric_certificate_id
    optional: candidate_norm_lo, candidate_norm_hi, normalization_error_bound

BoundResult:
    status: CERTIFIED | UNKNOWN
    score_domain: L2_SQUARED_FLOAT_KEY | NEGATIVE_IP_FLOAT_KEY
    lower_key_bound
    certificate_id

KthThreshold:
    ready, score_domain, kth_exact_key, kth_tid
    result_qualification, target_count, snapshot, batch_generation
    fallback_policy

ComputePageBound(query_context, centroid_context, PageBoundMeta) -> BoundResult
CanPrunePage(BoundResult, KthThreshold) -> decision
```

`CanPrunePage` 只有在身份/表示/数值证书/阈值资格均匹配、ready 且 certified lower_key_bound 严格大于 kth_exact_key 时为 true；其他情况正常扫描。score_domain 相同也不代表可以忽略不同 raw/normalized representation。未知 custom opclass 不能仅根据 SQL operator 名称推断 metric。

### Page 生命周期和工作量定义

插入 candidate 可能增大 Rmax 或减小 Rmin；必须在新 tuple 对 scan 可用之前原子更新/扩大 enclosure，或先使 metadata 无效。删除可保留旧宽 enclosure（安全但松）；vacuum 后复用、新插入和重建换中心必须重新验证。并发/WAL 协议未设计完成前不可启用生产剪枝。

现有 nextblkno 在 entry page 的 special space 中。如果只把 bound 放在本页，仍要 ReadBuffer/读取该页才能获知 bound 和下一页；此时只能说减少 candidate processing、distance calls、sort inputs，不能说减少 page reads。若未来要跳过 page read，PageMetaProvider 必须在进入页前拿到 bound 以及一致的后继遍历信息，且不改变 selected lists。本文不实现这种目录或 page-format 修改。

### 阶段结论

源码语义与实数推导支持三种 internal-key adapter。浮点证书、metadata consistency、SQL 阈值资格、fallback 完整性和潜力测量仍是独立未完成项。本阶段没有 D-0B、真实 pruning、page-format 变更、latency 或 Recall 实验结论。

## 7. 公式与源码映射

- L2 score / squared 表示：[S01 vector.sql:292–311](../../../contrib/pgvector/sql/vector.sql#L292)；[S21 vector.c:561–633](../../../contrib/pgvector/src/vector.c#L561)；[S28 ivfscan.c:161–424](../../../contrib/pgvector/src/ivfscan.c#L161)。
- IP score / normalization 区分：[S01 vector.sql:292–311](../../../contrib/pgvector/sql/vector.sql#L292)；[S05 ivfbuild.c:57–155](../../../contrib/pgvector/src/ivfbuild.c#L57)；[S10 ivfbuild.c:162–218](../../../contrib/pgvector/src/ivfbuild.c#L162)；[S22 vector.c:636–675](../../../contrib/pgvector/src/vector.c#L636)。
- Cosine actual representation / SQL 差异：[S19 ivfutils.c:71–115](../../../contrib/pgvector/src/ivfutils.c#L71)；[S20 ivfutils.c:301–389](../../../contrib/pgvector/src/ivfutils.c#L301)；[S23 vector.c:678–724](../../../contrib/pgvector/src/vector.c#L678)；[S25 vector.c:797–847](../../../contrib/pgvector/src/vector.c#L797)；[S26 ivfscan.c:430–492](../../../contrib/pgvector/src/ivfscan.c#L430)；[S33 ivfscan.c:767–884](../../../contrib/pgvector/src/ivfscan.c#L767)。
- Top-K / ties / SQL eligibility：[S29 ivfscan.c:498–512](../../../contrib/pgvector/src/ivfscan.c#L498)；[S31 ivfscan.c:707–761](../../../contrib/pgvector/src/ivfscan.c#L707)；[S32 ivfscan.c:518–568](../../../contrib/pgvector/src/ivfscan.c#L518)；[S34 nodeLimit.c:417–437](../../../src/backend/executor/nodeLimit.c#L417)；[S37 indexam.c:680–706](../../../src/backend/access/index/indexam.c#L680)；[S38 nodeIndexscan.c:523–541](../../../src/backend/executor/nodeIndexscan.c#L523)；[S39 itemptr.c:51–72](../../../src/backend/storage/page/itemptr.c#L51)。
- metadata 生命周期 / page-read 限制：[S14 ivfflat.h:272–299](../../../contrib/pgvector/src/ivfflat.h#L272)；[S18 ivfutils.c:147–202](../../../contrib/pgvector/src/ivfutils.c#L147)；[S40 ivfinsert.c:20–180](../../../contrib/pgvector/src/ivfinsert.c#L20)；[S41 ivfvacuum.c:85–135](../../../contrib/pgvector/src/ivfvacuum.c#L85)。
