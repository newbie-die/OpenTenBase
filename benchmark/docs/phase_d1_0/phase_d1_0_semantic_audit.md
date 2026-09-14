# Phase D1-0 — Threshold-Aware Progressive Distance Evaluation Semantic Audit

**本阶段完成源码审计、数学推导和正确性边界设计，未实现 progressive distance。** 当前参数的 production Gate 必须按 **K=physicalBound=40**；一般规则是动态取 `so->physicalBound`，不是硬编码40。它只证明 bounded 前缀安全，必须配合关闭D1的原始 full-sort fallback，不能宣称40能覆盖任意过滤选择率。

当前仓库HEAD：`2210a4a2a66c52b448097e073f8a1da25d4172e7`；生产来源commit：`6dab3886aa18a194f3cb955ad6c8c6addd8d7a8c`。生产 `.so`：`/workspace/install/lib/postgresql/vector.so`；SHA256：`d56dffd4db4e08d075cad229da5b04a7dc722f1d2cd45f03fb5570b0a64f6480`。源码定位以本次读取的工作区为准，完整路径/函数/行范围及hash见source map。

D-0B的GIST residual-shell No-Go保留，不把它当作D1逐维partial bound的潜力结果。D1-0无SQL查询、无smoke、无benchmark、无binary构建、无production/tuplesort/index-format改动；未进入D1-1。

## 1. LIMIT/OFFSET 到 xs_tuple_bound

| 环节 | 精确行为 | 源码 |
|---|---|---|
| LIMIT需求 | `compute_tuples_needed`返回count+offset；无限LIMIT/WITH TIES返回-1；负值代表不能推断有限需求 | [S01 nodeLimit.c:417–437](../../../src/backend/executor/nodeLimit.c#L417) |
| 计划树传播 | `ExecSetTupleBound`把值写入IndexScanState.iss_TupleBound；不是所有父节点都传播，有quals的SubqueryScan会中止传播 | [S03 execProcnode.c:832–985](../../../src/backend/executor/execProcnode.c#L832) |
| 默认值 | ExecInitIndexScan和AM scan初始化均为-1 | [S06 indexam.c:331–339](../../../src/backend/access/index/indexam.c#L331)、[S10 nodeIndexscan.c:920–927](../../../src/backend/executor/nodeIndexscan.c#L920) |
| ORDER BY index路径 | `ExecIndexScan`选择`IndexNextWithReorder`，它在index_beginscan之后、index_rescan之前写入xs_tuple_bound | [S29 nodeIndexscan.c:523–541](../../../src/backend/executor/nodeIndexscan.c#L523)、[S08 nodeIndexscan.c:200–223](../../../src/backend/executor/nodeIndexscan.c#L200) |
| 非ORDER BY与rescan | IndexNext也赋值；ExecReScanIndexScan再次传入当前bound，不可沿用旧query的threshold | [S07 nodeIndexscan.c:103–126](../../../src/backend/executor/nodeIndexscan.c#L103)、[S09 nodeIndexscan.c:587–598](../../../src/backend/executor/nodeIndexscan.c#L587) |
| AM读取 | ivfflatrescan直接复制`scan->xs_tuple_bound`到logicalBound | [S13 ivfscan.c:707–761](../../../contrib/pgvector/src/ivfscan.c#L707) |

这两个bound字段都是int64（[S04 execnodes.h:1705–1718](../../../src/include/nodes/execnodes.h#L1705)、[S05 relscan.h:136–146](../../../src/include/access/relscan.h#L136)）。`LIMIT 10 OFFSET 20`所需子节点行数是30，而非10；LIMIT节点实际丢弃已由子节点返回的前20行（[S02 nodeLimit.c:90–118](../../../src/backend/executor/nodeLimit.c#L90)）。不能在Index AM中读取SQL文本猜K。

## 2. logical=10 / physical=40 的真实条件

GUC注册（[S11 ivfflat.c:75–102](../../../contrib/pgvector/src/ivfflat.c#L75)）给出：

- `bounded_scan`默认false；当前实验显式on。
- `bound_overfetch`默认4，允许1..INT_MAX：乘以logicalBound。
- `bound_min`默认40，允许1..INT_MAX：physicalBound下限。
- `bound_fastpath_limit`默认100，允许1..INT_MAX：**logicalBound资格上限**，不是threshold heap容量上限。
- `experimental_sort_bound`默认0；正值优先走实验oracle分支，D1首版不支持该分支。

在`experimental_sort_bound=0`、bounded_scan=on、`0<logicalBound<=fastpath_limit`、iterative_scan=off时：

```
logicalBound = xs_tuple_bound
physicalBound = max(bound_min, logicalBound * bound_overfetch)
```

因此LIMIT10/OFFSET0得到10和40。LIMIT10/OFFSET20得到30和120；LIMIT10/OFFSET90得到100和400；LIMIT10/OFFSET91得到101，超过默认fastpath100，自动bounded路径不启用，D1必须关闭。实际值以rescan后的`so->physicalBound`为准，不在D1重复推测或截断。[S12 ivfscan.c:574–615](../../../contrib/pgvector/src/ivfscan.c#L574)、[S13 ivfscan.c:707–761](../../../contrib/pgvector/src/ivfscan.c#L707)

D1容量须能表示physicalBound并通过分配大小/溢出检查；若无法提供相应容量，就关闭D1，不能把P=120截成40，也不能为了内存预算改变既有physicalBound。

## 3. candidate→distance→tuplesort 的准确路径

`GetScanValue`处理query和NULL；generic L2使用索引proc1 `vector_l2_squared_distance`，其本体调用float32 reduction并将结果提升为float8；SQL `<->`使用的sqrt不在内部candidate排序键上。[S14 ivfscan.c:438–492](../../../contrib/pgvector/src/ivfscan.c#L438)、[S15 ivfscan.c:624–632](../../../contrib/pgvector/src/ivfscan.c#L624)、[S20 vector.c:561–574](../../../contrib/pgvector/src/vector.c#L561)、[S21 vector.c:609–633](../../../contrib/pgvector/src/vector.c#L609)、[S36 vector.sql:292–296](../../../contrib/pgvector/sql/vector.sql#L292)

`GetScanLists`的选择及顺序保持原样（[S16 ivfscan.c:85–155](../../../contrib/pgvector/src/ivfscan.c#L85)）。每个selected list的entry page沿nextblkno完整遍历：

1. `ReadBufferExtended`、SHARE buffer lock、PageGetMaxOffsetNumber：ivfscan.c:204–207。
2. `PageGetItemId`→`PageGetItem`→`index_getattr`取得candidate datum：230–233。
3. generic exact score：`so->distfunc(so->procinfo,so->collation,datum,value)`：328。
4. score、heap TID填入slot：351–354；`tuplesort_puttupleslot`：362。
5. 全部batch处理后`tuplesort_performsort`：405。

以上见[S17 ivfscan.c:161–247](../../../contrib/pgvector/src/ivfscan.c#L161)、[S18 ivfscan.c:251–362](../../../contrib/pgvector/src/ivfscan.c#L251)、[S19 ivfscan.c:375–405](../../../contrib/pgvector/src/ivfscan.c#L375)。D1候选检查点应位于datum/维度验证之后、完整exact调用和sort输入之前；被证明可abandon的candidate不向sort写partial或伪造distance。页仍必须读取，list选择不变，不宣称减少page reads。

## 4. visibility、WHERE 和 post-index filters 晚于上述处理

`ivfflatgettuple`从sort拿TID交给上层，`RememberBoundedTid`记录的是**AM已返回的TID**，不是SQL通过过滤的row（[S23 ivfscan.c:827–888](../../../contrib/pgvector/src/ivfscan.c#L827)、[S25 ivfscan.c:518–550](../../../contrib/pgvector/src/ivfscan.c#L518)）。`index_getnext_slot`之后调用`index_fetch_heap`，通过`xs_snapshot`处理heap visibility/HOT；不可见时继续向AM索取（[S26 indexam.c:721–750](../../../src/backend/access/index/indexam.c#L721)、[S27 indexam.c:680–703](../../../src/backend/access/index/indexam.c#L680)）。

`IndexNextWithReorder`随后可能recheck index quals/orderby（[S28 nodeIndexscan.c:264–313](../../../src/backend/executor/nodeIndexscan.c#L264)）；当前IVFFlat把recheck和recheckorderby均设false，但这**不会关闭普通WHERE过滤**。`ExecScan`取得`node->ps.qual`，`ExecScanExtended`在fetch后才执行`ExecQual`，失败继续取下一行；还有可能存在更上层plan-node过滤（[S29 nodeIndexscan.c:523–541](../../../src/backend/executor/nodeIndexscan.c#L523)、[S30 execScan.c:46–64](../../../src/backend/executor/execScan.c#L46)、[S31 execScan.h:160–252](../../../src/include/executor/execScan.h#L160)）。有些上层过滤使bound根本不能传播到AM，届时必须禁用D1。

因此，即使SQL没有WHERE，也不能一般假定“最小10个index candidates=SQL返回10行”：仍有MVCC/HOT等检查。K10只是额外证明全部候选可见、无过滤、无OFFSET时的理想对照。

## 5. A/B/C 三种K的安全矩阵

下表的“安全”区分当前bounded候选前缀和完整SQL协议；**K40并非无限可用的SQL结果池**。

| 场景 | A：K=LIMIT=10 | B：K=LIMIT+OFFSET | C：K=实际physicalBound，当前40 |
|---|---|---|---|
| 无WHERE、OFFSET0且所有候选可见/合格 | 数学上保全SQL Top10，但不保全当前Top40前缀；仅ideal诊断 | 同A | 保全当前bounded前缀，推荐 |
| 无WHERE但候选有不可见tuple | 不保证得到SQL10行 | 不保证 | 初始前缀安全；不足时必须原始full fallback |
| post-index/WHERE filter | 不安全：近邻可能全部被过滤 | 不安全：可见合格数量未知 | 前缀安全；即使前40全部过滤，关闭D1的fallback恢复后续结果 |
| OFFSET>0、均可见无过滤 | K10不保全跳过行及所需输出 | 数学上保全前LIMIT+OFFSET；仍不是本项目bounded前缀目标 | 必须按动态P；例如OFFSET20时P=120，固定40不合格 |
| 自动bounded scan、iterative off | 不作为production Gate | 不作为production Gate | 支持的首版模式；heap容量恰为P |
| full/unbounded sort | 固定K会丢掉后续结果 | 即使SQL有限需求，本设计也不改变full路径语义 | P=0或未建立，D1关闭；不能猜40 |
| StartFullSortFallback | 禁止使用旧K10继续abandon | 禁止 | 禁止继续使用旧K40；完整原始distance+sort重扫同一selected lists |
| iterative scan | 不支持 | 不支持 | 不支持；batch排序重置与全局阈值关系未证明 |
| WITH TIES / LIMIT ALL /未知或非正bound | 不支持 | 不支持 | 不支持，原始distance |

项目正式D1 potential Gate取`production_safe_K40`（本配置）以及完整fallback/数值保持协议。`ideal_no_filter_K10`只作标明假设的辅助结果，不能决定Production Go。本阶段不运行这两个workload。

## 6. D1与原有tuplesort/fallback的正确性证明

推荐“独立D1 threshold heap + 原始tuplesort”，**不替代tuplesort，不从其私有结构读取τ**。所有未early-abandon的candidate仍完整计算，并原样交给原始sort；即使其完整distance不是threshold heap的改进，也保留原sort输入路径。阈值heap只做插入/替换最差者的辅助判断。

令P为当前physicalBound，H在每一步保留原始candidate多重集按(distance,TID)排序的前P个；未满前所有candidate必须完整算。满后只在certified lower key严格大于worst distance时跳过，已有P个完整candidate均严格更好，所以被跳过者不属于当前或最终TopP。加入更多candidate只会使worst key变好；归纳可知threshold heap和原始bounded tuplesort的最终TopP与未启用D1时一致。tie由原 comparator处理，不可使用>=。

这使上层visibility/quals接收到相同bounded前缀。若请求超过前缀或sort耗尽，`StartFullSortFallback`设置fallbackTriggered、重建unbounded sort、listIndex=0（[S24 ivfscan.c:556–568](../../../contrib/pgvector/src/ivfscan.c#L556)）。**进入该函数重扫前必须禁用/清空D1阈值，原本abandon的candidate也必须重新完整计算。** 沿用既有ReturnedFromBounded抑制已交付TID。于是fallback路径也保持原始顺序与语义；不能只重扫幸存者、沿用abandon标记或重新选择lists。

反例：前40个candidate全部被WHERE过滤，第41个才合格。K40永久裁剪不能返回正确结果；K40保全前缀+原始full fallback才具备本项目所需协议。若只用K10改变初始前缀，即便可另外设计更早fallback来补救，也已改变当前bounded/fallback行为，不是本阶段选定方案。

## 7. 最终RQ回答

1. **RQ1：当前取40，通用取实际physicalBound。** 它保全现有bounded候选前缀；SQL整体安全来自这个前缀加完整fallback，非“40总够”。
2. **RQ2：OFFSET/filter可在上述协议内保持安全。** OFFSET要动态扩大P，filter可能触发full fallback；无法传播bound或超出fastpath时禁用D1。
3. **RQ3：实数条件P_m>τ_sq；实现条件是certified lower bound on旧生产score>τ_key。** heap未满、相等、数值证书未知都不能abandon。
4. **RQ4：heap按(distance,TID)构建max-heap，root为最大distance、同distance最大TID。** 等于阈值继续算，完整distance相等才由TID决定替换；survivor score须保持旧kernel值。
5. **RQ5：首版仅已建立自动physical bound的非iterative、当前vector L2 generic、正常bounded阶段。** 其他mode不启用；具体能力/数值验证完成前没有production实现获准。
6. **RQ6：无正bound、P不可用、full/fallback/iterative、oracle、非L2、direct/FUSED2、NULL、未知表示/数值条件和分配失败等回原始distance。** rescan重新初始化，不能继承旧阈值。

partial浮点推导、heap生命周期和接口契约见 `phase_d1_0_threshold_semantics.md`。完成D1-0后停止，不进入D1-1。
