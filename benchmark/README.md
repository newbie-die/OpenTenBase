# OpenTenBase IVFFlat Benchmark

本目录保存 IVFFlat 性能优化、正确性验证和正式实验的可版本化源码。数据集、数据库、编译产物、日志、检查点和结果统一保存在仓库外的运行目录中。标准环境的运行根目录是 `/workspace/benchmark`，可通过环境变量 `BENCHMARK_RUNTIME_ROOT` 或统一入口的 `--runtime-root` 修改。

## 目录约定

| 位置 | 内容 |
| --- | --- |
| `benchmark/scripts/` | 统一命令入口、实验调度、结果汇总、恢复逻辑和公共路径处理 |
| `benchmark/configs/` | 受版本控制的 workload 与数据集配置 |
| `benchmark/microbench/` | 不经过 PostgreSQL executor 的独立 C 微基准 |
| `benchmark/d0b/`、`d1_1/`、`d1_2/`、`d1_2b/` | D 阶段各项在线或离线机制验证程序 |
| `benchmark/d2/` | 自适应策略的离线训练、风险控制和 oracle 分析 |
| `benchmark/d2p/` | 已部署 D2-P 策略的采集、校准、验证代码和冻结策略 |
| `benchmark/docs/` | 阶段设计、审计、实验协议与结论 |
| `benchmark/tests/` | runner、checkpoint、resume、汇总和 C/Python 一致性测试 |
| `/workspace/benchmark/data/` | GIST、SIFT、GloVe、Cohere 等数据集 |
| `/workspace/benchmark/runs/` | 每次运行的日志、manifest、checkpoint、raw CSV 和 summary |
| `/workspace/benchmark/frozen/` | 已冻结的历史正式实验快照 |

新实验应从仓库中的 `benchmark/scripts/benchmark.py` 启动。运行目录里保存的源码副本、临时修复脚本和 `before` 快照仅作为历史证据，不作为当前执行入口。

## 阶段名称

整个研究过程按 A、B、C、D 四个阶段组织：

| 阶段 | 研究对象 | 当前可运行名称 |
| --- | --- | --- |
| A | IVFFlat 扫描、候选保留和排序开销；2A 为 bounded scan 优化 | `a`、`2a`、`2a2`、`2a34` |
| B | exact distance 计算路径；2B 为 direct/FUSED2 优化 | `b`、`2b`、`2b-correctness`、`2b-production` |
| C | 多系统正式对照、结果完整性和可恢复执行 | `formal` |
| D | 安全 page pruning、自适应 probes 与 D2-P 策略 | `all-fomal-exp`；当前正式复现推荐使用 `final-multi` |

最终结果中的 `2A+B` 表示同时启用 2A bounded scan 和 B/FUSED2。阶段 C 是正式验证框架，不是一个单独的查询优化方法。最终表中的方法 `D` 指已经部署的 D2-P 自适应 probes 策略。

面向代码评审和结果复现的简明提交报告见 [`docs/IVFFLAT_OPTIMIZATION.md`](docs/IVFFLAT_OPTIMIZATION.md)。

## 环境依赖

### 硬件与操作系统

- 使用 Linux 环境；调度脚本依赖 Bash、POSIX 文件权限、进程信号和文件锁。
- 当前冻结的编译参数包含 `-march=haswell -mavx2 -mfma`，运行 CPU 必须支持 AVX2 和 FMA。正式复现时不要擅自改变这些参数，否则 binary identity 和性能结果不可直接比较。
- 正式实验会同时保存多个 `vector.so`、隔离源码副本、checkpoint 和 raw CSV。数据集及数据库占用不计入仓库，运行前应确认 `/workspace/benchmark` 和 PostgreSQL 数据目录有足够空间。

### 编译与系统工具

需要以下命令可从 `PATH` 找到，或位于下面的默认安装路径：

| 依赖 | 用途 |
| --- | --- |
| GCC 11.5.0 | 编译 pgvector、FUSED2、D2-P 和 C/Python 特性一致性程序；Phase C 会严格检查该版本 |
| GNU Make | 构建 `contrib/pgvector` |
| Git | 记录源码提交和 dirty diff；Phase C 使用 detached worktree 构建 pristine binary |
| GNU binutils | `objdump`、`strings`、`nm` 用于检查汇编、profiling 字符串和未解析符号 |
| coreutils | `sha256sum`、`sort`、`tee` 等用于生成身份和运行证据 |
| util-linux | 非 PostgreSQL OS 用户运行时通过 `runuser` 切换到数据库用户 |
| OpenTenBase/PostgreSQL 开发环境 | 提供 `pg_config`、`pg_ctl`、server headers、client 和运行中的数据库实例 |

当前验证环境为：

```text
Python       3.9.25
GCC          11.5.0
GNU Make     4.3
PostgreSQL   18.6（由 /workspace/install/bin/pg_config 报告）
CPU flags    Haswell / AVX2 / FMA
```

可执行以下命令检查核心工具：

```bash
python3 --version
gcc -dumpfullversion
make --version | head -1
git --version
/workspace/install/bin/pg_config --version
/workspace/install/bin/pg_ctl --version
command -v objdump strings nm sha256sum runuser
grep -m1 '^flags' /proc/cpuinfo | grep -qw avx2
grep -m1 '^flags' /proc/cpuinfo | grep -qw fma
```

### Python 依赖

Python 至少需要 3.9。基础 A/B/C runner 使用 NumPy、h5py 和 psycopg2；Final Multi 读取 Cohere Parquet 时还需要 PyArrow，SIFT D 校准和 `benchmark/d2/` 离线分析还需要 scikit-learn。

| Python 包 | 当前验证版本 | 用途 |
| --- | ---: | --- |
| `numpy` | 2.0.2 | 向量、统计、归一化和 percentile |
| `h5py` | 3.14.0 | 读取 GIST、SIFT 和 GloVe HDF5 数据 |
| `psycopg2-binary` | 2.9.12 | 连接 OpenTenBase/PostgreSQL |
| `pyarrow` | 21.0.0 | 读取 Cohere Parquet 数据和 ground truth |
| `scikit-learn` | 1.6.1 | D/D2-P 的决策树、GBDT、校准和离线审计 |

标准环境使用 `/workspace/benchmark/.venv`。新环境可以按当前已验证版本创建：

```bash
python3.9 -m venv /workspace/benchmark/.venv
/workspace/benchmark/.venv/bin/python -m pip install --upgrade pip
/workspace/benchmark/.venv/bin/python -m pip install \
  numpy==2.0.2 \
  h5py==3.14.0 \
  psycopg2-binary==2.9.12 \
  pyarrow==21.0.0 \
  scikit-learn==1.6.1
```

`/workspace/benchmark/requirements.txt` 只覆盖基础 runner 的三个包。执行 Final Multi 或 D 阶段时仍需安装上表中的 PyArrow 和 scikit-learn。统一入口默认优先使用 `/workspace/benchmark/.venv/bin/python`，也可以通过 `--python /path/to/python` 显式指定解释器。

检查 Python 环境：

```bash
/workspace/benchmark/.venv/bin/python - <<'PY'
import h5py
import numpy
import psycopg2
import pyarrow
import sklearn

print('numpy', numpy.__version__)
print('h5py', h5py.__version__)
print('psycopg2', psycopg2.__version__)
print('pyarrow', pyarrow.__version__)
print('scikit-learn', sklearn.__version__)
PY
```

### OpenTenBase、数据库与权限

- OpenTenBase 应已编译并安装到 `/workspace/install`；`/workspace/install/bin/pg_config` 必须与要运行的 server 和扩展安装目录一致。
- PostgreSQL 数据目录默认为 `/workspace/data`，数据库 OS 用户默认为 `dev`。需要重新编译或切换 `vector.so` 的阶段必须有权停止、安装并重新启动这个实例；可由 `dev` 直接运行，也可由 root 通过 `runuser` 执行。
- 默认数据库是 `taskdb`，连接用户是 `dev`。该用户需要读取 benchmark 表、设置实验使用的 `ivfflat.*` GUC，并执行查询与元数据检查。
- 数据库需要安装当前源码对应的 vector 扩展。Final Multi 所需表和索引是 `gist_base/gist_ivf_l2`、`sift_base/sift_ivf_l2`、`glove_base/glove_ivf_cosine`、`glove_ip_base/glove_ivf_ip` 和 `cohere_base/cohere_ivf_cosine`，索引均使用 `lists=1000`。
- Phase C 需要环境变量 `PRISTINE_COMMIT` 指向可解析且可建立 detached worktree 的提交。正式 resume 必须继续使用 Part 1 记录的同一提交。

可先检查服务和扩展：

```bash
/workspace/install/bin/pg_ctl -D /workspace/data status
/workspace/install/bin/psql \
  -h 127.0.0.1 -p 5432 -U dev -d taskdb \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

### 数据集

数据路径由 [`configs/final_multi_datasets.json`](configs/final_multi_datasets.json) 定义，相对路径均以运行根目录为基准。标准布局需要：

```text
/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5
/workspace/benchmark/data/gist1m/gist-960-euclidean-formal.hdf5
/workspace/benchmark/data/sift1m/sift-128-euclidean.hdf5
/workspace/benchmark/data/glove100/glove-100-angular.hdf5
/workspace/benchmark/data/cohere1m/shuffle_train.parquet
/workspace/benchmark/data/cohere1m/test.parquet
/workspace/benchmark/data/cohere1m/neighbors.parquet
```

Final Multi 的 preparation 会检查文件 SHA256、维度、行数、base ID 与向量映射、ground truth、查询切分、索引 opclass 和 EXPLAIN 计划。任一检查失败都应先修复环境或数据，不应跳过 preparation 直接运行 formal。

## 运行前提与默认路径

默认路径如下：

```text
源码仓库        /workspace/OpenTenBase
运行根目录      /workspace/benchmark
Python 环境     /workspace/benchmark/.venv/bin/python
OpenTenBase 安装 /workspace/install
PGDATA          /workspace/data
运行结果        /workspace/benchmark/runs/<run-name>
```

数据库中需要预先存在配置文件声明的表和 IVFFlat 索引。实验 runner 会核对索引、opclass、lists、数据文件和查询映射；需要复用已有索引的阶段不会静默重建索引。默认连接参数为 `127.0.0.1:5432`、数据库 `taskdb`、用户 `dev`，可通过 `--host`、`--port`、`--dbname` 和 `--user` 修改。

所有命令以下列目录为当前目录：

```bash
cd /workspace/OpenTenBase
```

可先添加 `--dry-run` 检查最终命令和输出路径。`--dry-run` 不创建运行目录，也不执行查询。

早期 `a`、`b`、`2a`、`2a2` 和 `2a34` 流程会按阶段要求重建目标 IVFFlat 索引，应在专用 benchmark 数据库上运行。`2b*`、Phase C resume、D2-P resume 和 Final Multi 会复用并核对已有索引。

## A：扫描与排序优化

### A 做什么

早期 Phase A 在相同的 GloVe 100 维数据上比较 L2、Cosine 和 Inner Product 三种距离语义，并遍历 `probes=1..256`。它记录 list、candidate、distance、sort 和完整 scan 的耗时，用来确定 IVFFlat 查询中的主要成本分布。

Phase 2A 随后把 SQL `LIMIT` 传入 IndexScan 和 tuplesort，为候选排序设置物理 bound。正式使用的参数是逻辑 Top10、`bound_overfetch=4`、`bound_min=40`、`bound_fastpath_limit=100`。这项优化减少 tuplesort 保留与排序工作，但仍计算当前 probes 范围内的全部候选距离，因此不改变扫描集合和 Recall。

围绕 2A 已完成以下验证：

- `2a`：在 GloVe-L2 上扫描 probes、sort bound 和多轮延迟，验证 Top10 返回数量和 Recall。
- `2a2`：在 GloVe-Cosine 上比较 full、固定 oracle40 和自动 bounded scan，核对完整 Top10。
- `2a34`：加入不同过滤选择率，验证结果不足 Top10 时的 fallback、结果唯一性和 full/auto 一致性。
- Final Multi：在 GIST-L2、SIFT-L2、GloVe-Cosine、Cohere-Cosine 和 GloVe-IP 上正式比较 Baseline 与 2A。

### A 如何复现

复现早期算子与 probes 剖析：

```bash
benchmark/scripts/benchmark.py run a \
  --run-dir phase_a_repro \
  --warmup 100 --queries 1000 \
  --background
```

复现 2A 主实验：

```bash
benchmark/scripts/benchmark.py run 2a \
  --run-dir phase_2a_repro \
  --warmup 500 --queries 10000 \
  --probes 16,64,128 --rounds 3 \
  --background
```

复现 Cosine 自动 bound 和过滤 fallback：

```bash
benchmark/scripts/benchmark.py run 2a2 \
  --run-dir phase_2a_cosine_repro \
  --warmup 500 --queries 10000 \
  --probes 16,64,128 \
  --background

benchmark/scripts/benchmark.py run 2a34 \
  --run-dir phase_2a_filter_repro \
  --warmup 5 --queries 20 --probes 64 \
  --background
```

这些命令的主要结果是运行目录中的 `phase_a_profile_*.csv`、`phase2a_raw.csv`、`phase2a_summary.csv` 或相应的 `phase_2a2/2a34` CSV。

## B：距离计算与 FUSED2

### B 做什么

早期 Phase B 固定 `probes=64`，在 GloVe-100、SIFT-128 和 GIST-960 三个 L2 workload 上观察维度增长对 candidate distance 和 scan 耗时的影响。结论推动了后续 exact distance kernel 优化。

Phase 2B 将距离计算路径拆成 generic、direct 和 FUSED2。FUSED2 一次处理两个 L2 candidate，并保留单数尾项和不支持输入的回退路径。它只改变 exact L2 distance 的计算实现，不改变 selected lists、scanned pages、candidate 数量、tuplesort 输入或结果排序。

围绕 2B 已完成以下工作：

- 独立 raw C 微基准，检查 kernel 指令、checksum、不同 candidate pool 和冷热数据表现。
- PostgreSQL 内部 profiling，对 generic/direct/FUSED2 的 workload counters 和耗时进行配对比较。
- correctness 测试，逐查询核对 result TID、ID、distance bit、返回行数和边界输入。
- production 测试，使用无 profiling 宏的构建，在 full/auto、多个 probes、交错顺序和四轮数据上比较 direct 与 FUSED2。
- Final Multi 中只在 GIST-L2 和 SIFT-L2 运行 `2A+B`；Cosine 和 IP 不运行 FUSED2。

### B 如何复现

复现早期跨维度 L2 剖析：

```bash
benchmark/scripts/benchmark.py run b \
  --run-dir phase_b_repro \
  --warmup 100 --queries 1000 \
  --background
```

先运行正确性门禁，再运行 production 对照：

```bash
benchmark/scripts/benchmark.py run 2b-correctness \
  --run-dir phase_2b_correctness_repro \
  --warmup 100 --queries 100 --probes 64 \
  --background

benchmark/scripts/benchmark.py run 2b-production \
  --run-dir phase_2b_production_repro \
  --warmup 100 --queries 1000 \
  --probes 16,64,128 --rounds 4 \
  --background
```

独立 kernel 微基准的构建和运行方式见 [`microbench/README.md`](microbench/README.md)。

## C：正式对照与可恢复执行

### C 做什么

Phase C 建立统一的正式实验协议，用于比较 `pristine`、`vanilla_eq`、`phase2a`、`phase2b` 和 `final` 五个系统。它的重点是实验可比性和结果完整性：

- Part 1 使用 GIST-L2、3 条查询、1 次 warmup 和 1 轮执行 smoke，并验证 pristine/optimized 两套 binary 的来源、编译参数和 SHA256。
- Part 2 使用 GIST 的 1000 条官方查询、9 个 probes、10 轮以及每配置 100 次 warmup，共执行 450000 次 measured queries。
- 同一 round 内共享确定性的 query permutation 和 probes order，并轮换 binary family 与优化系统顺序，减少运行顺序偏差。
- 每个配置写入原子 checkpoint；resume 会验证 manifest、数据、索引、binary、查询顺序、结果 checksum 和 Recall，完成的配置不会重跑。
- raw CSV 和 round summary 可由 checkpoint 重建，避免中断造成半行、重复行或 completed 状态丢失。

Phase C 的详细协议见 [`docs/phase_c_part1.md`](docs/phase_c_part1.md) 和 [`docs/phase_c_part2.md`](docs/phase_c_part2.md)。

### C 如何复现

首先指定用于 pristine binary 的提交并运行 Part 1：

```bash
PRISTINE_COMMIT=d65ea656a91b38c84dea234a83524b43e080e427 \
benchmark/scripts/benchmark.py run formal \
  --run-dir phase_c_part1_repro \
  --background
```

Part 1 通过后，以其 `phase_c_artifacts` 启动 Part 2：

```bash
PRISTINE_COMMIT=d65ea656a91b38c84dea234a83524b43e080e427 \
benchmark/scripts/benchmark.py run formal \
  --run-dir phase_c_part2_repro \
  --background \
  --part2 \
  --part1-artifacts /workspace/benchmark/runs/phase_c_part1_repro/phase_c_artifacts
```

中断后使用同一运行目录恢复：

```bash
PRISTINE_COMMIT=d65ea656a91b38c84dea234a83524b43e080e427 \
benchmark/scripts/benchmark.py run formal \
  --run-dir phase_c_part2_repro \
  --resume --background
```

主要证据位于 `<run-dir>/phase_c_artifacts/`，包括 manifest、permutation、run order、checkpoints、raw CSV、round summaries、binary/source hash 和最终报告。

## D：安全剪枝与自适应 probes

### D 做什么

D 的目标是在不改变 Top-K 排序语义的前提下减少无效 page processing、candidate evaluation、exact distance calls 和 tuplesort 输入，并根据单条查询的难度在 `16 -> 32 -> 64` probes 之间安全提前停止。

D 按以下路径推进：

- D-0A：审计 L2、Cosine 和 IP 的距离语义、LIMIT/threshold 传播以及 metric-aware 安全上下界，只完成数学与代码路径设计。
- D-0B：在 GIST-L2 上以 shadow replay 验证 radial-shell page bound。正确性检查全部通过，但可安全跳过的页面比例过低，因此该候选方案得到 No-Go，没有进入 production。
- D1：研究 threshold-aware progressive distance。D1.1 做扫描内 shadow 对照；D1.2 和 D1.2b 用独立 kernel/稀疏页实验判断计算潜力。
- D2：基于 progressive shadow 数据进行离线策略研究，包括固定阈值、风险排序、loss budget、GBDT、收敛分析和 oracle gap attribution。
- D2-P：冻结可部署的浅层决策树，在 stage 16 和 stage 32 根据 centroid 与扫描中间特征决定停止或继续到 64；同时保留固定 probes fallback、C/Python 特征一致性和逐查询 Top10 校验。

当前正式 D 只覆盖具备独立校准和部署验证的 GIST-L2 与 SIFT-L2。GloVe-Cosine、Cohere-Cosine 和 GloVe-IP 没有进入 D 正式矩阵；这几项显示为 N/A，不应解释为性能为零。D 也没有与 2A 组合为 `2A+D`。

设计和审计材料见：

- [`docs/phase_d0a/phase_d0a_semantic_audit.md`](docs/phase_d0a/phase_d0a_semantic_audit.md)
- [`docs/phase_d0b/phase_d0b_l2_report.md`](docs/phase_d0b/phase_d0b_l2_report.md)
- [`docs/phase_d1_0/phase_d1_0_semantic_audit.md`](docs/phase_d1_0/phase_d1_0_semantic_audit.md)
- [`docs/phase_d2/phase_d2_oracle_report.json`](docs/phase_d2/phase_d2_oracle_report.json)

### D 如何复现

旧版单数据集 D2-P 正式入口仍可用于历史协议复现：

```bash
benchmark/scripts/benchmark.py run all-fomal-exp \
  --run-dir d2p_gist_repro \
  --background
```

中断后恢复：

```bash
benchmark/scripts/benchmark.py run all-fomal-exp \
  --run-dir d2p_gist_repro \
  --resume --background
```

当前推荐通过 Final Multi 同时复现 Baseline、2A、2A+B 和 D，因为它会执行数据验证、SIFT 独立校准、策略一致性门禁和最终两轮汇总。

## Final Multi-Dataset 正式复现

Final Multi 使用 [`configs/final_multi_datasets.json`](configs/final_multi_datasets.json) 定义 workload，并复用 Phase C 的 binary 激活、GUC、随机顺序、原子 checkpoint、CSV 和 hash 机制。

每个有效 round 有 14 个配置：五个 workload 的 Baseline 与 2A 共 10 项，GIST/SIFT 的 2A+B 共 2 项，GIST/SIFT 的 D 共 2 项。最终只使用两个完整 round，共 28 config-rounds 和 164000 次正式 query execution。统计输出保留 Round A、Round B、mean 和 sample std，不选择最好的一轮。

先执行准备与 smoke。该步骤验证数据、GT、表、索引、opclass、lists、EXPLAIN、query split 和 hash，不会自动启动正式长实验：

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_prepare_repro
```

准备结果中的 `smoke_gate.json` 为 PASS 后启动正式实验：

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_formal_repro \
  --formal \
  --prepared final_multi_prepare_repro \
  --background
```

同一命令加 `--resume` 即可安全恢复；已验证完成的 config 会被跳过：

```bash
benchmark/scripts/benchmark.py run final-multi \
  --run-dir final_multi_formal_repro \
  --formal \
  --prepared final_multi_prepare_repro \
  --resume --background
```

最终目录包含 `plan.json`、`manifest.json`、`progress.json`、`checkpoints/`、`raw.csv`、`summary.csv`、`final_comparison.csv` 和 `final_comparison.json`。详细矩阵、查询数和产物定义见 [`docs/final_multi_comparison.md`](docs/final_multi_comparison.md)。

## 查看进度与日志

```bash
benchmark/scripts/benchmark.py status \
  --run-dir final_multi_formal_repro

tail -f /workspace/benchmark/runs/final_multi_formal_repro/experiment.log
```

后台运行还会在目录内生成 `launcher.json` 和 `runner.pid`。`status` 只读取现有状态，不执行查询。

## 源码与运行目录审计

以下命令扫描运行根目录中的 Python、Shell、C 和头文件，并报告与源码仓库完全相同的 SHA256 副本：

```bash
benchmark/scripts/benchmark.py audit \
  --output audits/benchmark-code-layout.json
```

审计报告写入 `/workspace/benchmark/audits/benchmark-code-layout.json`。可复用的实现必须提交到本目录；运行目录只保存输入身份、日志、编译证据、manifest、checkpoint 和结果。历史运行和 frozen 目录中的源码副本保留用于追溯，不参与当前代码加载。
