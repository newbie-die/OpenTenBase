# FUSED2 Raw microbenchmark

`fused2_raw_bench.c` 保持独立 C 程序，直接链接真实 `vector.c` 对象；不经过 PostgreSQL 查询、fmgr 或 IVFFlat。它原位于 `benchmark/scripts/`，此次仅移动位置，算法和参数不变。

固定 dimension=960、seed=20260907，hot pool=4 candidates，IVFFlat-like pool=4096 candidates（15 MiB），含预热、12 轮 DFFD/FDDF 和 checksum 消费。运行参数是 CPU 编号，stdout 为逐样本 CSV，stderr 为配置及 pool 正确性检查。请从当前 affinity 允许的 CPU 中选择。

从仓库根目录构建到独立证据目录，复用 Makefile 的完整 flags，不覆盖 PG_CFLAGS：

```bash
python3 - <<'PY'
import os
import shlex
import subprocess
from pathlib import Path

root = Path.cwd()
cwd = root / 'contrib/pgvector'
out = Path('/tmp/fused2-raw')
out.mkdir(exist_ok=True)
env = os.environ.copy()
env.pop('PG_CFLAGS', None)
make = ['make', '-n', '-B', 'src/vector.o',
        'PG_CONFIG=/workspace/install/bin/pg_config',
        'OPTFLAGS=-march=haswell -mtune=haswell -mavx2 -mfma',
        'IVFFLAT_PROFILE_CFLAGS=-DIVFFLAT_BENCH -DIVFFLAT_PROFILE_2B']
text = subprocess.check_output(make, cwd=cwd, env=env, text=True)
cmd = next(shlex.split(line) for line in text.splitlines()
           if line.startswith('gcc ') and 'src/vector.c' in line)
cmd[cmd.index('-o') + 1] = str(out / 'vector.o')
subprocess.run(cmd, cwd=cwd, env=env, check=True)
driver = cmd.copy()
driver.insert(1, '-Isrc')
driver[driver.index('-o') + 1] = str(out / 'driver.o')
driver[-1] = str(root / 'benchmark/microbench/fused2_raw_bench.c')
subprocess.run(driver, cwd=cwd, env=env, check=True)
link = ['gcc', '-no-pie', '-Wl,--unresolved-symbols=ignore-in-object-files',
        '-o', str(out / 'bench'), str(out / 'driver.o'), str(out / 'vector.o'), '-lm']
subprocess.run(link, check=True)
(out / 'compile-commands.txt').write_text('\n'.join(shlex.join(c) for c in (cmd, driver, link)) + '\n')
with (out / 'bench.asm').open('w') as f:
    subprocess.run(['objdump', '-dwC', '-Mintel', str(out / 'bench')], stdout=f, check=True)
PY
/tmp/fused2-raw/bench 0 > /tmp/fused2-raw/raw.csv 2> /tmp/fused2-raw/run.log
```

独立链接允许完整 `vector.o` 中未调用的 PostgreSQL 函数引用未解析；两个 Raw kernel 不依赖这些函数。此可执行文件仅用于该 driver，不是数据库扩展。保存并检查 `bench.asm` 中 driver 的两次 Raw / 一次 PairRaw 调用。

每 case/path 共 24 samples；ns/candidate=ns/pair÷2。统计全部样本的 median/min/max 和 population CV，不只选最好的一轮。此 Raw 层流程独立于数据库 correctness，不新增专用 Python runner。
