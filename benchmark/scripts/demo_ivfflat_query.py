#!/usr/bin/env python3
"""Run one audited vector query as a baseline-versus-2A live demonstration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import final_multi
from benchmark_paths import BenchmarkPaths


DEFAULT_DATASET = "glove-cosine"
DEFAULT_QID = 7097
DEFAULT_REFERENCE_RUN = "final_multi_formal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_enter(enabled: bool, message: str) -> None:
    if enabled:
        input(f"\n{message}")


def execute(cur, config: dict, vector_literal: str) -> tuple[list[int], float]:
    started = time.perf_counter_ns()
    cur.execute(final_multi.sql(config), (vector_literal,))
    result_ids = [int(row[0]) for row in cur.fetchall()]
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return result_ids, elapsed_ms


def recall_at_10(result_ids: list[int], ground_truth: list[int]) -> float:
    return len(set(result_ids) & set(ground_truth)) / 10


def print_result(label: str, settings: dict, result_ids: list[int],
                 ground_truth: list[int], elapsed_ms: float) -> None:
    print(f"\n=== {label} ===")
    print("配置:", json.dumps(settings, ensure_ascii=False, sort_keys=True))
    print("查询结果:", json.dumps(result_ids, ensure_ascii=False))
    print(f"Recall@10: {recall_at_10(result_ids, ground_truth):.6f}")
    print(f"查询时间: {elapsed_ms:.3f} ms")


def historical_evidence(reference_run: Path, dataset: str, qid: int) -> list[dict]:
    with (reference_run / "raw.csv").open(newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row["dataset"] == dataset and int(row["qid"]) == qid
                and row["method"] in ("baseline", "2a")
                and row["selection_status"] in
                ("USED_FINAL_ROUND_A", "USED_FINAL_ROUND_B")]
    by_round: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_round.setdefault(int(row["original_round_id"]), {})[row["method"]] = row
    evidence = []
    for round_id, methods in sorted(by_round.items()):
        if set(methods) != {"baseline", "2a"}:
            continue
        baseline = methods["baseline"]
        optimized = methods["2a"]
        if baseline["result_checksum"] != optimized["result_checksum"]:
            raise RuntimeError(f"正式 Round {round_id} 的 Baseline/2A 结果不一致")
        evidence.append({
            "round": round_id,
            "baseline_ms": float(baseline["latency_us"]) / 1000,
            "optimized_ms": float(optimized["latency_us"]) / 1000,
            "recall": float(optimized["recall_at_10"]),
        })
    return evidence


def run(args: argparse.Namespace) -> int:
    paths = BenchmarkPaths.from_environment(args.runtime_root)
    reference_run = paths.require_runtime_output(paths.run_path(args.reference_run))
    installed_binary = final_multi.INSTALLED_VECTOR
    reference_binary = reference_run / "binaries/production/vector.so"
    if not installed_binary.is_file() or not reference_binary.is_file():
        raise RuntimeError("找不到当前或正式实验的 production vector.so")
    installed_hash = sha256(installed_binary)
    reference_hash = sha256(reference_binary)
    if installed_hash != reference_hash:
        raise RuntimeError(
            "当前安装的 vector.so 与正式实验不一致："
            f" installed={installed_hash} reference={reference_hash}"
        )

    config = final_multi.DATASETS[args.dataset]
    qids, queries, ground_truth = final_multi.load(config)
    try:
        position = qids.index(args.qid)
    except ValueError as error:
        raise RuntimeError(f"qid {args.qid} 不在 GIST 正式查询范围内") from error
    query = queries[position]
    gt_ids = [int(value) for value in ground_truth[position]]
    literal = final_multi.common.vector_literal(query)
    show_complete_vector = args.show_full_vector or config["dimension"] <= 128
    preview = literal if show_complete_vector else (
        "[" + ",".join(str(float(value)) for value in query[:12])
        + ", ..., " + ",".join(str(float(value)) for value in query[-4:]) + "]"
    )

    import psycopg2

    connection = psycopg2.connect(
        host=args.host, port=args.port, dbname=args.dbname, user=args.user,
        application_name="ivfflat_query_demo",
    )
    connection.autocommit = True
    try:
        with connection.cursor() as cur:
            cur.execute("SET statement_timeout='5min'")
            cur.execute("SET enable_seqscan=off")
            cur.execute("SET enable_indexscan=on")
            cur.execute("SET max_parallel_workers_per_gather=0")

            # The first index query loads vector.so and registers its GUCs.
            execute(cur, config, literal)
            identity = final_multi.index_identity(cur, config)
            validation = json.loads((reference_run / "validation.json").read_text())
            if identity != validation[args.dataset]["index"]:
                raise RuntimeError(f"当前 {args.dataset} 索引与正式实验记录不一致")

            # Prewarm both paths before the visible timed sequence.
            for method in ("baseline", "2a"):
                final_multi.settings(cur, method)
                for _ in range(args.warmup):
                    execute(cur, config, literal)

            print("=== 演示查询 ===")
            print(f"数据集: {args.dataset} | qid: {args.qid} | 维度: {config['dimension']}")
            print(f"索引: {config['index']} | lists=1000 | probes=64 | topk=10")
            print(f"query SHA256: {hashlib.sha256(query.tobytes()).hexdigest()}")
            print("SQL:")
            print(f"SELECT id FROM {config['table']}")
            print(f"ORDER BY embedding {config['operator']} :query_vector::vector LIMIT 10;")
            print("query_vector:", preview)
            if not show_complete_vector:
                print(f"提示: 添加 --show-full-vector 可打印完整 {config['dimension']} 维向量。")
            print("Ground Truth:", json.dumps(gt_ids))
            print(f"预热: Baseline 和 2A 各 {args.warmup} 次，不计入展示时间。")

            wait_for_enter(args.pause, "按 Enter 运行无优化 Baseline ...")
            baseline_settings = final_multi.settings(cur, "baseline")
            baseline_ids, baseline_ms = execute(cur, config, literal)
            print_result("1. 无优化环境（Baseline）", baseline_settings,
                         baseline_ids, gt_ids, baseline_ms)

            wait_for_enter(args.pause, "按 Enter 运行最优 2A ...")
            optimized_settings = final_multi.settings(cur, "2a")
            optimized_ids, optimized_ms = execute(cur, config, literal)
            print_result("2. 最优环境（2A LIMIT Top-K）", optimized_settings,
                         optimized_ids, gt_ids, optimized_ms)

            if baseline_ids != optimized_ids:
                raise RuntimeError("Baseline 与 2A 返回结果不一致")
            print("\n=== 对比 ===")
            print("结果一致: PASS")
            print(f"现场 Speedup: {baseline_ms / optimized_ms:.3f}x")
            print(f"现场延迟降低: {(1 - optimized_ms / baseline_ms) * 100:.1f}%")
            history = historical_evidence(reference_run, args.dataset, args.qid)
            if history:
                speeds = []
                for row in history:
                    speedup = row["baseline_ms"] / row["optimized_ms"]
                    speeds.append(speedup)
                    print(
                        f"正式 Round {row['round']}: {row['baseline_ms']:.3f} -> "
                        f"{row['optimized_ms']:.3f} ms, {speedup:.3f}x, "
                        f"Recall@10={row['recall']:.6f}"
                    )
                geomean = math.exp(sum(math.log(value) for value in speeds) / len(speeds))
                print(f"正式 rounds 几何平均 Speedup: {geomean:.3f}x")
    finally:
        connection.close()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=tuple(final_multi.DATASETS),
                        default=DEFAULT_DATASET)
    result.add_argument("--qid", type=int, default=DEFAULT_QID)
    result.add_argument("--reference-run", default=DEFAULT_REFERENCE_RUN,
                        help="正式实验目录名称或 runtime root 内的绝对路径")
    result.add_argument("--runtime-root", type=Path)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--pause", action="store_true",
                        help="展示查询后，分别等待 Enter 再执行 Baseline 和 2A")
    result.add_argument("--show-full-vector", action="store_true")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=5432)
    result.add_argument("--dbname", default="taskdb")
    result.add_argument("--user", default="dev")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.warmup < 0:
        raise SystemExit("--warmup 必须大于或等于 0")
    if any(importlib.util.find_spec(name) is None for name in ("h5py", "psycopg2")):
        paths = BenchmarkPaths.from_environment(args.runtime_root)
        venv_python = paths.runtime / ".venv/bin/python"
        if venv_python.is_file() and Path(sys.prefix).resolve() != venv_python.parent.parent.resolve():
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit("演示环境缺少 h5py 或 psycopg2，且未找到 runtime .venv")
    try:
        return run(args)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        raise SystemExit(f"演示失败: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
