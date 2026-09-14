#!/usr/bin/env python3
"""Single entry point for OpenTenBase benchmark runs and runtime inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_paths import BenchmarkPaths


PHASES = (
    "all", "a", "b", "2a", "2a2", "2a34", "2b", "2b-correctness",
    "2b-production", "formal", "all-fomal-exp", "final-multi",
)
RUNTIME_CODE_EXCLUDES = {
    ".venv", "__pycache__", "binaries", "build", "assembly",
    "feature_parity", "sift_feature_parity", "site-packages",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_python(paths: BenchmarkPaths) -> Path:
    candidate = paths.runtime / ".venv/bin/python"
    return candidate if candidate.is_file() else Path(sys.executable).resolve()


def resolve_run_dir(paths: BenchmarkPaths, value: str | None, phase: str) -> Path:
    if value:
        return paths.require_runtime_output(paths.run_path(value))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return paths.runs / f"{phase.replace('-', '_')}_{timestamp}"


def apply_tuning(environment: dict[str, str], args: argparse.Namespace) -> None:
    if args.phase == "final-multi":
        supplied = [name for name in ("dataset", "warmup", "queries", "probes", "rounds", "topk", "lists")
                    if getattr(args, name) is not None]
        if supplied:
            raise ValueError(f"final-multi has a frozen matrix; unsupported tuning: {', '.join(supplied)}")
        return
    phase_prefix = {
        "2a": "PHASE2A", "2a2": "PHASE2A2", "2a34": "PHASE2A34",
        "2b": "PHASE2B", "2b-correctness": "PHASE2B_CORRECTNESS",
        "2b-production": "PHASE2B_PRODUCTION", "formal": "FORMAL",
    }.get(args.phase)
    if args.warmup is not None:
        environment[f"{phase_prefix}_WARMUP" if phase_prefix else "WARMUP"] = str(args.warmup)
    if args.queries is not None:
        environment[f"{phase_prefix}_QUERIES" if phase_prefix else "QUERIES"] = str(args.queries)
    if args.probes is not None:
        environment[f"{phase_prefix}_PROBES" if phase_prefix else "PROBES"] = args.probes
    if args.rounds is not None:
        if args.phase not in ("2a", "2b", "2b-production", "formal"):
            raise ValueError(f"--rounds is not supported for phase {args.phase}")
        environment[f"{phase_prefix}_ROUNDS"] = str(args.rounds)
    if args.topk is not None:
        environment["TOPK"] = str(args.topk)
    if args.lists is not None:
        environment["LISTS"] = str(args.lists)


def final_multi_command(args: argparse.Namespace, paths: BenchmarkPaths,
                        run_dir: Path, python: Path, extra: list[str]) -> list[str]:
    command = [str(python), str(paths.source / "scripts/ivfflat_profile.py"), "final-multi",
               "--output", str(run_dir), "--host", args.host, "--port", str(args.port),
               "--dbname", args.dbname, "--user", args.user]
    if args.formal:
        if not args.prepared:
            raise ValueError("final-multi --formal requires --prepared")
        command.extend(("--formal", "--prepared", str(paths.run_path(args.prepared))))
    elif args.prepared:
        raise ValueError("--prepared is only valid with final-multi --formal")
    if args.resume:
        command.append("--resume")
    command.extend(extra)
    return command


def legacy_command(args: argparse.Namespace, paths: BenchmarkPaths,
                   run_dir: Path, extra: list[str]) -> list[str]:
    command = [str(paths.source / "scripts/run_ivfflat_experiments.sh"), args.phase,
               "--foreground"]
    if args.resume:
        if args.phase not in ("formal", "all-fomal-exp"):
            raise ValueError(f"legacy phase {args.phase} does not support --resume")
        command.extend(("--resume", str(run_dir)))
    else:
        command.extend(("--run-dir", str(run_dir)))
    if args.dataset:
        command.extend(("--dataset", args.dataset))
    command.extend(extra)
    return command


def write_launcher_manifest(path: Path, command: list[str], args: argparse.Namespace,
                            paths: BenchmarkPaths) -> None:
    path.mkdir(parents=True, exist_ok=True)
    value = {
        "phase": args.phase,
        "command": command,
        "source_root": str(paths.source),
        "runtime_root": str(paths.runtime),
        "run_dir": str(path),
        "background": args.background,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path / ".launcher.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path / "launcher.json")


def run_command(args: argparse.Namespace, extra: list[str]) -> int:
    paths = BenchmarkPaths.from_environment(args.runtime_root)
    run_dir = resolve_run_dir(paths, args.run_dir, args.phase)
    python = Path(args.python).expanduser().resolve() if args.python else default_python(paths)
    environment = dict(os.environ)
    environment.update({
        "BENCHMARK_ROOT": str(paths.source),
        "BENCHMARK_RUNTIME_ROOT": str(paths.runtime),
        "RUN_ROOT": str(paths.runs),
        "OPENTENBASE_INSTALL_ROOT": str(paths.install),
        "PGDATA": str(paths.pgdata),
        "PYTHON_BIN": str(python),
    })
    apply_tuning(environment, args)
    command = (final_multi_command(args, paths, run_dir, python, extra)
               if args.phase == "final-multi"
               else legacy_command(args, paths, run_dir, extra))
    description = {"command": command, "run_dir": str(run_dir), "runtime_root": str(paths.runtime)}
    if args.dry_run:
        print(json.dumps(description, indent=2))
        return 0
    if args.resume and not run_dir.is_dir():
        raise ValueError(f"resume run directory does not exist: {run_dir}")
    write_launcher_manifest(run_dir, command, args, paths)
    log_path = run_dir / "experiment.log"
    if args.background:
        log = log_path.open("ab")
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (run_dir / "runner.pid").write_text(f"{process.pid}\n")
        print(json.dumps({**description, "pid": process.pid, "log": str(log_path)}, indent=2))
        return 0
    if args.phase != "final-multi":
        return subprocess.call(command, env=environment)
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status_command(args: argparse.Namespace) -> int:
    paths = BenchmarkPaths.from_environment(args.runtime_root)
    run_dir = paths.require_runtime_output(paths.run_path(args.run_dir))
    pid_path = run_dir / "runner.pid"
    pid = int(pid_path.read_text()) if pid_path.is_file() else None
    running = False
    if pid:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            pass
    value = {"run_dir": str(run_dir), "pid": pid, "running": running}
    for name in ("progress.json", "manifest.json", "final_comparison.json"):
        content = read_json(run_dir / name)
        if content is not None:
            value[name.removesuffix(".json")] = content
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def audit_command(args: argparse.Namespace) -> int:
    paths = BenchmarkPaths.from_environment(args.runtime_root)
    source_by_hash: dict[str, list[str]] = {}
    source_roots = (paths.source, paths.repository / "contrib/pgvector")
    for source_root in source_roots:
        for path in source_root.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".sh", ".c", ".h") and "__pycache__" not in path.parts:
                source_by_hash.setdefault(sha256(path), []).append(str(path))
    runtime_files = []
    for path in paths.runtime.rglob("*"):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".c", ".h"):
            continue
        relative = path.relative_to(paths.runtime)
        if any(part in RUNTIME_CODE_EXCLUDES for part in relative.parts):
            continue
        digest = sha256(path)
        runtime_files.append({"path": str(path), "size": path.stat().st_size,
                              "sha256": digest, "source_matches": source_by_hash.get(digest, [])})
    value = {
        "source_roots": [str(path) for path in source_roots],
        "runtime_root": str(paths.runtime),
        "policy": "source code lives in OpenTenBase/benchmark; data, logs and run artifacts live in the runtime root",
        "runtime_code_files": runtime_files,
        "exact_source_duplicates": sum(bool(item["source_matches"]) for item in runtime_files),
    }
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = paths.require_runtime_output(paths.data_path(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-root", type=Path,
                        help="data/log/artifact root (default: BENCHMARK_RUNTIME_ROOT or <workspace>/benchmark)")
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume an experiment")
    run.add_argument("phase", choices=PHASES)
    run.add_argument("--run-dir", help="absolute path or name below <runtime>/runs")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--background", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--python")
    run.add_argument("--formal", action="store_true", help="enable final-multi formal mode")
    run.add_argument("--prepared", help="final-multi preparation run path/name")
    run.add_argument("--dataset")
    run.add_argument("--warmup", type=int)
    run.add_argument("--queries", type=int)
    run.add_argument("--probes")
    run.add_argument("--rounds", type=int)
    run.add_argument("--topk", type=int)
    run.add_argument("--lists", type=int)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=5432)
    run.add_argument("--dbname", default="taskdb")
    run.add_argument("--user", default="dev")
    status = commands.add_parser("status", help="read one run without changing it")
    status.add_argument("--run-dir", required=True)
    audit = commands.add_parser("audit", help="find code files stored in the runtime tree")
    audit.add_argument("--output", help="optional JSON path below the runtime root")
    return result


def main() -> int:
    args, extra = parser().parse_known_args()
    try:
        if args.command == "run":
            return run_command(args, extra)
        if extra:
            raise ValueError(f"unexpected arguments: {' '.join(extra)}")
        if args.command == "status":
            return status_command(args)
        return audit_command(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
