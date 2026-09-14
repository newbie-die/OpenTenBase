"""Shared source/runtime path resolution for benchmark commands."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SOURCE_ROOT.parent
WORKSPACE_ROOT = REPOSITORY_ROOT.parent


def _environment_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class BenchmarkPaths:
    source: Path
    repository: Path
    workspace: Path
    runtime: Path
    runs: Path
    install: Path
    pgdata: Path

    @classmethod
    def from_environment(cls, runtime_root: Path | None = None) -> "BenchmarkPaths":
        runtime = (runtime_root.expanduser().resolve() if runtime_root else
                   _environment_path("BENCHMARK_RUNTIME_ROOT", WORKSPACE_ROOT / "benchmark"))
        return cls(
            source=SOURCE_ROOT,
            repository=REPOSITORY_ROOT,
            workspace=WORKSPACE_ROOT,
            runtime=runtime,
            runs=_environment_path("RUN_ROOT", runtime / "runs"),
            install=_environment_path("OPENTENBASE_INSTALL_ROOT", WORKSPACE_ROOT / "install"),
            pgdata=_environment_path("PGDATA", WORKSPACE_ROOT / "data"),
        )

    def run_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.runs / path).resolve()

    def data_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.runtime / path).resolve()

    def require_runtime_output(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(self.runtime)
        except ValueError as error:
            raise ValueError(
                f"benchmark output must be under runtime root {self.runtime}: {resolved}"
            ) from error
        return resolved
