import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/benchmark.py"


class BenchmarkCliTests(unittest.TestCase):
    def run_cli(self, runtime, *arguments):
        return subprocess.run(
            [sys.executable, str(CLI), "--runtime-root", str(runtime), *arguments],
            text=True, capture_output=True, check=False,
        )

    def test_final_multi_dry_run_resolves_runtime_paths_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            result = self.run_cli(
                runtime, "run", "final-multi", "--run-dir", "formal",
                "--formal", "--prepared", "prepare", "--resume", "--dry-run",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["run_dir"], str(runtime / "runs/formal"))
            self.assertIn(str(runtime / "runs/prepare"), plan["command"])
            self.assertFalse((runtime / "runs/formal").exists())

    def test_legacy_dry_run_uses_same_entry_and_explicit_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            result = self.run_cli(
                runtime, "run", "2a", "--run-dir", "smoke",
                "--queries", "7", "--probes", "64", "--dry-run",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["command"][1], "2a")
            self.assertIn("--run-dir", plan["command"])
            self.assertFalse((runtime / "runs/smoke").exists())

    def test_output_outside_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            result = self.run_cli(
                Path(directory), "run", "final-multi", "--run-dir", external,
                "--dry-run",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be under runtime root", result.stderr)

    def test_final_multi_rejects_matrix_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(
                Path(directory), "run", "final-multi", "--run-dir", "formal",
                "--queries", "1", "--dry-run",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("frozen matrix", result.stderr)


if __name__ == "__main__":
    unittest.main()
