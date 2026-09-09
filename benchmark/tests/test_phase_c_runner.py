"""Integrity regressions for Phase C durable checkpoint and pristine GUC handling."""
import importlib.util
import copy
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('profile', Path(__file__).resolve().parents[1] / 'scripts/ivfflat_profile.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


class PhaseCIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(phase='C', experiment='phase_c', smoke=True, system='pristine',
                        probes=16, round=0, query_id=0, result_ids=list(range(10)),
                        recall_at_10=1.0, latency_us=10.0, returned_rows=10,
                        binary_sha256='p')
        self.expected = {('pristine', 16, 0, 0), ('final', 16, 0, 0)}
        self.manifest = dict(pristine_binary_sha256='p', optimized_binary_sha256='o')
        self.neighbors = [list(range(10))]

    def validate(self, rows):
        return b.phase_c_validate_checkpoint(rows, self.expected, self.manifest, self.neighbors)

    def test_partial_checkpoint_preserves_missing_key(self):
        keys = self.validate([self.row])
        self.assertEqual(self.expected - set(keys), {('final', 16, 0, 0)})

    def test_duplicate_and_unknown_execution_keys_rejected(self):
        with self.assertRaises(RuntimeError):
            self.validate([self.row, self.row])
        row = dict(self.row, round=1)
        with self.assertRaises(RuntimeError):
            self.validate([row])

    def test_corrupt_checkpoint_rejected(self):
        for field, value in [('binary_sha256', 'wrong'), ('result_ids', [0] * 10),
                             ('recall_at_10', 0.5), ('latency_us', float('nan')),
                             ('experiment', 'formal'), ('smoke', False), ('extra', 1)]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate([{**self.row, field: value}])

    def test_atomic_checkpoint_retains_previous_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            b.write_json_atomic(path, [self.row])
            original = path.read_bytes()
            with self.assertRaises(TypeError):
                b.write_json_atomic(path, object())
            self.assertEqual(path.read_bytes(), original)

    def test_pristine_never_sets_optimized_gucs(self):
        class Cursor:
            def __init__(self):
                self.settings = []
                self.value = None
            def execute(self, sql, params=None):
                if sql.startswith('SELECT setting'):
                    self.value = 'registered'
                elif sql.startswith('SELECT set_config'):
                    self.settings.append(params[0])
                    self.value = params[1]
            def fetchone(self):
                return (self.value,)
        cur = Cursor()
        b.phase_c_settings(cur, 'pristine', 16)
        self.assertEqual(cur.settings, ['ivfflat.probes', 'ivfflat.iterative_scan'])


if __name__ == '__main__':
    unittest.main()
