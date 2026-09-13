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


class PhaseCFormalTests(unittest.TestCase):
    def test_schedule_balances_families_and_covers_full_workload(self):
        permutations, schedule = b.phase_c_schedule()
        self.assertEqual(len(schedule), 450)
        self.assertEqual(len({(c['system'], c['probes'], c['round']) for c in schedule}), 450)
        self.assertEqual(sum(p['family_order'][0] == 'pristine' for p in permutations), 5)
        self.assertEqual(len({p['permutation_sha256'] for p in permutations}), 10)
        for p in permutations:
            self.assertEqual(sorted(p['permutation']), list(range(1000)))
            self.assertEqual(p['permutation_sha256'], b.phase_c_json_hash(p['permutation']))
            self.assertEqual(sorted(p['probes_order']), list(b.PROBES))
            for system in b.PHASE_C_SYSTEMS:
                self.assertEqual([c['probes'] for c in schedule if c['round'] == p['round'] and c['system'] == system], p['probes_order'])
        self.assertEqual(permutations[1]['optimized_system_order'], ['phase2a', 'phase2b', 'final', 'vanilla_eq'])

    def test_formal_checkpoint_rejects_missing_duplicate_and_corrupt_rows(self):
        permutations, schedule = b.phase_c_schedule()
        config = schedule[0]
        manifest = {'query_permutations': permutations, 'pristine_commit': 'c', 'pristine_binary_sha256': 'h'}
        ids = ';'.join(map(str, range(10)))
        rows = [{**config, 'phase': 'C', 'experiment': 'phase_c', 'dataset': 'GIST1M',
                 'dimension': 960, 'metric': 'L2', 'source_commit': 'c', 'binary_sha256': 'h',
                 'lists': 1000, 'topk': 10, 'query_order_position': pos, 'query_id': qid,
                 'latency_us': 10, 'returned_rows': 10, 'result_ids': ids,
                 'result_checksum': b.hashlib.sha256(ids.encode()).hexdigest(), 'recall_at_10': 1.0}
                for pos, qid in enumerate(permutations[0]['permutation'])]
        summary = {**config, 'queries': 1000, 'warmup_queries': 100, 'p50_us': 10,
                   'p95_us': 10, 'p99_us': 10, 'mean_us': 10, 'wall_seconds': 1,
                   'qps': 1000, 'mean_recall_at_10': 1, 'minimum_returned_rows': 10}
        block = {'complete': True, 'config': config, 'manifest_sha256': b.phase_c_json_hash(manifest),
                 'rows': rows, 'rows_sha256': b.phase_c_json_hash(rows), 'summary': summary}
        gt = [set(range(10)) for _ in range(1000)]
        self.assertEqual(len(b.phase_c_validate_block(block, config, manifest, gt)), 1000)
        short = copy.deepcopy(block)
        short_ids = ';'.join(map(str, range(6)))
        short['rows'][0].update(result_ids=short_ids, returned_rows=6, recall_at_10=0.6,
            result_checksum=b.hashlib.sha256(short_ids.encode()).hexdigest())
        short['rows_sha256'] = b.phase_c_json_hash(short['rows'])
        short['summary']['minimum_returned_rows'] = 6
        self.assertEqual(len(b.phase_c_validate_block(short, config, manifest, gt)[0]['result_ids'].split(';')), 6)
        for change in ('missing', 'duplicate', 'checksum', 'manifest', 'completion', 'qps'):
            damaged = copy.deepcopy(block)
            if change == 'missing': damaged['rows'].pop()
            if change == 'duplicate': damaged['rows'][1] = damaged['rows'][0]
            if change == 'checksum': damaged['rows'][0]['result_checksum'] = 'wrong'
            if change == 'manifest': damaged['manifest_sha256'] = 'wrong'
            if change == 'completion': damaged['complete'] = False
            if change == 'qps': damaged['summary']['qps'] = 100000
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                b.phase_c_validate_block(damaged, config, manifest, gt)
