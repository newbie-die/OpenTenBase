"""Regression checks for path selection, accounting, and shell dispatch (no DB)."""
import importlib.util
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('profile', ROOT / 'benchmark/scripts/ivfflat_profile.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def row(path, eligible=True):
    r = dict.fromkeys(b.PHASE2B_WORKLOAD_FIELDS + b.FUSED2_COUNTERS, 0)
    active = eligible and path != 'generic'
    fused = active and path == 'fused2'
    r.update(query_id=0, selected_lists=1, scanned_pages=2, scanned_candidates=3,
             distance_calls=3, tuplesort_input_calls=3, returned_rows=2,
             page_candidates_1=1, page_candidates_2=1, max_candidates_per_page=2,
             direct_l2_eligible=int(eligible), direct_l2_active=int(active), fused2_active=int(fused),
             generic_distance_calls=0 if active else 3,
             direct_distance_calls=(1 if fused else 3) if active else 0,
             generic_distance_ns=0 if active else 30,
             direct_distance_ns=(5 if fused else 30) if active else 0,
             fused_distance_ns=10 if fused else 0, distance_ns=15 if fused else 30,
             scan_items_total_ns=100, candidate_extract_ns=10, tuple_materialization_ns=10,
             sort_insert_ns=10, sort_finalize_ns=10,
             result_tids='(1,1);(1,2)', result_ids='1;2', result_distances='1.0;2.0',
             result_distance_bits='3ff0000000000000;4000000000000000')
    if fused:
        r.update(fused_pair_calls=1, fused_candidates=2, single_tail_candidates=1)
    return r


class AccountingTests(unittest.TestCase):
    def test_both_comparisons_and_unsupported(self):
        for a, c in [('generic', 'direct'), ('direct', 'fused2')]:
            for eligible in [True, False]:
                summary = b.compare_phase2b_paths([row(a, eligible)], [row(c, eligible)],
                                                  'full', a, c, int(eligible))
                b.check_phase2b_comparison(summary)

    def test_coverage_includes_tail(self):
        r = row('fused2')
        summary = b.summarize_phase2b_path([r], 'full', 64, 1, 'fused2', 2)
        self.assertEqual(summary['distance_ns_per_candidate'], 5)
        self.assertAlmostEqual(summary['fused_coverage_pct'], 200 / 3)
        b.verify_fused2_tail([r])

    def test_bad_accounting_is_rejected(self):
        for key in ['fused_pair_calls', 'single_tail_candidates', 'distance_calls',
                    'tuplesort_input_calls', 'direct_distance_calls', 'fused_distance_ns']:
            r = row('fused2'); r[key] += 1
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                b.validate_phase2b_dispatch([r], 'fused2')

    def test_workload_and_bit_changes_are_rejected(self):
        for key in ['physical_bound', 'bounded_active', 'fallback_triggered', 'result_distance_bits']:
            r = row('fused2')
            r[key] = 'changed' if isinstance(r[key], str) else r[key] + 1
            summary = b.compare_phase2b_paths([row('direct')], [r], 'full', 'direct', 'fused2')
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                b.check_phase2b_comparison(summary)

    def test_duplicate_query_ids_rejected(self):
        with self.assertRaises(RuntimeError):
            b.compare_phase2b_paths([row('direct')] * 2, [row('fused2')] * 2,
                                   'full', 'direct', 'fused2')

    def test_legacy_profile_columns(self):
        a = b.summarize_phase2b_path([row('generic')], 'full', 64, 1, 'generic', 1)
        c = b.summarize_phase2b_path([row('direct')], 'full', 64, 1, 'direct', 2)
        pair = b.compare_phase2b_profile_pair(a, c)
        self.assertEqual(pair['generic_ns_per_candidate'], pair['baseline_ns_per_candidate'])
        summary = b.summarize_phase2b_pairs([pair])[0]
        self.assertEqual(summary['direct_ns_per_candidate_median'], 10)


    def test_production_schedule_rotates_and_interleaves(self):
        schedule = b.phase2b_production_schedule((16, 64, 128), 4)
        self.assertEqual(len(schedule), 48)
        for round_no, expected_probes in enumerate(b.PRODUCTION_PROBE_ORDERS, start=1):
            rows = [entry for entry in schedule if entry['round'] == round_no]
            actual_probes = []
            for entry in rows:
                if entry['probes'] not in actual_probes:
                    actual_probes.append(entry['probes'])
            self.assertEqual(tuple(actual_probes), expected_probes)
            expected_paths = ('direct', 'fused2') if round_no % 2 else ('fused2', 'direct')
            for position in range(0, len(rows), 2):
                self.assertEqual(tuple(entry['distance_path']
                                       for entry in rows[position:position + 2]),
                                 expected_paths)

    def test_production_pair_checks_results_and_statistics(self):
        raw = []
        runs = []
        for distance_path, order, latencies, ids in [
                ('direct', 1, (100.0, 200.0), ('1;2', '3;4')),
                ('fused2', 2, (80.0, 160.0), ('1;2', '3;4'))]:
            path_rows = []
            for query_id, (latency, result_ids) in enumerate(zip(latencies, ids)):
                item = {
                    'mode': 'full', 'probes': 64, 'round': 1,
                    'distance_path': distance_path, 'query_id': query_id,
                    'latency_us': latency, 'returned_rows': 2,
                    'result_checksum': result_ids,
                }
                path_rows.append(item)
                raw.append(item)
            runs.append(b.summarize_phase2b_production_run(
                path_rows, 'full', 64, 1, distance_path, 1, order))
        pair = b.compare_phase2b_production_pair(runs[0], runs[1], raw)
        self.assertEqual(pair['result_mismatch_queries'], 0)
        self.assertAlmostEqual(pair['paired_p50_improvement_pct'], 20.0)
        summary = b.summarize_phase2b_production([pair])
        self.assertEqual(summary[0]['paired_p50_positive_rounds'], 1)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['bin', 'run', 'pgdata']:
            (self.root / name).mkdir()
        self.log = self.root / 'calls.jsonl'
        mock = self.root / 'bin/mock'
        mock.write_text('#!' + sys.executable + '\n' + '''import json, os, sys, subprocess
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['MOCK_CALLS'], 'a') as f:
    f.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'python' and '--validate-only' in sys.argv:
    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))
if name == 'make':
    print('gcc -O2 -g -march=haswell -mtune=haswell -mavx2 -mfma -DIVFFLAT_FUSED2 -c -o src/vector.o src/vector.c')
else:
    print('mock ' + name)
''')
        mock.chmod(0o755)
        for name in ['python', 'make', 'pg_config', 'pg_ctl', 'objdump', 'strings']:
            (self.root / 'bin' / name).symlink_to(mock)
        self.vector_so = self.root / 'vector.so'
        self.vector_so.write_bytes(b'mock vector')
        self.env = dict(os.environ, PATH=str(self.root / 'bin') + ':' + os.environ['PATH'],
                        PYTHON_BIN=str(self.root / 'bin/python'),
                        PG_CONFIG=str(self.root / 'bin/pg_config'), PG_CTL=str(self.root / 'bin/pg_ctl'),
                        PGDATA=str(self.root / 'pgdata'), PG_OS_USER=pwd.getpwuid(os.getuid()).pw_name,
                        RUN_DIR=str(self.root / 'run'), PID_FILE=str(self.root / 'pid'),
                        MOCK_CALLS=str(self.log), INSTALL_DEPS='0',
                        PGVECTOR_INSTALLED_SO=str(self.vector_so))

    def run_worker(self, *args):
        return subprocess.run(['bash', str(ROOT / 'benchmark/scripts/run_ivfflat_experiments_worker.sh'), *args],
                              env=self.env, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_fused2_cli_forwarding_and_flags(self):
        r = self.run_worker('2b-correctness', '--baseline-path', 'direct', '--test-path', 'fused2',
                            '--check-fused2-edges', '--queries', '3', '--warmup', '0')
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.calls()
        validation = next(i for i,c in enumerate(calls) if '--validate-only' in c)
        build = next(i for i,c in enumerate(calls) if c[0] == 'make')
        self.assertLess(validation, build)
        self.assertIn('OPTFLAGS=-march=haswell -mtune=haswell -mavx2 -mfma', calls[build])
        final = [c for c in calls if c[0] == 'python' and 'run' in c][-1]
        self.assertTrue(any('phase_2b_correctness' in value for value in final))
        self.assertEqual(final[-9:], ['--baseline-path', 'direct', '--test-path', 'fused2',
                                      '--check-fused2-edges', '--queries', '3', '--warmup', '0'])
        self.assertEqual((self.root / 'run/status').read_text().strip(), 'complete')

    def test_launcher_forwards_correctness_options(self):
        self.env['RUN_ROOT'] = str(self.root / 'launched')
        r = subprocess.run(['bash', str(ROOT / 'benchmark/scripts/run_ivfflat_experiments.sh'),
                            '2b-correctness', '--baseline-path', 'direct', '--test-path', 'fused2',
                            '--queries', '3', '--warmup', '0'],
                           env=self.env, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        status = None
        for _ in range(100):
            paths = list((self.root / 'launched').glob('*/status'))
            if paths:
                status = paths[0].read_text().strip()
                if status != 'running':
                    break
            time.sleep(0.05)
        self.assertEqual(status, 'complete')
        final = [c for c in self.calls() if c[0] == 'python' and 'run' in c][-1]
        self.assertEqual(final[-8:], ['--baseline-path', 'direct', '--test-path', 'fused2',
                                      '--queries', '3', '--warmup', '0'])


    def test_profile_cli_paths_select_fused2_output_name(self):
        r = self.run_worker('2b', '--distance-path', 'interleaved',
                            '--baseline-path', 'direct', '--test-path', 'fused2',
                            '--queries', '3', '--warmup', '0', '--rounds', '1')
        self.assertEqual(r.returncode, 0, r.stderr)
        final = [c for c in self.calls() if c[0] == 'python' and 'run' in c][-1]
        output = final[final.index('--output') + 1]
        self.assertTrue(output.endswith('/phase_2b_fused2'), output)

    def test_production_build_has_switch_without_profiling(self):
        r = self.run_worker('2b-production')
        self.assertEqual(r.returncode, 0, r.stderr)
        compile_command = (self.root / 'run/compile_command.txt').read_text()
        self.assertIn('-DIVFFLAT_FUSED2', compile_command)
        self.assertNotIn('-DIVFFLAT_BENCH', compile_command)
        self.assertNotIn('-DIVFFLAT_PROFILE_2B', compile_command)
        final = [call for call in self.calls()
                 if call[0] == 'python' and 'run' in call][-1]
        self.assertEqual(final[final.index('--phase') + 1], '2b-production')
        self.assertTrue(final[final.index('--output') + 1].endswith(
            '/phase_2b_fused2_production'))

    def test_invalid_path_rejected_before_build(self):
        r = self.run_worker('2b-correctness', '--test-path', 'invalid')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(any(c[0] == 'make' for c in self.calls()))

    def test_formal_resume_still_skips_build(self):
        r = self.run_worker('formal', '--resume', '--dataset', 'gist-l2')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(any(c[0] in ('make', 'pg_ctl') for c in self.calls()))
        final = [c for c in self.calls() if c[0] == 'python' and 'run' in c][-1]
        self.assertEqual(final[-3:], ['--resume', '--dataset', 'gist-l2'])


if __name__ == '__main__':
    unittest.main()
