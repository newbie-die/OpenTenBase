#!/usr/bin/env python3
"""D2 feature collection and one-round fixed-64 versus adaptive validation."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

ROOT = Path('/workspace')
DATASET = ROOT / 'benchmark/data/gist1m/gist-960-euclidean.hdf5'
TRACE_RE = re.compile(r'IVFFLAT_ADAPTIVE\s+(.*)')
PROFILE_RE = re.compile(r'IVFFLAT_PROFILE\s+(.*)')
FIELD_RE = re.compile(r'([a-z0-9_]+)=(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)')


def save_json(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def percentile(values, q):
    values = sorted(values)
    return values[math.ceil(q / 100 * len(values)) - 1]


def vector_literal(value):
    return '[' + ','.join(str(float(x)) for x in value) + ']'


def parse_notice(notices, expression, required):
    for notice in reversed(notices):
        match = expression.search(notice)
        if match:
            fields = {key: float(value) for key, value in FIELD_RE.findall(match.group(1))}
            if not required <= set(fields):
                raise RuntimeError(f'incomplete profile notice: {notice}')
            return fields
    raise RuntimeError(f'missing {expression.pattern}; install the D2 build and use IVFFLAT_PROFILE_CFLAGS=-DIVFFLAT_BENCH for validation')


def verify_setting(cur, name, value):
    cur.execute('SELECT setting FROM pg_settings WHERE name=%s', (name,))
    if cur.fetchone() is None:
        raise RuntimeError(f'missing registered GUC: {name}; LOAD the D2 vector.so')
    cur.execute('SELECT set_config(%s,%s,false)', (name, str(value)))
    return cur.fetchone()[0]


def setup(cur, probes, adaptive, trace, thresholds=None):
    cur.execute("LOAD 'vector'")
    for name, value in {
        'ivfflat.probes': probes,
        'ivfflat.iterative_scan': 'off',
        'ivfflat.adaptive_probes': 'on' if adaptive else 'off',
        'ivfflat.adaptive_probes_trace': 'on' if trace else 'off',
    }.items():
        verify_setting(cur, name, value)
    if thresholds:
        for suffix, value in thresholds.items():
            verify_setting(cur, 'ivfflat.adaptive_probes_' + suffix, format(value, '.17g'))
    # D2 deliberately avoids the D1 distance paths and bounded-scan experiment.
    cur.execute("SELECT setting FROM pg_settings WHERE name='ivfflat.distance_path'")
    if cur.fetchone() is not None:
        verify_setting(cur, 'ivfflat.distance_path', 'generic')
    cur.execute("SELECT setting FROM pg_settings WHERE name='ivfflat.bounded_scan'")
    if cur.fetchone() is not None:
        verify_setting(cur, 'ivfflat.bounded_scan', 'off')
    for sql in ('SET enable_seqscan=off', 'SET enable_indexscan=on',
                'SET max_parallel_workers_per_gather=0'):
        cur.execute(sql)


def load_workload(count):
    import h5py
    import numpy as np
    with h5py.File(DATASET, 'r') as source:
        queries = np.asarray(source['test'][:count], dtype=np.float32)
        truth = [set(int(value) for value in row) for row in source['neighbors'][:count, :10]]
    return queries, truth


def query(cur, conn, literal):
    conn.notices.clear()
    started = time.perf_counter_ns()
    cur.execute('SELECT id FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10', (literal,))
    ids = [int(row[0]) for row in cur.fetchall()]
    return ids, (time.perf_counter_ns() - started) / 1000.0, list(conn.notices)


def progress(path, config, configs, done, expected, probes, recalls, started):
    elapsed = time.perf_counter() - started
    average = statistics.fmean(probes) if probes else 0.0
    recall = statistics.fmean(recalls) if recalls else 0.0
    eta = elapsed * (expected - done) / done if done else None
    value = {'status': 'RUNNING', 'config': f'{config}/{configs}', 'queries': f'{done}/{expected}',
             'average_probes_so_far': average, 'recall_so_far': recall,
             'elapsed_seconds': elapsed, 'eta_seconds': eta}
    save_json(path, value)
    print(' '.join(f'{key}={value[key]}' for key in ('config', 'queries', 'average_probes_so_far',
          'recall_so_far', 'elapsed_seconds', 'eta_seconds')), flush=True)


def require_new_output(path):
    if path.exists():
        raise RuntimeError(f'output must be a new directory: {path}')
    path.mkdir(parents=True)


def run_features(args):
    import psycopg2
    require_new_output(args.output)
    queries, truth = load_workload(args.queries)
    literals = [vector_literal(query) for query in queries]
    rows, probes, recalls = [], [], []
    started = time.perf_counter()
    conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname, user=args.user,
                            application_name='d2_feature_collection')
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            setup(cur, 128, False, True)
            for qid, literal in enumerate(literals):
                ids, latency_us, notices = query(cur, conn, literal)
                feature = parse_notice(notices, TRACE_RE, {'d1', 'd2_d1', 'd4_d1', 'd8_d1', 'd16_d1', 'probes'})
                recall = len(set(ids) & truth[qid]) / 10
                row = {'query_id': qid, 'd1': feature['d1'], 'd2_d1': feature['d2_d1'],
                       'd4_d1': feature['d4_d1'], 'd8_d1': feature['d8_d1'],
                       'd16_d1': feature['d16_d1'], 'gap': feature['gap'],
                       'selected_probes': int(feature['probes']), 'recall_at_10': recall,
                       'latency_us': latency_us}
                rows.append(row); probes.append(row['selected_probes']); recalls.append(recall)
                progress(args.output / 'progress.json', 1, 1, qid + 1, args.queries, probes, recalls, started)
    finally:
        conn.close()
    with (args.output / 'features.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    save_json(args.output / 'summary.json', {'status': 'COMPLETE', 'mode': 'features', 'queries': args.queries,
              'average_probes': statistics.fmean(probes), 'recall_at_10': statistics.fmean(recalls),
              'elapsed_seconds': time.perf_counter() - started})
    save_json(args.output / 'progress.json', {'status': 'COMPLETE', 'config': '1/1',
              'queries': f'{args.queries}/{args.queries}', 'average_probes_so_far': statistics.fmean(probes),
              'recall_so_far': statistics.fmean(recalls), 'elapsed_seconds': time.perf_counter() - started,
              'eta_seconds': 0})


def read_rule(path):
    value = json.loads(path.read_text())
    value = value.get('heuristic', value)
    thresholds = value['thresholds']
    result = {'ratio_16': float(thresholds['ratio_16']), 'ratio_32': float(thresholds['ratio_32']),
              'ratio_64': float(thresholds['ratio_64'])}
    if not all(math.isfinite(value) and value >= 1 for value in result.values()):
        raise ValueError('rule thresholds must be finite and >= 1')
    if not result['ratio_16'] >= result['ratio_32'] >= result['ratio_64']:
        raise ValueError('thresholds must satisfy ratio_16 >= ratio_32 >= ratio_64')
    return result


def run_mode(cur, conn, output, config_no, config_count, mode, queries, truth, order, warmup, thresholds):
    adaptive = mode == 'adaptive'
    max_probes = 128 if adaptive else 64
    setup(cur, max_probes, adaptive, False, thresholds if adaptive else None)
    for qid in order[:warmup]:
        query(cur, conn, vector_literal(queries[qid]))
    rows, probes, recalls, latencies, candidates = [], [], [], [], []
    started = time.perf_counter()
    for position, qid in enumerate(order, 1):
        ids, latency_us, notices = query(cur, conn, vector_literal(queries[qid]))
        profile = parse_notice(notices, PROFILE_RE, {'probes', 'candidates'})
        probe = int(profile['probes'])
        recall = len(set(ids) & truth[qid]) / 10
        rows.append({'mode': mode, 'query_order_position': position, 'query_id': qid,
                     'selected_probes': probe, 'scanned_candidates': int(profile['candidates']),
                     'recall_at_10': recall, 'latency_us': latency_us})
        probes.append(probe); recalls.append(recall); latencies.append(latency_us); candidates.append(int(profile['candidates']))
        progress(output / 'progress.json', config_no, config_count, position, len(order), probes, recalls, started)
    elapsed = time.perf_counter() - started
    return rows, {'mode': mode, 'queries': len(rows), 'recall_at_10': statistics.fmean(recalls),
                  'p50_latency_us': percentile(latencies, 50), 'p95_latency_us': percentile(latencies, 95),
                  'qps': len(rows) / elapsed, 'average_probes': statistics.fmean(probes),
                  'probe_distribution': {str(p): probes.count(p) for p in (16, 32, 64, 128)},
                  'average_scanned_candidates': statistics.fmean(candidates), 'elapsed_seconds': elapsed}


def run_validation(args):
    import psycopg2
    require_new_output(args.output)
    thresholds = read_rule(args.rule)
    queries, truth = load_workload(args.queries)
    order = list(range(args.queries))
    import random
    random.Random(args.seed).shuffle(order)
    conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname, user=args.user,
                            application_name='d2_one_round_validation')
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            fixed_rows, fixed = run_mode(cur, conn, args.output, 1, 2, 'fixed_64', queries, truth, order,
                                         args.warmup, thresholds)
            adaptive_rows, adaptive = run_mode(cur, conn, args.output, 2, 2, 'adaptive', queries, truth, order,
                                               args.warmup, thresholds)
    finally:
        conn.close()
    rows = fixed_rows + adaptive_rows
    with (args.output / 'raw.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    recall_delta = adaptive['recall_at_10'] - fixed['recall_at_10']
    probe_reduction = 1.0 - adaptive['average_probes'] / fixed['average_probes']
    p50_improvement = 1.0 - adaptive['p50_latency_us'] / fixed['p50_latency_us']
    if recall_delta >= -0.005 and probe_reduction >= 0.20 and p50_improvement >= 0.10:
        decision = 'Go'
    elif recall_delta >= -0.005 and probe_reduction >= 0.20:
        decision = 'Marginal'
    else:
        decision = 'No-Go'
    summary = {'status': 'COMPLETE', 'rounds': 1, 'rule_thresholds': thresholds,
               'fixed_probes_64': fixed, 'adaptive_probes': adaptive,
               'recall_delta': recall_delta, 'probe_reduction': probe_reduction,
               'p50_latency_improvement': p50_improvement, 'decision': decision,
               'requirements': {'recall_delta_min': -0.005, 'probe_reduction_min': 0.20,
                                'p50_latency_improvement_for_go': 0.10}}
    save_json(args.output / 'summary.json', summary)
    save_json(args.output / 'progress.json', {'status': 'COMPLETE', 'config': '2/2', 'queries': f'{args.queries}/{args.queries}',
              'average_probes_so_far': adaptive['average_probes'], 'recall_so_far': adaptive['recall_at_10'],
              'elapsed_seconds': adaptive['elapsed_seconds'], 'eta_seconds': 0, 'decision': decision})
    print(json.dumps({'decision': decision, 'recall_delta': recall_delta,
                      'probe_reduction': probe_reduction, 'p50_latency_improvement': p50_improvement}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('features', 'validate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rule', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--dbname', default='taskdb')
    parser.add_argument('--user', default='dev')
    parser.add_argument('--queries', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260911)
    args = parser.parse_args()
    if args.mode == 'features':
        if not 1 <= args.queries <= 1000:
            raise ValueError('--queries must be 1..1000')
        run_features(args)
    else:
        if not args.rule:
            raise ValueError('validate requires --rule')
        if not 1 <= args.queries <= 1000:
            raise ValueError('--queries must be 1..1000')
        run_validation(args)

if __name__ == '__main__':
    main()
