"""Repair-aware selection, classification and two-round final summaries."""
import csv
import json
import math
import statistics
from pathlib import Path

import ivfflat_profile as common


def config_key(item):
    return (int(item['round']), item['dataset'], item['method'])


def selected_maps(control):
    result = {}
    for round_name, label in [('round_a_schedule', 'USED_FINAL_ROUND_A'),
                              ('round_b_schedule', 'USED_FINAL_ROUND_B')]:
        for item in control[round_name]:
            result[config_key(item)] = (label, item['family'])
    return result


def disposition(item, control):
    match = selected_maps(control).get(config_key(item))
    if match and item['family'] == match[1]:
        return match[0]
    if item['dataset'] == 'sift' and item['method'] == 'baseline' and item['family'] == 'sift':
        return 'DUPLICATE_UNUSED'
    return 'HISTORICAL_UNUSED'


def load_checkpoints(root):
    blocks = {}
    for path in sorted((root / 'checkpoints').glob('r*.json')):
        block = json.loads(path.read_text())
        item = block['item']
        blocks[(int(item['round']), item['dataset'], item['method'], item['family'])] = (path, block)
    return blocks


def _available_values(*values):
    return [float(value) for value in values if value is not None]


def _paired_mean(value_a, value_b):
    values = _available_values(value_a, value_b)
    return statistics.fmean(values) if values else None


def _paired_std(value_a, value_b):
    values = _available_values(value_a, value_b)
    return statistics.stdev(values) if len(values) == 2 else None


def _summary_value(block, metric):
    if block is None:
        return None
    value = block.get('summary', {}).get(metric)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f'Non-finite summary metric {metric} in {block.get("item")}')
    return value


def _geomean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise RuntimeError('Geometric-mean speedup requires positive finite values')
    return math.exp(statistics.fmean(math.log(value) for value in values))


def update_progress(root, control, active=None, status='RUNNING'):
    root = Path(root)
    blocks = load_checkpoints(root)
    selected = selected_maps(control)
    completed_a = completed_b = historical = duplicate = 0
    completed_rows = 0
    for (_, dataset, method, family), (_, block) in blocks.items():
        r = int(block['item']['round'])
        label, selected_family = selected.get((r, dataset, method), (None, None))
        if label and family == selected_family:
            completed_a += label == 'USED_FINAL_ROUND_A'
            completed_b += label == 'USED_FINAL_ROUND_B'
            completed_rows += len(block['rows'])
        else:
            historical += 1
            if dataset == 'sift' and method == 'baseline' and family == 'sift':
                duplicate += 1
    running_rows = 0
    if active:
        item = active.get('item')
        label, family = selected.get(config_key(item), (None, None)) if item else (None, None)
        if item and label and item['family'] == family:
            running_rows = min(int(active.get('queries', 0)), int(active.get('total', 0)))
    planned_queries = 164000
    last_block = (max(blocks.values(), key=lambda value: value[0].stat().st_mtime)[1]
                  if blocks else None)
    progress = {
        'status': status,
        'effective_rounds': 2,
        'final_round_a': control['final_round_a'],
        'final_round_b': control['final_round_b'],
        'effective_total_config_rounds': 28,
        'final_round_a_completed_configs': completed_a,
        'final_round_a_total_configs': 14,
        'final_round_b_completed_configs': completed_b,
        'final_round_b_total_configs': 14,
        'completed_effective_config_rounds': completed_a + completed_b,
        'historical_completed_config_rounds': historical,
        'duplicate_unused_config_rounds': duplicate,
        'effective_formal_query_executions': planned_queries,
        'completed_effective_formal_queries': completed_rows,
        'inflight_effective_formal_queries': running_rows,
        'remaining_effective_formal_queries': max(0, planned_queries - completed_rows - running_rows),
        'original_round_a_order': control['round_a_order'],
        'original_round_b_order': control['round_b_order'],
        'last_completed_block': (None if last_block is None else
            f"{last_block['item']['round']}:{last_block['item']['dataset']}:{last_block['item']['method']}"),
        'next_unstarted_config': next(({'original_round_id':x['round'],'dataset':x['dataset'],'method':x['method']}
            for x in control['schedule'] if not (root/'checkpoints'/(f"r{x['round']}_{x['dataset']}_{x['family']}_{x['method']}.json")).exists()),None),
    }
    if active:
        progress['current_original_round'] = active.get('round')
        progress['current_config'] = active.get('block')
        progress['current_config_queries'] = active.get('queries', 0)
        progress['current_config_total'] = active.get('total', 0)
    common.write_json_atomic(root / 'progress.json', progress)
    return progress


def refresh_outputs(root, control, active=None, status='RUNNING'):
    root = Path(root)
    blocks = load_checkpoints(root)
    selected = selected_maps(control)
    disposition_rows = []
    raw = []
    summaries = []
    selected_blocks = {}
    for (_, dataset, method, family), (path, block) in sorted(blocks.items()):
        item = block['item']
        label, expected_family = selected.get(config_key(item), (None, None))
        state = label if label and family == expected_family else disposition(item, control)
        disposition_rows.append({'original_round_id': item['round'], 'dataset': dataset,
                                 'method': method, 'binary_family': family,
                                 'config_round': path.stem, 'status': state,
                                 'queries': len(block['rows']), 'checkpoint': str(path)})
        summaries.append({**block['summary'], 'original_round_id': item['round'], 'selection_status': state})
        for row in block['rows']:
            raw.append({**row, 'original_round_id': item['round'], 'selection_status': state,
                        'result_ids': json.dumps(row['result_ids'])})
        if label and family == expected_family:
            selected_blocks[(int(item['round']), dataset, method)] = block
    if active and active.get('block'):
        present = any(x['config_round'] == active['block'] for x in disposition_rows)
        if not present:
            item = active['item']
            disposition_rows.append({'original_round_id': item['round'], 'dataset': item['dataset'],
                                     'method': item['method'], 'binary_family': item['family'],
                                     'config_round': active['block'], 'status': 'INCOMPLETE',
                                     'queries': active.get('queries', 0), 'checkpoint': ''})
    round_a, round_b = int(control['final_round_a']), int(control['final_round_b'])
    metrics = ('recall_at_10','mean_us','p50_us','p95_us','p99_us','qps')
    rows = []
    missing = []
    for item_a in control['round_a_schedule']:
        item_b = next(x for x in control['round_b_schedule']
                      if x['dataset'] == item_a['dataset'] and x['method'] == item_a['method'])
        block_a = selected_blocks.get((round_a,item_a['dataset'],item_a['method']))
        block_b = selected_blocks.get((round_b,item_b['dataset'],item_b['method']))
        if not block_a: missing.append({'round':round_a,'dataset':item_a['dataset'],'method':item_a['method']})
        if not block_b: missing.append({'round':round_b,'dataset':item_b['dataset'],'method':item_b['method']})
        queries_a=_summary_value(block_a,'queries')
        queries_b=_summary_value(block_b,'queries')
        row={'dataset':item_a['dataset'],'method':item_a['method'],
             'round_a_id':round_a,'round_b_id':round_b,
             'round_a_family':item_a['family'],'round_b_family':item_b['family'],
             'round_a_queries':queries_a,'round_b_queries':queries_b,
             'queries_per_round':queries_a if queries_a is not None else queries_b}
        for metric in metrics:
            a,b=_summary_value(block_a,metric),_summary_value(block_b,metric)
            row['round_a_'+metric]=a;row['round_b_'+metric]=b
            row['mean_'+metric]=_paired_mean(a,b)
            row['std_'+metric]=_paired_std(a,b)
        def speedup(block,base_method):
            if block is None:
                return None
            base=selected_blocks.get((int(block['item']['round']),block['item']['dataset'],base_method))
            base_mean=_summary_value(base,'mean_us')
            block_mean=_summary_value(block,'mean_us')
            if base_mean is None or block_mean is None:return None
            if block_mean <= 0:raise RuntimeError(f'Non-positive latency in {block["item"]}')
            return base_mean/block_mean
        if item_a['method']=='2a':
            row['round_a_speedup']=speedup(block_a,'baseline');row['round_b_speedup']=speedup(block_b,'baseline')
        elif item_a['method']=='2a_b':
            row['round_a_speedup']=speedup(block_a,'2a');row['round_b_speedup']=speedup(block_b,'2a')
        elif item_a['method']=='d':
            row['round_a_speedup']=speedup(block_a,'baseline');row['round_b_speedup']=speedup(block_b,'baseline')
        else:
            row['round_a_speedup']=row['round_b_speedup']=None
        row['mean_speedup']=_paired_mean(row['round_a_speedup'],row['round_b_speedup'])
        row['std_speedup']=_paired_std(row['round_a_speedup'],row['round_b_speedup'])
        if item_a['method']=='d':
            baseline_a=selected_blocks.get((round_a,item_a['dataset'],'baseline'))
            baseline_b=selected_blocks.get((round_b,item_b['dataset'],'baseline'))
            recall_a=_summary_value(block_a,'recall_at_10')
            recall_b=_summary_value(block_b,'recall_at_10')
            baseline_recall_a=_summary_value(baseline_a,'recall_at_10')
            baseline_recall_b=_summary_value(baseline_b,'recall_at_10')
            delta_a=(recall_a-baseline_recall_a if recall_a is not None and baseline_recall_a is not None else None)
            delta_b=(recall_b-baseline_recall_b if recall_b is not None and baseline_recall_b is not None else None)
        else:
            delta_a=delta_b=None
        row['round_a_recall_delta']=delta_a
        row['round_b_recall_delta']=delta_b
        row['mean_recall_delta']=_paired_mean(delta_a,delta_b)
        row['std_recall_delta']=_paired_std(delta_a,delta_b)
        rows.append(row)

    complete = not missing and len(rows)==14 and len(selected_blocks)==28
    if status == 'COMPLETE' and not complete:
        raise RuntimeError(f'Cannot finalize two-round comparison; {len(missing)} selected config-rounds are missing')
    if complete:
        required_speedup_counts={'2a':5,'2a_b':2,'d':2}
        for method,expected_count in required_speedup_counts.items():
            for round_label in ('a','b'):
                values=[row[f'round_{round_label}_speedup'] for row in rows if row['method']==method]
                if len(values)!=expected_count or any(value is None for value in values):
                    raise RuntimeError(f'Cannot finalize: incomplete {round_label.upper()} speedups for {method}')
        for row in rows:
            if row['round_a_queries'] is None or row['round_b_queries'] is None or row['round_a_queries']!=row['round_b_queries']:
                raise RuntimeError(f'Cannot finalize: missing or mismatched round query counts for {row["dataset"]}/{row["method"]}')
            for metric in metrics:
                if row['round_a_'+metric] is None or row['round_b_'+metric] is None:
                    raise RuntimeError(f'Cannot finalize: missing two-round {metric} for {row["dataset"]}/{row["method"]}')
            if row['method']=='d' and (row['round_a_recall_delta'] is None or row['round_b_recall_delta'] is None):
                raise RuntimeError(f'Cannot finalize: missing two-round Recall delta for {row["dataset"]}/d')

    geomeans={}
    for method in ('2a','2a_b','d'):
        round_a_values=[row['round_a_speedup'] for row in rows if row['method']==method]
        round_b_values=[row['round_b_speedup'] for row in rows if row['method']==method]
        geo_a=_geomean(round_a_values)
        geo_b=_geomean(round_b_values)
        geomeans[method]={'round_a':geo_a,'round_b':geo_b,
                          'mean':_paired_mean(geo_a,geo_b),'std':_paired_std(geo_a,geo_b)}

    raw_fields = list(raw[0]) if raw else ['dataset','method','family','round','position','qid','latency_us',
                                           'recall_at_10','result_ids','result_checksum','binary_sha256',
                                           'original_round_id','selection_status']
    summary_fields = list(summaries[0]) if summaries else ['dataset','method','family','round','queries','mean_us',
                                                            'p50_us','p95_us','p99_us','qps','recall_at_10',
                                                            'avg_probes','original_round_id','selection_status']
    if raw:
        common.write_csv_atomic(root / 'raw.csv', raw, raw_fields)
    if summaries:
        common.write_csv_atomic(root / 'summary.csv', summaries, summary_fields)
    common.write_csv_atomic(root / 'result_disposition.csv', disposition_rows,
                            ('original_round_id','dataset','method','binary_family','config_round',
                             'status','queries','checkpoint'))
    comparison_fields=(tuple(rows[0]) if rows else
        ('dataset','method','round_a_id','round_b_id','round_a_family','round_b_family',
         'round_a_queries','round_b_queries','queries_per_round',
         *tuple(k+'_'+m for m in metrics for k in ('round_a','round_b','mean','std')),
         'round_a_speedup','round_b_speedup','mean_speedup','std_speedup',
         'round_a_recall_delta','round_b_recall_delta','mean_recall_delta','std_recall_delta'))
    common.write_csv_atomic(root/'final_comparison.csv',rows,comparison_fields)
    fields={'effective_rounds':2,'final_round_a':round_a,'final_round_b':round_b,
            'final_round_a_config_count':14,'final_round_b_config_count':14,
            'effective_query_executions_per_round':82000,
            'effective_formal_query_executions':164000,
            'completed_config_rounds':len(selected_blocks),'missing_configs':missing,
            'status':'COMPLETE' if complete else status,
            'aggregation':'A and B only; mean and sample standard deviation across the two round summaries; no best-round selection',
            'geomean_speedups':geomeans,
            'final_comparison':rows}
    common.write_json_atomic(root/'final_comparison.json',fields)
    progress=update_progress(root,control,active,status='COMPLETE' if complete else status)
    if complete:
        manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest.update({'effective_rounds':2,'final_round_a':round_a,'final_round_b':round_b,
                         'effective_total_config_rounds':28,'effective_formal_query_executions':164000,
                         'status':'COMPLETE'})
        common.write_json_atomic(manifest_path,manifest)
    return progress
