"""Dataset-parametric extension of the Phase C runner; prepare/smoke by default.

Never rebuilds indexes or edits algorithm sources in the working tree. D policy
compilation uses an archived source copy, with the original calibration features.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import ivfflat_profile as common

BENCH = common.SCRIPT_ROOT
CONFIG_PATH = BENCH / 'configs/final_multi_datasets.json'
D2P = BENCH / 'd2p'
INSTALL_ROOT = Path(os.environ.get('OPENTENBASE_INSTALL_ROOT', common.WORKSPACE_ROOT / 'install')).resolve()
INSTALLED_VECTOR = INSTALL_ROOT / 'lib/postgresql/vector.so'
POSTGRES_BINARY = INSTALL_ROOT / 'bin/postgres'
sys.path.insert(0, str(D2P))
import calibrate_policy as calibration
import collect_progressive_shadow as shadow_parser

sha = common.phase_c_hash
jhash = common.phase_c_json_hash
atomic = common.write_json_atomic
FLAGS = common.PHASE_C_FLAGS


def load_dataset_configs():
    configs = json.loads(CONFIG_PATH.read_text())
    for config in configs.values():
        data_path = Path(config['data_path'])
        config['data_path'] = str(data_path if data_path.is_absolute() else common.ROOT / data_path)
        if config['format'] == 'parquet':
            for key in ('query_source', 'gt'):
                path = Path(config[key])
                config[key] = str(path if path.is_absolute() else common.ROOT / path)
        if 'policy' in config:
            path = Path(config['policy'])
            config['policy'] = str(path if path.is_absolute() else BENCH / path)
    return configs


DATASETS = load_dataset_configs()


def cmd(argv, **kwargs):
    return common.phase_c_command(argv, **kwargs)


def load(config, calibration_only=False):
    import numpy as np
    if config['format'] == 'hdf5':
        import h5py
        with h5py.File(config['data_path']) as f:
            start = 0 if calibration_only else config['formal_start']
            n = 1000 if calibration_only else config['formal_count']
            q = np.asarray(f['test'][start:start+n], dtype=np.float32)
            gt = np.asarray(f['neighbors'][start:start+n, :10], dtype=np.int64)
            qids = list(range(start, start+n))
    else:
        import pyarrow.parquet as pq
        test = pq.read_table(config['query_source']).to_pydict()
        neighbors = pq.read_table(config['gt']).to_pydict()
        query_map = dict(zip(test['id'], test['emb']))
        truth_map = dict(zip(neighbors['id'], neighbors['neighbors_id']))
        if len(query_map) != len(test['id']) or set(query_map) != set(truth_map):
            raise RuntimeError('Duplicate or mismatched Cohere query/GT IDs')
        qids = sorted(query_map)
        q = np.asarray([query_map[i] for i in qids], dtype=np.float32)
        gt = np.asarray([truth_map[i][:10] for i in qids], dtype=np.int64)
    if config['normalize']:
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.where(norms == 0, 1, norms)
    if q.shape != (len(qids), config['dimension']) or gt.shape != (len(qids), 10):
        raise RuntimeError('Query/GT dimensions mismatch')
    if not np.isfinite(q).all() or (gt < 0).any() or (gt >= config['rows']).any() or any(len(set(row)) != 10 for row in gt.tolist()):
        raise RuntimeError('Nonfinite query or duplicate GT IDs')
    return qids, q, gt


def index_identity(cur, config):
    cur.execute('SELECT c.oid,c.relfilenode,pg_relation_size(c.oid),pg_get_indexdef(c.oid),'
                'i.indisvalid,o.opcname,t.relname,c.reloptions FROM pg_class c '
                'JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_class t ON t.oid=i.indrelid '
                'JOIN pg_opclass o ON o.oid=i.indclass[0] WHERE c.oid=to_regclass(%s)',
                (config['index'],))
    row = cur.fetchone()
    if not row or not row[4] or row[5] != config['opclass'] or row[6] != config['table'] or 'lists=1000' not in row[7]:
        raise RuntimeError('Missing/wrong index: '+config['index'])
    return dict(zip(('oid','relfilenode','bytes','definition','valid','opclass','table','options'),row))


def sql(config):
    return f"SELECT id FROM {config['table']} ORDER BY embedding {config['operator']} %s::vector LIMIT 10"


def settings(cur, method, probes=64, debug=False):
    if method not in ('baseline','2a','2a_b','d','shadow'):
        raise ValueError(method)
    system = {'baseline':'vanilla_eq','2a':'phase2a','2a_b':'final','d':'vanilla_eq','shadow':'vanilla_eq'}[method]
    result = common.phase_c_settings(cur, system, probes)
    extra = {'ivfflat.adaptive_probes':'off','ivfflat.adaptive_probes_trace':'on' if method=='shadow' else 'off',
             'ivfflat.progressive_scan':'on' if method=='d' else ('shadow' if method=='shadow' else 'off'),
             'ivfflat.progressive_scan_debug':'on' if debug else 'off'}
    for name,value in extra.items():
        cur.execute('SELECT setting FROM pg_settings WHERE name=%s',(name,))
        if cur.fetchone() is None: raise RuntimeError('Unregistered GUC '+name)
        cur.execute('SELECT set_config(%s,%s,false)',(name,value))
        if cur.fetchone()[0] != value: raise RuntimeError('GUC mismatch')
    result.update(extra)
    cur.execute("SET statement_timeout='5min'")
    return result


def production_check(path):
    strings = cmd(['strings',path])
    if 'IVFFLAT_PROFILE' in strings: raise RuntimeError('Profiling in production binary')
    if re.search(r'\b(clock_gettime|gettimeofday)\b',cmd(['nm','-u',path])):
        raise RuntimeError('Timer dependency in production binary')


def build(root, family='production', policy_dir=None, profiled=False):
    target = root/'binaries'/family
    target.mkdir(parents=True,exist_ok=True)
    source = root/'build'/family
    if not source.exists():
        shutil.copytree(BENCH.parent/'contrib/pgvector',source,
                        ignore=shutil.ignore_patterns('*.o','*.so','*.bc','__pycache__'))
    if policy_dir is not None:
        compile_policy(source, policy_dir)
    flags = '-DIVFFLAT_FUSED2' + (' -DIVFFLAT_BENCH -DIVFFLAT_PROFILE_2B' if profiled else '')
    command = ['make','-C',str(source),'-B','-j8','CC=gcc',
               'PG_CONFIG='+str(INSTALL_ROOT/'bin/pg_config'),'OPTFLAGS='+FLAGS,'IVFFLAT_PROFILE_CFLAGS='+flags]
    env = dict(os.environ)
    for name in ('PG_CFLAGS','CFLAGS','CPPFLAGS','IVFFLAT_PROFILE_CFLAGS','MAKEFLAGS'): env.pop(name,None)
    with (target/'compile.log').open('w') as f:
        f.write(json.dumps(command)+'\n'); f.flush()
        subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(source/'vector.so',target/'vector.so')
    if not profiled: production_check(target/'vector.so')
    metadata = {'sha256':sha(target/'vector.so'),'compiler':cmd(['gcc','--version']),
                'argv':command,'profiling':profiled,'policy_sha256':sha(policy_dir/'d2p_policy.json') if policy_dir else sha(D2P/'artifacts/d2p_policy.json'),
                'source_hashes':{str(p.relative_to(source)):sha(p) for p in sorted((source/'src').glob('*')) if p.suffix in ('.h','.c')}}
    atomic(target/'build.json',metadata)
    return metadata


def activate(args,root,family):
    info = json.loads((root/'binaries'/family/'build.json').read_text())
    meta = {family+'_binary_sha256':info['sha256'],
            'installed_so':str(INSTALLED_VECTOR)}
    conn = common.phase_c_activate(args,root,meta,family)
    return conn


def validate(args,root,conn):
    import numpy as np
    import h5py
    result = {}
    with conn.cursor() as cur:
        cur.execute("LOAD 'vector'")
        settings(cur,'baseline')
        for name,config in DATASETS.items():
            qids,q,gt = load(config)
            identity = index_identity(cur,config)
            cur.execute(f"SELECT count(*),count(DISTINCT id),min(id),max(id) FROM {config['table']}")
            counts = list(cur.fetchone())
            if counts != [config['rows'],config['rows'],0,config['rows']-1]:
                raise RuntimeError('Unexpected base IDs/counts: '+name)
            cur.execute('EXPLAIN (FORMAT JSON) '+sql(config),(common.vector_literal(q[0]),))
            plan = cur.fetchone()[0]
            if config['index'] not in json.dumps(plan): raise RuntimeError('Wrong query index '+name)
            paths = [config['data_path']]
            if config['format']=='parquet': paths += [config['query_source'],config['gt']]
            hashes = {p:sha(p) for p in paths}
            samples = []
            if config['format']=='hdf5':
                with h5py.File(config['data_path']) as f:
                    for i in (0,config['rows']//2,config['rows']-1):
                        v = np.asarray(f['train'][i],dtype=np.float32)
                        if config['normalize']:
                            norm=np.linalg.norm(v); v=v/norm if norm else v
                        samples.append((i,v))
            else:
                import pyarrow.parquet as pq
                b = next(pq.ParquetFile(config['data_path']).iter_batches(batch_size=3)).to_pydict()
                samples = list(zip(b['id'], b['emb']))
            for i,v in samples:
                cur.execute(f"SELECT embedding::text FROM {config['table']} WHERE id=%s",(int(i),))
                actual = np.asarray(json.loads(cur.fetchone()[0]),dtype=np.float32)
                if not np.array_equal(actual,np.asarray(v,dtype=np.float32)):
                    raise RuntimeError(f'Base ID/vector mismatch {name} {i}')
            result[name]={'config':config,'queries':len(qids),'qid_range':[min(qids),max(qids)],
                          'query_sha256':jhash(q.tolist()),'gt_sha256':jhash(gt.tolist()),
                          'file_sha256':hashes,'index':identity,'counts':counts,'plan':plan,
                          'mapping_sample_ids':[int(i) for i,v in samples],'status':'PASS'}
            print('validated',name,len(qids),flush=True)
    # Split is frozen before collecting any SIFT calibration outcomes.
    with h5py.File(DATASETS['sift']['data_path']) as f:
        cal = {row.tobytes() for row in np.asarray(f['test'][:1000],dtype=np.float32)}
        formal = {row.tobytes() for row in np.asarray(f['test'][1000:],dtype=np.float32)}
        overlap = len(cal & formal)
    if overlap: raise RuntimeError(f'SIFT split has {overlap} duplicate query vectors across boundary')
    result['sift']['split']={'calibration_qids':[0,999],'formal_qids':[1000,9999],
        'cross_split_duplicate_vectors':overlap,'formal_unique_vectors':len(formal),'calibration_unique_vectors':len(cal),'frozen_before_calibration':True,
        'rule':'First 1000 calibration, remaining 9000 formal; no outcome-based selection.'}
    with h5py.File(common.ROOT/'data/gist1m/gist-960-euclidean-formal.hdf5') as f, h5py.File(DATASETS['gist']['data_path']) as original:
        if not np.array_equal(f['test'][1000:2000],original['test'][:]):
            raise RuntimeError('GIST official holdout mismatch')
        cal={row.tobytes() for row in np.asarray(f['test'][:1000],dtype=np.float32)}
        if any(row.tobytes() in cal for row in original['test'][:]): raise RuntimeError('GIST overlap')
    result['gist']['frozen_policy_sha256']=sha(D2P/'artifacts/d2p_policy.json')
    result['gist']['frozen_header_sha256']=sha(BENCH.parent/'contrib/pgvector/src/ivfd2ppolicy.h')
    atomic(root/'validation.json',result)
    return result


def schedule(smoke=False):
    # Reconnaissance respects strict coverage priority. Only after it completes,
    # repetitions rotate whole method groups, including the B and D comparisons.
    work=[]
    for priority,names,methods in [(1,list(DATASETS),['baseline','2a']),
                                  (2,['gist','sift'],['2a_b']),
                                  (3,['gist'],['d']),
                                  (3,['sift'],['baseline','d'])]:
        for name in names:
            for method in methods:
                work.append({'dataset':name,'method':method,'round':0,'priority':priority,
                             'family':'sift' if name=='sift' and priority==3 else 'production'})
    if smoke:return work
    for round_no in (1,2,3):
        names=list(DATASETS)
        offset=(round_no-1)%len(names)
        for name in names[offset:]+names[:offset]:
            methods=['baseline','2a']+(['2a_b'] if name in ('gist','sift') else [])+(['d'] if name=='gist' else [])
            rotation=(round_no-1)%len(methods)
            for method in methods[rotation:]+methods[:rotation]:
                work.append({'dataset':name,'method':method,'round':round_no,
                             'priority':2 if method=='2a_b' else (3 if method=='d' else 1),'family':'production'})
        for method in (['baseline','d'] if round_no%2 else ['d','baseline']):
            work.append({'dataset':'sift','method':method,'round':round_no,'priority':3,'family':'sift'})
    return work


def permutation(config, round_no, smoke=False):
    ids=list(range(config['formal_count']))
    random.Random(20260908+round_no+1).shuffle(ids)
    return ids[:3] if smoke else ids


def block_id(item):
    return f"r{item['round']}_{item['dataset']}_{item['family']}_{item['method']}"


def run_block(args,root,conn,item,manifest,smoke=False):
    name,method=item['dataset'],item['method']; config=DATASETS[name]
    if method=='2a_b' and config['metric']!='l2': raise RuntimeError('FUSED2 is L2-only')
    if method=='d' and name not in ('gist','sift'): raise RuntimeError('D unsupported')
    if method=='d' and name=='sift' and item['family']!='sift': raise RuntimeError('Never use GIST policy for SIFT')
    qids,queries,gt=load(config); order=permutation(config,item['round'],smoke)
    path=root/'checkpoints'/(block_id(item)+'.json'); path.parent.mkdir(exist_ok=True)
    if path.exists():
        block=json.loads(path.read_text())
        expected_binary=sha(root/'binaries'/item['family']/'vector.so')
        if any(r['binary_sha256']!=expected_binary for r in block['rows']): raise RuntimeError('Checkpoint binary mismatch')
        allowed_manifests={jhash(manifest)}
        control_path=root/'round_control.json'
        if control_path.exists():
            old_hash=json.loads(control_path.read_text()).get('source_manifest_hash')
            if old_hash:allowed_manifests.add(old_hash)
        if block['manifest_hash'] not in allowed_manifests or block['item']!=item or block['rows_hash']!=jhash(block['rows']):
            raise RuntimeError('Invalid checkpoint '+str(path))
        if [r['qid'] for r in block['rows']] != [qids[i] for i in order]: raise RuntimeError('Invalid checkpoint qid order')
        if block.get('summary_hash')!=jhash(block['summary']):raise RuntimeError('Invalid checkpoint summary')
        for row,i in zip(block['rows'],order):
            ids=row['result_ids']
            if (len(ids)!=10 or len(set(ids))!=10 or jhash(ids)!=row['result_checksum'] or
                not math.isfinite(row['latency_us']) or row['latency_us']<=0 or
                row['recall_at_10']!=len(set(ids)&set(map(int,gt[i])))/10):
                raise RuntimeError('Invalid checkpoint result')
        with (root/'resume.jsonl').open('a') as log:
            log.write(json.dumps({'block':block_id(item),'validated_rows':len(block['rows']),'time':time.time()})+'\n')
        return block
    binary=json.loads((root/'binaries'/item['family']/'build.json').read_text())['sha256']
    if sha(INSTALLED_VECTOR)!=binary: raise RuntimeError('Installed binary changed')
    rows=[]
    with conn.cursor() as cur:
        cur.execute('BEGIN')
        cur.execute(f"LOCK TABLE {config['table']} IN SHARE MODE")
        identity=index_identity(cur,config)
        if identity!=manifest['validation'][name]['index']: raise RuntimeError('Index changed')
        gucs=settings(cur,method)
        warmup=2 if smoke else 100
        for i in order[:warmup] if len(order)>=warmup else [order[k%len(order)] for k in range(warmup)]:
            cur.execute(sql(config),(common.vector_literal(queries[i]),));cur.fetchall()
        for pos,i in enumerate(order):
            literal=common.vector_literal(queries[i]);conn.notices.clear()
            before=time.perf_counter_ns();cur.execute(sql(config),(literal,));ids=[int(r[0]) for r in cur.fetchall()]
            latency=(time.perf_counter_ns()-before)/1000
            if any('IVFFLAT_' in n for n in conn.notices): raise RuntimeError('Instrumentation NOTICE during timing')
            if len(ids)!=10 or len(set(ids))!=10 or latency<=0: raise RuntimeError('Invalid Top10/latency')
            rows.append({'dataset':name,'method':method,'family':item['family'],'round':item['round'],
                         'position':pos,'qid':qids[i],'latency_us':latency,
                         'recall_at_10':len(set(ids)&set(map(int,gt[i])))/10,
                         'result_ids':ids,'result_checksum':jhash(ids),'binary_sha256':binary})
            if (pos+1)%100==0:
                active={'item':item,'block':block_id(item),'queries':pos+1,'total':len(order),'round':item['round']}
                cp=root/'round_control.json'
                if cp.exists():
                    from final_multi_control import update_progress
                    update_progress(root,json.loads(cp.read_text()),active,'RUNNING')
                else:
                    atomic(root/'progress.json',{'status':'RUNNING','block':block_id(item),'queries':pos+1,'total':len(order)})
                print(block_id(item),pos+1,'/',len(order),flush=True)
        # Collect stage counts separately with debug, never inside timed execution.
        probes=None
        if method=='d':
            probe_path=root/'diagnostics'/(name+'.json');probe_path.parent.mkdir(exist_ok=True)
            if probe_path.exists():
                replay=json.loads(probe_path.read_text())
                if replay['binary_sha256']!=binary: raise RuntimeError('Diagnostic binary mismatch')
            else:
                settings(cur,method,debug=True); records={}
                for row,i in zip(rows,order):
                    conn.notices.clear();cur.execute(sql(config),(common.vector_literal(queries[i]),));ids=[int(r[0]) for r in cur.fetchall()]
                    if ids!=row['result_ids']: raise RuntimeError('D diagnostic replay changed results')
                    notices='\n'.join(conn.notices)
                    decisions=re.findall(r'IVFFLAT_PROGRESSIVE_POLICY stage=(16|32).*?stop=([01])',notices)
                    stop=next((int(stage) for stage,yes in decisions if yes=='1'),64)
                    if not decisions: raise RuntimeError('Missing D stage diagnostic')
                    records[str(qids[i])]={'probes':stop,'result_checksum':jhash(ids)}
                replay={'binary_sha256':binary,'timed':False,'queries':records}
                atomic(probe_path,replay);settings(cur,method)
            for row in rows:
                if replay['queries'][str(row['qid'])]['result_checksum']!=row['result_checksum']:
                    raise RuntimeError('D stage replay checksum mismatch')
            probes=statistics.fmean(replay['queries'][str(r['qid'])]['probes'] for r in rows)
        cur.execute('COMMIT')
    lat=[r['latency_us'] for r in rows]
    summary={'dataset':name,'method':method,'family':item['family'],'round':item['round'],'queries':len(rows),
             'mean_us':statistics.fmean(lat),'p50_us':common.percentile(lat,50),'p95_us':common.percentile(lat,95),
             'p99_us':common.percentile(lat,99),'qps':1e6/statistics.fmean(lat),
             'recall_at_10':statistics.fmean(r['recall_at_10'] for r in rows),'avg_probes':probes if probes is not None else 64}
    block={'item':item,'rows':rows,'rows_hash':jhash(rows),'manifest_hash':jhash(manifest),'summary':summary,'summary_hash':jhash(summary),'settings':gucs,'warmup':warmup}
    atomic(path,block)
    print('completed',block_id(item),json.dumps(summary),flush=True)
    return block


def compile_policy(source,policy_dir):
    """Populate the EXISTING training features in a private SIFT build.

    The frozen GIST production path computes only its tree's selected fields.
    SIFT may select any field from the same frozen training feature dictionary.
    No probing, feature definition, threshold search, or kernel is changed.
    """
    policy=json.loads((policy_dir/'d2p_policy.json').read_text())
    if policy['stage16_features']!=list(calibration.STAGE16) or policy['stage32_features']!=list(calibration.STAGE32):
        raise RuntimeError('D feature definitions changed')
    header='typedef struct D2PFeatures {\n'+''.join('double '+f+';\n' for f in calibration.STAGE32)+'} D2PFeatures;\n'
    emitted=(policy_dir/'d2p_policy.c.inc').read_text()
    emitted=re.sub(r'\binf\b','INFINITY',emitted)
    (source/'src/ivfd2ppolicy.h').write_text(header+emitted)
    lines=['static void\nD2PBuildFeatures(IvfflatScanOpaque so, D2PFeatures *f, int stage)\n{',
           'double means[2] = {0,0};', 'MemSet(f, 0, sizeof(*f));']
    for n in (1,2,4,8,16,32,64): lines.append(f'f->d{n} = so->listDistances[{n-1}];')
    lines.append('f->d2_d1 = f->d1 > 0 ? f->d2 / f->d1 : (f->d2 > 0 ? DBL_MAX : 1.0);')
    for n in (4,8,16,32,64): lines.append(f'f->d{n}_d1 = f->d{n} / f->d1;')
    lines += ['f->d32_d16=f->d32/f->d16;', 'f->d64_d32=f->d64/f->d32;',
              'f->gap=f->d2-f->d1;', 'f->gap32_16_d1=(f->d32-f->d16)/f->d1;',
              'f->gap64_32_d1=(f->d64-f->d32)/f->d1;']
    for index,stage in enumerate((16,32)):
        lines += ['{',f'IvfflatShadowSnapshot *s = &so->shadowSnapshots[{index}];',
                  f'if ({stage} == 16 || stage == 32) {{',
                  'double sum=0, variance=0, minimum=s->items[0].distance, maximum=minimum;',
                  'for (int i=0;i<10;i++) { double d=s->items[i].distance; sum+=d; if(d<minimum) minimum=d; if(d>maximum) maximum=d; }',
                  f'means[{index}]=sum/10.0;',
                  f'for(int i=0;i<10;i++) {{ double delta=s->items[i].distance-means[{index}]; variance+=delta*delta; }}',
                  f'f->s{stage}_top10_min=minimum; f->s{stage}_top10_mean=means[{index}];',
                  f'f->s{stage}_top10_std=sqrt(variance/10.0); f->s{stage}_top10_max=maximum;',
                  f'f->s{stage}_top10_spread=maximum-minimum;',
                  f'f->s{stage}_candidates=s->candidatesSeen; f->s{stage}_pages=s->pagesSeen; f->s{stage}_replacements=s->replacements;',
                  f'f->s{stage}_replacement_rate=s->candidatesSeen ? (double)s->replacements/s->candidatesSeen : 0;',
                  f'f->s{stage}_candidates_per_page=s->pagesSeen ? (double)s->candidatesSeen/s->pagesSeen : 0;', '}', '}']
    lines += ['if(stage==32) {','double a=so->shadowSnapshots[0].items[9].distance;',
              'double b=so->shadowSnapshots[1].items[9].distance;',
              'for(int i=0;i<10;i++) for(int j=0;j<10;j++)',
              'if(ItemPointerEquals(&so->shadowSnapshots[0].items[i].tid,&so->shadowSnapshots[1].items[j].tid)) {f->top10_overlap_16_32++; break;}',
              'f->top10_replaced_16_32=10-f->top10_overlap_16_32;',
              'f->kth10_relative_change_16_32=fabs(a)>1e-30 ? (a-b)/a : 0;',
              'f->mean10_relative_change_16_32=fabs(means[0])>1e-30 ? (means[0]-means[1])/means[0] : 0;',
              'f->candidate_growth_16_32=f->s16_candidates ? (f->s32_candidates-f->s16_candidates)/f->s16_candidates : 0;', '}', '}\n']
    path=source/'src/ivfscan.c'; code=path.read_text()
    start=code.index('static void\nD2PBuildFeatures(')
    end=code.index('static bool\nD2PShouldStop(',start)
    path.write_text(code[:start]+'\n'.join(lines)+'\n'+code[end:])


def calibration_commands(args, root, queries=1000):
    config=DATASETS['sift']; dest=root/'calibration/sift';dest.mkdir(parents=True,exist_ok=True)
    shared=['--host',args.host,'--port',str(args.port),'--dbname',args.dbname,'--user',args.user]
    env=dict(os.environ,D2P_DATASET=config['data_path'],D2P_TABLE=config['table'],D2P_INDEX=config['index'])
    commands=[[sys.executable,str(D2P/'collect_progressive_shadow.py'),'--output',str(dest/'shadow'),'--queries',str(queries),*shared]]
    commands += [[sys.executable,str(D2P/'collect_fixed_baseline.py'),'--output',str(dest/'fixed'),'--queries',str(queries),'--probes',str(p),*shared] for p in (16,32,64,40,48,56)]
    return dest,env,commands


def calibrate_sift(args,root,smoke=False):
    dest,env,commands=calibration_commands(args,root,3 if smoke else 1000)
    if not (root/'binaries/calibration/build.json').exists(): build(root,'calibration',profiled=True)
    connection=activate(args,root,'calibration');connection.close()
    static_satisfied=False
    for argv in commands:
        probe=int(argv[argv.index('--probes')+1]) if '--probes' in argv else None
        if not smoke and probe in (40,48,56):
            def mean_recall(p):
                with (dest/'fixed'/f'fixed{p}.csv').open() as f:
                    return statistics.fmean(float(r['recall_at_10']) for r in csv.DictReader(f))
            target=mean_recall(64)-0.005
            if mean_recall(32)>=target or static_satisfied:continue
        with (dest/'collection.log').open('a') as log: subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not smoke and probe in (40,48,56):static_satisfied=mean_recall(probe)>=target
    if smoke:
        atomic(dest/'smoke.json',{'status':'PASS','shadow_queries':3,'fixed_queries':18,'trained':False})
        return
    records=[]
    for p in (16,32,64):
        with (dest/'fixed'/f'fixed{p}.csv').open() as f:
            records.extend({'system':'vanilla_eq','round':1,'probes':p,'query_id':r['query_id'],'recall_at_10':r['recall_at_10']} for r in csv.DictReader(f))
    common.write_csv_atomic(dest/'phase_c_calibration.csv',records,tuple(records[0]))
    argv=[sys.executable,str(D2P/'calibrate_policy.py'),'--shadow',str(dest/'shadow/shadow_snapshots.csv'),
          '--phase-c',str(dest/'phase_c_calibration.csv'),'--fixed-dir',str(dest/'fixed'),'--output',str(dest/'policy')]
    with (dest/'training.log').open('w') as log: subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    build(root,'sift',policy_dir=dest/'policy')
    # Verify production decisions against frozen Python features on calibration ONLY.
    conn=activate(args,root,'sift')
    try: policy_parity(args,root,conn,dest)
    finally: conn.close()
    atomic(dest/'complete.json',{'policy_sha256':sha(dest/'policy/d2p_policy.json'),
                                'binary_sha256':sha(root/'binaries/sift/vector.so'),'formal_data_used':False})


def policy_parity(args,root,conn,dest):
    features=calibration.feature_rows(calibration.read_csv(dest/'shadow/shadow_snapshots.csv'))
    policy=json.loads((dest/'policy/d2p_policy.json').read_text())
    def predict(tree,f):
        while 'feature' in tree: tree=tree['if_le' if f[tree['feature']]<=tree['threshold'] else 'if_gt']
        return tree['safe_probability']
    config=DATASETS['sift'];qids,q,gt=load(config,True);records=[]
    with conn.cursor() as cur:
        settings(cur,'d',debug=True)
        for i in range(1000):
            expected=16 if predict(policy['stage16_tree'],features[i])>=policy['stage16_threshold'] else (32 if predict(policy['stage32_tree'],features[i])>=policy['stage32_threshold'] else 64)
            conn.notices.clear();cur.execute(sql(config),(common.vector_literal(q[i]),));ids=[int(r[0]) for r in cur.fetchall()]
            decisions=re.findall(r'IVFFLAT_PROGRESSIVE_POLICY stage=(16|32).*?stop=([01])','\n'.join(conn.notices))
            actual=next((int(s) for s,b in decisions if b=='1'),64)
            if not decisions or expected!=actual: raise RuntimeError(f'SIFT policy parity failure qid={i}')
            with (dest/'fixed'/f'fixed{expected}.csv').open() as f:
                fixed=next(r for r in csv.DictReader(f) if int(r['query_id'])==i)
            if ids!=list(map(int,fixed['result_ids'].split(';'))): raise RuntimeError('SIFT prefix equivalence failed')
            records.append({'qid':i,'expected':expected,'actual':actual})
    atomic(dest/'policy_parity.json',{'status':'PASS','calibration_only':True,'records':records})


def outputs(root,blocks,smoke=False):
    common.write_csv_atomic(root/'raw.csv',[{**r,'result_ids':json.dumps(r['result_ids'])} for b in blocks for r in b['rows']],tuple(blocks[0]['rows'][0]))
    common.write_csv_atomic(root/'summary.csv',[b['summary'] for b in blocks],tuple(blocks[0]['summary']))
    selected=[b for b in blocks if smoke or b['item']['round']>0]
    lookup={}
    for b in selected:
        key=(b['item']['dataset'],b['item']['family'],b['item']['method'])
        lookup.setdefault(key,[]).extend(b['rows'])
    def get(name,method,family='production'):
        rows=lookup.get((name,family,method))
        if not rows:return None
        return statistics.fmean(r['latency_us'] for r in rows),statistics.fmean(r['recall_at_10'] for r in rows)
    a=[];b=[];c=[]
    for name,config in DATASETS.items():
        base=get(name,'baseline');two=get(name,'2a');fused=get(name,'2a_b')
        if base and two:a.append({'Dataset':name,'Baseline latency us':base[0],'2A latency us':two[0],'2A speedup':base[0]/two[0],'Recall baseline':base[1],'Recall 2A':two[1]})
        if two and fused:b.append({'Dataset':name,'2A latency us':two[0],'2A+B latency us':fused[0],'B incremental speedup':two[0]/fused[0],'Recall 2A':two[1],'Recall 2A+B':fused[1]})
        family='sift' if name=='sift' else 'production';d=get(name,'d',family);dbase=get(name,'baseline',family)
        row={'Dataset':name,'Baseline latency us':None,'D latency us':None,'D speedup':None,'Baseline Recall':None,'D Recall':None,'Recall delta':None,'Avg probes':None,'Status':'N/A','Reason':config.get('d_reason','pending calibration/formal execution')}
        if d and dbase:
            diags=json.loads((root/'diagnostics'/f'{name}.json').read_text())
            row.update({'Baseline latency us':dbase[0],'D latency us':d[0],'D speedup':dbase[0]/d[0],'Baseline Recall':dbase[1],'D Recall':d[1],'Recall delta':d[1]-dbase[1],'Avg probes':statistics.fmean(v['probes'] for v in diags['queries'].values()),'Status':'RUN','Reason':''})
        c.append(row)
    for letter,rows in [('A',a),('B',b),('C',c)]:
        if rows:common.write_csv_atomic(root/f'table_{letter}.csv',rows,tuple(rows[0]))
    def geo(values):
        values=[float(v) for v in values if v is not None]
        if not values:return None
        if any(not math.isfinite(v) or v<=0 for v in values):
            raise RuntimeError('Geometric-mean speedup requires positive finite values')
        return math.exp(statistics.fmean(math.log(v) for v in values))
    atomic(root/'comparison.json',{'smoke_only':smoke,'aggregation':'pooled equally sized rounds 1..3; round 0 reconnaissance excluded',
        'table_A':a,'table_B':b,'table_C':c,'geomeans':{
            '2a':geo([r['2A speedup'] for r in a]) if len(a)==5 else None,
            '2a_b_incremental':geo([r['B incremental speedup'] for r in b]) if len(b)==2 else None,
            'd':geo([r['D speedup'] for r in c if r['D speedup'] is not None])}})


def equality_gate(blocks,round_no):
    groups={}
    for b in blocks:
        if b['item']['round']==round_no and b['item']['family']=='production':
            groups[(b['item']['dataset'],b['item']['method'])]=b
    for name in DATASETS:
        for left,right in [('baseline','2a'),('2a','2a_b')]:
            a=groups.get((name,left));b=groups.get((name,right))
            if not a or not b:continue
            if [(r['qid'],r['result_checksum']) for r in a['rows']]!=[(r['qid'],r['result_checksum']) for r in b['rows']]:
                raise RuntimeError(f'Top10 equivalence failed {name} {left}/{right}; inspect before repetitions')


def plan(root):
    control_path=Path(root)/'round_control.json'
    control=json.loads(control_path.read_text()) if control_path.exists() else None
    work=control['schedule'] if control else schedule()
    measured=sum(DATASETS[i['dataset']]['formal_count'] for i in work)
    details={'scope':'Adaptive two-effective-round formal schedule' if control else 'Preparation and smoke only unless --formal is explicitly given',
        'dataset_configs':DATASETS,'schedule':work,'probes':64,'topk':10,'warmup_per_block':100,
        'rounds':({'effective_rounds':2,'final_round_a':control['final_round_a'],'final_round_b':control['final_round_b'],
                   'effective_total_config_rounds':28,'round_a_order':control['round_a_order'],
                   'round_b_order':control['round_b_order'],'rotation':control['rotation']} if control else '1 reconnaissance + 3 balanced formal; round 0 excluded from final tables'),
        'query_orders':{name:{str(r):{'seed':20260909+r,'permutation_sha256':jhash(permutation(config,r)),
                        'first_ten_source_qids':[config['formal_start']+i for i in permutation(config,r)[:10]]}
                        for r in range(4)} for name,config in DATASETS.items()},
        'measured_query_executions':164000 if control else measured,'effective_query_executions_per_round':82000 if control else None,
        'effective_total_config_rounds':28 if control else len(work),'warmup_query_executions':len(work)*100,
        'sift_calibration_executions_max':7000,'sift_policy_parity_executions':1000,
        'd_untimed_stage_replay_executions':10000,
        'total_query_executions_max':measured+len(work)*100+7000+1000+10000,
        'sift_d':'16 -> 32 -> 64; existing feature definitions; DecisionTreeClassifier; depth {2,3}; leaf {20,40}; qid%5 OOF; seed 20260913; loss budget 0.005; calibration qids 0..999; formal qids 1000..9999',
        'sift_binary_note':'General comparisons share production; Baseline vs D remeasured in same SIFT-policy binary after calibration.',
        'glove_cosine_d':'N/A: no existing clean split, skipped as instructed',
        'production':'No BENCH/PROFILE macros or timers; debug off during timing; independent untimed stage replay',
        'command':f'{BENCH}/scripts/benchmark.py run final-multi --run-dir {root} --formal --prepared <PREPARED_RUN> --resume --background'}
    atomic(root/'plan.json',details)
    return details


def exact_gt_validation(root,conn):
    records=[]
    with conn.cursor() as cur:
        settings(cur,'baseline')
        cur.execute('SET enable_indexscan=off')
        cur.execute('SET enable_indexonlyscan=off')
        cur.execute('SET enable_bitmapscan=off')
        cur.execute('SET enable_seqscan=on')
        for name,config in DATASETS.items():
            qids,q,gt=load(config)
            literal=common.vector_literal(q[0])
            cur.execute('EXPLAIN (FORMAT JSON) '+sql(config),(literal,))
            query_plan=cur.fetchone()[0]
            if 'Seq Scan' not in json.dumps(query_plan) or 'Index Scan' in json.dumps(query_plan):
                raise RuntimeError('Exact GT validation did not use a sequential scan')
            cur.execute(sql(config),(literal,));ids=[int(row[0]) for row in cur.fetchall()]
            expected=list(map(int,gt[0]))
            if set(ids)!=set(expected):raise RuntimeError('Exact GT mismatch for '+name)
            records.append({'dataset':name,'qid':qids[0],'result_ids':ids,'gt':expected,'status':'PASS'})
            print('exact GT PASS',name,flush=True)
    atomic(root/'exact_gt_validation.json',{'queries':records,'query_executions':len(records),'timed':False})


def runtime_environment(args):
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('numpy','h5py','pyarrow','psycopg2-binary','scikit-learn')}
    conn=common.connect(args)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name,setting,unit FROM pg_settings WHERE name=ANY(%s) ORDER BY name",
                        (['shared_buffers','work_mem','maintenance_work_mem','effective_cache_size','jit',
                          'random_page_cost','seq_page_cost','max_parallel_workers_per_gather','track_io_timing'],))
            settings=cur.fetchall()
    finally:conn.close()
    return {'python':sys.version,'packages':versions,'server_settings':[list(row) for row in settings]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--formal',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--dataset',choices=['all',*DATASETS],default='all')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=5432)
    p.add_argument('--dbname',default='taskdb');p.add_argument('--user',default='dev')
    p.add_argument('--stop-after-configs',type=int,default=0)
    args=p.parse_args(argv);root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with (common.ROOT/'.phase_c_database.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: execute(args,root)
        except BaseException as exc:
            atomic(root/'progress.json',{'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED','error':str(exc)})
            if isinstance(exc,KeyboardInterrupt):raise SystemExit(75)
            raise


def execute(args,root):
    plan(root)
    (root/'source_diff.patch').write_text(cmd(['git','-C',BENCH.parent,'diff','--','contrib/pgvector'])+'\n')
    if args.formal:
        if not args.prepared:raise ValueError('--formal requires --prepared with PASS smoke')
        certificate=json.loads((args.prepared/'smoke_gate.json').read_text())
        if certificate['status']!='PASS':raise RuntimeError('Preparation did not pass')
        for name,digest in certificate['runner_hashes'].items():
            if sha(name)!=digest:raise RuntimeError('Runner changed after smoke; repeat preparation')
        for relative,digest in certificate['source_hashes'].items():
            if sha(BENCH.parent/'contrib/pgvector'/relative)!=digest:raise RuntimeError('Algorithm source changed after smoke')
        if sha(POSTGRES_BINARY)!=certificate['postgres_sha256']:raise RuntimeError('Server changed after smoke')
        if not (root/'binaries/production').exists():
            shutil.copytree(args.prepared/'binaries/production',root/'binaries/production')
        if not (root/'binaries/calibration').exists():
            shutil.copytree(args.prepared/'binaries/calibration',root/'binaries/calibration')
    elif not (root/'binaries/production/build.json').exists():build(root)
    conn=activate(args,root,'production')
    try:
        validation=validate(args,root,conn)
        if not args.formal and not (root/'exact_gt_validation.json').exists():exact_gt_validation(root,conn)
    finally:conn.close()
    if args.validate_only:return
    control_path=root/'round_control.json'
    control=json.loads(control_path.read_text()) if args.formal and control_path.exists() else None
    manifest={'version':1,'formal':args.formal,'git_commit':cmd(['git','-C',BENCH.parent,'rev-parse','HEAD']),
              'source_diff_sha256':jhash(cmd(['git','-C',BENCH.parent,'diff','--','contrib/pgvector'])),
              'config_sha256':sha(CONFIG_PATH),'validation':validation,'probes':64,'topk':10,
              'postgres_sha256':sha(POSTGRES_BINARY),
              'environment':runtime_environment(args),
              'binaries':{'production':json.loads((root/'binaries/production/build.json').read_text())},
              'schedule':control['schedule'] if control else schedule(not args.formal),'warmup':100 if args.formal else 2}
    if control:
        manifest.update({'effective_rounds':2,'final_round_a':control['final_round_a'],'final_round_b':control['final_round_b'],
                         'effective_total_config_rounds':28,'effective_formal_query_executions':164000,
                         'source_manifest_sha256':control['source_manifest_hash'],'round_control_sha256':sha(control_path)})
    if args.formal:
        prepared_validation=json.loads((args.prepared/'validation.json').read_text())
        if validation!=prepared_validation:raise RuntimeError('Data/index/policy changed since preparation')
    mp=root/'manifest.json'
    if mp.exists():
        if not args.resume:raise RuntimeError('Output exists; use --resume')
        if json.loads(mp.read_text())!=manifest:raise RuntimeError('Resume manifest mismatch')
    else:atomic(mp,manifest)
    blocks=[];family=None;conn=None;executed=0
    try:
        for item in manifest['schedule']:
            if args.dataset!='all' and item['dataset']!=args.dataset:continue
            if item['family']=='sift':
                if not args.formal:continue # No uncalibrated SIFT policy is ever run.
                if not (root/'calibration/sift/complete.json').exists():
                    if conn:conn.close();conn=None
                    print('Priority 3: SIFT clean calibration',flush=True)
                    calibrate_sift(args,root);family=None
                done=json.loads((root/'calibration/sift/complete.json').read_text())
                if done['policy_sha256']!=sha(root/'calibration/sift/policy/d2p_policy.json') or done['binary_sha256']!=sha(root/'binaries/sift/vector.so'):
                    raise RuntimeError('Frozen SIFT calibration artifacts changed')
            if family!=item['family']:
                if conn:conn.close()
                conn=activate(args,root,item['family']);family=item['family']
            path=root/'checkpoints'/(block_id(item)+'.json');was_done=path.exists()
            block=run_block(args,root,conn,item,manifest,not args.formal);blocks.append(block)
            equality_gate(blocks,item['round'])
            if args.formal and control:
                from final_multi_control import refresh_outputs
                refresh_outputs(root,control,status='RUNNING')
            else:
                outputs(root,blocks,not args.formal)
            if not was_done:executed+=1
            if not (args.formal and control):
                atomic(root/'progress.json',{'status':'RUNNING','completed_blocks':len(blocks),'planned_blocks':len(manifest['schedule']),'last_block':block_id(item)})
            if args.stop_after_configs and executed>=args.stop_after_configs:
                raise KeyboardInterrupt('Requested stop after durable checkpoint')
    finally:
        if conn:conn.close()
    if not args.formal and args.dataset=='all':
        calibrate_sift(args,root,smoke=True)
        conn=activate(args,root,'production');conn.close()
        runner_files=[Path(__file__),Path(common.__file__),Path(common.__file__).with_name('benchmark_paths.py'),
                      CONFIG_PATH,D2P/'collect_progressive_shadow.py',D2P/'collect_fixed_baseline.py',
                      D2P/'calibrate_policy.py']
        atomic(root/'smoke_gate.json',{'status':'PASS','production_blocks':len(blocks),'measured_queries':sum(len(b['rows']) for b in blocks),
            'sift_D':'shadow and fixed calibration smoke PASS; no trained SIFT D yet',
            'runner_hashes':{str(f):sha(f) for f in runner_files+[Path(__file__).with_name('final_multi_control.py')]},
            'source_hashes':json.loads((root/'binaries/production/build.json').read_text())['source_hashes'],
            'postgres_sha256':sha(POSTGRES_BINARY),'long_experiments_started':False})
    if args.formal and control:
        from final_multi_control import refresh_outputs
        refresh_outputs(root,control,status='COMPLETE')
    else:
        atomic(root/'progress.json',{'status':'COMPLETE' if args.formal else 'PREPARED','completed_blocks':len(blocks),'formal':args.formal})


if __name__=='__main__':main()
