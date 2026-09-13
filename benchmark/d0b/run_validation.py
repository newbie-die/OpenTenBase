#!/usr/bin/env python3
"""D-0B: one immutable 100 x 6 measured workload, no resume/second round."""
import csv,datetime,hashlib,json,os,random,shutil,struct,subprocess,sys,traceback
from pathlib import Path
import h5py,numpy as np,psycopg2
REPO=Path('/workspace/OpenTenBase');SRC=REPO/'benchmark/d0b'
OLD=Path('/workspace/benchmark/runs/20260909T083926Z')
PRODUCTION=Path('/workspace/install/lib/postgresql/vector.so')
PROBES=[8,16,32,64,128,256];SEED=20260911
BINARY='d56dffd4db4e08d075cad229da5b04a7dc722f1d2cd45f03fb5570b0a64f6480'
DATASET=Path('/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()

def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def protect():
    provenance=json.loads((OLD/'freeze_evidence/production_provenance.json').read_text())
    assert sha(PRODUCTION)==BINARY
    for name,digest in provenance['source_files_sha256'].items():assert sha(REPO/name)==digest
    assert not git('diff',provenance['source_commit'],'--','contrib/pgvector','src')
    snapshot=Path(json.loads((OLD/'freeze_evidence/termination_receipt.json').read_text())['snapshot'])
    inventory=json.loads((snapshot/'inventory.json').read_text())
    for name,item in inventory.items():
        if item['type']=='file':assert sha(OLD/name)==item['sha256'],name
    assert not Path('/proc/516593').exists()
    return dict(production_sha256=BINARY,protected_sources_unchanged=True,phase_c_files_unchanged=True,phase_c_verified_files=sum(x['type']=='file' for x in inventory.values()),old_runner_absent=True)

def identity(cur):
    cur.execute("SELECT c.oid,c.relfilenode,pg_relation_size(c.oid),pg_get_indexdef(c.oid),i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid WHERE c.oid='gist_ivf_l2'::regclass")
    return dict(zip(['oid','relfilenode','bytes','definition','valid'],cur.fetchone()))

def analyze(root,rows,correctness,pages):
    summary=[];half=[];occupancy=[]
    for p in PROBES:
        rs=[r for r in rows if r['probes']==p];pg=[r for r in pages if r['probes']==p]
        npages=sum(r['scanned_entry_pages'] for r in rs);nc=sum(r['scanned_candidates'] for r in rs)
        npp=sum(r['prunable_pages'] for r in rs);npc=sum(r['prunable_candidates'] for r in rs)
        out=dict(probes=p,queries=len(rs),scanned_entry_pages=npages,scanned_candidates=nc,
                 mean_scanned_entry_pages=float(np.mean([r['scanned_entry_pages'] for r in rs])),median_scanned_entry_pages=float(np.median([r['scanned_entry_pages'] for r in rs])),
                 mean_scanned_candidates=float(np.mean([r['scanned_candidates'] for r in rs])),median_scanned_candidates=float(np.median([r['scanned_candidates'] for r in rs])),
                 prunable_pages=npp,prunable_candidates=npc,page_prune_ratio=npp/npages,candidate_prune_ratio=npc/nc,distance_avoid_ratio=npc/nc,sort_input_avoid_ratio=npc/nc,
                 candidate_prune_ratio_p25=float(np.quantile([r['candidate_prune_ratio'] for r in rs],.25)),candidate_prune_ratio_p50=float(np.quantile([r['candidate_prune_ratio'] for r in rs],.5)),candidate_prune_ratio_p75=float(np.quantile([r['candidate_prune_ratio'] for r in rs],.75)),
                 page_prune_ratio_p25=float(np.quantile([r['page_prune_ratio'] for r in rs],.25)),page_prune_ratio_p50=float(np.quantile([r['page_prune_ratio'] for r in rs],.5)),page_prune_ratio_p75=float(np.quantile([r['page_prune_ratio'] for r in rs],.75)),
                 zero_pruning_query_fraction=sum(r['prunable_pages']==0 for r in rs)/len(rs),mean_recall_at_10=float(np.mean([r['real_recall_at_10'] for r in rs])),shadow_mismatch_count=sum(r['shadow_mismatch'] for r in rs))
        for name in ['pages_before_topk_ready','pages_after_topk_ready','lb_zero_pages','dc_inside_interval_pages','dc_inside_interval_ready_pages','positive_lb_insufficient_pages','numeric_guard_rejected_pages','width_ge_tau_pages','singleton_bound_prunable_candidates','singleton_all_prunable_page_retained','oracle_min_distance_prunable_pages','oracle_min_distance_prunable_candidates']:
            out[name]=sum(r[name] for r in rs)
        out['lb_zero_page_ratio']=out['lb_zero_pages']/npages
        out['pages_before_ready_ratio']=out['pages_before_topk_ready']/npages
        out['dc_inside_interval_ratio']=out['dc_inside_interval_pages']/npages
        out['width_ge_tau_after_ready_ratio']=out['width_ge_tau_pages']/out['pages_after_topk_ready']
        out['singleton_radial_bound_candidate_prune_ratio']=out['singleton_bound_prunable_candidates']/nc
        out['oracle_page_candidate_prune_ratio']=out['oracle_min_distance_prunable_candidates']/nc
        for name in ['rmin_mean','rmax_mean','residual_width_mean','lb_radius_mean']:
            out[name]=sum(r[name]*r['scanned_entry_pages'] for r in rs)/npages
        for group_name,values,destination in [('list_half',['first','second'],half),('occupancy',['0','1','2','3+'],occupancy)]:
            for value in values:
                g=[r for r in pg if r[group_name]==value];gp=sum(r['scanned_entry_pages'] for r in g);gc=sum(r['scanned_candidates'] for r in g)
                if not gp:continue
                destination.append(dict(probes=p,group=value,scanned_entry_pages=gp,scanned_candidates=gc,prunable_pages=sum(r['prunable_pages'] for r in g),prunable_candidates=sum(r['prunable_candidates'] for r in g),page_prune_ratio=sum(r['prunable_pages'] for r in g)/gp,candidate_prune_ratio=sum(r['prunable_candidates'] for r in g)/gc if gc else 0,lb_zero_page_ratio=sum(r['lb_zero_pages'] for r in g)/gp,inside_interval_page_ratio=sum(r['dc_inside_interval_pages'] for r in g)/gp,singleton_radial_candidate_prune_ratio=sum(r['singleton_bound_prunable_candidates'] for r in g)/gc if gc else 0))
        summary.append(out)
    high=[r['distance_avoid_ratio'] for r in summary if r['probes'] in [64,128,256]]
    decision='Strong Go' if sum(x>=.30 for x in high)>=2 else 'Go' if sum(x>=.15 for x in high)>=2 else 'No-Go' if all(x<.05 for x in high) else 'Marginal'
    csvwrite(root/'phase_d0b_l2_query_summary.csv',rows);csvwrite(root/'phase_d0b_l2_probe_summary.csv',summary);csvwrite(root/'phase_d0b_l2_correctness.csv',correctness);csvwrite(root/'phase_d0b_l2_page_stats.csv',pages)
    csvwrite(root/'phase_d0b_l2_list_half_summary.csv',half);csvwrite(root/'phase_d0b_l2_occupancy_summary.csv',occupancy)
    result=dict(status='COMPLETE',decision=decision,rounds=[1],measured_query_executions=600,warmup_query_executions=120,shadow_mismatch_count=0,real_sql_vs_shadow_mismatch_count=0,real_sql_vs_frozen_phase_c_mismatch_count=0,
                probe_summary=summary,list_half_summary=half,occupancy_summary=occupancy,
                definition='600 actual measured Top10 SQL executions plus read-only observer helper per execution; 20 warmup executions per probes excluded. Counters describe index-entry traversal replay with actual selected-list order; observer distance calls and metadata work are instrumentation overhead.',
                page_stats_unit='query x selected-list rank x candidate occupancy; exact aggregation of every replayed entry-page visit, not sampled pages or unique physical page count',
                singleton_diagnostic='Same page-entry tau and residual bound per individual candidate; counterfactual only, never changes page pruning decision',
                oracle_diagnostic='Hindsight exact page-minimum distance > page-entry tau; query-dependent ceiling only, not feasible static metadata or a pruning decision',
                gate_rule='Strong Go >=30% at >=2 high probes; Go >=15% at >=2; No-Go all 3 high probes <5%; otherwise Marginal (mixed cases conservative)',
                no_latency_conclusion=True,production_pruning=False,page_format_changed=False,d0c_started=False,protection_after=protect())
    save(root/'phase_d0b_l2_analysis.json',result)
    return result

def main():
    assert len(sys.argv)==2,'Pass one NEW run directory; no rounds/resume option exists'
    root=Path(sys.argv[1]);assert root.parent==Path('/workspace/benchmark/runs') and root.name.startswith('phase_d0b_l2_') and not root.exists()
    before=protect();prior=json.loads((OLD/'phase_c_artifacts/phase_c_manifest.json').read_text());assert sha(DATASET)==prior['dataset_sha256']
    root.mkdir()
    for name in ['observer.c','run_validation.py','build_command.json','build.log','numerical_certificate.md']:
        shutil.copy2(SRC/name,root/name)
    shutil.copy2('/workspace/benchmark/d0b_observer.so',root/'d0b_observer.so')
    permutation=list(range(1000));random.Random(SEED).shuffle(permutation);ids=permutation[:100]
    permhash=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    save(root/'query_permutation.json',dict(seed=SEED,method='Python random.Random(seed).shuffle(range(1000)); first 100; same permutation per probes',query_ids=ids,warmup_query_ids=ids[:20],sha256=permhash))
    (root/'query_ids.txt').write_text(''.join(str(q)+'\n' for q in ids))
    with h5py.File(DATASET,'r') as f:
        queries={q:np.asarray(f['test'][q],dtype=np.float32) for q in ids};truth={q:set(int(v) for v in f['neighbors'][q,:10]) for q in ids}
    literals={q:'['+','.join(str(float(v)) for v in x)+']' for q,x in queries.items()}
    frozen={}
    with (OLD/'phase_c_artifacts/phase_c_raw.csv').open() as f:
        for r in csv.DictReader(f):
            if r['system']=='phase2a' and int(r['round'])==1 and int(r['probes']) in PROBES and int(r['query_id']) in queries:
                frozen[(int(r['probes']),int(r['query_id']))]=[int(v) for v in r['result_ids'].split(';')]
    assert len(frozen)==600
    manifest=dict(phase='D-0B',status='PREPARED',dataset='GIST1M',dimension=960,metric='L2',lists=1000,topk=10,rounds=1,queries=100,warmup_per_probes=20,probes=PROBES,total_measured_query_executions=600,
                  query_permutation_sha256=permhash,dataset_sha256=prior['dataset_sha256'],ground_truth_reference=prior['ground_truth_sha256'],ground_truth_changed=False,
                  production_binary=str(PRODUCTION),production_sha256=BINARY,production_source_commit=prior['optimized_commit'],repository_head=git('rev-parse','HEAD'),observer_sha256=sha(root/'d0b_observer.so'),observer_source_sha256=sha(root/'observer.c'),runner_sha256=sha(root/'run_validation.py'),
                  metadata_norm='non-squared L2 radii in outward binary64 intervals; squared final LB compared with unchanged production squared-L2 threshold',
                  measurement='Read-only ExecutorEnd captures actual selected list order and logical/physical bounds. Independent observer replays all corresponding candidates through unchanged generic support proc under stable table lock; real AM never prunes.',
                  threshold='Two independent Top10 heaps ordered (distance,TID); tau captured BEFORE reading page candidates; metadata cache only query-independent radii; real SQL and both heaps verified',
                  diagnostic_queries_excluded_from_workload='Temporary helper calls replay metadata/candidates; no additional Top10 workload beyond 600 measured +120 warmup; no SIFT/GLOVE/Cohere/IP',
                  instrumentation_compiler_flags=json.loads((SRC/'build_command.json').read_text()),abi_macro='IVFFLAT_DISTANCE_PATH matches installed scan struct; no FUSED2 macro or FUSED2 kernel call in observer',protection_before=before,
                  latency_claims=False,second_round_allowed=False,d0c_allowed=False)
    save(root/'phase_d0b_l2_manifest.json',manifest)
    (root/'environment.txt').write_text(subprocess.check_output(['uname','-a'],text=True)+subprocess.check_output(['gcc','--version'],text=True)+git('status','--short','--untracked-files=all')+'\n')
    (root/'pid').write_text(str(os.getpid())+'\n')
    conn=psycopg2.connect(host='127.0.0.1',port=5432,dbname='taskdb',user='dev',application_name='phase_d0b_l2_single_round')
    conn.autocommit=True;rows=[];correctness=[];pages=[];actual_count=0;warmup_count=0
    handles=[]
    try:
        with conn.cursor() as cur:
            cur.execute("LOAD 'vector'")
            cur.execute("CREATE FUNCTION pg_temp.d0_selftest() RETURNS text AS %s,'d0_selftest' LANGUAGE C STRICT",(str(root/'d0b_observer.so'),))
            cur.execute("CREATE FUNCTION pg_temp.d0_observe(regclass,vector,int) RETURNS text AS %s,'d0_observe' LANGUAGE C STRICT",(str(root/'d0b_observer.so'),))
            cur.execute('SELECT pg_temp.d0_selftest()');save(root/'selftest.json',dict(result=cur.fetchone()[0]))
            cur.execute('BEGIN ISOLATION LEVEL REPEATABLE READ');cur.execute('SET LOCAL lock_timeout=10000');cur.execute('LOCK TABLE gist_base IN SHARE MODE')
            assert identity(cur)==prior['index'];manifest['index']=identity(cur)
            cur.execute('SELECT pg_backend_pid(),version()');manifest['backend_pid'],manifest['postgres_version']=cur.fetchone()
            settings={'ivfflat.iterative_scan':'off','ivfflat.distance_path':'generic','ivfflat.bounded_scan':'on','ivfflat.experimental_sort_bound':'0','ivfflat.bound_overfetch':'4','ivfflat.bound_min':'40','ivfflat.bound_fastpath_limit':'100'}
            for name,value in settings.items():
                cur.execute('SELECT setting FROM pg_settings WHERE name=%s',(name,));assert cur.fetchone() is not None
                cur.execute('SELECT set_config(%s,%s,false)',(name,value));assert cur.fetchone()[0]==value
            for sql in ['SET LOCAL enable_seqscan=off','SET LOCAL enable_indexscan=on','SET LOCAL max_parallel_workers_per_gather=0','SET LOCAL statement_timeout=300000']:cur.execute(sql)
            manifest.update(settings=settings,status='RUNNING');save(root/'phase_d0b_l2_manifest.json',manifest)
            sentinel=Path('/workspace/benchmark/phase_d0b_l2_once.json')
            with sentinel.open('x') as f:json.dump(dict(run=str(root),pid=os.getpid(),rounds=1,measured_limit=600),f)
            sql='SELECT id,ctid::text FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10'
            raw=(root/'phase_d0b_l2_raw.csv').open('w',newline='');handles.append(raw)
            writer=csv.DictWriter(raw,fieldnames=['round','probes','query_order_position','query_id','actual_ids','actual_tids','shadow_full_ids','shadow_pruned_ids','shadow_full_top10','shadow_pruned_top10','real_recall_at_10','shadow_mismatch','observer_json'])
            writer.writeheader();raw.flush()
            warm=(root/'warmup.jsonl').open('w');handles.append(warm)
            last_serial=0
            for p in PROBES:
                cur.execute("SELECT set_config('ivfflat.probes',%s,false)",(str(p),));assert cur.fetchone()[0]==str(p)
                cur.execute('EXPLAIN (FORMAT JSON) '+sql,(literals[ids[0]],));plan=cur.fetchone()[0]
                assert 'gist_ivf_l2' in json.dumps(plan) and 'Index Scan' in json.dumps(plan);save(root/f'plan_p{p}.json',plan)
                for is_warm,work in [(True,ids[:20]),(False,ids)]:
                    for position,q in enumerate(work,1):
                        cur.execute(sql,(literals[q],));actual=cur.fetchall()
                        if is_warm:warmup_count+=1
                        else:actual_count+=1
                        save(root/'progress.json',dict(status='VALIDATING',round=1,actual_measured_executions=actual_count,validated_measured_executions=len(rows),warmup_executions=warmup_count,last_probes=p,last_query_id=q))
                        cur.execute("SELECT pg_temp.d0_observe('gist_ivf_l2'::regclass,%s::vector,%s)",(literals[q],p));obs=json.loads(cur.fetchone()[0])
                        assert obs['scan_serial']>last_serial;last_serial=obs['scan_serial']
                        full=obs['shadow_full_top10'];pruned=obs['shadow_pruned_top10'];real_ids=[r[0] for r in actual];real_tids=[r[1] for r in actual];mapping={tid:rid for rid,tid in actual}
                        full_ids=[mapping.get(x['tid']) for x in full];pruned_ids=[mapping.get(x['tid']) for x in pruned]
                        record=dict(round=1,probes=p,query_id=q,ordered_tid_mismatch=int([x['tid'] for x in full]!=[x['tid'] for x in pruned]),set_tid_mismatch=int({x['tid'] for x in full}!={x['tid'] for x in pruned}),distance_value_mismatch=int([struct.pack('!d',x['distance_squared']) for x in full]!=[struct.pack('!d',x['distance_squared']) for x in pruned]),ordered_id_mismatch=int(full_ids!=pruned_ids),real_sql_tid_mismatch=int(real_tids!=[x['tid'] for x in full]),real_sql_id_mismatch=int(real_ids!=full_ids),frozen_phase_c_id_mismatch=int(real_ids!=frozen[(p,q)]),returned_rows=len(actual))
                        mismatch=int(any(v for k,v in record.items() if k.endswith('mismatch')) or obs['shadow_mismatch'])
                        record['shadow_mismatch']=mismatch
                        if mismatch:
                            save(root/'correctness_failure.json',dict(warmup=is_warm,record=record,actual=actual,observer=obs));raise RuntimeError('D-0B FAIL: Top10 mismatch; stop immediately')
                        recall=len(set(real_ids)&truth[q])/10
                        if is_warm:
                            warm.write(json.dumps(record)+'\n');warm.flush()
                        else:
                            rawrow=dict(round=1,probes=p,query_order_position=position,query_id=q,actual_ids=json.dumps(real_ids),actual_tids=json.dumps(real_tids),shadow_full_ids=json.dumps(full_ids),shadow_pruned_ids=json.dumps(pruned_ids),shadow_full_top10=json.dumps(full),shadow_pruned_top10=json.dumps(pruned),real_recall_at_10=recall,shadow_mismatch=0,observer_json=json.dumps(obs,separators=(',',':')))
                            writer.writerow(rawrow);raw.flush()
                            r={k:v for k,v in obs.items() if not isinstance(v,(list,dict))};r.update(round=1,probes=p,query_order_position=position,query_id=q,real_recall_at_10=recall,
                              exact_distance_calls=obs['scanned_candidates'],tuplesort_inputs=obs['scanned_candidates'],avoidable_distance_calls=obs['prunable_candidates'],avoidable_sort_inputs=obs['prunable_candidates'],
                              page_prune_ratio=obs['prunable_pages']/obs['scanned_entry_pages'],candidate_prune_ratio=obs['prunable_candidates']/obs['scanned_candidates'],distance_avoid_ratio=obs['prunable_candidates']/obs['scanned_candidates'],lb_zero_page_ratio=obs['lb_zero_pages']/obs['scanned_entry_pages'],lb_gt_tau_page_ratio=obs['prunable_pages']/obs['scanned_entry_pages'],
                              shadow_full_ids=json.dumps(full_ids),shadow_pruned_ids=json.dumps(pruned_ids),shadow_full_tids=json.dumps([x['tid'] for x in full]),shadow_pruned_tids=json.dumps([x['tid'] for x in pruned]),shadow_full_distances_squared=json.dumps([x['distance_squared'] for x in full]),shadow_pruned_distances_squared=json.dumps([x['distance_squared'] for x in pruned]))
                            for o in ['0','1','2','3+']:
                                r['occupancy_'+o+'_pages']=sum(g['scanned_entry_pages'] for g in obs['page_groups'] if g['occupancy']==o)
                                r['prunable_occupancy_'+o+'_pages']=sum(g['prunable_pages'] for g in obs['page_groups'] if g['occupancy']==o)
                            assert sum(g['scanned_entry_pages'] for g in obs['page_groups'])==obs['scanned_entry_pages']
                            assert sum(g['scanned_candidates'] for g in obs['page_groups'])==obs['scanned_candidates']
                            rows.append(r);correctness.append(record)
                            for g in obs['page_groups']:pages.append(dict(round=1,probes=p,query_id=q,**g))
                            save(root/'progress.json',dict(status='RUNNING',round=1,actual_measured_executions=actual_count,validated_measured_executions=len(rows),expected_measured_executions=600,warmup_executions=warmup_count,last_probes=p,last_query_id=q))
                        if position%20==0:print(f'probes={p} {"warmup" if is_warm else "measured"}={position}/{len(work)} total={len(rows)}/600 mismatches=0',flush=True)
            assert actual_count==len(rows)==600 and warmup_count==120
            assert len({(r['round'],r['probes'],r['query_id']) for r in rows})==600
            assert identity(cur)==manifest['index'];cur.execute('ROLLBACK')
        conn.close()
        for f in handles:f.close()
        result=analyze(root,rows,correctness,pages)
        manifest.update(status='COMPLETE',completed_rounds=[1],actual_measured_executions=600,warmup_executions=120,shadow_mismatch_count=0,decision=result['decision'],finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(root/'phase_d0b_l2_manifest.json',manifest)
        save(root/'progress.json',dict(status='COMPLETE',completed_rounds=[1],actual_measured_executions=600,validated_measured_executions=600,warmup_executions=120,decision=result['decision']))
        save(root/'artifact_hashes.json',{p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name not in ['artifact_hashes.json','experiment.log']})
        print(json.dumps(dict(status='COMPLETE',run=str(root),decision=result['decision'],measured=600,mismatches=0)),flush=True)
    except BaseException as error:
        conn.close()
        for f in handles:f.close()
        manifest.update(status='FAIL',actual_measured_executions=actual_count,validated_measured_executions=len(rows),warmup_executions=warmup_count,error=str(error));save(root/'phase_d0b_l2_manifest.json',manifest)
        save(root/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),no_automatic_retry=True,actual_measured_executions=actual_count))
        raise
if __name__=='__main__':main()
