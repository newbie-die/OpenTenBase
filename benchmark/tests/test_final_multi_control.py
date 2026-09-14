import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import final_multi_control as c

class FinalRoundControlTests(unittest.TestCase):
    def control(self):
        a=[]
        for d in ('gist','sift','glove-cosine','cohere-cosine','glove-ip'):
            a.extend([{'round':1,'dataset':d,'method':'baseline','family':'production'},
                      {'round':1,'dataset':d,'method':'2a','family':'production'}])
        a += [{'round':1,'dataset':d,'method':'2a_b','family':'production'} for d in ('gist','sift')]
        a += [{'round':1,'dataset':'gist','method':'d','family':'production'},
              {'round':1,'dataset':'sift','method':'d','family':'sift'}]
        b=[{**x,'round':2} for x in a[5:]+a[:5]]
        return {'final_round_a':1,'final_round_b':2,'round_a_schedule':a,'round_b_schedule':b,'schedule':a+b,
                'round_a_order':[{'dataset':x['dataset'],'method':x['method']} for x in a],
                'round_b_order':[{'dataset':x['dataset'],'method':x['method']} for x in b]}
    def test_exact_14_unique_configs_and_rotation(self):
        x=self.control();self.assertEqual(len(x['schedule']),28)
        self.assertEqual(len({(i['dataset'],i['method']) for i in x['round_a_schedule']}),14)
        self.assertNotEqual(x['round_a_order'],x['round_b_order'])
        self.assertEqual({i['round'] for i in x['round_b_schedule']},{2})
    def test_duplicate_sift_baseline_classified_unused(self):
        x=self.control()
        self.assertEqual(c.disposition({'round':1,'dataset':'sift','method':'baseline','family':'sift'},x),'DUPLICATE_UNUSED')
        self.assertEqual(c.disposition({'round':0,'dataset':'gist','method':'baseline','family':'production'},x),'HISTORICAL_UNUSED')
        self.assertEqual(c.disposition(x['round_a_schedule'][0],x),'USED_FINAL_ROUND_A')
    def test_progress_uses_28_and_164000_target(self):
        x=self.control()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoints').mkdir()
            item=x['round_a_schedule'][0]
            row={'item':item,'rows':[{},{}]}
            (root/'checkpoints'/'r1_gist_production_baseline.json').write_text(json.dumps(row))
            p=c.update_progress(root,x)
            self.assertEqual(p['effective_total_config_rounds'],28)
            self.assertEqual(p['final_round_a_completed_configs'],1)
            self.assertEqual(p['final_round_b_completed_configs'],0)
            self.assertEqual(p['historical_completed_config_rounds'],0)
            self.assertEqual(p['remaining_effective_formal_queries'],164000-2)

    def write_block(self, root, item, index, round_offset=0):
        family=item['family'];dataset=item['dataset'];method=item['method'];round_id=item['round']
        latency=100.0+index*7+round_offset
        recall=0.90+(index%5)*0.01+round_offset/10000
        summary={'queries':1,'mean_us':latency,'p50_us':latency-1,'p95_us':latency+2,
                 'p99_us':latency+3,'qps':1e6/latency,'recall_at_10':recall,'avg_probes':32}
        row={'dataset':dataset,'method':method,'family':family,'round':round_id,'position':0,
             'qid':100+index,'latency_us':latency,'recall_at_10':recall,'result_ids':[1,2],
             'result_checksum':'checksum','binary_sha256':'binary'}
        block={'item':item,'rows':[row],'summary':summary}
        name=f"r{round_id}_{dataset}_{family}_{method}.json"
        (root/'checkpoints'/name).write_text(json.dumps(block))
        return block

    def test_partial_round_two_aggregation_is_none_safe(self):
        control=self.control()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoints').mkdir()
            for i,item in enumerate(control['round_a_schedule']):
                self.write_block(root,item,i)
            fused_sift=next(x for x in control['round_b_schedule']
                            if x['dataset']=='sift' and x['method']=='2a_b')
            self.write_block(root,fused_sift,20,round_offset=5)

            progress=c.refresh_outputs(root,control,status='RUNNING')
            result=json.loads((root/'final_comparison.json').read_text())
            self.assertEqual(result['status'],'RUNNING')
            self.assertEqual(len(result['final_comparison']),14)
            self.assertEqual(progress['final_round_a_completed_configs'],14)
            self.assertEqual(progress['final_round_b_completed_configs'],1)
            self.assertEqual(len(result['missing_configs']),13)
            fused=next(x for x in result['final_comparison']
                       if x['dataset']=='sift' and x['method']=='2a_b')
            self.assertIsNotNone(fused['round_a_speedup'])
            self.assertIsNone(fused['round_b_speedup'])
            self.assertEqual(fused['mean_speedup'],fused['round_a_speedup'])
            self.assertIsNone(fused['std_speedup'])
            baseline=next(x for x in result['final_comparison']
                          if x['dataset']=='gist' and x['method']=='baseline')
            for metric in ('recall_at_10','mean_us','p50_us','p95_us','p99_us','qps'):
                self.assertEqual(baseline['mean_'+metric],baseline['round_a_'+metric])
                self.assertIsNone(baseline['round_b_'+metric])
                self.assertIsNone(baseline['std_'+metric])
            self.assertIsNone(baseline['round_b_queries'])
            sift_d=next(x for x in result['final_comparison']
                        if x['dataset']=='sift' and x['method']=='d')
            self.assertIsNone(sift_d['round_b_recall_delta'])
            self.assertEqual(sift_d['mean_recall_delta'],sift_d['round_a_recall_delta'])
            self.assertIsNone(result['geomean_speedups']['2a_b']['round_b'])
            self.assertEqual(progress['completed_effective_config_rounds'],15)

    def test_complete_round_pair_mean_std_comparison_and_recall_delta(self):
        control=self.control()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoints').mkdir()
            for i,item in enumerate(control['round_a_schedule']):
                self.write_block(root,item,i)
            for i,item in enumerate(control['round_b_schedule']):
                self.write_block(root,item,i,round_offset=10)
            (root/'manifest.json').write_text('{}')

            progress=c.refresh_outputs(root,control,status='RUNNING')
            result=json.loads((root/'final_comparison.json').read_text())
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(result['missing_configs'],[])
            self.assertEqual(len(result['final_comparison']),14)
            self.assertEqual(progress['final_round_a_completed_configs'],14)
            self.assertEqual(progress['final_round_b_completed_configs'],14)
            self.assertEqual(progress['status'],'COMPLETE')
            gist=next(x for x in result['final_comparison']
                      if x['dataset']=='gist' and x['method']=='baseline')
            a=json.loads((root/'checkpoints/r1_gist_production_baseline.json').read_text())['summary']['mean_us']
            b=json.loads((root/'checkpoints/r2_gist_production_baseline.json').read_text())['summary']['mean_us']
            self.assertEqual(gist['mean_mean_us'],statistics.fmean((a,b)))
            self.assertEqual(gist['std_mean_us'],statistics.stdev((a,b)))
            for metric in ('recall_at_10','mean_us','p50_us','p95_us','p99_us','qps'):
                self.assertEqual(gist['mean_'+metric],statistics.fmean((
                    gist['round_a_'+metric],gist['round_b_'+metric])))
                self.assertEqual(gist['std_'+metric],statistics.stdev((
                    gist['round_a_'+metric],gist['round_b_'+metric])))
            sift_d=next(x for x in result['final_comparison']
                        if x['dataset']=='sift' and x['method']=='d')
            self.assertIsNotNone(sift_d['mean_recall_delta'])
            self.assertIsNotNone(result['geomean_speedups']['2a']['mean'])

    def test_complete_status_rejects_missing_round_without_final_comparison(self):
        control=self.control()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoints').mkdir()
            for i,item in enumerate(control['round_a_schedule']):
                self.write_block(root,item,i)
            with self.assertRaisesRegex(RuntimeError,'Cannot finalize two-round comparison'):
                c.refresh_outputs(root,control,status='COMPLETE')
            self.assertFalse((root/'final_comparison.json').exists())
            self.assertFalse((root/'final_comparison.csv').exists())

if __name__=='__main__':unittest.main()
