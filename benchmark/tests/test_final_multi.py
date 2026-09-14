"""Regression checks for comparison isolation and durable checkpoint guards."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import final_multi as m

class FinalMultiTests(unittest.TestCase):
    def test_priority_and_coverage(self):
        work=m.schedule()
        self.assertEqual(len(work),60)
        first=[x for x in work if x['round']==0]
        self.assertEqual([x['priority'] for x in first],sorted(x['priority'] for x in first))
        self.assertEqual({(x['dataset'],x['method']) for x in first[:10]},
                         {(d,k) for d in m.DATASETS for k in ('baseline','2a')})
        self.assertEqual(len({m.block_id(x) for x in work}),len(work))
        for x in work:
            if x['method']=='2a_b':self.assertIn(x['dataset'],('gist','sift'))
            if x['method']=='d':self.assertIn(x['dataset'],('gist','sift'))
            if x['dataset']=='sift' and x['method']=='d':self.assertEqual(x['family'],'sift')
    def test_balanced_orders_and_shared_queries(self):
        work=m.schedule()
        orders=[]
        for r in (1,2,3):
            orders.append([x['method'] for x in work if x['round']==r and x['dataset']=='gist' and x['priority']==1])
            q=m.permutation(m.DATASETS['sift'],r)
            self.assertEqual(set(q),set(range(9000)))
            self.assertEqual(q,m.permutation(m.DATASETS['sift'],r))
        self.assertNotEqual(orders[0],orders[1])
    def test_sift_holdout_never_calibration(self):
        qids,_,_=m.load(m.DATASETS['sift'])
        cal,_,_=m.load(m.DATASETS['sift'],True)
        self.assertEqual(len(qids),9000)
        self.assertFalse(set(qids)&set(cal))
    def test_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=m.plan(Path(d))
        self.assertEqual(p['measured_query_executions'],364000)
        self.assertEqual(p['total_query_executions_max'],388000)
    def test_checkpoint_corruption_rejected_without_queries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);item=m.schedule(True)[0];manifest={'test':True}
            (root/'checkpoints').mkdir();(root/'binaries/production').mkdir(parents=True)
            (root/'binaries/production/vector.so').write_bytes(b'binary')
            bad={'item':item,'manifest_hash':m.jhash(manifest),'rows':[], 'rows_hash':'bad'}
            (root/'checkpoints'/(m.block_id(item)+'.json')).write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError,'Invalid checkpoint'):
                m.run_block(None,root,None,item,manifest,True)
    def test_equivalence_gate(self):
        blocks=[{'item':{'round':0,'family':'production','dataset':'gist','method':k},
                 'rows':[{'qid':1,'result_checksum':v}]} for k,v in [('baseline','a'),('2a','b')]]
        with self.assertRaisesRegex(RuntimeError,'Top10 equivalence'):
            m.equality_gate(blocks,0)
    def test_profile_binary_rejected(self):
        with patch.object(m,'cmd',return_value='IVFFLAT_PROFILE_2B'):
            with self.assertRaisesRegex(RuntimeError,'Profiling'):m.production_check('unused')

if __name__=='__main__':unittest.main()
