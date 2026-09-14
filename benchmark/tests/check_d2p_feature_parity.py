"""Offline C/Python feature parity on archived calibration snapshots (no SQL)."""
import argparse
import csv
import ctypes
import json
import math
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import final_multi as m


def run(source,shadow,output):
    code=(source/'src/ivfscan.c').read_text()
    start=code.index('static void\nD2PBuildFeatures(')
    end=code.index('static bool\nD2PShouldStop(',start)
    header=(source/'src/ivfd2ppolicy.h').read_text()
    fields=re.findall(r'^\s*double\s+([A-Za-z_][A-Za-z0-9_]*)\s*;',header,re.MULTILINE)
    if not fields:
        raise RuntimeError('D2PFeatures has no double fields')
    wrapper=r'''
#include <assert.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <string.h>
#define MemSet memset
#define Assert assert
typedef long long ItemPointerData;
#define ItemPointerEquals(a,b) (*(a)==*(b))
typedef struct {double distance; ItemPointerData tid;} Item;
typedef struct {uint64_t candidatesSeen,pagesSeen,replacements; int count; Item *items;} IvfflatShadowSnapshot;
typedef struct {double listDistances[64];IvfflatShadowSnapshot shadowSnapshots[2];} Scan;
typedef Scan *IvfflatScanOpaque;
'''+header+'\n'+code[start:end]+r'''
void compute(double *dist, double *top16, double *top32, double *stats, long long *ids16,long long *ids32,int stage,double *out) {
 Scan scan; Item a[10],b[10]; D2PFeatures features;
 memset(&scan,0,sizeof(scan));
 int offsets[]={0,1,3,7,15,31,63};
 for(int i=0;i<7;i++)scan.listDistances[offsets[i]]=dist[i];
 for(int i=0;i<10;i++){a[i].distance=top16[i];a[i].tid=ids16[i];b[i].distance=top32[i];b[i].tid=ids32[i];}
 scan.shadowSnapshots[0].items=a;scan.shadowSnapshots[1].items=b;
 scan.shadowSnapshots[0].count=10;scan.shadowSnapshots[1].count=10;
 for(int i=0;i<2;i++){scan.shadowSnapshots[i].candidatesSeen=stats[i*3];scan.shadowSnapshots[i].pagesSeen=stats[i*3+1];scan.shadowSnapshots[i].replacements=stats[i*3+2];}
 D2PBuildFeatures(&scan,&features,stage);
'''+''.join(f'out[{i}]=features.{name};\n' for i,name in enumerate(fields))+'}\n'
    output.mkdir(parents=True,exist_ok=True)
    (output/'feature_parity.c').write_text(wrapper)
    subprocess.run(['gcc','-shared','-fPIC','-O2',*m.FLAGS.split(),'-ftree-vectorize','-fassociative-math','-fno-signed-zeros','-fno-trapping-math',str(output/'feature_parity.c'),'-lm','-o',str(output/'feature_parity.so')],check=True)
    lib=ctypes.CDLL(str(output/'feature_parity.so'));double=ctypes.c_double;ll=ctypes.c_longlong
    lib.compute.argtypes=[ctypes.POINTER(double)]*4+[ctypes.POINTER(ll)]*2+[ctypes.c_int,ctypes.POINTER(double)]
    rows=m.calibration.read_csv(shadow);features=m.calibration.feature_rows(rows);maximum=0;checked=0
    def tid(s):
        a,b=s.split('/');return (int(a)<<16)+int(b)
    for row in rows:
        qid=int(row['query_id']);dist=(double*7)(*[float(row['d'+str(i)]) for i in (1,2,4,8,16,32,64)])
        top=[(double*10)(*m.calibration.parse_floats(row[f'stage{s}_topk_distances'])[:10]) for s in (16,32)]
        ids=[(ll*10)(*[tid(x) for x in row[f'stage{s}_topk_tids'].split(';')[:10]]) for s in (16,32)]
        stats=(double*6)(*[float(row[f'stage{s}_{f}']) for s in (16,32) for f in ('candidates','pages','replacements')])
        for stage,available in [(16,m.calibration.STAGE16),(32,m.calibration.STAGE32)]:
            names=[name for name in fields if name in available]
            out=(double*len(fields))();lib.compute(dist,*top,stats,*ids,stage,out)
            for name in names:
                i=fields.index(name)
                a,b=out[i],features[qid][name]
                if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise RuntimeError(f'{qid} {stage} {name}: C={a} Python={b}')
                maximum=max(maximum,abs(a-b));checked+=1
    result={'status':'PASS','snapshots':len(rows),'scalar_checks':checked,'maximum_absolute_error':maximum,
            'source_sha256':m.sha(source/'src/ivfscan.c'),'fixture_sha256':m.sha(shadow),'sql_executions':0}
    m.atomic(output/'result.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--shadow',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.shadow,a.output)
