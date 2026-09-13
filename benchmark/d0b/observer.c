/* D-0B read-only replay observer. Never replaces vector.so or an index AM.
 * ExecutorEnd captures the actual scan's selected-list order and bounds.
 * A separate SQL function replays those pages without affecting real results.
 */
#include "postgres.h"
#include <float.h>
#include <math.h>
#include <fenv.h>
#include "utils/lsyscache.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "access/itup.h"
#include "access/relscan.h"
#include "executor/executor.h"
#include "lib/stringinfo.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/memutils.h"
#include "ivfflat.h"
PG_MODULE_MAGIC;
void _PG_init(void);
Datum d0_observe(PG_FUNCTION_ARGS);
Datum d0_selftest(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(d0_observe);
PG_FUNCTION_INFO_V1(d0_selftest);
static ExecutorEnd_hook_type previous_end;
static BlockNumber actual_lists[1000];
static int actual_n, actual_probes;
static Oid actual_index;
static int64 logical_bound, physical_bound;
static bool bounded, direct_active, fused_active, fallback;
static uint64 scan_serial;
static Vector *captured_query;
typedef struct { double d; ItemPointerData tid; } Candidate;
typedef struct { Candidate items[10]; int n; } TopK;
typedef struct { double lo,hi,mid; } D0Interval;
typedef struct { bool valid; BlockNumber list; int n; double lo,hi,min_mid,max_mid; D0Interval *radii; } Metadata;
static Metadata *metadata;
static BlockNumber metadata_blocks;
static Oid metadata_index;
static int compare(const Candidate *a, const Candidate *b)
{
 if(a->d<b->d) return -1;
 if(a->d>b->d) return 1;
 return ItemPointerCompare((ItemPointer)&a->tid,(ItemPointer)&b->tid);
}
static void insert(TopK *t, Candidate x)
{
 int i=t->n;
 if(i==10) { if(compare(&x,&t->items[9])>=0) return; i=9; }
 else t->n++;
 while(i>0 && compare(&x,&t->items[i-1])<0) { t->items[i]=t->items[i-1]; i--; }
 t->items[i]=x;
}
static double tau(const TopK *t) { return t->n==10?t->items[9].d:INFINITY; }

/* Strict scalar binary64 arithmetic; no custom production distance kernel.
 * All actual distances below still call the existing index support function.
 */
static double down(double x) { return nextafter(x,-INFINITY); }
static double up(double x) { return nextafter(x,INFINITY); }
static double gamma_up(double u) { double z=(3*960+8)*u; return up(z/(1-z)); }
static void domain(Vector *v)
{
 if(v->dim!=960) elog(ERROR,"D0B dimension mismatch");
 for(int i=0;i<v->dim;i++) {
  double x=fabs((double)v->x[i]);
  if(!isfinite(x)||x>2.0||(x!=0 && x<0x1p-32)) elog(ERROR,"D0B numeric domain outside certificate");
 }
}
static D0Interval norm_interval(Vector *a,Vector *b)
{
 double sum=0,g=gamma_up(DBL_EPSILON/2); D0Interval r;
 domain(a);domain(b);
 for(int i=0;i<960;i++) { double d=(double)a->x[i]-(double)b->x[i];sum+=d*d; }
 if(sum==0) { r.lo=r.hi=r.mid=0;return r; }
 r.mid=sqrt(sum);
 r.lo=fmax(0,down(sqrt(fmax(0,down(sum/up(1+g))))));
 r.hi=up(sqrt(up(sum/down(1-g))));
 return r;
}
static double radial_lb(double rmin_lo,double rmax_hi,D0Interval dc)
{ return fmax(0,fmax(down(rmin_lo-dc.hi),down(dc.lo-rmax_hi))); }
static double key_lb(double radius)
{
 double g=gamma_up(FLT_EPSILON/2);
 if(radius<=0)return 0;
 return fmax(0,down(down(radius*radius)*down(1-g)));
}
static bool can_prune(double lb,double threshold) { return isfinite(threshold)&&lb>threshold; }
static void capture(PlanState *ps)
{
 if(!ps) return;
 if(IsA(ps,IndexScanState)) {
  IndexScanState *is=(IndexScanState*)ps;
  if(is->iss_ScanDesc && is->iss_ScanDesc->opaque && is->iss_RelationDesc && strcmp(RelationGetRelationName(is->iss_RelationDesc),"gist_ivf_l2")==0) {
   IvfflatScanOpaque so=(IvfflatScanOpaque)is->iss_ScanDesc->opaque;
   if(so->maxProbes>1000 || so->listIndex>1000) elog(ERROR,"D0 unexpected maxProbes");
   actual_n=so->listIndex; actual_probes=so->probes;
   memcpy(actual_lists,so->listPages,sizeof(BlockNumber)*actual_n);
   actual_index=RelationGetRelid(is->iss_RelationDesc);
   logical_bound=so->logicalBound; physical_bound=so->physicalBound;
   bounded=so->boundedActive; fallback=so->fallbackTriggered;
   direct_active=so->useDirectL2; fused_active=so->useFused2; scan_serial++;
   if(DatumGetPointer(so->value)==NULL) elog(ERROR,"D0B NULL scan value");
   if(!captured_query) captured_query=MemoryContextAlloc(TopMemoryContext,VECTOR_SIZE(960));
   memcpy(captured_query,DatumGetPointer(so->value),VECTOR_SIZE(960));
  }
 }
 capture(outerPlanState(ps)); capture(innerPlanState(ps));
}
static void end_hook(QueryDesc *qd)
{
 capture(qd->planstate);
 if(previous_end) previous_end(qd); else standard_ExecutorEnd(qd);
}
void _PG_init(void) { previous_end=ExecutorEnd_hook; ExecutorEnd_hook=end_hook; }
static void top_json(StringInfo out,TopK *t)
{
 appendStringInfoChar(out,'[');
 for(int i=0;i<t->n;i++) {
  Candidate *c=&t->items[i];
  appendStringInfo(out,"%s{\"tid\":\"(%u,%u)\",\"distance_squared\":%.17g}",i?",":"",ItemPointerGetBlockNumber(&c->tid),ItemPointerGetOffsetNumber(&c->tid),c->d);
 }
 appendStringInfoChar(out,']');
}

Datum d0_selftest(PG_FUNCTION_ARGS)
{
 TopK a={0},b={0}; Candidate x; D0Interval dc={0,0,0}; double lb=key_lb(radial_lb(5,5,dc));
 for(int i=19;i>=0;i--) { x.d=1;ItemPointerSet(&x.tid,1,i+1);insert(&a,x); }
 if(a.n!=10 || ItemPointerGetOffsetNumber(&a.items[0].tid)!=1 || ItemPointerGetOffsetNumber(&a.items[9].tid)!=10)elog(ERROR,"D0B tie comparator selftest FAIL");
 if(fegetround()!=FE_TONEAREST || DBL_MANT_DIG!=53 || FLT_MANT_DIG!=24)elog(ERROR,"D0B rounding environment unsupported");
 if(!(lb>4) || lb>25 || can_prune(lb,lb) || can_prune(lb,INFINITY))elog(ERROR,"D0B boundary selftest FAIL");
 dc.lo=dc.hi=dc.mid=3;if(radial_lb(1,5,dc)!=0)elog(ERROR,"D0B interval zero selftest FAIL");
 b=a;x.d=100;ItemPointerSet(&x.tid,9,1);insert(&a,x);
 for(int i=0;i<10;i++)if(compare(&a.items[i],&b.items[i])!=0)elog(ERROR,"D0B synthetic pruning selftest FAIL");
 PG_RETURN_TEXT_P(cstring_to_text("PASS: strict equality retained; infinity retained; distance/TID order; zero LB; conservative squared bound; synthetic nonzero pruning"));
}
/* Each block is reported by query/list/occupancy. Aggregates partition all page
 * visits exactly; no page samples or latency estimates enter potential gates.
 */
enum { PAGES,CANDS,BEFORE,AFTER,PRUNED_P,PRUNED_C,LBZERO,INSIDE,INSIDE_READY,
 POSITIVE_INSUFFICIENT,GUARD_REJECTED,WIDTH_GE_TAU,SINGLE_PRUNED_C,
 SINGLE_ALL_PAGE_RETAINED,ORACLE_P,ORACLE_C,NCOUNTS };
static const char *names[NCOUNTS]={"scanned_entry_pages","scanned_candidates","pages_before_topk_ready","pages_after_topk_ready","prunable_pages","prunable_candidates","lb_zero_pages","dc_inside_interval_pages","dc_inside_interval_ready_pages","positive_lb_insufficient_pages","numeric_guard_rejected_pages","width_ge_tau_pages","singleton_bound_prunable_candidates","singleton_all_prunable_page_retained","oracle_min_distance_prunable_pages","oracle_min_distance_prunable_candidates"};
typedef struct {
 uint64 count[NCOUNTS];
 double rmin_sum,rmax_sum,width_sum,lb_sum,dc_sum,tau_sum;
 double rmin_min,rmin_max,rmax_min,rmax_max,width_max,min_gap;
} Stats;
static void initstats(Stats *s) { memset(s,0,sizeof(*s));s->rmin_min=s->rmax_min=s->min_gap=DBL_MAX; }
static void addstats(Stats *to,const Stats *s)
{
 for(int i=0;i<NCOUNTS;i++)to->count[i]+=s->count[i];
 to->rmin_sum+=s->rmin_sum;to->rmax_sum+=s->rmax_sum;to->width_sum+=s->width_sum;to->lb_sum+=s->lb_sum;to->dc_sum+=s->dc_sum;to->tau_sum+=s->tau_sum;
 to->rmin_min=fmin(to->rmin_min,s->rmin_min);to->rmin_max=fmax(to->rmin_max,s->rmin_max);
 to->rmax_min=fmin(to->rmax_min,s->rmax_min);to->rmax_max=fmax(to->rmax_max,s->rmax_max);
 to->width_max=fmax(to->width_max,s->width_max);to->min_gap=fmin(to->min_gap,s->min_gap);
}
static void statsjson(StringInfo o,Stats *s)
{
 double n=(double)s->count[PAGES];
 for(int i=0;i<NCOUNTS;i++)appendStringInfo(o,"\"%s\":%llu,",names[i],(unsigned long long)s->count[i]);
 appendStringInfo(o,"\"rmin_mean\":%.17g,\"rmin_min\":%.17g,\"rmin_max\":%.17g,\"rmax_mean\":%.17g,\"rmax_min\":%.17g,\"rmax_max\":%.17g,\"residual_width_mean\":%.17g,\"residual_width_max\":%.17g,\"lb_radius_mean\":%.17g,\"dc_mean\":%.17g,\"tau_radius_mean_when_ready\":%.17g,\"minimum_exact_squared_minus_safe_lb_squared\":%.17g",s->rmin_sum/n,s->rmin_min,s->rmin_max,s->rmax_sum/n,s->rmax_min,s->rmax_max,s->width_sum/n,s->width_max,s->lb_sum/n,s->dc_sum/n,s->count[AFTER]?s->tau_sum/s->count[AFTER]:0,s->min_gap==DBL_MAX?0:s->min_gap);
}
Datum d0_observe(PG_FUNCTION_ARGS)
{
 Oid idx=PG_GETARG_OID(0);Vector *q=(Vector*)PG_DETOAST_DATUM(PG_GETARG_DATUM(1));int probes=PG_GETARG_INT32(2);
 Relation rel;FmgrInfo *proc;TupleDesc desc;BlockNumber blk;Vector **centers;
 TopK full={0},sim={0};StringInfoData out;Stats total,*groups;uint64 serial=scan_serial;
 if(fegetround()!=FE_TONEAREST)elog(ERROR,"D0B rounding mode changed");
 domain(q);
 if(!captured_query || memcmp(q->x,captured_query->x,960*sizeof(float)))elog(ERROR,"D0B actual/replay query mismatch");
 if(idx!=actual_index || actual_n!=probes || actual_probes!=probes || logical_bound!=10 || physical_bound!=40 || !bounded || direct_active || fused_active || fallback)elog(ERROR,"D0B actual path/bounds mismatch");
 rel=index_open(idx,AccessShareLock);desc=RelationGetDescr(rel);proc=index_getprocinfo(rel,1,IVFFLAT_DISTANCE_PROC);
 if(strcmp(get_func_name(proc->fn_oid),"vector_l2_squared_distance"))elog(ERROR,"D0B non-L2 support function");
 blk=RelationGetNumberOfBlocks(rel);
 if(!metadata){metadata_blocks=blk;metadata_index=idx;metadata=MemoryContextAllocZero(TopMemoryContext,sizeof(Metadata)*blk);}
 if(metadata_blocks!=blk || metadata_index!=idx)elog(ERROR,"D0B index metadata identity changed");
 centers=palloc0(sizeof(Vector*)*blk);groups=palloc(sizeof(Stats)*probes*4);initstats(&total);
 for(int i=0;i<probes*4;i++)initstats(&groups[i]);
 blk=IVFFLAT_HEAD_BLKNO;
 while(BlockNumberIsValid(blk)){
  Buffer buf=ReadBuffer(rel,blk);Page page;LockBuffer(buf,BUFFER_LOCK_SHARE);page=BufferGetPage(buf);
  for(OffsetNumber off=FirstOffsetNumber;off<=PageGetMaxOffsetNumber(page);off++){
   IvfflatList l=(IvfflatList)PageGetItem(page,PageGetItemId(page,off));
   for(int j=0;j<actual_n;j++)if(l->startPage==actual_lists[j]){centers[l->startPage]=palloc(VARSIZE_ANY(&l->center));memcpy(centers[l->startPage],&l->center,VARSIZE_ANY(&l->center));break;}
  }
  blk=IvfflatPageGetOpaque(page)->nextblkno;UnlockReleaseBuffer(buf);
 }
 for(int li=0;li<actual_n;li++){
  BlockNumber start=actual_lists[li];Vector *center=centers[start];D0Interval dc;
  if(!center)elog(ERROR,"D0B missing centroid");
  dc=norm_interval(q,center);blk=start;
  while(BlockNumberIsValid(blk)){
   Buffer buf;Page page;int n=0,single=0;Candidate items[MaxIndexTuplesPerPage];D0Interval radii[MaxIndexTuplesPerPage];Metadata *m;
   double page_tau=tau(&sim),full_tau=tau(&full),lb,lk,rawlb,page_min=DBL_MAX;bool skip;Stats one;
   if(blk>=metadata_blocks)elog(ERROR,"D0B entry page beyond index");
   m=&metadata[blk];
   CHECK_FOR_INTERRUPTS();buf=ReadBuffer(rel,blk);LockBuffer(buf,BUFFER_LOCK_SHARE);page=BufferGetPage(buf);
   if(!m->valid){m->lo=m->min_mid=DBL_MAX;m->hi=m->max_mid=0;m->list=start;}
   if(m->list!=start)elog(ERROR,"D0B metadata belongs to another centroid");
   for(OffsetNumber off=FirstOffsetNumber;off<=PageGetMaxOffsetNumber(page);off++){
    IndexTuple it=(IndexTuple)PageGetItem(page,PageGetItemId(page,off));bool isnull;Datum datum=index_getattr(it,1,desc,&isnull);
    if(isnull||n>=MaxIndexTuplesPerPage)elog(ERROR,"D0B unexpected candidate representation");
    items[n].tid=it->t_tid;items[n].d=DatumGetFloat8(FunctionCall2Coll(proc,rel->rd_indcollation[0],datum,PointerGetDatum(q)));
    if(!isfinite(items[n].d)||items[n].d<0)elog(ERROR,"D0B invalid actual support distance");
    page_min=fmin(page_min,items[n].d);
    if(!m->valid){Vector *v=(Vector*)PG_DETOAST_DATUM(datum);radii[n]=norm_interval(v,center);m->lo=fmin(m->lo,radii[n].lo);m->hi=fmax(m->hi,radii[n].hi);m->min_mid=fmin(m->min_mid,radii[n].mid);m->max_mid=fmax(m->max_mid,radii[n].mid);if((Pointer)v!=DatumGetPointer(datum))pfree(v);}
    n++;
   }
   if(!m->valid){m->n=n;if(n){m->radii=MemoryContextAlloc(TopMemoryContext,n*sizeof(D0Interval));memcpy(m->radii,radii,n*sizeof(D0Interval));}else m->lo=m->min_mid=0;m->valid=true;}
   if(m->n!=n)elog(ERROR,"D0B page changed during locked validation");
   lb=radial_lb(m->lo,m->hi,dc);lk=key_lb(lb);rawlb=fmax(0,fmax(m->min_mid-dc.mid,dc.mid-m->max_mid));
   skip=n>0&&can_prune(lk,page_tau);
   /* The entire actual page is checked, but this oracle never chooses skip. */
   if(n && lk>page_min)elog(ERROR,"D0B correctness FAIL: lower bound violates page exact distances");
   if(can_prune(lk,full_tau)!=skip && n)elog(ERROR,"D0B full/pruned page-entry decisions differ");
   for(int i=0;i<n;i++){
    double sl=key_lb(radial_lb(m->radii[i].lo,m->radii[i].hi,dc));
    if(sl>items[i].d)elog(ERROR,"D0B singleton bound correctness FAIL");
    if(can_prune(sl,page_tau))single++;
   }
   initstats(&one);one.count[PAGES]=1;one.count[CANDS]=n;
   if(isfinite(page_tau)){one.count[AFTER]=1;one.tau_sum=sqrt(page_tau);if(m->hi-m->lo>=sqrt(page_tau))one.count[WIDTH_GE_TAU]=1;}else one.count[BEFORE]=1;
   if(lb==0)one.count[LBZERO]=1;
   if(dc.mid>=m->min_mid&&dc.mid<=m->max_mid){one.count[INSIDE]=1;if(isfinite(page_tau))one.count[INSIDE_READY]=1;}
   if(lb>0&&isfinite(page_tau)&&!skip)one.count[POSITIVE_INSUFFICIENT]=1;
   if(rawlb*rawlb>page_tau&&!skip)one.count[GUARD_REJECTED]=1;
   if(skip){one.count[PRUNED_P]=1;one.count[PRUNED_C]=n;}
   one.count[SINGLE_PRUNED_C]=single;
   if(n&&single==n&&!skip)one.count[SINGLE_ALL_PAGE_RETAINED]=1;
   if(n&&isfinite(page_tau)&&page_min>page_tau){one.count[ORACLE_P]=1;one.count[ORACLE_C]=n;}
   one.rmin_sum=one.rmin_min=one.rmin_max=m->lo;one.rmax_sum=one.rmax_min=one.rmax_max=m->hi;
   one.width_sum=one.width_max=m->hi-m->lo;one.lb_sum=lb;one.dc_sum=dc.mid;if(n)one.min_gap=page_min-lk;
   addstats(&total,&one);addstats(&groups[li*4+Min(n,3)],&one);
   /* page_tau was captured before any item on this page enters either heap. */
   for(int i=0;i<n;i++){insert(&full,items[i]);if(!skip)insert(&sim,items[i]);}
   if(full.n!=sim.n)elog(ERROR,"D0B shadow count mismatch");
   for(int i=0;i<full.n;i++)if(compare(&full.items[i],&sim.items[i]))elog(ERROR,"D0B shadow Top10 mismatch");
   blk=IvfflatPageGetOpaque(page)->nextblkno;UnlockReleaseBuffer(buf);
  }
 }
 initStringInfo(&out);appendStringInfo(&out,"{\"scan_serial\":%llu,\"selected_lists\":%d,\"logical_bound\":%lld,\"physical_bound\":%lld,\"bounded_active\":true,\"direct_active\":false,\"fused2_active\":false,\"shadow_mismatch\":0,",(unsigned long long)serial,actual_n,(long long)logical_bound,(long long)physical_bound);
 statsjson(&out,&total);appendStringInfoString(&out,",\"selected_list_start_pages\":[");
 for(int i=0;i<actual_n;i++)appendStringInfo(&out,"%s%u",i?",":"",actual_lists[i]);
 appendStringInfoString(&out,"],\"shadow_full_top10\":");top_json(&out,&full);appendStringInfoString(&out,",\"shadow_pruned_top10\":");top_json(&out,&sim);
 appendStringInfoString(&out,",\"page_groups\":[");
 {bool comma=false;for(int li=0;li<actual_n;li++)for(int o=0;o<4;o++){
  Stats *s=&groups[li*4+o];if(!s->count[PAGES])continue;
  appendStringInfo(&out,"%s{\"list_rank\":%d,\"list_start_page\":%u,\"list_half\":\"%s\",\"occupancy\":\"%s\",",comma?",":"",li+1,actual_lists[li],li<actual_n/2?"first":"second",o==3?"3+":o==2?"2":o==1?"1":"0");statsjson(&out,s);appendStringInfoChar(&out,'}');comma=true;
 }}
 appendStringInfoString(&out,"]}");index_close(rel,AccessShareLock);PG_RETURN_TEXT_P(cstring_to_text(out.data));
}
