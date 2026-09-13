/* Phase D1-1 read-only progressive-distance shadow observer.
 *
 * This module never replaces vector.so and never changes an index AM.  An
 * ExecutorEnd hook captures the list order and bounds used by the real query;
 * d1_observe() then replays those entry pages under the runner's stable
 * snapshot.  Every candidate first receives the unchanged support-proc full
 * distance.  The progressive work is a separate, hindsight-only simulation.
 */
#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/itup.h"
#include "access/relscan.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "ivfflat.h"

PG_MODULE_MAGIC;

void _PG_init(void);
Datum d1_observe(PG_FUNCTION_ARGS);
Datum d1_selftest(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(d1_observe);
PG_FUNCTION_INFO_V1(d1_selftest);

#define D1_DIM 960
#define D1_MAX_LISTS 1000
#define D1_CONFIGS 8
#define D1_BUCKETS 7
#define D1_ANCHORS 14

static const int capacities[D1_CONFIGS] = {10, 10, 10, 10, 40, 40, 40, 40};
static const int block_sizes[D1_CONFIGS] = {32, 64, 128, 256, 32, 64, 128, 256};
static const char *config_names[D1_CONFIGS] = {
	"k10_b32", "k10_b64", "k10_b128", "k10_b256",
	"k40_b32", "k40_b64", "k40_b128", "k40_b256"
};
static const uint64 anchors[D1_ANCHORS] = {
	10, 40, 80, 200, 500, 1000, 2000, 5000, 10000, 20000,
	50000, 100000, 200000, 400000
};

static ExecutorEnd_hook_type previous_end;
static BlockNumber actual_lists[D1_MAX_LISTS];
static int actual_n;
static int actual_probes;
static Oid actual_index;
static int64 logical_bound;
static int64 physical_bound;
static bool bounded;
static bool direct_active;
static bool fused_active;
static bool fallback;
static uint64 scan_serial;
static Vector *captured_query;

typedef struct Candidate
{
	double d;
	ItemPointerData tid;
} Candidate;

typedef struct TopN
{
	Candidate items[40];
	int n;
	int cap;
} TopN;

typedef struct ConfigStats
{
	uint64 total;
	uint64 before_ready;
	uint64 early;
	uint64 evaluated_dims;
	uint64 avoided_dims;
	uint64 threshold_checks;
	uint64 eval_hist[D1_DIM + 1];
	uint64 abort_hist[D1_BUCKETS];
	uint64 half_total[2];
	uint64 half_early[2];
	uint64 half_avoided[2];
	double anchor_tau[D1_ANCHORS];
	bool anchor_seen[D1_ANCHORS];
} ConfigStats;

static int
candidate_compare(const Candidate *a, const Candidate *b)
{
	if (a->d < b->d)
		return -1;
	if (a->d > b->d)
		return 1;
	return ItemPointerCompare((ItemPointer) &a->tid, (ItemPointer) &b->tid);
}

static void
topn_init(TopN *top, int capacity)
{
	memset(top, 0, sizeof(*top));
	top->cap = capacity;
}

static void
topn_insert(TopN *top, Candidate candidate)
{
	int i = top->n;

	if (i == top->cap)
	{
		if (candidate_compare(&candidate, &top->items[top->cap - 1]) >= 0)
			return;
		i = top->cap - 1;
	}
	else
		top->n++;

	while (i > 0 && candidate_compare(&candidate, &top->items[i - 1]) < 0)
	{
		top->items[i] = top->items[i - 1];
		i--;
	}
	top->items[i] = candidate;
}

static bool
topn_ready(const TopN *top)
{
	return top->n == top->cap;
}

static double
topn_tau(const TopN *top)
{
	return topn_ready(top) ? top->items[top->cap - 1].d : INFINITY;
}

static int
eval_bucket(int evaluated)
{
	if (evaluated <= 32)
		return 0;
	if (evaluated <= 64)
		return 1;
	if (evaluated <= 128)
		return 2;
	if (evaluated <= 256)
		return 3;
	if (evaluated <= 512)
		return 4;
	if (evaluated <= 768)
		return 5;
	return 6;
}

/* Float inputs are exactly representable in binary64.  A float-float
 * subtraction and its square fit in binary64 precision; stepping one ulp down
 * after every positive addition supplies a real-partial lower enclosure.
 * The extra factor is the deliberately conservative, GIST1M-only bridge used
 * by this potential study.  Every resulting key is checked against the already
 * known production full score before it may drive a shadow rejection.
 */
static double
down(double value)
{
	return nextafter(value, -INFINITY);
}

static double
gamma_bound(void)
{
	double u = FLT_EPSILON / 2.0;
	double z = (3.0 * D1_DIM + 8.0) * u;
	return nextafter(z / (1.0 - z), INFINITY);
}

static double
certified_key(double partial_lo)
{
	double factor = down(1.0 - gamma_bound());
	return fmax(0.0, down(partial_lo * factor));
}

static void
check_vector(Vector *vector)
{
	int i;

	if (vector->dim != D1_DIM)
		elog(ERROR, "D1-1 dimension mismatch");
	for (i = 0; i < D1_DIM; i++)
	{
		double x = fabs((double) vector->x[i]);
		if (!isfinite(x) || x > 2.0 || (x != 0.0 && x < 0x1p-32))
			elog(ERROR, "D1-1 vector outside GIST1M numerical certificate");
	}
}

static void
capture_scan(PlanState *planstate)
{
	if (planstate == NULL)
		return;
	if (IsA(planstate, IndexScanState))
	{
		IndexScanState *indexstate = (IndexScanState *) planstate;

		if (indexstate->iss_ScanDesc && indexstate->iss_ScanDesc->opaque &&
			indexstate->iss_RelationDesc &&
			strcmp(RelationGetRelationName(indexstate->iss_RelationDesc),
				   "gist_ivf_l2") == 0)
		{
			IvfflatScanOpaque so = (IvfflatScanOpaque) indexstate->iss_ScanDesc->opaque;

			if (so->maxProbes > D1_MAX_LISTS || so->listIndex > D1_MAX_LISTS)
				elog(ERROR, "D1-1 unexpected selected-list count");
			actual_n = so->listIndex;
			actual_probes = so->probes;
			memcpy(actual_lists, so->listPages, sizeof(BlockNumber) * actual_n);
			actual_index = RelationGetRelid(indexstate->iss_RelationDesc);
			logical_bound = so->logicalBound;
			physical_bound = so->physicalBound;
			bounded = so->boundedActive;
			fallback = so->fallbackTriggered;
			direct_active = so->useDirectL2;
			fused_active = so->useFused2;
			scan_serial++;
			if (DatumGetPointer(so->value) == NULL)
				elog(ERROR, "D1-1 NULL captured query");
			if (captured_query == NULL)
				captured_query = MemoryContextAlloc(TopMemoryContext,
												 VECTOR_SIZE(D1_DIM));
			memcpy(captured_query, DatumGetPointer(so->value), VECTOR_SIZE(D1_DIM));
		}
	}
	capture_scan(outerPlanState(planstate));
	capture_scan(innerPlanState(planstate));
}

static void
executor_end(QueryDesc *querydesc)
{
	capture_scan(querydesc->planstate);
	if (previous_end)
		previous_end(querydesc);
	else
		standard_ExecutorEnd(querydesc);
}

void
_PG_init(void)
{
	previous_end = ExecutorEnd_hook;
	ExecutorEnd_hook = executor_end;
}

static void
append_top10(StringInfo output, const TopN *top)
{
	int i;

	appendStringInfoChar(output, '[');
	for (i = 0; i < Min(top->n, 10); i++)
	{
		const Candidate *candidate = &top->items[i];
		appendStringInfo(output,
						 "%s{\"tid\":\"(%u,%u)\",\"distance_squared\":%.17g}",
						 i ? "," : "",
						 ItemPointerGetBlockNumber(&candidate->tid),
						 ItemPointerGetOffsetNumber(&candidate->tid), candidate->d);
	}
	appendStringInfoChar(output, ']');
}

static const char *
key_relation(const Candidate *candidate, const TopN *top)
{
	int cmp;

	if (!topn_ready(top))
		return "not_ready";
	cmp = candidate_compare(candidate, &top->items[top->cap - 1]);
	return cmp < 0 ? "better" : cmp > 0 ? "worse" : "equal";
}

static int
quantile_from_hist(const uint64 *hist, uint64 total, double quantile)
{
	uint64 target = (uint64) ceil(quantile * (double) total);
	uint64 cumulative = 0;
	int i;

	if (target == 0)
		target = 1;
	for (i = 0; i <= D1_DIM; i++)
	{
		cumulative += hist[i];
		if (cumulative >= target)
			return i;
	}
	return D1_DIM;
}

Datum
d1_selftest(PG_FUNCTION_ARGS)
{
	TopN top;
	Candidate candidate;
	double partial = 0.0;
	int i;

	topn_init(&top, 10);
	for (i = 19; i >= 0; i--)
	{
		candidate.d = 1.0;
		ItemPointerSet(&candidate.tid, 1, i + 1);
		topn_insert(&top, candidate);
	}
	if (top.n != 10 || ItemPointerGetOffsetNumber(&top.items[0].tid) != 1 ||
		ItemPointerGetOffsetNumber(&top.items[9].tid) != 10)
		elog(ERROR, "D1-1 distance/TID selftest failed");
	for (i = 0; i < D1_DIM; i++)
		partial = down(partial + 1.0);
	if (!(certified_key(partial) > 0.0) || certified_key(partial) > partial)
		elog(ERROR, "D1-1 conservative partial selftest failed");
	if (!(INFINITY == topn_tau(&(TopN) {.cap = 40})))
		elog(ERROR, "D1-1 heap readiness selftest failed");
	PG_RETURN_TEXT_P(cstring_to_text(
		"PASS: strict threshold; K10/K40; distance/TID ties; conservative partial"));
}

Datum
d1_observe(PG_FUNCTION_ARGS)
{
	Oid index_oid = PG_GETARG_OID(0);
	Vector *query = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(1));
	int probes = PG_GETARG_INT32(2);
	char *raw_path = text_to_cstring(PG_GETARG_TEXT_PP(3));
	int query_id = PG_GETARG_INT32(4);
	Relation relation;
	TupleDesc tupledesc;
	FmgrInfo *proc;
	TopN full;
	TopN progressive[D1_CONFIGS];
	ConfigStats stats[D1_CONFIGS];
	FILE *raw;
	uint64 candidate_ordinal = 0;
	uint64 serial = scan_serial;
	StringInfoData output;
	int li;
	int c;

	if (strncmp(raw_path, "/workspace/benchmark/runs/phase_d1_1_l2_",
			strlen("/workspace/benchmark/runs/phase_d1_1_l2_")) != 0 ||
		strchr(raw_path, '\n') != NULL)
		elog(ERROR, "D1-1 invalid raw output path");
	check_vector(query);
	if (captured_query == NULL ||
		memcmp(query->x, captured_query->x, D1_DIM * sizeof(float)) != 0)
		elog(ERROR, "D1-1 actual/replay query mismatch");
	if (index_oid != actual_index || actual_n != probes || actual_probes != probes ||
		logical_bound != 10 || physical_bound != 40 || !bounded || fallback ||
		direct_active || fused_active)
		elog(ERROR, "D1-1 actual path or bound mismatch");

	relation = index_open(index_oid, AccessShareLock);
	tupledesc = RelationGetDescr(relation);
	proc = index_getprocinfo(relation, 1, IVFFLAT_DISTANCE_PROC);
	if (strcmp(get_func_name(proc->fn_oid), "vector_l2_squared_distance") != 0)
		elog(ERROR, "D1-1 support proc is not squared L2");

	topn_init(&full, 10);
	for (c = 0; c < D1_CONFIGS; c++)
	{
		topn_init(&progressive[c], capacities[c]);
		memset(&stats[c], 0, sizeof(stats[c]));
	}

	raw = AllocateFile(raw_path, "a");
	if (raw == NULL)
		elog(ERROR, "D1-1 cannot append raw CSV: %m");
	setvbuf(raw, NULL, _IOFBF, 8 * 1024 * 1024);

	for (li = 0; li < actual_n; li++)
	{
		BlockNumber block = actual_lists[li];
		int half = li < actual_n / 2 ? 0 : 1;

		while (BlockNumberIsValid(block))
		{
			Buffer buffer;
			Page page;
			OffsetNumber offset;

			CHECK_FOR_INTERRUPTS();
			buffer = ReadBuffer(relation, block);
			LockBuffer(buffer, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buffer);
			for (offset = FirstOffsetNumber;
				 offset <= PageGetMaxOffsetNumber(page); offset++)
			{
				IndexTuple index_tuple;
				bool isnull;
				Datum datum;
				Vector *vector;
				Candidate candidate;
				double threshold[D1_CONFIGS];
				bool ready[D1_CONFIGS];
				bool abandoned[D1_CONFIGS] = {false};
				int first_abort[D1_CONFIGS] = {0};
				double abort_raw[D1_CONFIGS] = {0};
				double abort_key[D1_CONFIGS] = {0};
				double partial = 0.0;
				double partial_lo = 0.0;
				int dim;

				index_tuple = (IndexTuple) PageGetItem(page,
											 PageGetItemId(page, offset));
				datum = index_getattr(index_tuple, 1, tupledesc, &isnull);
				if (isnull)
					elog(ERROR, "D1-1 NULL index candidate");
				candidate.tid = index_tuple->t_tid;
				candidate.d = DatumGetFloat8(FunctionCall2Coll(proc,
														relation->rd_indcollation[0], datum,
														PointerGetDatum(query)));
				if (!isfinite(candidate.d) || candidate.d < 0.0)
					elog(ERROR, "D1-1 invalid production full score");
				vector = (Vector *) PG_DETOAST_DATUM(datum);
				check_vector(vector);
				candidate_ordinal++;

				for (c = 0; c < D1_CONFIGS; c++)
				{
					ready[c] = topn_ready(&progressive[c]);
					threshold[c] = topn_tau(&progressive[c]);
					stats[c].total++;
					stats[c].half_total[half]++;
					if (!ready[c])
						stats[c].before_ready++;
					for (int a = 0; a < D1_ANCHORS; a++)
					{
						if (!stats[c].anchor_seen[a] && candidate_ordinal == anchors[a])
						{
							stats[c].anchor_seen[a] = true;
							stats[c].anchor_tau[a] = threshold[c];
						}
					}
				}

				/* The unchanged full score above is known before this shadow loop,
				 * but it is used only for the numerical guard and final validation. */
				for (dim = 1; dim <= D1_DIM; dim++)
				{
					double diff = (double) vector->x[dim - 1] -
						(double) query->x[dim - 1];
					double term = diff * diff;
					double lower_key;

					partial += term;
					partial_lo = down(partial_lo + term);
					if (dim % 32 != 0 && dim != D1_DIM)
						continue;
					lower_key = certified_key(partial_lo);
					if (lower_key > candidate.d)
						elog(ERROR, "D1-1 numerical certificate violated at candidate %llu dim %d",
							 (unsigned long long) candidate_ordinal, dim);
					for (c = 0; c < D1_CONFIGS; c++)
					{
						int block_size = block_sizes[c];

						if (abandoned[c] || !ready[c])
							continue;
						if (dim != D1_DIM && dim % block_size != 0)
							continue;
						stats[c].threshold_checks++;
						if (lower_key > threshold[c])
						{
							abandoned[c] = true;
							first_abort[c] = dim;
							abort_raw[c] = partial;
							abort_key[c] = lower_key;
						}
					}
				}

				fprintf(raw, "1,%d,%d,%llu,%d,%s,\"(%u,%u)\",%.9g,32;64;128;256",
						query_id, probes, (unsigned long long) candidate_ordinal,
						li + 1, half == 0 ? "first" : "second",
						ItemPointerGetBlockNumber(&candidate.tid),
						ItemPointerGetOffsetNumber(&candidate.tid), candidate.d);
				for (c = 0; c < D1_CONFIGS; c++)
				{
					int evaluated = abandoned[c] ? first_abort[c] : D1_DIM;
					int avoided = D1_DIM - evaluated;
					int bucket = eval_bucket(evaluated);

					stats[c].evaluated_dims += evaluated;
					stats[c].avoided_dims += avoided;
					stats[c].eval_hist[evaluated]++;
					if (abandoned[c])
					{
						stats[c].early++;
						stats[c].half_early[half]++;
						stats[c].half_avoided[half] += avoided;
						stats[c].abort_hist[bucket]++;
					}
					fprintf(raw, ",%.9g,%d,%d,%d,%d,%d,%s,%.9g,%.9g",
							threshold[c], ready[c] ? 1 : 0, first_abort[c],
							evaluated, avoided, abandoned[c] ? 1 : 0,
							key_relation(&candidate, &progressive[c]),
							abort_raw[c], abort_key[c]);
					if (!abandoned[c])
						topn_insert(&progressive[c], candidate);
				}
				fputc('\n', raw);
				topn_insert(&full, candidate);
				if ((Pointer) vector != DatumGetPointer(datum))
					pfree(vector);
			}
			block = IvfflatPageGetOpaque(page)->nextblkno;
			UnlockReleaseBuffer(buffer);
		}
	}

	if (FreeFile(raw) != 0)
		elog(ERROR, "D1-1 cannot close raw CSV: %m");

	initStringInfo(&output);
	appendStringInfo(&output,
					 "{\"scan_serial\":%llu,\"selected_lists\":%d,"
					 "\"logical_bound\":%lld,\"physical_bound\":%lld,"
					 "\"candidates_total\":%llu,\"shadow_full_top10\":",
					 (unsigned long long) serial, actual_n,
					 (long long) logical_bound, (long long) physical_bound,
					 (unsigned long long) candidate_ordinal);
	append_top10(&output, &full);
	appendStringInfoString(&output, ",\"selected_list_start_pages\":[");
	for (li = 0; li < actual_n; li++)
		appendStringInfo(&output, "%s%u", li ? "," : "", actual_lists[li]);
	appendStringInfoString(&output, "],\"configs\":[");
	for (c = 0; c < D1_CONFIGS; c++)
	{
		ConfigStats *s = &stats[c];
		int mismatch = 0;
		int i;

		for (i = 0; i < 10; i++)
		{
			if (progressive[c].n <= i || full.n <= i ||
				candidate_compare(&progressive[c].items[i], &full.items[i]) != 0)
				mismatch = 1;
		}
		appendStringInfo(&output,
						 "%s{\"config\":\"%s\",\"capacity\":%d,"
						 "\"block_size\":%d,\"candidates_total\":%llu,"
						 "\"candidates_before_threshold_ready\":%llu,"
						 "\"candidates_after_threshold_ready\":%llu,"
						 "\"early_abandoned_candidates\":%llu,"
						 "\"full_evaluation_candidates\":%llu,"
						 "\"evaluated_dimensions\":%llu,"
						 "\"avoided_dimensions\":%llu,\"threshold_checks\":%llu,"
						 "\"p50_dims_evaluated\":%d,\"p75_dims_evaluated\":%d,"
						 "\"p90_dims_evaluated\":%d,\"p95_dims_evaluated\":%d,"
						 "\"threshold_ready_ordinal\":%d,\"shadow_mismatch\":%d,"
						 "\"final_threshold\":%.17g,",
						 c ? "," : "", config_names[c], capacities[c], block_sizes[c],
						 (unsigned long long) s->total,
						 (unsigned long long) s->before_ready,
						 (unsigned long long) (s->total - s->before_ready),
						 (unsigned long long) s->early,
						 (unsigned long long) (s->total - s->early),
						 (unsigned long long) s->evaluated_dims,
						 (unsigned long long) s->avoided_dims,
						 (unsigned long long) s->threshold_checks,
						 quantile_from_hist(s->eval_hist, s->total, 0.50),
						 quantile_from_hist(s->eval_hist, s->total, 0.75),
						 quantile_from_hist(s->eval_hist, s->total, 0.90),
						 quantile_from_hist(s->eval_hist, s->total, 0.95),
						 capacities[c], mismatch, topn_tau(&progressive[c]));
		appendStringInfoString(&output, "\"eval_hist\":[");
		{
			bool comma = false;
			for (i = 0; i <= D1_DIM; i++)
			{
				if (s->eval_hist[i] == 0)
					continue;
				appendStringInfo(&output, "%s{\"dims\":%d,\"count\":%llu}",
								 comma ? "," : "", i,
								 (unsigned long long) s->eval_hist[i]);
				comma = true;
			}
		}
		appendStringInfoString(&output, "],\"abort_hist\":[");
		for (i = 0; i < D1_BUCKETS; i++)
			appendStringInfo(&output, "%s%llu", i ? "," : "",
							 (unsigned long long) s->abort_hist[i]);
		appendStringInfo(&output,
						 "],\"first_half_total\":%llu,\"first_half_early\":%llu,"
						 "\"first_half_avoided_dimensions\":%llu,"
						 "\"second_half_total\":%llu,\"second_half_early\":%llu,"
						 "\"second_half_avoided_dimensions\":%llu,\"trajectory\":[",
						 (unsigned long long) s->half_total[0],
						 (unsigned long long) s->half_early[0],
						 (unsigned long long) s->half_avoided[0],
						 (unsigned long long) s->half_total[1],
						 (unsigned long long) s->half_early[1],
						 (unsigned long long) s->half_avoided[1]);
		for (i = 0; i < D1_ANCHORS; i++)
		{
			if (!s->anchor_seen[i])
				continue;
			appendStringInfo(&output,
							 "%s{\"candidate_ordinal\":%llu,\"threshold\":",
							 output.data[output.len - 1] == '[' ? "" : ",",
							 (unsigned long long) anchors[i]);
			if (isfinite(s->anchor_tau[i]))
				appendStringInfo(&output, "%.17g}", s->anchor_tau[i]);
			else
				appendStringInfoString(&output, "null}");
		}
		appendStringInfoString(&output, "],\"shadow_progressive_top10\":");
		append_top10(&output, &progressive[c]);
		appendStringInfoChar(&output, '}');
		if (mismatch)
		{
			index_close(relation, AccessShareLock);
			elog(ERROR, "D1-1 FAIL: shadow Top10 mismatch for %s", config_names[c]);
		}
	}
	appendStringInfoString(&output, "]}");
	index_close(relation, AccessShareLock);
	PG_RETURN_TEXT_P(cstring_to_text(output.data));
}
