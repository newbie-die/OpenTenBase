#include "postgres.h"

#include <float.h>

#include "access/genam.h"
#include "access/itup.h"
#include "access/relscan.h"
#include "access/tupdesc.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "fmgr.h"
#include "lib/pairingheap.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/tuplesort.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)

#ifdef IVFFLAT_BENCH
/*
 * Calculate L2 distance without fmgr while preserving varlena and dimension
 * safety.  The common IVFFlat entry representation takes the first branch.
 */
static Datum
DirectL2Distance(IvfflatScanOpaque so, Datum datum)
{
	Pointer		original = DatumGetPointer(datum);
	Vector	   *candidate;
	bool		freeCandidate = false;
	Datum		result;

	if (likely(!VARATT_IS_EXTENDED(original)))
		candidate = (Vector *) original;
	else
	{
		candidate = DatumGetVector(datum);
		freeCandidate = ((Pointer) candidate != original);
	}

	if (unlikely(candidate->dim != so->directQuery->dim))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d",
						candidate->dim, so->directQuery->dim)));

	result = Float8GetDatum((double) VectorL2SquaredDistanceRaw(candidate->dim,
												 candidate->x, so->directQuery->x));

	if (freeCandidate)
		pfree(candidate);

	return result;
}
#endif

/*
 * Compare list distances
 */
static int
CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (GetScanListConst(a)->distance > GetScanListConst(b)->distance)
		return 1;

	if (GetScanListConst(a)->distance < GetScanListConst(b)->distance)
		return -1;

	return 0;
}

/*
 * Get lists and sort by distance
 */
static void
GetScanLists(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	int			listCount = 0;
	double		maxDistance = DBL_MAX;

	/* Search all list pages */
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);

		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
			double		distance;

			/* Use procinfo from the index instead of scan key for performance */
			distance = DatumGetFloat8(so->distfunc(so->procinfo, so->collation, PointerGetDatum(&list->center), value));

			if (listCount < so->maxProbes)
			{
				IvfflatScanList *scanlist;

				scanlist = &so->lists[listCount];
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				listCount++;

				/* Add to heap */
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Calculate max distance */
				if (listCount == so->maxProbes)
					maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance)
			{
				IvfflatScanList *scanlist;

				/* Remove */
				scanlist = GetScanList(pairingheap_remove_first(so->listQueue));

				/* Reuse */
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Update max distance */
				maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);
	}

	for (int i = listCount - 1; i >= 0; i--)
		so->listPages[i] = GetScanList(pairingheap_remove_first(so->listQueue))->startPage;

	Assert(pairingheap_is_empty(so->listQueue));
}

/*
 * Get items
 */
static void
GetScanItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);
	TupleTableSlot *slot = so->vslot;
	int			batchProbes = 0;

#ifdef IVFFLAT_BENCH
	instr_time	getitems_start;

	INSTR_TIME_SET_CURRENT(getitems_start);
	so->profile_getitems_calls++;
#endif

	tuplesort_reset(so->sortstate);
	/* tuplesort_reset() clears the per-batch bounded state */
	if (so->boundedActive && !so->fallbackTriggered)
		tuplesort_set_bound(so->sortstate, so->physicalBound);

	/* Search closest probes lists */
	while (so->listIndex < so->maxProbes && (++batchProbes) <= so->probes)
	{
		BlockNumber searchPage = so->listPages[so->listIndex++];

		/* Search all entry pages for list */
		while (BlockNumberIsValid(searchPage))
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoffno;

#ifdef IVFFLAT_BENCH
			so->profile_pages++;
#endif

			buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoffno = PageGetMaxOffsetNumber(page);
#ifdef IVFFLAT_PROFILE_2B
			so->profile_page_candidates[Min((uint64) maxoffno, 4)]++;
			so->profile_max_candidates_per_page = Max(so->profile_max_candidates_per_page, (uint64) maxoffno);
#endif

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
			{
				IndexTuple	itup;
				Datum		datum;
				bool		isnull;
				ItemId		itemid;
#ifdef IVFFLAT_BENCH
				instr_time	candidate_start;
				instr_time	distance_start;
				instr_time	elapsed;
#endif
#ifdef IVFFLAT_PROFILE_2B
				instr_time	segment_start;
#endif
#ifdef IVFFLAT_BENCH
				INSTR_TIME_SET_CURRENT(candidate_start);
#endif
#ifdef IVFFLAT_PROFILE_2B
				INSTR_TIME_SET_CURRENT(segment_start);
#endif
				itemid = PageGetItemId(page, offno);

				itup = (IndexTuple) PageGetItem(page, itemid);
				datum = index_getattr(itup, 1, tupdesc, &isnull);
#ifdef IVFFLAT_PROFILE_2B
				INSTR_TIME_SET_CURRENT(elapsed);
				INSTR_TIME_SUBTRACT(elapsed, segment_start);
				so->profile_candidate_extract_ns += INSTR_TIME_GET_NANOSEC(elapsed);
#endif

				/*
				 * Add virtual tuple
				 *
				 * Use procinfo from the index instead of scan key for
				 * performance
				 */
#ifdef IVFFLAT_PROFILE_2B
				INSTR_TIME_SET_CURRENT(segment_start);
#endif
				ExecClearTuple(slot);
#ifdef IVFFLAT_PROFILE_2B
				INSTR_TIME_SET_CURRENT(elapsed);
				INSTR_TIME_SUBTRACT(elapsed, segment_start);
				so->profile_tuple_materialization_ns += INSTR_TIME_GET_NANOSEC(elapsed);
#endif
#ifdef IVFFLAT_BENCH
				INSTR_TIME_SET_CURRENT(distance_start);
#endif
#ifdef IVFFLAT_BENCH
				if (so->useDirectL2)
				{
					slot->tts_values[0] = DirectL2Distance(so, datum);
#ifdef IVFFLAT_PROFILE_2B
					so->profile_direct_distance_calls++;
#endif
				}
				else
#endif
				{
					slot->tts_values[0] = so->distfunc(so->procinfo, so->collation, datum, value);
#ifdef IVFFLAT_PROFILE_2B
					so->profile_generic_distance_calls++;
#endif
				}
#ifdef IVFFLAT_BENCH
				INSTR_TIME_SET_CURRENT(elapsed);
				INSTR_TIME_SUBTRACT(elapsed, distance_start);
				so->profile_distance_us += INSTR_TIME_GET_MICROSEC(elapsed);
#ifdef IVFFLAT_PROFILE_2B
				so->profile_distance_ns += INSTR_TIME_GET_NANOSEC(elapsed);
				if (so->useDirectL2)
					so->profile_direct_distance_ns += INSTR_TIME_GET_NANOSEC(elapsed);
				else
					so->profile_generic_distance_ns += INSTR_TIME_GET_NANOSEC(elapsed);
				so->profile_distance_calls++;
#endif
#endif
#ifdef IVFFLAT_PROFILE_2B
				INSTR_TIME_SET_CURRENT(segment_start);
#endif
				slot->tts_isnull[0] = false;
				slot->tts_values[1] = PointerGetDatum(&itup->t_tid);
				slot->tts_isnull[1] = false;
				ExecStoreVirtualTuple(slot);
#ifdef IVFFLAT_PROFILE_2B
				INSTR_TIME_SET_CURRENT(elapsed);
				INSTR_TIME_SUBTRACT(elapsed, segment_start);
				so->profile_tuple_materialization_ns += INSTR_TIME_GET_NANOSEC(elapsed);
				INSTR_TIME_SET_CURRENT(segment_start);
#endif

				tuplesort_puttupleslot(so->sortstate, slot);
#ifdef IVFFLAT_PROFILE_2B
				so->profile_tuplesort_input_calls++;
				INSTR_TIME_SET_CURRENT(elapsed);
				INSTR_TIME_SUBTRACT(elapsed, segment_start);
				so->profile_sort_insert_ns += INSTR_TIME_GET_NANOSEC(elapsed);
#endif
#ifdef IVFFLAT_BENCH
				INSTR_TIME_SET_CURRENT(elapsed);
				INSTR_TIME_SUBTRACT(elapsed, candidate_start);
				so->profile_candidate_us += INSTR_TIME_GET_MICROSEC(elapsed);
				so->profile_candidates++;
#endif
			}

			searchPage = IvfflatPageGetOpaque(page)->nextblkno;

			UnlockReleaseBuffer(buf);
		}
	}

#ifdef IVFFLAT_BENCH
	{
		instr_time	sort_start;
		instr_time	elapsed;

		INSTR_TIME_SET_CURRENT(sort_start);
		tuplesort_performsort(so->sortstate);
		INSTR_TIME_SET_CURRENT(elapsed);
		INSTR_TIME_SUBTRACT(elapsed, sort_start);
		so->profile_sort_us += INSTR_TIME_GET_MICROSEC(elapsed);
#ifdef IVFFLAT_PROFILE_2B
		so->profile_sort_finalize_ns += INSTR_TIME_GET_NANOSEC(elapsed);
#endif
	}
#else
	tuplesort_performsort(so->sortstate);
#endif

#ifdef IVFFLAT_BENCH
	{
		instr_time	elapsed;

		INSTR_TIME_SET_CURRENT(elapsed);
		INSTR_TIME_SUBTRACT(elapsed, getitems_start);
		so->profile_getitems_us += INSTR_TIME_GET_MICROSEC(elapsed);
#ifdef IVFFLAT_PROFILE_2B
		so->profile_scan_items_total_ns += INSTR_TIME_GET_NANOSEC(elapsed);
#endif
	}
#endif

#if defined(IVFFLAT_MEMORY)
	elog(INFO, "memory: %zu MB", MemoryContextMemAllocated(CurrentMemoryContext, true) / (1024 * 1024));
#endif
}

/*
 * Zero distance
 */
static Datum
ZeroDistance(FmgrInfo *flinfo, Oid collation, Datum arg1, Datum arg2)
{
	return Float8GetDatum(0.0);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
	{
		value = PointerGetDatum(NULL);
		so->distfunc = ZeroDistance;
	}
	else
	{
		value = scan->orderByData->sk_argument;
		so->distfunc = FunctionCall2Coll;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->normprocinfo != NULL)
		{
			MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

			value = IvfflatNormValue(so->typeInfo, so->collation, value);

			MemoryContextSwitchTo(oldCtx);
		}

#ifdef IVFFLAT_BENCH
		so->useDirectL2 = so->directL2Eligible &&
			ivfflat_distance_path == IVFFLAT_DISTANCE_PATH_DIRECT;
		if (so->useDirectL2)
		{
			MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

			so->directQuery = DatumGetVector(value);
			so->directQueryNeedsFree =
				((Pointer) so->directQuery != DatumGetPointer(value));
			MemoryContextSwitchTo(oldCtx);

			if (so->directQuery->dim != so->dimensions)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("different vector dimensions %d and %d",
								so->dimensions, so->directQuery->dim)));
		}
#endif
	}

	return value;
}

/*
 * Initialize scan sort state
 */
static Tuplesortstate *
InitScanSortState(TupleDesc tupdesc, bool bounded, int64 bound)
{
	AttrNumber	attNums[] = {1, 2};
	Oid			sortOperators[] = {Float8LessOperator, TIDLessOperator};
	Oid			sortCollations[] = {InvalidOid, InvalidOid};
	bool		nullsFirstFlags[] = {false, false};
	Tuplesortstate *sortstate;
	int			sortopt = bounded ? TUPLESORT_ALLOWBOUNDED : TUPLESORT_NONE;

	sortstate = tuplesort_begin_heap(tupdesc, 2, attNums, sortOperators, sortCollations, nullsFirstFlags, work_mem, NULL, sortopt);
	if (bounded)
		tuplesort_set_bound(sortstate, bound);

	return sortstate;
}

/*
 * Check whether a TID was already returned during the bounded phase
 */
static bool
ReturnedFromBounded(IvfflatScanOpaque so, ItemPointer heaptid)
{
	for (Size i = 0; i < so->returnedTidsCount; i++)
	{
		if (ItemPointerEquals(&so->returnedTids[i], heaptid))
			return true;
	}

	return false;
}

/*
 * Remember a TID returned to the index AM caller during the bounded phase
 */
static void
RememberBoundedTid(IvfflatScanOpaque so, ItemPointer heaptid)
{
	if (so->returnedTidsCount == so->returnedTidsCapacity)
	{
		Size		newCapacity = so->returnedTidsCapacity == 0 ? 64 :
			so->returnedTidsCapacity * 2;

		if (so->returnedTids == NULL)
			so->returnedTids = MemoryContextAlloc(so->tmpCtx,
											 mul_size(sizeof(ItemPointerData), newCapacity));
		else
			so->returnedTids = repalloc_array(so->returnedTids,
											  ItemPointerData, newCapacity);
		so->returnedTidsCapacity = newCapacity;
	}

	ItemPointerCopy(heaptid, &so->returnedTids[so->returnedTidsCount++]);
}

/*
 * Rebuild an unbounded sort over the already selected lists
 */
static void
StartFullSortFallback(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	so->boundedExhausted = true;
	so->fallbackTriggered = true;
	so->profile_fallback_triggered = true;

	tuplesort_end(so->sortstate);
	so->sortstate = InitScanSortState(so->tupdesc, false, 0);
	so->listIndex = 0;
	GetScanItems(scan, so->value);
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
ivfflatbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	IvfflatScanOpaque so;
	int			lists;
	int			dimensions;
	int			probes = ivfflat_probes;
	int			maxProbes;
	MemoryContext oldCtx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	/* Get lists and dimensions from metapage */
	IvfflatGetMetaPageInfo(index, &lists, &dimensions);

	if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
		maxProbes = Max(ivfflat_max_probes, probes);
	else
		maxProbes = probes;

	if (probes > lists)
		probes = lists;

	if (maxProbes > lists)
		maxProbes = lists;

	so = palloc_object(IvfflatScanOpaqueData);
	so->typeInfo = IvfflatGetTypeInfo(index);
	so->first = true;
	so->probes = probes;
	so->maxProbes = maxProbes;
	so->dimensions = dimensions;
	so->sortBound = ivfflat_experimental_sort_bound;
	so->logicalBound = -1;
	so->physicalBound = 0;
	so->boundedActive = false;
	so->boundedExhausted = false;
	so->fallbackTriggered = false;
	so->returnedTids = NULL;
	so->returnedTidsCount = 0;
	so->returnedTidsCapacity = 0;
	so->value = PointerGetDatum(NULL);
#ifdef IVFFLAT_BENCH
	so->directL2Eligible = false;
	so->useDirectL2 = false;
	so->directQuery = NULL;
	so->directQueryNeedsFree = false;
#endif

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	so->collation = index->rd_indcollation[0];
#ifdef IVFFLAT_BENCH
	/* Exact function identity also excludes IP, cosine, halfvec, and bit. */
	so->directL2Eligible =
		so->procinfo->fn_addr == vector_l2_squared_distance;
#endif

	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Ivfflat scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/* Create tuple description for sorting */
	so->tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 1, "distance", FLOAT8OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 2, "heaptid", TIDOID, -1, 0);
#if PG_VERSION_NUM >= 190000
	TupleDescFinalize(so->tupdesc);
#endif

	/* Prep sort */
	so->sortstate = NULL;

	/* Need separate slots for puttuple and gettuple */
	so->vslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsVirtual);
	so->mslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsMinimalTuple);

	/*
	 * Reuse same set of shared buffers for scan
	 *
	 * See postgres/src/backend/storage/buffer/README for description
	 */
	so->bas = GetAccessStrategy(BAS_BULKREAD);

	so->listQueue = pairingheap_allocate(CompareLists, scan);
	so->listPages = palloc_array_checked(BlockNumber, maxProbes);
	so->listIndex = 0;
	so->lists = palloc_array_checked(IvfflatScanList, maxProbes);

	// profiling 检测查询耗时
		/* Initialize profiling counters */
	so->profile_candidates = 0;
	so->profile_pages = 0;
	so->profile_getitems_calls = 0;

	so->profile_list_us = 0.0;
	so->profile_getitems_us = 0.0;
	so->profile_candidate_us = 0.0;
	so->profile_distance_us = 0.0;
	so->profile_sort_us = 0.0;
	so->profile_return_us = 0.0;
	so->profile_fallback_triggered = false;
	so->profile_returned_from_bounded = 0;
	so->profile_returned_after_fallback = 0;
#ifdef IVFFLAT_PROFILE_2B
	so->profile_distance_calls = so->profile_candidate_extract_ns = 0;
	so->profile_generic_distance_calls = so->profile_direct_distance_calls = 0;
	so->profile_tuplesort_input_calls = so->profile_returned_rows = 0;
	so->profile_distance_ns = so->profile_generic_distance_ns = 0;
	so->profile_direct_distance_ns = so->profile_tuple_materialization_ns = 0;
	so->profile_sort_insert_ns = so->profile_sort_finalize_ns = 0;
	so->profile_scan_items_total_ns = so->profile_max_candidates_per_page = 0;
	MemSet(so->profile_page_candidates, 0, sizeof(so->profile_page_candidates));
#endif

	MemoryContextSwitchTo(oldCtx);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
ivfflatrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	so->logicalBound = scan->xs_tuple_bound;
	so->physicalBound = 0;
	so->boundedActive = false;
	so->boundedExhausted = false;
	so->fallbackTriggered = false;
	so->returnedTidsCount = 0;
#ifdef IVFFLAT_BENCH
	if (so->directQueryNeedsFree)
		pfree(so->directQuery);
	so->directQuery = NULL;
	so->directQueryNeedsFree = false;
	so->useDirectL2 = false;
#endif

	/* Oracle bound has priority over the automatic limit-aware path */
	if (so->sortBound > 0)
	{
		so->physicalBound = so->sortBound;
		so->boundedActive = true;
	}
	else if (ivfflat_bounded_scan &&
			 so->logicalBound > 0 &&
			 so->logicalBound <= ivfflat_bound_fastpath_limit &&
			 ivfflat_iterative_scan == IVFFLAT_ITERATIVE_SCAN_OFF)
	{
		so->physicalBound = Max((int64) ivfflat_bound_min,
								so->logicalBound * ivfflat_bound_overfetch);
		so->boundedActive = true;
	}

	if (so->sortstate != NULL)
		tuplesort_end(so->sortstate);
	so->sortstate = InitScanSortState(so->tupdesc, so->boundedActive,
								  so->physicalBound);
	so->first = true;
	pairingheap_reset(so->listQueue);
	so->listIndex = 0;

	if (so->normprocinfo != NULL && DatumGetPointer(so->value) != NULL)
	{
		pfree(DatumGetPointer(so->value));
		so->value = PointerGetDatum(NULL);
	}

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
ivfflatgettuple(IndexScanDesc scan, ScanDirection dir)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	ItemPointer heaptid;
	bool		isnull;

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan ivfflat index without order");

		/* Requires MVCC-compliant snapshot as not able to pin during sorting */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with ivfflat");

		value = GetScanValue(scan);
#ifdef IVFFLAT_BENCH
		{
			instr_time	start;
			instr_time	elapsed;

			INSTR_TIME_SET_CURRENT(start);
			GetScanLists(scan, value);
			INSTR_TIME_SET_CURRENT(elapsed);
			INSTR_TIME_SUBTRACT(elapsed, start);
			so->profile_list_us += INSTR_TIME_GET_MICROSEC(elapsed);
		}
#else
		GetScanLists(scan, value);
#endif
		GetScanItems(scan, value);
		so->first = false;
		so->value = value;
	}

	for (;;)
	{
		bool		found;
#ifdef IVFFLAT_BENCH
		instr_time	start;
		instr_time	elapsed;
#endif

		/* Avoid requesting tuple physicalBound + 1 from bounded tuplesort */
		if (so->boundedActive && !so->fallbackTriggered &&
			ivfflat_iterative_scan == IVFFLAT_ITERATIVE_SCAN_OFF &&
			so->returnedTidsCount >= (uint64) so->physicalBound)
		{
			StartFullSortFallback(scan);
			continue;
		}

#ifdef IVFFLAT_BENCH
		INSTR_TIME_SET_CURRENT(start);
#endif
		found = tuplesort_gettupleslot(so->sortstate, true, false, so->mslot, NULL);
#ifdef IVFFLAT_BENCH
		INSTR_TIME_SET_CURRENT(elapsed);
		INSTR_TIME_SUBTRACT(elapsed, start);
		so->profile_return_us += INSTR_TIME_GET_MICROSEC(elapsed);
#endif
		if (found)
		{
			heaptid = (ItemPointer) DatumGetPointer(slot_getattr(so->mslot, 2, &isnull));

			/* The full-sort fallback contains the bounded results as well */
			if (so->fallbackTriggered && ReturnedFromBounded(so, heaptid))
				continue;

			break;
		}

		/*
		 * Exhausting a non-iterative bounded sort while the caller still asks
		 * for tuples means the conservative bound was insufficient.  Reuse the
		 * selected listPages, rebuild an unbounded sort, and suppress TIDs that
		 * were already returned during the bounded phase.
		 */
		if (so->boundedActive && !so->fallbackTriggered &&
			ivfflat_iterative_scan == IVFFLAT_ITERATIVE_SCAN_OFF)
		{
			StartFullSortFallback(scan);
			continue;
		}

		if (so->listIndex == so->maxProbes)
			return false;

		GetScanItems(scan, so->value);
	}

	scan->xs_heaptid = *heaptid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	if (so->boundedActive && !so->fallbackTriggered)
	{
		RememberBoundedTid(so, heaptid);
		so->profile_returned_from_bounded++;
	}
	else if (so->fallbackTriggered)
		so->profile_returned_after_fallback++;
#ifdef IVFFLAT_PROFILE_2B
	so->profile_returned_rows++;
#endif
	return true;
}

/*
 * End a scan and release resources
 */
void
ivfflatendscan(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

#ifdef IVFFLAT_BENCH
	elog(INFO, "IVFFLAT_PROFILE probes=%d dimensions=%d sort_bound=%d logical_bound=" INT64_FORMAT " physical_bound=" INT64_FORMAT " bounded_active=%d bounded_exhausted=%d fallback_triggered=%d returned_from_bounded=%llu returned_after_fallback=%llu candidates=%llu pages=%llu getitems_calls=%llu list_us=%.3f getitems_us=%.3f candidate_us=%.3f distance_us=%.3f sort_us=%.3f return_us=%.3f scan_us=%.3f",
		 so->probes, so->dimensions, so->sortBound, so->logicalBound,
		 so->physicalBound, so->boundedActive ? 1 : 0,
		 so->boundedExhausted ? 1 : 0,
		 so->profile_fallback_triggered ? 1 : 0,
		 (unsigned long long) so->profile_returned_from_bounded,
		 (unsigned long long) so->profile_returned_after_fallback,
		 (unsigned long long) so->profile_candidates,
		 (unsigned long long) so->profile_pages,
		 (unsigned long long) so->profile_getitems_calls,
		 so->profile_list_us, so->profile_getitems_us,
		 so->profile_candidate_us, so->profile_distance_us,
		 so->profile_sort_us, so->profile_return_us,
		 so->profile_list_us + so->profile_getitems_us + so->profile_return_us);
#ifdef IVFFLAT_PROFILE_2B
	elog(INFO, "IVFFLAT_PROFILE_2B probes=%d dimensions=%d selected_lists=%d scanned_candidates=%llu scanned_pages=%llu distance_calls=%llu tuplesort_input_calls=%llu returned_rows=%llu distance_path_requested=%d direct_l2_eligible=%d direct_l2_active=%d generic_distance_calls=%llu direct_distance_calls=%llu generic_distance_ns=%llu direct_distance_ns=%llu candidate_extract_ns=%llu distance_ns=%llu tuple_materialization_ns=%llu sort_insert_ns=%llu sort_finalize_ns=%llu scan_items_total_ns=%llu page_candidates_0=%llu page_candidates_1=%llu page_candidates_2=%llu page_candidates_3=%llu page_candidates_4plus=%llu max_candidates_per_page=%llu",
		 so->probes, so->dimensions, so->listIndex,
		 (unsigned long long) so->profile_candidates, (unsigned long long) so->profile_pages,
		 (unsigned long long) so->profile_distance_calls, (unsigned long long) so->profile_tuplesort_input_calls,
		 (unsigned long long) so->profile_returned_rows, ivfflat_distance_path,
		 so->directL2Eligible ? 1 : 0, so->useDirectL2 ? 1 : 0,
		 (unsigned long long) so->profile_generic_distance_calls,
		 (unsigned long long) so->profile_direct_distance_calls,
		 (unsigned long long) so->profile_generic_distance_ns,
		 (unsigned long long) so->profile_direct_distance_ns,
		 (unsigned long long) so->profile_candidate_extract_ns,
		 (unsigned long long) so->profile_distance_ns, (unsigned long long) so->profile_tuple_materialization_ns,
		 (unsigned long long) so->profile_sort_insert_ns, (unsigned long long) so->profile_sort_finalize_ns,
		 (unsigned long long) so->profile_scan_items_total_ns, (unsigned long long) so->profile_page_candidates[0],
		 (unsigned long long) so->profile_page_candidates[1], (unsigned long long) so->profile_page_candidates[2],
		 (unsigned long long) so->profile_page_candidates[3], (unsigned long long) so->profile_page_candidates[4],
		 (unsigned long long) so->profile_max_candidates_per_page);
#endif
#endif

	/* Free any temporary files */
	tuplesort_end(so->sortstate);

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
