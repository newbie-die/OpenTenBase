#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/genam.h"
#include "access/itup.h"
#include "access/relscan.h"
#include "access/tupdesc.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "fmgr.h"
#include "lib/pairingheap.h"
#include "lib/stringinfo.h"
#include "ivfflat.h"
#include "ivfd2ppolicy.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/tuplesort.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)

#ifdef IVFFLAT_DISTANCE_PATH
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
 * A larger d2/d1 means the nearest centroid is more clearly separated from
 * its runner-up and can use fewer probes. ivfflat.probes remains the maximum
 * number retained and the fallback for an unset or inapplicable rule.
 */
static int
AdaptiveProbeCount(IvfflatScanOpaque so, int listCount)
{
	double		d1;
	double		d2;
	double		ratio;
	int			probes;

	if (!ivfflat_adaptive_probes || listCount < 16 || so->maxProbes < 16)
		return so->probes;

	d1 = so->listDistances[0];
	d2 = so->listDistances[1];
	if (d1 > 0)
		ratio = d2 / d1;
	else
		ratio = d2 > 0 ? DBL_MAX : 1.0;

	if (ratio >= ivfflat_adaptive_probes_ratio_16)
		probes = 16;
	else if (ratio >= ivfflat_adaptive_probes_ratio_32)
		probes = 32;
	else if (ratio >= ivfflat_adaptive_probes_ratio_64)
		probes = 64;
	else
		probes = 128;

	return Min(probes, so->maxProbes);
}

/*
 * Emit the exact ordered list identifiers and centroid distances used by a
 * scan. This is audit-only output and does not read or reorder any list.
 */
static void
DebugOrderedLists(IvfflatScanOpaque so, int listCount)
{
	StringInfoData lists;

	if (!ivfflat_progressive_scan_debug)
		return;

	initStringInfo(&lists);
	for (int i = 0; i < listCount; i++)
		appendStringInfo(&lists, "%s%u:%.17g", i == 0 ? "" : ",",
						 so->listPages[i], so->listDistances[i]);

	elog(INFO, "IVFFLAT_PROGRESSIVE_LISTS mode=%s count=%d pages_distances=%s",
		 so->progressiveEarlyStop ? "on" : (so->progressiveShadow ? "shadow" : "fixed"),
		 listCount, lists.data);
	pfree(lists.data);
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
	{
		IvfflatScanList *scanlist = GetScanList(pairingheap_remove_first(so->listQueue));

		so->listPages[i] = scanlist->startPage;
		so->listDistances[i] = scanlist->distance;
	}

	Assert(pairingheap_is_empty(so->listQueue));
	so->listCount = listCount;
	DebugOrderedLists(so, listCount);
	if (so->progressiveShadow)
		so->probes = 64;
	else
		so->probes = AdaptiveProbeCount(so, listCount);

	if (ivfflat_adaptive_probes_trace && listCount >= 64)
	{
		double		d1 = so->listDistances[0];
		double		ratio = d1 > 0 ? so->listDistances[1] / d1 :
			(so->listDistances[1] > 0 ? DBL_MAX : 1.0);

		elog(INFO, "IVFFLAT_ADAPTIVE d1=%.17g d2=%.17g d4=%.17g d8=%.17g d16=%.17g d32=%.17g d64=%.17g d2_d1=%.17g d4_d1=%.17g d8_d1=%.17g d16_d1=%.17g d32_d1=%.17g d64_d1=%.17g d32_d16=%.17g d64_d32=%.17g gap=%.17g gap32_16_d1=%.17g gap64_32_d1=%.17g probes=%d max_probes=%d",
			 d1, so->listDistances[1], so->listDistances[3],
			 so->listDistances[7], so->listDistances[15],
			 so->listDistances[31], so->listDistances[63], ratio,
			 so->listDistances[3] / d1, so->listDistances[7] / d1,
			 so->listDistances[15] / d1, so->listDistances[31] / d1,
			 so->listDistances[63] / d1,
			 so->listDistances[31] / so->listDistances[15],
			 so->listDistances[63] / so->listDistances[31],
			 so->listDistances[1] - d1,
			 (so->listDistances[31] - so->listDistances[15]) / d1,
			 (so->listDistances[63] - so->listDistances[31]) / d1,
			 so->probes, so->maxProbes);
	}
}

/*
 * Compare shadow results in the same order as the real tuplesort: distance,
 * then heap TID. This also gives deterministic snapshots for exact ties.
 */
static int
CompareShadowTopItems(const void *va, const void *vb)
{
	const IvfflatShadowTopItem *a = (const IvfflatShadowTopItem *) va;
	const IvfflatShadowTopItem *b = (const IvfflatShadowTopItem *) vb;
	int			cmp = float8_cmp_internal(a->distance, b->distance);

	if (cmp != 0)
		return cmp;
	return ItemPointerCompare((ItemPointer) &a->tid, (ItemPointer) &b->tid);
}

/*
 * Observe an already computed candidate distance. No index or heap access and
 * no distance function call is performed here.
 */
static void
ShadowTopKUpdate(IvfflatScanOpaque so, Datum distanceDatum, ItemPointer tid)
{
	IvfflatShadowTopItem candidate;

	if (!so->progressiveShadow)
		return;

	candidate.distance = DatumGetFloat8(distanceDatum);
	ItemPointerCopy(tid, &candidate.tid);
	so->shadowCandidatesSeen++;
	so->shadowDistanceCalls++;

	if (so->shadowTopCount < so->shadowK)
	{
		int			inserted = so->shadowTopCount++;

		so->shadowTopItems[inserted] = candidate;
		if (inserted == 0 ||
			CompareShadowTopItems(&so->shadowTopItems[so->shadowWorst],
								 &candidate) < 0)
			so->shadowWorst = inserted;
		return;
	}

	/* Most candidates need only this one comparison. */
	if (CompareShadowTopItems(&candidate,
						  &so->shadowTopItems[so->shadowWorst]) < 0)
	{
		so->shadowTopItems[so->shadowWorst] = candidate;
		so->shadowWorst = 0;
		for (int i = 1; i < so->shadowTopCount; i++)
		{
			if (CompareShadowTopItems(&so->shadowTopItems[so->shadowWorst],
									 &so->shadowTopItems[i]) < 0)
				so->shadowWorst = i;
		}
		so->shadowReplacements++;
	}
}

/*
 * Persist a stage snapshot in scan state and optionally emit parseable debug
 * output. The output is entirely gated by progressive_scan_debug.
 */
static void
ShadowSnapshot(IvfflatScanOpaque so, int snapshotIndex, int stage)
{
	IvfflatShadowSnapshot *snapshot = &so->shadowSnapshots[snapshotIndex];

	qsort(so->shadowTopItems, so->shadowTopCount,
		  sizeof(IvfflatShadowTopItem), CompareShadowTopItems);
	if (so->shadowTopCount > 0)
		so->shadowWorst = so->shadowTopCount - 1;

	snapshot->stage = stage;
	snapshot->count = so->shadowTopCount;
	snapshot->kthDistance = so->shadowTopCount == so->shadowK ?
		so->shadowTopItems[so->shadowTopCount - 1].distance :
		get_float8_infinity();
	snapshot->candidatesSeen = so->shadowCandidatesSeen;
	snapshot->pagesSeen = so->shadowPagesSeen;
	snapshot->distanceCalls = so->shadowDistanceCalls;
	snapshot->replacements = so->shadowReplacements;
	memcpy(snapshot->items, so->shadowTopItems,
		   mul_size(sizeof(IvfflatShadowTopItem), so->shadowTopCount));

	if (ivfflat_progressive_scan_debug)
	{
		StringInfoData topk;
		int			previousStage = snapshotIndex == 0 ? 0 :
			so->shadowSnapshots[snapshotIndex - 1].stage;

		elog(INFO, "IVFFLAT_PROGRESSIVE stage=%d probes_scanned=%d new_lists=%d candidates=%llu pages=%llu distance_calls=%llu replacements=%llu kth_distance=%.17g topk_count=%d list_mask=%016llx",
			 stage, stage, stage - previousStage,
			 (unsigned long long) snapshot->candidatesSeen,
			 (unsigned long long) snapshot->pagesSeen,
			 (unsigned long long) snapshot->distanceCalls,
			 (unsigned long long) snapshot->replacements,
			 snapshot->kthDistance, snapshot->count,
			 (unsigned long long) so->shadowScannedLists);

		initStringInfo(&topk);
		for (int i = 0; i < snapshot->count; i++)
		{
			BlockNumber block = ItemPointerGetBlockNumber(&snapshot->items[i].tid);
			OffsetNumber offset = ItemPointerGetOffsetNumber(&snapshot->items[i].tid);

			appendStringInfo(&topk, "%s%u/%u:%.17g", i == 0 ? "" : ",",
						 block, offset, snapshot->items[i].distance);
		}
		elog(INFO, "IVFFLAT_PROGRESSIVE_TOPK stage=%d tids_distances=%s",
			 stage, topk.data);
		pfree(topk.data);
	}
}

/*
 * Get items from the half-open ordered-list range [startIndex, endIndex).
 * resetSort is true only for the first range; performSort is true only for
 * the final range.
 */
static void
GetScanItemsRange(IndexScanDesc scan, Datum value, int startIndex, int endIndex,
				  bool resetSort, bool performSort)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);
	TupleTableSlot *slot = so->vslot;
#ifdef IVFFLAT_BENCH
	instr_time	getitems_start;
#endif

	Assert(startIndex >= 0);
	Assert(startIndex == so->listIndex);
	Assert(endIndex >= startIndex);
	Assert(endIndex <= so->maxProbes);

#ifdef IVFFLAT_BENCH
	INSTR_TIME_SET_CURRENT(getitems_start);
	so->profile_getitems_calls++;
#endif

	if (resetSort)
	{
		tuplesort_reset(so->sortstate);
		/* tuplesort_reset() clears the per-batch bounded state */
		if (so->boundedActive && !so->fallbackTriggered)
			tuplesort_set_bound(so->sortstate, so->physicalBound);
	}

	/* Search exactly this range of the closest ordered lists */
	while (so->listIndex < endIndex)
	{
		int			currentListIndex = so->listIndex++;
		BlockNumber searchPage = so->listPages[currentListIndex];

		if (so->progressiveShadow)
		{
			uint64		listBit = UINT64CONST(1) << currentListIndex;

			if ((so->shadowScannedLists & listBit) != 0)
				elog(ERROR, "IVFFlat progressive shadow attempted to scan list %d twice",
					 currentListIndex);
			so->shadowScannedLists |= listBit;
		}

		/* Search all entry pages for list */
		while (BlockNumberIsValid(searchPage))
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoffno;
#ifdef IVFFLAT_DISTANCE_PATH
			/* Only a scalar distance survives to the next offset, never a Vector *. */
			OffsetNumber pairedOffset = InvalidOffsetNumber;
			float		pairedDistance = 0.0f;
#endif
#ifdef IVFFLAT_PROFILE_2B
			uint64		pageCandidates = 0;
#endif

#ifdef IVFFLAT_BENCH
			so->profile_pages++;
#endif
			if (so->progressiveShadow)
				so->shadowPagesSeen++;

			buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoffno = PageGetMaxOffsetNumber(page);

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
				bool		fusedDistance = false;
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
				if (!isnull)
					pageCandidates++;
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
#ifdef IVFFLAT_DISTANCE_PATH
				if (so->useFused2 && offno == pairedOffset)
				{
					slot->tts_values[0] = Float8GetDatum((double) pairedDistance);
					pairedOffset = InvalidOffsetNumber;
#ifdef IVFFLAT_PROFILE_2B
					fusedDistance = true;
#endif
				}
				else if (so->useDirectL2)
				{
					bool		paired = false;

					/* Preserve the existing offset scan and tuplesort input order. */
					if (so->useFused2 && offno < maxoffno && !isnull &&
						!VARATT_IS_EXTENDED(DatumGetPointer(datum)))
					{
						OffsetNumber nextoff = OffsetNumberNext(offno);
						ItemId		nextid = PageGetItemId(page, nextoff);

						if (ItemIdIsNormal(nextid))
						{
							IndexTuple nexttuple = (IndexTuple) PageGetItem(page, nextid);
							bool		nextnull;
							Datum		nextdatum = index_getattr(nexttuple, 1, tupdesc, &nextnull);

							if (!nextnull && !VARATT_IS_EXTENDED(DatumGetPointer(nextdatum)))
							{
								Vector	   *a = (Vector *) DatumGetPointer(datum);
								Vector	   *b = (Vector *) DatumGetPointer(nextdatum);

								/* Unusual representations/dimensions retain DIRECT safety. */
								if (a->dim == so->directQuery->dim && b->dim == a->dim)
								{
									float		distance;

									VectorL2SquaredDistancePairRaw(a->dim, so->directQuery->x,
																 a->x, b->x, &distance, &pairedDistance);
									slot->tts_values[0] = Float8GetDatum((double) distance);
									pairedOffset = nextoff;
									paired = true;
#ifdef IVFFLAT_PROFILE_2B
									fusedDistance = true;
									so->profile_fused_pair_calls++;
									so->profile_fused_candidates += 2;
#endif
								}
							}
						}
					}
					if (!paired)
					{
						slot->tts_values[0] = DirectL2Distance(so, datum);
#ifdef IVFFLAT_PROFILE_2B
						so->profile_direct_distance_calls++;
						if (so->useFused2)
						{
							if (offno == maxoffno)
								so->profile_single_tail_candidates++;
							else
								so->profile_fused_fallback_candidates++;
						}
#endif
					}
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
				if (fusedDistance)
					so->profile_fused_distance_ns += INSTR_TIME_GET_NANOSEC(elapsed);
				else if (so->useDirectL2)
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
				ShadowTopKUpdate(so, slot->tts_values[0], &itup->t_tid);
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

#ifdef IVFFLAT_DISTANCE_PATH
			Assert(pairedOffset == InvalidOffsetNumber);
#endif
#ifdef IVFFLAT_PROFILE_2B
			so->profile_page_candidates[Min(pageCandidates, 4)]++;
			so->profile_max_candidates_per_page = Max(so->profile_max_candidates_per_page, pageCandidates);
#endif
			searchPage = IvfflatPageGetOpaque(page)->nextblkno;

			UnlockReleaseBuffer(buf);
		}
	}

	if (performSort)
	{
#ifdef IVFFLAT_BENCH
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
#else
		tuplesort_performsort(so->sortstate);
#endif
	}

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

/* Preserve the original one-batch behavior when progressive scan is off. */
static void
GetScanItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	int			startIndex = so->listIndex;
	int			endIndex = Min(startIndex + so->probes, so->maxProbes);

	GetScanItemsRange(scan, value, startIndex, endIndex, true, true);
}

/* Build only the frozen tree's low-cost fields from existing scan state. */
static void
D2PBuildFeatures(IvfflatScanOpaque so, D2PFeatures *features, int stage)
{
	IvfflatShadowSnapshot *snapshot = stage == 16 ?
		&so->shadowSnapshots[0] : &so->shadowSnapshots[1];
	double		d1 = so->listDistances[0];
	double		mean = 0;
	double		variance = 0;

	Assert(snapshot->count >= 10);
	MemSet(features, 0, sizeof(*features));
	features->d16 = so->listDistances[15];
	features->d64 = so->listDistances[63];
	if (d1 != 0)
	{
		features->d4_d1 = so->listDistances[3] / d1;
		features->d32_d1 = so->listDistances[31] / d1;
		features->d64_d1 = so->listDistances[63] / d1;
		features->gap64_32_d1 =
			(so->listDistances[63] - so->listDistances[31]) / d1;
		features->d2_d1 = so->listDistances[1] / d1;
		features->gap32_16_d1 =
			(so->listDistances[31] - so->listDistances[15]) / d1;
	}

	features->s16_pages = (double) so->shadowSnapshots[0].pagesSeen;
	features->s16_candidates = (double) so->shadowSnapshots[0].candidatesSeen;
	if (so->shadowSnapshots[0].candidatesSeen > 0)
		features->s16_replacement_rate =
			(double) so->shadowSnapshots[0].replacements /
			(double) so->shadowSnapshots[0].candidatesSeen;

	if (stage == 32)
	{
		for (int i = 0; i < 10; i++)
			mean += snapshot->items[i].distance;
		mean /= 10.0;
		for (int i = 0; i < 10; i++)
		{
			double		delta = snapshot->items[i].distance - mean;

			variance += delta * delta;
		}
		features->s32_top10_std = sqrt(variance / 10.0);
		if (snapshot->candidatesSeen > 0)
			features->s32_replacement_rate =
				(double) snapshot->replacements / (double) snapshot->candidatesSeen;
		if (so->shadowSnapshots[0].candidatesSeen > 0)
			features->candidate_growth_16_32 =
				(double) (snapshot->candidatesSeen - so->shadowSnapshots[0].candidatesSeen) /
				(double) so->shadowSnapshots[0].candidatesSeen;
		if (so->shadowSnapshots[0].items[9].distance != 0)
			features->kth10_relative_change_16_32 =
				(so->shadowSnapshots[0].items[9].distance - snapshot->items[9].distance) /
				so->shadowSnapshots[0].items[9].distance;
	}
}

static bool
D2PShouldStop(IvfflatScanOpaque so, int stage)
{
	D2PFeatures features;
	double		probability;

	D2PBuildFeatures(so, &features, stage);
	probability = stage == 16 ? D2PStage16SafeProbability(features) :
		D2PStage32SafeProbability(features);
	if (ivfflat_progressive_scan_debug)
		elog(INFO, "IVFFLAT_PROGRESSIVE_POLICY stage=%d safe_probability=%.17g threshold=%.17g stop=%d",
			 stage, probability,
			 stage == 16 ? D2P_STAGE16_THRESHOLD : D2P_STAGE32_THRESHOLD,
			 probability >= (stage == 16 ? D2P_STAGE16_THRESHOLD :
								 D2P_STAGE32_THRESHOLD));
	return probability >= (stage == 16 ? D2P_STAGE16_THRESHOLD :
						   D2P_STAGE32_THRESHOLD);
}

/* Shadow completes all stages; on may finalize after the 16/32 snapshot. */
static void
GetProgressiveItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	Assert(so->listIndex == 0);
	Assert(so->listCount >= 64);
	Assert(so->maxProbes == 64);
	Assert(!so->boundedActive);

	GetScanItemsRange(scan, value, 0, 16, true, false);
	ShadowSnapshot(so, 0, 16);
	if (so->progressiveEarlyStop && D2PShouldStop(so, 16))
	{
		so->probes = 16;
		so->progressiveStopStage = 16;
		GetScanItemsRange(scan, value, 16, 16, false, true);
		return;
	}
	GetScanItemsRange(scan, value, 16, 32, false, false);
	ShadowSnapshot(so, 1, 32);
	if (so->progressiveEarlyStop && D2PShouldStop(so, 32))
	{
		so->probes = 32;
		so->progressiveStopStage = 32;
		GetScanItemsRange(scan, value, 32, 32, false, true);
		return;
	}
	GetScanItemsRange(scan, value, 32, 64, false, true);
	ShadowSnapshot(so, 2, 64);
	so->probes = 64;
	so->progressiveStopStage = 64;

	if (!so->progressiveEarlyStop && so->shadowScannedLists != UINT64_MAX)
		elog(ERROR, "IVFFlat progressive shadow did not scan each of 64 lists exactly once");
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

#ifdef IVFFLAT_DISTANCE_PATH
		so->useFused2 = so->directL2Eligible &&
			ivfflat_distance_path == IVFFLAT_DISTANCE_PATH_FUSED2;
		so->useDirectL2 = so->directL2Eligible &&
			(ivfflat_distance_path == IVFFLAT_DISTANCE_PATH_DIRECT || so->useFused2);
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
	bool		progressiveShadow =
		ivfflat_progressive_scan != IVFFLAT_PROGRESSIVE_SCAN_OFF;
	bool		progressiveEarlyStop =
		ivfflat_progressive_scan == IVFFLAT_PROGRESSIVE_SCAN_ON;
	MemoryContext oldCtx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	/* Get lists and dimensions from metapage */
	IvfflatGetMetaPageInfo(index, &lists, &dimensions);

	if (progressiveShadow)
	{
		if (lists < 64)
			ereport(ERROR,
					(errmsg("ivfflat progressive scan requires an index with at least 64 lists")));
		if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
			ereport(ERROR,
					(errmsg("ivfflat progressive scan is incompatible with iterative_scan")));
		if (ivfflat_adaptive_probes)
			ereport(ERROR,
					(errmsg("ivfflat progressive scan is incompatible with adaptive_probes")));
		if (ivfflat_experimental_sort_bound > 0 || ivfflat_bounded_scan)
			ereport(ERROR,
					(errmsg("ivfflat progressive scan is incompatible with bounded scan experiments")));

		/* Shadow semantics are fixed at 16 -> 32 -> 64. */
		probes = 64;
		maxProbes = 64;
	}
	else
	{
		if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
			maxProbes = Max(ivfflat_max_probes, probes);
		else
			maxProbes = probes;

		if (probes > lists)
			probes = lists;

		if (maxProbes > lists)
			maxProbes = lists;
	}

	so = palloc_object(IvfflatScanOpaqueData);
	so->typeInfo = IvfflatGetTypeInfo(index);
	so->first = true;
	so->probes = probes;
	so->maxProbes = maxProbes;
	so->dimensions = dimensions;
	so->progressiveShadow = progressiveShadow;
	so->progressiveEarlyStop = progressiveEarlyStop;
	so->progressiveStopStage = 0;
	so->shadowK = 0;
	so->shadowCapacity = 0;
	so->shadowTopCount = 0;
	so->shadowWorst = 0;
	so->shadowTopItems = NULL;
	MemSet(so->shadowSnapshots, 0, sizeof(so->shadowSnapshots));
	so->shadowCandidatesSeen = 0;
	so->shadowPagesSeen = 0;
	so->shadowDistanceCalls = 0;
	so->shadowReplacements = 0;
	so->shadowScannedLists = 0;
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
#ifdef IVFFLAT_DISTANCE_PATH
	so->directL2Eligible = false;
	so->useDirectL2 = false;
	so->useFused2 = false;
	so->directQuery = NULL;
	so->directQueryNeedsFree = false;
#endif

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	so->collation = index->rd_indcollation[0];
#ifdef IVFFLAT_DISTANCE_PATH
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
	so->listDistances = palloc_array_checked(double, maxProbes);
	so->listCount = 0;
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
	so->profile_fused_pair_calls = so->profile_fused_candidates = 0;
	so->profile_single_tail_candidates = so->profile_fused_fallback_candidates = 0;
	so->profile_fused_distance_ns = 0;
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

	if (so->progressiveShadow &&
		(ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF ||
		 ivfflat_adaptive_probes || so->sortBound > 0 || ivfflat_bounded_scan))
		ereport(ERROR,
				(errmsg("ivfflat progressive scan cannot be combined with adaptive, iterative, or bounded scan modes")));
#ifdef IVFFLAT_DISTANCE_PATH
	if (so->directQueryNeedsFree)
		pfree(so->directQuery);
	so->directQuery = NULL;
	so->directQueryNeedsFree = false;
	so->useDirectL2 = false;
	so->useFused2 = false;
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
	so->listCount = 0;
	so->listIndex = 0;
	so->progressiveStopStage = 0;

	if (so->progressiveShadow)
	{
		int64		requestedK = 40;

		/* xs_tuple_bound includes LIMIT+OFFSET when the executor can supply it. */
		if (so->logicalBound > 0)
		{
			if (so->logicalBound > INT_MAX / 4)
				ereport(ERROR,
						(errmsg("LIMIT+OFFSET is too large for IVFFlat shadow Top-K")));
			requestedK = Max(INT64CONST(40), so->logicalBound * 4);
		}
		so->shadowK = (int) requestedK;

		if (so->shadowCapacity < so->shadowK)
		{
			if (so->shadowTopItems == NULL)
				so->shadowTopItems = MemoryContextAlloc(so->tmpCtx,
					mul_size(sizeof(IvfflatShadowTopItem), so->shadowK));
			else
				so->shadowTopItems = repalloc_array(so->shadowTopItems,
					IvfflatShadowTopItem, so->shadowK);

			for (int i = 0; i < 3; i++)
			{
				if (so->shadowSnapshots[i].items == NULL)
					so->shadowSnapshots[i].items = MemoryContextAlloc(so->tmpCtx,
						mul_size(sizeof(IvfflatShadowTopItem), so->shadowK));
				else
					so->shadowSnapshots[i].items = repalloc_array(
						so->shadowSnapshots[i].items,
						IvfflatShadowTopItem, so->shadowK);
			}
			so->shadowCapacity = so->shadowK;
		}

		so->shadowTopCount = 0;
		so->shadowWorst = 0;
		so->shadowCandidatesSeen = 0;
		so->shadowPagesSeen = 0;
		so->shadowDistanceCalls = 0;
		so->shadowReplacements = 0;
		so->shadowScannedLists = 0;
		for (int i = 0; i < 3; i++)
		{
			so->shadowSnapshots[i].stage = 0;
			so->shadowSnapshots[i].count = 0;
			so->shadowSnapshots[i].kthDistance = get_float8_infinity();
			so->shadowSnapshots[i].candidatesSeen = 0;
			so->shadowSnapshots[i].pagesSeen = 0;
			so->shadowSnapshots[i].distanceCalls = 0;
			so->shadowSnapshots[i].replacements = 0;
		}
	}

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
		if (so->progressiveShadow)
			GetProgressiveItems(scan, value);
		else
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
	elog(INFO, "IVFFLAT_PROFILE_2B probes=%d dimensions=%d selected_lists=%d progressive_stop_stage=%d scanned_candidates=%llu scanned_pages=%llu distance_calls=%llu tuplesort_input_calls=%llu returned_rows=%llu distance_path_requested=%d direct_l2_eligible=%d direct_l2_active=%d generic_distance_calls=%llu direct_distance_calls=%llu generic_distance_ns=%llu direct_distance_ns=%llu candidate_extract_ns=%llu distance_ns=%llu tuple_materialization_ns=%llu sort_insert_ns=%llu sort_finalize_ns=%llu scan_items_total_ns=%llu page_candidates_0=%llu page_candidates_1=%llu page_candidates_2=%llu page_candidates_3=%llu page_candidates_4plus=%llu max_candidates_per_page=%llu fused2_active=%d fused_pair_calls=%llu fused_candidates=%llu single_tail_candidates=%llu fused_fallback_candidates=%llu fused_distance_ns=%llu",
		 so->probes, so->dimensions, so->listIndex, so->progressiveStopStage,
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
		 (unsigned long long) so->profile_max_candidates_per_page, so->useFused2 ? 1 : 0,
		 (unsigned long long) so->profile_fused_pair_calls,
		 (unsigned long long) so->profile_fused_candidates,
		 (unsigned long long) so->profile_single_tail_candidates,
		 (unsigned long long) so->profile_fused_fallback_candidates,
		 (unsigned long long) so->profile_fused_distance_ns);
#endif
#endif

	/* Free any temporary files */
	tuplesort_end(so->sortstate);

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
