/* Standalone timing driver; kernels are linked from the real vector.o. */
#include "postgres.h"
#include "vector.h"
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Use libc I/O in this standalone driver, not PostgreSQL port wrappers. */
#undef printf
#undef fprintf

#define DIM 960
#define ROUNDS 12
#define PAIRS 524288
static volatile double sink;
static uint32_t rng = 20260907;

static float
random_float(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	return (float) (rng & 65535) / 65536.0f;
}

static uint64_t
now_ns(void)
{
	struct timespec t;
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &t) != 0)
		abort();
	return (uint64_t) t.tv_sec * 1000000000 + t.tv_nsec;
}

/* External calls and consumed outputs prevent DCE; no per-pair volatile access. */
static __attribute__((noinline)) double
run_direct(const float *q, const float *pool, size_t candidates, size_t pairs)
{
	double checksum = 0;
	size_t index = 0;
	for (size_t i = 0; i < pairs; i++)
	{
		const float *a = pool + index * DIM;
		const float *b = a + DIM;
		float da = VectorL2SquaredDistanceRaw(DIM, a, q);
		float db = VectorL2SquaredDistanceRaw(DIM, b, q);
		checksum += (double) da + (double) db;
		index += 2;
		if (index == candidates)
			index = 0;
	}
	return checksum;
}

static __attribute__((noinline)) double
run_fused(const float *q, const float *pool, size_t candidates, size_t pairs)
{
	double checksum = 0;
	size_t index = 0;
	for (size_t i = 0; i < pairs; i++)
	{
		const float *a = pool + index * DIM;
		const float *b = a + DIM;
		float da, db;
		VectorL2SquaredDistancePairRaw(DIM, q, a, b, &da, &db);
		checksum += (double) da + (double) db;
		index += 2;
		if (index == candidates)
			index = 0;
	}
	return checksum;
}

static void
run_case(const char *name, size_t candidates)
{
	float *q, *pool;
	size_t mismatches = 0;
	if (posix_memalign((void **) &q, 64, DIM * sizeof(float)) != 0 ||
		posix_memalign((void **) &pool, 64, candidates * DIM * sizeof(float)) != 0)
		abort();
	for (size_t i = 0; i < DIM; i++)
		q[i] = random_float();
	for (size_t i = 0; i < candidates * DIM; i++)
		pool[i] = random_float();
	/* Untimed full-pool check against the actual DIRECT kernel. */
	for (size_t i = 0; i < candidates; i += 2)
	{
		float da = VectorL2SquaredDistanceRaw(DIM, pool + i * DIM, q);
		float db = VectorL2SquaredDistanceRaw(DIM, pool + (i + 1) * DIM, q);
		float fa, fb;
		VectorL2SquaredDistancePairRaw(DIM, q, pool + i * DIM,
									 pool + (i + 1) * DIM, &fa, &fb);
		mismatches += memcmp(&da, &fa, sizeof(float)) != 0;
		mismatches += memcmp(&db, &fb, sizeof(float)) != 0;
	}
	fprintf(stderr, "%s candidates=%zu candidate_bytes=%zu query_bytes=%zu bit_mismatches=%zu\n",
			name, candidates, candidates * DIM * sizeof(float), DIM * sizeof(float), mismatches);
	if (mismatches != 0)
		abort();
	for (int w = 0; w < 4; w++)
	{
		sink += run_direct(q, pool, candidates, PAIRS / 8);
		sink += run_fused(q, pool, candidates, PAIRS / 8);
	}
	for (int round = 0; round < ROUNDS; round++)
	{
		const char *order = round % 2 == 0 ? "DFFD" : "FDDF";
		for (int slot = 0; slot < 4; slot++)
		{
			uint64_t start, end;
			double checksum;
			start = now_ns();
			if (order[slot] == 'D')
				checksum = run_direct(q, pool, candidates, PAIRS);
			else
				checksum = run_fused(q, pool, candidates, PAIRS);
			end = now_ns();
			sink += checksum;
			printf("%s,%d,%d,%c,%d,%llu,%.9f,%.17g\n", name, round + 1, slot + 1,
				   order[slot], PAIRS, (unsigned long long) (end - start),
				   (double) (end - start) / PAIRS, checksum);
			fflush(stdout);
		}
	}
	free(pool);
	free(q);
}

int
main(int argc, char **argv)
{
	cpu_set_t cpus;
	int cpu;
	if (argc != 2)
		return 2;
	cpu = atoi(argv[1]);
	if (cpu < 0 || cpu >= CPU_SETSIZE)
		return 2;
	CPU_ZERO(&cpus);
	CPU_SET(cpu, &cpus);
	if (sched_setaffinity(0, sizeof(cpus), &cpus) != 0)
		return 3;
	fprintf(stderr, "cpu=%d seed=20260907 dim=%d rounds=%d pairs_per_sample=%d\n", cpu, DIM, ROUNDS, PAIRS);
	puts("case,round,slot,path,pairs,elapsed_ns,ns_per_pair,checksum");
	run_case("hot", 4);
	run_case("ivfflat_like", 4096);
	fprintf(stderr, "sink=%.17g\n", sink);
	return 0;
}
