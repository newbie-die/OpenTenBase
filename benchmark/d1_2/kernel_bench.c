#define _GNU_SOURCE
#include <errno.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DIM 960
#define DEFAULT_TRIALS 9
#define DEFAULT_TARGET_CANDIDATES 1000000

typedef struct __attribute__((packed)) SampleMeta
{
	int32_t query_slot;
	int32_t query_id;
	int32_t probes;
	uint32_t tid_block;
	uint16_t tid_offset;
	uint16_t reserved;
	uint64_t source_row;
	double threshold;
	double production_full;
} SampleMeta;

typedef struct KernelResult
{
	float score;
	uint16_t evaluated_dims;
	uint16_t checks;
	uint8_t early;
} KernelResult;

typedef struct Workload
{
	const char *name;
	int probes;
	size_t begin;
	size_t count;
	int no_abandon;
} Workload;

static volatile double timing_sink;

static inline float
horizontal_sum(__m256 value)
{
	__m128 low = _mm256_castps256_ps128(value);
	__m128 high = _mm256_extractf128_ps(value, 1);
	__m128 sum = _mm_add_ps(low, high);
	__m128 upper = _mm_movehl_ps(sum, sum);
	sum = _mm_add_ps(sum, upper);
	__m128 lane1 = _mm_shuffle_ps(sum, sum, 0x55);
	sum = _mm_add_ss(sum, lane1);
	return _mm_cvtss_f32(sum);
}

__attribute__((noinline)) float
l2_full_avx2(const float *candidate, const float *query)
{
	__m256 accumulated = _mm256_setzero_ps();
	for (int dim = 0; dim < DIM; dim += 8)
	{
		__m256 a = _mm256_loadu_ps(candidate + dim);
		__m256 q = _mm256_loadu_ps(query + dim);
		__m256 difference = _mm256_sub_ps(a, q);
		accumulated = _mm256_fmadd_ps(difference, difference, accumulated);
	}
	return horizontal_sum(accumulated);
}

__attribute__((noinline)) KernelResult
l2_progressive_avx2(const float *candidate, const float *query,
					float threshold, int block_size)
{
	KernelResult result = {0};
	__m256 accumulated = _mm256_setzero_ps();
	int start;

	for (start = 0; start < DIM; start += block_size)
	{
		int end = start + block_size;
		if (end > DIM)
			end = DIM;
		for (int dim = start; dim < end; dim += 8)
		{
			__m256 a = _mm256_loadu_ps(candidate + dim);
			__m256 q = _mm256_loadu_ps(query + dim);
			__m256 difference = _mm256_sub_ps(a, q);
			accumulated = _mm256_fmadd_ps(difference, difference, accumulated);
		}
		result.checks++;
		result.score = horizontal_sum(accumulated);
		result.evaluated_dims = (uint16_t) end;
		if (result.score > threshold)
		{
			result.early = 1;
			return result;
		}
	}
	return result;
}

static double
seconds_now(void)
{
	struct timespec timestamp;
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0)
	{
		perror("clock_gettime");
		exit(2);
	}
	return timestamp.tv_sec + timestamp.tv_nsec * 1e-9;
}

static int
pin_first_allowed_cpu(void)
{
	cpu_set_t allowed;
	cpu_set_t selected;
	if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
	{
		perror("sched_getaffinity");
		exit(2);
	}
	for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
	{
		if (!CPU_ISSET(cpu, &allowed))
			continue;
		CPU_ZERO(&selected);
		CPU_SET(cpu, &selected);
		if (sched_setaffinity(0, sizeof(selected), &selected) != 0)
		{
			perror("sched_setaffinity");
			exit(2);
		}
		return cpu;
	}
	fprintf(stderr, "no allowed CPU\n");
	exit(2);
}

static double
run_trial(const Workload *workload, const SampleMeta *metadata,
		  const float *candidates, const float *queries,
		  int block_size, int baseline, int passes)
{
	double start = seconds_now();
	double local_sink = 0.0;
	uint64_t local_dims = 0;
	for (int pass = 0; pass < passes; pass++)
	{
		for (size_t relative = 0; relative < workload->count; relative++)
		{
			size_t index = workload->begin + relative;
			const SampleMeta *meta = &metadata[index];
			const float *candidate = candidates + index * DIM;
			const float *query = queries + (size_t) meta->query_slot * DIM;
			if (baseline)
			{
				local_sink += l2_full_avx2(candidate, query);
				local_dims += DIM;
			}
			else
			{
				float threshold = workload->no_abandon ? INFINITY : (float) meta->threshold;
				KernelResult result = l2_progressive_avx2(candidate, query, threshold, block_size);
				local_sink += result.score;
				local_dims += result.evaluated_dims;
			}
		}
	}
	timing_sink += local_sink + (double) local_dims * 1e-30;
	return seconds_now() - start;
}

static void
write_progress(const char *path, const char *workload, const char *config,
			   int complete, int total, double elapsed)
{
	FILE *file = fopen(path, "w");
	if (!file)
	{
		perror("progress file");
		exit(2);
	}
	double eta = complete ? elapsed * (total - complete) / complete : 0.0;
	fprintf(file,
			"{\"status\":\"RUNNING\",\"current_workload\":\"%s\","
			"\"current_config\":\"%s\",\"completed_configs\":%d,"
			"\"total_configs\":%d,\"elapsed_seconds\":%.6f,"
			"\"eta_seconds\":%.6f}\n",
			workload, config, complete, total, elapsed, eta);
	fclose(file);
}

int
main(int argc, char **argv)
{
	FILE *sample;
	FILE *timing;
	FILE *correctness;
	char magic[8];
	uint32_t version;
	uint32_t dimension;
	uint32_t query_count;
	uint32_t record_count;
	int32_t *query_ids;
	float *queries;
	SampleMeta *metadata;
	float *candidates;
	Workload workloads[5];
	size_t probe_begin[3] = {0};
	size_t probe_count[3] = {0};
	const int blocks[2] = {64, 128};
	char timing_path[4096];
	char correctness_path[4096];
	char progress_path[4096];
	double benchmark_start;
	int pinned_cpu;
	int completed = 0;
	const int total_configs = 15;
	int trials = DEFAULT_TRIALS;
	int target_candidates = DEFAULT_TARGET_CANDIDATES;
	const char *trials_environment = getenv("D1_BENCH_TRIALS");
	const char *target_environment = getenv("D1_BENCH_TARGET_CANDIDATES");

	if (trials_environment)
		trials = atoi(trials_environment);
	if (target_environment)
		target_candidates = atoi(target_environment);
	if (trials <= 0 || trials > 100 || target_candidates <= 0)
	{
		fprintf(stderr, "invalid smoke/timing environment override\n");
		return 2;
	}

	if (argc != 3)
	{
		fprintf(stderr, "usage: %s phase_d1_2_sample.bin output_directory\n", argv[0]);
		return 2;
	}
	sample = fopen(argv[1], "rb");
	if (!sample)
	{
		fprintf(stderr, "cannot open %s: %s\n", argv[1], strerror(errno));
		return 2;
	}
	if (fread(magic, 1, 8, sample) != 8 || memcmp(magic, "D1SAMP2\0", 8) != 0 ||
		fread(&version, 4, 1, sample) != 1 || fread(&dimension, 4, 1, sample) != 1 ||
		fread(&query_count, 4, 1, sample) != 1 || fread(&record_count, 4, 1, sample) != 1 ||
		version != 1 || dimension != DIM || query_count == 0 || record_count == 0)
	{
		fprintf(stderr, "invalid sample header\n");
		return 2;
	}
	query_ids = malloc(query_count * sizeof(int32_t));
	if (posix_memalign((void **) &queries, 64, (size_t) query_count * DIM * sizeof(float)) != 0 ||
		posix_memalign((void **) &candidates, 64, (size_t) record_count * DIM * sizeof(float)) != 0)
	{
		fprintf(stderr, "aligned allocation failed\n");
		return 2;
	}
	metadata = malloc((size_t) record_count * sizeof(SampleMeta));
	if (!query_ids || !metadata)
	{
		fprintf(stderr, "allocation failed\n");
		return 2;
	}
	for (uint32_t q = 0; q < query_count; q++)
	{
		if (fread(&query_ids[q], sizeof(int32_t), 1, sample) != 1 ||
			fread(queries + (size_t) q * DIM, sizeof(float), DIM, sample) != DIM)
		{
			fprintf(stderr, "short query data\n");
			return 2;
		}
	}
	for (uint32_t r = 0; r < record_count; r++)
	{
		if (fread(&metadata[r], sizeof(SampleMeta), 1, sample) != 1 ||
			fread(candidates + (size_t) r * DIM, sizeof(float), DIM, sample) != DIM)
		{
			fprintf(stderr, "short candidate data\n");
			return 2;
		}
	}
	if (fgetc(sample) != EOF)
	{
		fprintf(stderr, "sample has trailing bytes\n");
		return 2;
	}
	fclose(sample);

	for (uint32_t r = 0; r < record_count; r++)
	{
		int probe_index = metadata[r].probes == 64 ? 0 : metadata[r].probes == 128 ? 1 : metadata[r].probes == 256 ? 2 : -1;
		if (probe_index < 0 || metadata[r].query_slot < 0 || (uint32_t) metadata[r].query_slot >= query_count)
		{
			fprintf(stderr, "invalid record metadata\n");
			return 2;
		}
		if (probe_count[probe_index] == 0)
			probe_begin[probe_index] = r;
		else if (probe_begin[probe_index] + probe_count[probe_index] != r)
		{
			fprintf(stderr, "records are not probe-contiguous\n");
			return 2;
		}
		probe_count[probe_index]++;
	}
	workloads[0] = (Workload) {"no_abandon", 0, 0, record_count, 1};
	workloads[1] = (Workload) {"realistic_all", 0, 0, record_count, 0};
	workloads[2] = (Workload) {"realistic_p64", 64, probe_begin[0], probe_count[0], 0};
	workloads[3] = (Workload) {"realistic_p128", 128, probe_begin[1], probe_count[1], 0};
	workloads[4] = (Workload) {"realistic_p256", 256, probe_begin[2], probe_count[2], 0};

	snprintf(timing_path, sizeof(timing_path), "%s/phase_d1_2_timing_raw.csv", argv[2]);
	snprintf(correctness_path, sizeof(correctness_path), "%s/phase_d1_2_correctness.csv", argv[2]);
	snprintf(progress_path, sizeof(progress_path), "%s/progress.json", argv[2]);
	timing = fopen(timing_path, "wx");
	correctness = fopen(correctness_path, "wx");
	if (!timing || !correctness)
	{
		fprintf(stderr, "cannot create benchmark output: %s\n", strerror(errno));
		return 2;
	}
	fprintf(timing, "workload,probes,config,block_size,trial,passes,candidates_per_pass,total_candidates,elapsed_seconds,ns_per_candidate,sink\n");
	fprintf(correctness, "workload,probes,block_size,records,early_abandoned,false_abandon,nonabandon_score_mismatch,evaluated_dimensions,avoided_dimensions,dimension_avoid_ratio,threshold_checks,checks_per_candidate,baseline_vs_production_bitwise_mismatch,max_abs_score_difference\n");

	uint64_t production_mismatch = 0;
	double maximum_difference = 0.0;
	for (uint32_t r = 0; r < record_count; r++)
	{
		float score = l2_full_avx2(candidates + (size_t) r * DIM,
								 queries + (size_t) metadata[r].query_slot * DIM);
		float production = (float) metadata[r].production_full;
		if (memcmp(&score, &production, sizeof(float)) != 0)
			production_mismatch++;
		maximum_difference = fmax(maximum_difference, fabs((double) score - metadata[r].production_full));
	}
	for (int w = 0; w < 5; w++)
	{
		const Workload *workload = &workloads[w];
		for (int b = 0; b < 2; b++)
		{
			uint64_t early = 0, false_abandon = 0, score_mismatch = 0;
			uint64_t evaluated = 0, checks = 0;
			for (size_t relative = 0; relative < workload->count; relative++)
			{
				size_t r = workload->begin + relative;
				const float *candidate = candidates + r * DIM;
				const float *query = queries + (size_t) metadata[r].query_slot * DIM;
				float full = l2_full_avx2(candidate, query);
				float threshold = workload->no_abandon ? INFINITY : (float) metadata[r].threshold;
				KernelResult result = l2_progressive_avx2(candidate, query,
											 threshold, blocks[b]);
				evaluated += result.evaluated_dims;
				checks += result.checks;
				if (result.early)
				{
					early++;
					if (!workload->no_abandon &&
						!(metadata[r].production_full > metadata[r].threshold))
						false_abandon++;
				}
				else if (memcmp(&result.score, &full, sizeof(float)) != 0)
					score_mismatch++;
			}
			uint64_t possible = (uint64_t) workload->count * DIM;
			fprintf(correctness, "%s,%d,%d,%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.17g,%" PRIu64 ",%.17g,%" PRIu64 ",%.17g\n",
					workload->name, workload->probes, blocks[b], workload->count,
					early, false_abandon, score_mismatch, evaluated,
					possible - evaluated, (double) (possible - evaluated) / possible,
					checks, (double) checks / workload->count,
					production_mismatch, maximum_difference);
			if (false_abandon || score_mismatch)
			{
				fprintf(stderr, "correctness failure for %s block %d\n",
						workload->name, blocks[b]);
				return 3;
			}
		}
	}
	fflush(correctness);
	fclose(correctness);

	pinned_cpu = pin_first_allowed_cpu();
	benchmark_start = seconds_now();
	for (int w = 0; w < 5; w++)
	{
		const Workload *workload = &workloads[w];
		int passes = (target_candidates + (int) workload->count - 1) / (int) workload->count;
		for (int config = 0; config < 3; config++)
		{
			int baseline = config == 0;
			int block_size = baseline ? 0 : blocks[config - 1];
			char config_name[32];
			snprintf(config_name, sizeof(config_name), "%s", baseline ? "baseline" : block_size == 64 ? "progressive_b64" : "progressive_b128");
			write_progress(progress_path, workload->name, config_name, completed, total_configs,
						   seconds_now() - benchmark_start);
			double elapsed_so_far = seconds_now() - benchmark_start;
			fprintf(stderr, "current workload/config=%s/%s completed=%d/%d elapsed=%.1fs ETA=%.1fs\n",
					workload->name, config_name, completed, total_configs, elapsed_so_far,
					completed ? elapsed_so_far * (total_configs - completed) / completed : 0.0);
			(void) run_trial(workload, metadata, candidates, queries, block_size, baseline, 1);
			for (int trial = 1; trial <= trials; trial++)
			{
				double elapsed = run_trial(workload, metadata, candidates, queries,
									   block_size, baseline, passes);
				uint64_t timed_candidates = (uint64_t) workload->count * passes;
				fprintf(timing, "%s,%d,%s,%d,%d,%d,%zu,%" PRIu64 ",%.9f,%.9f,%.17g\n",
						workload->name, workload->probes, config_name, block_size,
						trial, passes, workload->count, timed_candidates, elapsed,
						elapsed * 1e9 / timed_candidates, timing_sink);
				fflush(timing);
			}
			completed++;
			write_progress(progress_path, workload->name, config_name, completed,
						   total_configs, seconds_now() - benchmark_start);
		}
	}
	fclose(timing);
	FILE *progress = fopen(progress_path, "w");
	fprintf(progress,
			"{\"status\":\"COMPLETE\",\"completed_configs\":%d,"
			"\"total_configs\":%d,\"elapsed_seconds\":%.6f,"
			"\"pinned_cpu\":%d}\n",
			total_configs, total_configs, seconds_now() - benchmark_start, pinned_cpu);
	fclose(progress);
	fprintf(stderr, "COMPLETE configs=%d/%d elapsed=%.1fs pinned_cpu=%d sink=%.17g\n",
			total_configs, total_configs, seconds_now() - benchmark_start, pinned_cpu,
			timing_sink);
	free(query_ids);
	free(queries);
	free(metadata);
	free(candidates);
	return 0;
}
