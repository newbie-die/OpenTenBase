#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TARGET_PER_PROBE 16384
#define REQUIRED_FIELDS 81

typedef struct SampleRow
{
	uint64_t source_row;
	uint64_t candidate_ordinal;
	int query_id;
	int probes;
	unsigned block;
	unsigned offset;
	double full_distance;
	double threshold;
} SampleRow;

typedef struct Reservoir
{
	int probes;
	uint64_t seen;
	uint64_t state;
	int count;
	SampleRow rows[TARGET_PER_PROBE];
} Reservoir;

static uint64_t
next_random(Reservoir *reservoir)
{
	uint64_t x = reservoir->state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	reservoir->state = x;
	return x * UINT64_C(2685821657736338717);
}

/* Split in place while honoring the quoted TID field's embedded comma. */
static int
split_csv(char *line, char **fields, int capacity)
{
	char *read = line;
	char *write = line;
	int count = 0;
	int quoted = 0;

	if (capacity <= 0)
		return 0;
	fields[count++] = write;
	while (*read)
	{
		if (*read == '"')
		{
			quoted = !quoted;
			read++;
			continue;
		}
		if (*read == ',' && !quoted)
		{
			*write++ = '\0';
			read++;
			if (count < capacity)
				fields[count++] = write;
			continue;
		}
		if (*read == '\r' || *read == '\n')
		{
			read++;
			continue;
		}
		*write++ = *read++;
	}
	*write = '\0';
	return count;
}

static Reservoir *
find_reservoir(Reservoir reservoirs[3], int probes)
{
	for (int i = 0; i < 3; i++)
		if (reservoirs[i].probes == probes)
			return &reservoirs[i];
	return NULL;
}

static int
compare_source_row(const void *left, const void *right)
{
	const SampleRow *a = left;
	const SampleRow *b = right;
	return a->source_row < b->source_row ? -1 : a->source_row > b->source_row;
}

int
main(int argc, char **argv)
{
	FILE *input;
	FILE *output;
	char *line = NULL;
	size_t line_capacity = 0;
	ssize_t length;
	uint64_t source_row = 0;
	char *fields[REQUIRED_FIELDS];
	Reservoir reservoirs[3] = {
		{64, 0, UINT64_C(0x6400d11), 0, {{0}}},
		{128, 0, UINT64_C(0x12800d11), 0, {{0}}},
		{256, 0, UINT64_C(0x25600d11), 0, {{0}}}
	};

	if (argc != 3)
	{
		fprintf(stderr, "usage: %s phase_d1_1_raw.csv sample_metadata.csv\n", argv[0]);
		return 2;
	}
	input = fopen(argv[1], "r");
	if (!input)
	{
		fprintf(stderr, "cannot open %s: %s\n", argv[1], strerror(errno));
		return 2;
	}
	/* Discard and validate the header. */
	if ((length = getline(&line, &line_capacity, input)) <= 0 ||
		strstr(line, "k40_b64_threshold_before_candidate") == NULL)
	{
		fprintf(stderr, "unexpected D1-1 raw header\n");
		return 2;
	}
	while ((length = getline(&line, &line_capacity, input)) > 0)
	{
		Reservoir *reservoir;
		SampleRow row;
		uint64_t replacement;
		int count;
		int heap_ready;

		source_row++;
		count = split_csv(line, fields, REQUIRED_FIELDS);
		if (count != REQUIRED_FIELDS)
		{
			fprintf(stderr, "CSV field count %d at source row %" PRIu64 "\n", count, source_row);
			return 2;
		}
		row.probes = atoi(fields[2]);
		reservoir = find_reservoir(reservoirs, row.probes);
		if (!reservoir)
			continue;
		heap_ready = atoi(fields[55]);
		if (!heap_ready)
			continue;
		row.source_row = source_row;
		row.query_id = atoi(fields[1]);
		row.candidate_ordinal = strtoull(fields[3], NULL, 10);
		row.full_distance = strtod(fields[7], NULL);
		row.threshold = strtod(fields[54], NULL);
		if (sscanf(fields[6], "(%u,%u)", &row.block, &row.offset) != 2)
		{
			fprintf(stderr, "bad TID at source row %" PRIu64 "\n", source_row);
			return 2;
		}
		if (!(row.threshold >= 0.0) || !(row.full_distance >= 0.0))
		{
			fprintf(stderr, "bad score at source row %" PRIu64 "\n", source_row);
			return 2;
		}
		reservoir->seen++;
		if (reservoir->count < TARGET_PER_PROBE)
			reservoir->rows[reservoir->count++] = row;
		else
		{
			replacement = next_random(reservoir) % reservoir->seen;
			if (replacement < TARGET_PER_PROBE)
				reservoir->rows[replacement] = row;
		}
		if (source_row % UINT64_C(10000000) == 0)
			fprintf(stderr, "metadata scan rows=%" PRIu64 "\n", source_row);
	}
	fclose(input);
	free(line);
	if (source_row != UINT64_C(94323824))
	{
		fprintf(stderr, "unexpected raw row count: %" PRIu64 "\n", source_row);
		return 2;
	}
	output = fopen(argv[2], "wx");
	if (!output)
	{
		fprintf(stderr, "cannot create %s: %s\n", argv[2], strerror(errno));
		return 2;
	}
	fprintf(output, "sample_id,source_row,query_id,probes,candidate_ordinal,candidate_tid,full_distance_squared,threshold_before_candidate\n");
	int sample_id = 0;
	for (int r = 0; r < 3; r++)
	{
		Reservoir *reservoir = &reservoirs[r];
		if (reservoir->count != TARGET_PER_PROBE)
		{
			fprintf(stderr, "insufficient sample at probes=%d\n", reservoir->probes);
			return 2;
		}
		qsort(reservoir->rows, reservoir->count, sizeof(SampleRow), compare_source_row);
		for (int i = 0; i < reservoir->count; i++)
		{
			SampleRow *row = &reservoir->rows[i];
			fprintf(output, "%d,%" PRIu64 ",%d,%d,%" PRIu64 ",\"(%u,%u)\",%.17g,%.17g\n",
					sample_id++, row->source_row, row->query_id, row->probes,
					row->candidate_ordinal, row->block, row->offset,
					row->full_distance, row->threshold);
		}
		fprintf(stderr, "probes=%d eligible=%" PRIu64 " sampled=%d\n",
				reservoir->probes, reservoir->seen, reservoir->count);
	}
	fclose(output);
	return 0;
}
