#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "core.h"
#include "perf.h"
#include "nob.h"

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

PerfEntry __perf_entries[HASHTABLE_SIZE];
HashTable __perf_ht = {
    .entries = __perf_entries,
    .count = 0,
    .capacity = HASHTABLE_SIZE,
};

_Thread_local PerfEntry* __current_perf_entry = NULL;

// This uses FNV-1a hashing algorithm
static inline uint64_t hash_key(const char* key)
{
    uint64_t hash = FNV_OFFSET;
    for (const char* p = key; *p; p++) {
        hash ^= (uint64_t)(uint8_t)(*p);
        hash *= FNV_PRIME;
    }
    return hash;
}

static inline size_t get_idx(HashTable* ht, const char* key) {
    uint64_t hash = hash_key(key);
    return (size_t)(hash & (ht->capacity-1));
}

static PerfEntry* find_or_create_entry(HashTable* ht, const char* name) {
    size_t idx = get_idx(ht, name);
    PerfEntry* p = ht->entries + idx;
    PerfEntry* end = ht->entries + ht->capacity;

    while (p < end && p->name != NULL) {
        if (strcmp(p->name, name) == 0) {
            return p;
        }
        p++;
    }

    if (p >= end) {
        fprintf(stderr, "ERROR: Benchmark hash table is full! Cannot create entry for '%s'\n", name);
        return NULL;
    }

    p->name = name;
    p->total_time = 0.0;
    p->min_time = DBL_MAX;
    p->max_time = 0.0;
    p->count = 0;
    p->current_start = 0.0;
    p->flop = 0;
    p->bytes = 0;
    p->is_active = false;
    ht->count++;

    return p;
}

void perf_start(HashTable* ht, const char* name, double start_time)
{
    PerfEntry* entry = find_or_create_entry(ht, name);
    if (entry == NULL) return;

    if (entry->is_active) {
        fprintf(stderr, "WARNING: Benchmark '%s' is already active! Previous timing will be lost.\n", name);
    }
    __current_perf_entry = entry;

    entry->current_start = start_time;
    entry->is_active = true;
}

void perf_stop(HashTable* ht, const char* name, double stop_time)
{
    PerfEntry* entry;
    if (__current_perf_entry == NULL) entry = find_or_create_entry(ht, name);
    else entry = __current_perf_entry;

    if (!entry->is_active) {
        fprintf(stderr, "ERROR: Benchmark '%s' was never started!\n", name);
        return;
    }

    double elapsed = stop_time - entry->current_start;

    entry->total_time += elapsed;
    entry->count++;
    entry->min_time = fmin(elapsed, entry->min_time);
    entry->max_time = fmax(elapsed, entry->max_time);
    entry->is_active = false;
}

void perf_add_metric(HashTable* ht, const char* name, OpMetrics metrics)
{
    PerfEntry* entry;
    if (__current_perf_entry == NULL) entry = find_or_create_entry(ht, name);
    else entry = __current_perf_entry;

    if (!entry->is_active) {
        fprintf(stderr, "ERROR: Benchmark '%s' was never started!\n", name);
        return;
    }

    entry->has_metrics = true;
    entry->flop += metrics.flops;
    entry->bytes += metrics.bytes;
}

void perf_clear(HashTable* ht)
{
    memset(ht->entries, 0, sizeof(PerfEntry) * ht->capacity);
    ht->count = 0;
}

static int cmp_by_total_time(const void* a, const void* b)
{
    PerfEntry* ea = (PerfEntry*)a;
    PerfEntry* eb = (PerfEntry*)b;

    // Sort by avg time (descending)
    if ((eb->total_time) > (ea->total_time)) return 1;
    if ((eb->total_time) < (ea->total_time)) return -1;
    return 0;
}

static PerfEntry* get_sorted_entries(HashTable* ht)
{
    PerfEntry* valid_entries = malloc(ht->capacity * sizeof(PerfEntry));

    PerfEntry* p = ht->entries;
    PerfEntry* end = ht->entries + ht->capacity;
    size_t i = 0;
    while (p < end) {
        if (p->name != NULL && p->count > 0) {
            valid_entries[i++] = *p;
        }
        p++;
    }

    qsort(valid_entries, ht->count, sizeof(PerfEntry), cmp_by_total_time);
    return valid_entries;
}

void perf_print(HashTable* ht)
{

    // PerfEntry* p = ht->entries;
    // PerfEntry* end = ht->entries + ht->capacity;

    printf("%-30s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-8s\n",
           "name", "total(s)", "avg(s)", "min(s)", "max(s)",
           "GFLOP/s", "GB/s", "FLOP/byte", "calls");
    
    const int total_width = 30 + 12*7 + 8 + 8;
    for (int i = 0; i < total_width; i++) printf("-");
    printf("\n");

    PerfEntry* sorted_entries = get_sorted_entries(ht);

    for (size_t i = 0; i < ht->count; i++) {
        PerfEntry* e = &sorted_entries[i];
        double avg = e->total_time / e->count;

        printf("%-30s %-12.6f %-12.6f %-12.6f %-12.6f",
               e->name, e->total_time, avg, e->min_time, e->max_time);

	    if (e->has_metrics) {
	        double gflops = e->flop / e->total_time / 1e9;
	        double bandwidth = e->bytes / e->total_time / 1e9;
	        double intensity = (double)e->flop / e->bytes;
	        printf(" %-12.2f %-12.2f %-12.3f", gflops, bandwidth, intensity);
	    } else {
	        printf(" %-12s %-12s %-12s", "?", "?", "?");
	    }

	    printf(" %-8zu\n", e->count);
    }

    free(sorted_entries);
}

void perf_csv(HashTable* ht, FILE *file)
{
    fprintf(file, "name,total(s),avg(s),min(s),max(s),GFLOP/s,GB/s,FLOP/byte,calls\n");

    PerfEntry* sorted_entries = get_sorted_entries(ht);

    for (size_t i = 0; i < ht->count; i++) {
        PerfEntry* e = &sorted_entries[i];
        double avg = e->total_time / e->count;

        fprintf(file, "%s,%.6f,%.6f,%.6f,%.6f,",
                e->name, e->total_time, avg, e->min_time, e->max_time);

        if (e->has_metrics) {
            double gflops = e->flop / e->total_time / 1e9;
            double bandwidth = e->bytes / e->total_time / 1e9;
            double intensity = (double)e->flop / e->bytes;
            fprintf(file, "%.2f,%.2f,%.3f,", gflops, bandwidth, intensity);
        } else {
            fprintf(file, ",,,");  // Empty fields for missing metrics
        }

        fprintf(file, "%zu\n", e->count);
    }

    free(sorted_entries);
}
