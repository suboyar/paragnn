#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "core.h"

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

// typedef struct {
//     const char* name;
//     double elapsed_time;
// } TimeEntry;

typedef struct {
    const char* name;
    double      total_time;     // Sum of all measurements
    double      min_time;       // Minimum time observed
    double      max_time;       // Maximum time observed
    size_t      count;          // Number of measurements
    double      current_start;  // Current timing start (for active measurements)
    bool        is_active;      // Whether timing is currently active
} TimeEntry;

typedef struct {
    TimeEntry* entries;
    size_t count;
    size_t capacity;
} HashTable;

// 256 contexts should be enough (famous last words)
#define HASHTABLE_SIZE 256
static TimeEntry local_entries[HASHTABLE_SIZE];
static HashTable ht = {
    .entries  = local_entries,
    .count    = 0,
    .capacity = HASHTABLE_SIZE,
};

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

static inline size_t get_idx(const char* key) {
    uint64_t hash = hash_key(key);
    return (size_t)(hash & (ht.capacity-1));
}

static TimeEntry* find_or_create_entry(const char* name) {
    size_t idx = get_idx(name);
    TimeEntry* p = ht.entries + idx;
    TimeEntry* end = ht.entries + ht.capacity;

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
    p->is_active = false;
    ht.count++;

    return p;
}


void benchmark_start(const char* name, double start_time)
{
    TimeEntry* entry = find_or_create_entry(name);
    if (entry == NULL) return;

    if (entry->is_active) {
        fprintf(stderr, "WARNING: Benchmark '%s' is already active! Previous timing will be lost.\n", name);
    }

    entry->current_start = start_time;
    entry->is_active = true;
}

void benchmark_stop(const char* name, double stop_time)
{
    TimeEntry* entry = find_or_create_entry(name);
    if (entry == NULL) return;

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

void benchmark_clear()
{
    memset(ht.entries, 0, sizeof(TimeEntry) * ht.capacity);
    ht.count = 0;
}

void benchmark_print()
{
    TimeEntry* p = ht.entries;
    TimeEntry* end = ht.entries + ht.capacity;

    printf("%-30s %-10s %-10s %-10s %-10s %-8s\n",
           "name", "total(s)", "avg(s)", "min(s)", "max(s)", "calls");
    const int total_width = 30 + 8 + 10 + 10 + 10 + 10 + 5; // +5 for spaces between columns
    for (int i = 0; i < total_width; i++) {
        printf("-");
    }
    printf("\n");

    while (p < end) {
        if (p->name != NULL && p->count > 0) {
            double avg = p->total_time / p->count;
            printf("%-30s %-10.6f %-10.6f %-10.6f %-10.6f %-8zu\n",
                   p->name, p->total_time, avg, p->min_time, p->max_time, p->count);
        }
        p++;
    }
}
