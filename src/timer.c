#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "core.h"
#include "timer.h"
#include "../nob.h"

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

typedef struct {
    const char* name;
    double      total_time;
    double      min_time;
    double      max_time;
    double      current_start;
    int         thread_count;
    size_t      count;
    bool        is_active;
} TimerEntry;

typedef struct {
    TimerEntry* entries;
    size_t      count;
    size_t      capacity;
} TimerRegistry;

// 1024 contexts should be enough (famous last words)
#define HASHTABLE_SIZE 1024
static TimerEntry entries[HASHTABLE_SIZE];
static TimerRegistry reg = {
    .entries = entries,
    .count = 0,
    .capacity = HASHTABLE_SIZE,
};

// Thread-local tracking of current benchmark
// static _Thread_local TimerEntry* current_entry = NULL;

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
    return (size_t)(hash & (reg.capacity-1));
}

TimerEntry* find_entry(const char* name) {
    size_t idx = get_idx(name);
    TimerEntry* p = reg.entries + idx;
    TimerEntry* end = reg.entries + reg.capacity;

    while (p < end && p->name != NULL) {
        if (strcmp(p->name, name) == 0) {
            return p;
        }
        p++;
    }

    return NULL;
}

static TimerEntry* find_or_create_entry(const char* name) {
    if (reg.count >= reg.capacity) {
        return NULL;
    }

    size_t idx = get_idx(name);
    TimerEntry* p = reg.entries + idx;
    TimerEntry* end = reg.entries + reg.capacity;

    while (p < end && p->name != NULL) {
        if (strcmp(p->name, name) == 0) {
            return p;
        }
        p++;
    }

    p->name = name;
    p->total_time = 0.0;
    p->min_time = DBL_MAX;
    p->max_time = 0.0;
    p->thread_count = omp_in_parallel() ? omp_get_thread_num() : 1;
    p->count = 0;
    reg.count++;

    return p;
}

void timer_record(const char* name, double elapsed)
{
    TimerEntry* entry = find_or_create_entry(name);
    if (entry == NULL) {
        nob_log(NOB_ERROR, "Timer '%s': registry full", name);
        abort();
    }

    entry->total_time += elapsed;
    entry->min_time = fmin(elapsed, entry->min_time);
    entry->max_time = fmax(elapsed, entry->max_time);
    entry->count++;
}

void timer_record_parallel(const char* name, double* elapsed, int nthreads)
{
    double wall_time = 0.0;
    for (int t = 0; t < nthreads; t++) {
        wall_time = fmax(wall_time, elapsed[t]);
    }
    timer_record(name, wall_time);
}

void __timer_scope_end(TimerScope* scope)
{
    if (omp_in_parallel()) {
        NOB_TODO("Timer parallel region not supported");
    }

    timer_record(scope->name, omp_get_wtime() - scope->start_time);
}

double timer_get_time(const char* name, enum TimerMetric metric)
{
    TimerEntry* entry = find_entry(name);
    if (entry == NULL) {
        ERROR("Timer entry '%s' not found", name);
    }

    switch (metric) {
    case TIMER_TOTAL_TIME:
        return entry->total_time;
    case TIMER_MIN_TIME:
        return entry->min_time;
    case TIMER_MAX_TIME:
        return entry->max_time;
    }
    abort();
}

static int cmp_by_total_time(const void* a, const void* b)
{
    TimerEntry* ea = (TimerEntry*)a;
    TimerEntry* eb = (TimerEntry*)b;

    // Sort by avg time (descending)
    if ((eb->total_time) > (ea->total_time)) return 1;
    if ((eb->total_time) < (ea->total_time)) return -1;
    return 0;
}


static TimerEntry* get_sorted_entries()
{
    TimerEntry* valid_entries = malloc(reg.capacity * sizeof(TimerEntry));

    TimerEntry* p = reg.entries;
    TimerEntry* end = reg.entries + reg.capacity;
    size_t i = 0;
    while (p < end) {
        if (p->name != NULL && p->count > 0) {
            valid_entries[i++] = *p;
        }
        p++;
    }

    qsort(valid_entries, reg.count, sizeof(TimerEntry), cmp_by_total_time);
    return valid_entries;
}

void timer_print()
{
    printf("%-30s %-12s %-12s %-12s %-12s %-8s\n",
           "name", "avg(s)", "total(s)", "min(s)", "max(s)", "calls");

    printf("----------------------------------------------------------------------------------------\n");

    TimerEntry* sorted_entries = get_sorted_entries();

    for (size_t i = 0; i < reg.count; i++) {
        TimerEntry* e = &sorted_entries[i];
        double avg = e->total_time / e->count;

        printf("%-30s %-12.6f %-12.6f %-12.6f %-12.6f %-8zu\n",
               e->name, avg, e->min_time, e->total_time, e->max_time, e->count);
    }

    free(sorted_entries);
}

void timer_export_csv(FILE *fp)
{
    (void) fp;
    NOB_TODO("Implement timer_export_csv();");
}
