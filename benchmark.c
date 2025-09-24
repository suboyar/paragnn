#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "core.h"

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

typedef struct {
    const char* name;
    double elapsed_time;
} TimeEntry;

typedef struct {
    TimeEntry* entries;
    size_t count;
    size_t capacity;
} HashTable;

// 256 contexts should be enough (famous last words)
#define HASHTABLE_SIZE 256
TimeEntry local_entries[HASHTABLE_SIZE];
HashTable ht = {
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

void benchmark_start(const char* name, double start_time)
{
    size_t idx = get_idx(name);
    TimeEntry* p = ht.entries + idx;
    TimeEntry* end = ht.entries + ht.capacity;

    while (p < end && p->name != NULL) {
        p++;
    }

    if (p->name != NULL) {
        ERROR("Benchmark hash table is full! Cannot start timing '%s'. "
              "Consider increasing HASHTABLE_SIZE (currently %zu) or clearing old entries.",
              name, ht.capacity);
    }
    p->name = name;
    p->elapsed_time = start_time;
    ht.count++;
}

void benchmark_stop(const char* name, double stop_time)
{
    size_t idx = get_idx(name);
    TimeEntry* p = ht.entries + idx;
    TimeEntry* end = ht.entries + ht.capacity;

    while (p < end && p->name != NULL) {
        p++;
    }


    if (p->name != NULL) {
        ERROR("Benchmark '%s' was never started! "
              "Make sure to call benchmark_start() before benchmark_stop(). "
              "Active benchmarks: %zu", name, ht.count);
    }

    p->elapsed_time = stop_time - p->elapsed_time;
}


void benchmark_print()
{
    TimeEntry* p = ht.entries;
    TimeEntry* end = ht.entries + ht.count;

    printf("{\n");
    while (p < end) {
        // if (p->name != NULL)
            printf("\t%s: %f\n", p->name, p->elapsed_time);
        p++;
    }
    printf("}\n");
}
