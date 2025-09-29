#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdbool.h>
#include <stdio.h>

#include <omp.h>

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

// 1024 contexts should be enough (famous last words)
#define HASHTABLE_SIZE 1024
static TimeEntry __benchmark_entries[HASHTABLE_SIZE];
static HashTable __benchmark_ht = {
    .entries = __benchmark_entries,
    .count = 0,
    .capacity = HASHTABLE_SIZE,
};

#define BENCH_START(name) benchmark_start(&__benchmark_ht, (name), omp_get_wtime())
#define BENCH_STOP(name) benchmark_stop(&__benchmark_ht, (name), omp_get_wtime())
#define BENCH_CLEAR() benchmark_clear(&__benchmark_ht)
#define BENCH_PRINT() benchmark_print(&__benchmark_ht)
#define BENCH_CSV(file) benchmark_csv(&__benchmark_ht, (file))

void benchmark_start(HashTable* ht, const char* name, double start_time);
void benchmark_stop(HashTable* ht, const char* name, double stop_time);
void benchmark_clear(HashTable* ht);
void benchmark_print(HashTable* ht);
void benchmark_csv(HashTable* ht, FILE *f);

#endif // BENCHMARK_H
