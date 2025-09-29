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
extern TimeEntry __benchmark_entries[HASHTABLE_SIZE];
extern HashTable __benchmark_ht;

// Benchmarking: Track statistics across multiple calls

#define BENCH_START(name) benchmark_start(&__benchmark_ht, (name), omp_get_wtime())
#define BENCH_STOP(name) benchmark_stop(&__benchmark_ht, (name), omp_get_wtime())
#define BENCH_CLEAR() benchmark_clear(&__benchmark_ht)
#define BENCH_PRINT() benchmark_print(&__benchmark_ht)
#define BENCH_CSV(file) benchmark_csv(&__benchmark_ht, (file))

#define BENCH_CALL(name, func, ...) do {        \
        BENCH_START(name);                      \
        func(__VA_ARGS__);                      \
        BENCH_STOP(name);                       \
    } while(0)

#define BENCH_EXPR(name, expr) ({               \
            BENCH_START(name);                  \
            __typeof__(expr) __result = (expr); \
            BENCH_STOP(name);                   \
            __result;                           \
        })


// Timing: One-off measurements (e.g., for FLOP/s calculations)

#define TIME_IT(code) ({                        \
            double __start = omp_get_wtime();   \
            code;                               \
            omp_get_wtime() - __start;          \
        })

#define TIME_EXPR(expr, time_ptr) ({                    \
            double __start = omp_get_wtime();           \
            __typeof__(expr) __result = (expr);         \
            *(time_ptr) = omp_get_wtime() - __start;    \
            __result;                                   \
        })

void benchmark_start(HashTable* ht, const char* name, double start_time);
void benchmark_stop(HashTable* ht, const char* name, double stop_time);
void benchmark_clear(HashTable* ht);
void benchmark_print(HashTable* ht);
void benchmark_csv(HashTable* ht, FILE *f);

#endif // BENCHMARK_H
