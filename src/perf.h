#ifndef PERF_H
#define PERF_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <omp.h>

typedef struct {
    const char* name;
    double      total_time;     // Sum of all measurements
    double      min_time;       // Minimum time observed
    double      max_time;       // Maximum time observed
    size_t      count;          // Number of measurements
    double      current_start;  // Current timing start (for active measurements)
    uint64_t    flop;           // Optional: FLOP/s tracking
    uint64_t    bytes;          // Optional: Memory transfer tracking
    bool        has_metrics;  
    bool        is_active;      // Whether timing is currently active
} PerfEntry;

typedef enum {
    TOTAL_TIME,
    MIN_TIME,
    MAX_TIME,
    COUNT,
    CURRENT_START,
    FLOP,
    BYTES,
} PerfMetric;

typedef struct {
    uint64_t flops;
    uint64_t bytes;
} OpMetrics;

typedef struct {
    PerfEntry* entries;
    size_t count;
    size_t capacity;
} HashTable;

// 1024 contexts should be enough (famous last words)
#define HASHTABLE_SIZE 1024
extern PerfEntry __perf_entries[HASHTABLE_SIZE];
extern HashTable __perf_ht;

// Thread-local tracking of current benchmark
extern _Thread_local PerfEntry* __current_perf_entry;

// Track benchmarking statistics across multiple calls

#define PERF_FUNC_START() perf_start(&__perf_ht, __func__, omp_get_wtime())
#define PERF_START(name) perf_start(&__perf_ht, (name), omp_get_wtime())

#define PERF_FUNC_END() do {                          \
    perf_stop(&__perf_ht, __func__, omp_get_wtime()); \
    __current_perf_entry = NULL;                     \
} while(0)
#define PERF_END(name) do {                             \
        perf_stop(&__perf_ht, (name), omp_get_wtime()); \
        __current_perf_entry = NULL;                   \
    } while(0)

#define PERF_OP_METRICS(name, metrics) perf_add_metric(&__perf_ht, (name), (metrics))

#define PERF_CALL(func_name, expr) do {                  \
        PerfEntry* __parent = __current_perf_entry; \
        PERF_START((func_name));                         \
        OpMetrics __op = (expr);                    \
        PERF_OP_METRICS((func_name), __op);              \
        PERF_END((func_name));                           \
        __current_perf_entry = __parent;            \
        if (__parent) {                             \
            PERF_OP_METRICS((__parent->name), __op);    \
        }                                           \
    } while(0)

#define PERF_GET_ENTRY(name) find_entry(&__perf_ht, (name))


// Add metrics during function
#define PERF_ADD_METRICS(flops_val, bytes_val) do {                    \
    if (__current_perf_entry) {                                       \
        OpMetrics __m = {.flops = (flops_val), .bytes = (bytes_val)};  \
        perf_add_metric(&__perf_ht, __current_perf_entry->name, __m); \
    }                                                                  \
} while(0)

// One-off time measurements
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

#define PERF_CLEAR() perf_clear(&__perf_ht)
#define PERF_PRINT() perf_print(&__perf_ht)
#define PERF_CSV(file) perf_csv(&__perf_ht, (file))
#define PERF_GET_METRIC(name, metric, ret) perf_get_metric(&__perf_ht, (name), (metric), (ret))

void perf_start(HashTable* ht, const char* name, double start_time);
void perf_stop(HashTable* ht, const char* name, double stop_time);
PerfEntry* find_entry(HashTable* ht, const char* name);
void perf_add_metric(HashTable* ht, const char* name, OpMetrics metrics);
void perf_clear(HashTable* ht);
void perf_print(HashTable* ht);
void perf_csv(HashTable* ht, FILE *f);
void perf_get_metric(HashTable* ht, const char* name, PerfMetric metric, void* ret);

#endif // PERF_H
