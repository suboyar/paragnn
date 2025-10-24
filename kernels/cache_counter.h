#ifndef CACHE_COUNTER_H
#define CACHE_COUNTER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    struct { int fd; long long count; bool available; } demand_local_mem;
    struct { int fd; long long count; bool available; } llc_misses;
    struct { int fd; long long count; bool available; } cache_misses;
    struct { int fd; long long count; bool available; } cache_refs;
} cache_counter_t;

cache_counter_t cache_counter_init(void);
void cache_counter_start(cache_counter_t* counter);
void cache_counter_stop(cache_counter_t* counter);
void cache_counter_close(cache_counter_t* counter);
void cache_counter_print(cache_counter_t counter);
uint64_t cache_get_bytes_loaded(cache_counter_t* counter);

#endif // CACHE_COUNTER_H
