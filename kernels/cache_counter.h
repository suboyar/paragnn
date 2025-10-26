#ifndef CACHE_COUNTER_H
#define CACHE_COUNTER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    struct { int fd; long long count; bool available; } l3_miss_local;
    struct { int fd; long long count; bool available; } l3_miss_remote;
    struct { int fd; long long count; bool available; } l3_miss_generic;
} cache_counter_t;

cache_counter_t cache_counter_init(void);
void cache_counter_start(cache_counter_t* counter);
void cache_counter_stop(cache_counter_t* counter);
void cache_counter_close(cache_counter_t* counter);
void cache_counter_print(cache_counter_t counter);
void cache_counter_get_cache_misses(cache_counter_t* counter, long long* cache_misses_local, long long* cache_misses_remote);
uint64_t cache_counter_get_bytes_loaded(cache_counter_t* counter);

#endif // CACHE_COUNTER_H
