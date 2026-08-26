#ifndef CACHE_COUNTER_H
#define CACHE_COUNTER_H

#include <stdint.h>

void membw_init_1(void);
void membw_init_all(void);

void membw_start_1(void);
void membw_start_all(void);

void membw_stop_1(void);
void membw_stop_all(void);

void membw_close_1(void);
void membw_close_all(void);

uint64_t membw_get_llc_load_miss_1();
uint64_t membw_get_llc_load_miss_all();

uint64_t membw_get_llc_store_miss_1();
uint64_t membw_get_llc_store_miss_all();

uint64_t membw_get_l3_local_cache_miss_1();
uint64_t membw_get_l3_local_cache_miss_all();

uint64_t membw_get_l3_remote_cache_miss_1();
uint64_t membw_get_l3_remote_cache_miss_all();

uint64_t membw_get_bytes_loaded_1();
uint64_t membw_get_bytes_loaded_all();

double membw_get_bw_1(double time);
double membw_get_bw_all(double time);


#endif // CACHE_COUNTER_H
