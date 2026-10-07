#ifndef MEMBW_H
#define MEMBW_H

#include <stdint.h>

void membw_init_1(void);
void membw_init_all(void);

void membw_start_1(void);
void membw_start_all(void);

void membw_stop_1(void);
void membw_stop_all(void);

void membw_close_1(void);
void membw_close_all(void);

int64_t membw_get_llc_load_miss_1(void);
int64_t membw_get_llc_load_miss_all(void);

int64_t membw_get_llc_store_miss_1(void);
int64_t membw_get_llc_store_miss_all(void);

int64_t membw_get_l3_local_cache_miss_1(void);
int64_t membw_get_l3_local_cache_miss_all(void);

int64_t membw_get_l3_remote_cache_miss_1(void);
int64_t membw_get_l3_remote_cache_miss_all(void);

uint64_t membw_get_bytes_loaded_1(void);
uint64_t membw_get_bytes_loaded_all(void);

double membw_get_bw_1(double time);
double membw_get_bw_all(double time);


#endif // MEMBW_H
