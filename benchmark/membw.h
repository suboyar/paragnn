/*
 * ParaGNN
 * Copyright (C) 2026  Suleyman Boyar
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
