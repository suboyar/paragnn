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

#ifndef TIMER_H
#define TIMER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <omp.h>

enum TimerMetric {
    TIMER_INVALID,
    TIMER_MIN_TIME,
    TIMER_MAX_TIME,
    TIMER_TOTAL_TIME,
    TIMER_AVG_TIME,
    TIMER_STD_TIME,
    TIMER_P99_TIME,
    TIMER_P95_TIME,
    TIMER_COUNT,
};

typedef struct TimerEntry TimerEntry;

typedef struct {
    const char* name;
    TimerEntry* entry;
    double      start_time;
} TimerScope;

TimerEntry* __timer_scope_push(const char* name);
void __timer_scope_end(TimerScope* scope);

#define TIMER_FUNC()                                                    \
    TimerScope __timer_scope __attribute__((cleanup(__timer_scope_end))) = { \
        .name = __func__,                                               \
        .entry = __timer_scope_push(__func__),                          \
        .start_time = omp_get_wtime()                                   \
    }

#define TIMER_BLOCK(name_, code) do {                                    \
        TimerScope __timer_scope __attribute__((cleanup(__timer_scope_end))) = { \
            .name = (name_),                                             \
            .entry = __timer_scope_push((name_)),                       \
            .start_time = omp_get_wtime()                               \
        };                                                              \
        code;                                                           \
    } while(0)

#define TIMER_NORECORD(time_var, code) do {     \
        double _start = omp_get_wtime();        \
        code;                                   \
        (time_var) = omp_get_wtime() - _start;  \
    } while(0)

void timer_record(const char* name, double elapsed, TimerEntry* entry);
void timer_record_parallel(const char* name, double* elapsed, int nthreads);
void timer_enable(void);
void timer_disable(void);
void timer_reset(void);
void timer_record_counters(const char* name, uint64_t flops, uint64_t l3_local, uint64_t l3_remote, uint64_t bytes_loaded);
double timer_get_time(const char* name, enum TimerMetric metric);
void timer_print(void);
void timer_export_csv(FILE *fd);

#endif // TIMER_H
