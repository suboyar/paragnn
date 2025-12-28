#ifndef TIMER_H
#define TIMER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <omp.h>

enum TimerMetric {
    TIMER_TOTAL_TIME,
    TIMER_MIN_TIME,
    TIMER_MAX_TIME,
};

typedef struct {
    const char* name;
    double      start_time;
} TimerScope;

#define TIMER_FUNC()                                                    \
    TimerScope __timer_scope __attribute__((cleanup(__timer_scope_end))) = { \
        .name = __func__,                                               \
        .start_time = omp_get_wtime()                                   \
    }

#define TIMER_BLOCK(name, code) do {                                \
        double __timer_start = omp_get_wtime();                     \
        code;                                                       \
        timer_record((name), omp_get_wtime() - __timer_start);  \
    } while(0)

void timer_record(const char* name, double elapsed);
void timer_record_parallel(const char* name, double* elapsed, int nthreads);
void __timer_scope_end(TimerScope* scope);
double timer_get_time(const char* name, enum TimerMetric metric);
void timer_print();
void timer_export_csv(FILE *fp);

#endif // TIMER_H
