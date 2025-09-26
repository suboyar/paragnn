#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <omp.h>

#define BENCH_START(name) benchmark_start((name), omp_get_wtime())
#define BENCH_STOP(name)  benchmark_stop((name), omp_get_wtime())

void benchmark_start(const char* name, double start_time);
void benchmark_stop(const char* name, double stop_time);
void benchmark_print();

#endif // BENCHMARK_H
