#ifndef BENCHMARK_H
#define BENCHMARK_H

#define BENCH_START(name) benchmark_start((name), omp_get_wtime())
#define BENCH_STOP(name)  benchmark_stop((name), omp_get_wtime())

#endif // BENCHMARK_H
