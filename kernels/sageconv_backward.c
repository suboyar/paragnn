#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cblas.h>

#include "cache_counter.h"
#include "core.h"
#include "dataset.h"
#include "layers.h"
#include "sageconv_backward_fused.h"
#include "sageconv_backward_gemm_tn.h"
#include "timer.h"

#ifndef NTIMES
#define NTIMES 100
#endif

#ifndef DIMS
#define DIMS 256, 512, 1024
#endif

static inline void fill_uniform(Real *restrict x, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = (Real)rand() / (Real)RAND_MAX;
    }
}

static inline bool real_eq(Real a, Real b)
{
    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    const Real abs_tol = 1e-9;
    const Real rel_tol = 1e-6;

    Real abs_diff = real_fabs(a - b);
    Real abs_max = real_fmax(fabs(a), fabs(b));
    if (abs_diff > abs_tol && abs_diff > rel_tol * abs_max)
    {
        return false;
    }
    return true;
}

static bool is_valid(Real *x, Real *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (!real_eq(x[i], y[i])) return false;
    }

    return true;
}

typedef void (*fptr)(SageLayer *const l);

typedef struct {
    fptr func;
    const char *name;
} BenchFunc;

#define BENCH_FUNC(fn) { .func = &(fn), .name = #fn }

void benchmark(size_t in_dim, size_t out_dim, Dataset *ds)
{
    cache_counter_t* thread_counters = cache_counter_init_all();

    Real *input = cache_aligned_alloc(ds->num_nodes * in_dim * sizeof(Real));
    fill_uniform(input, ds->num_nodes * in_dim);
    Real *grad_output = cache_aligned_alloc(ds->num_nodes * out_dim * sizeof(Real));
    fill_uniform(grad_output, ds->num_nodes * out_dim);

    SageLayer *l = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, in_dim, out_dim, SOURCE_TO_TARGET);
    l->input = input;
    l->grad_output = grad_output;

    size_t num_nodes = l->num_nodes;

    BenchFunc funcs[] = {
        // BENCH_FUNC(sageconv_backward_gemm_tn_v1),
        BENCH_FUNC(sageconv_backward_gemm_tn_v2),
        // BENCH_FUNC(sageconv_backward_gemm_tn_v3),
        // BENCH_FUNC(sageconv_backward_gemm_tn_v4),
        // BENCH_FUNC(sageconv_backward_gemm_tn_blas),

        // BENCH_FUNC(sageconv_backward_fused_v1),
        // BENCH_FUNC(sageconv_backward_fused_v2),
    };

#if !defined(SKIP_VALID)
    // compute reference
    if (isatty(STDOUT_FILENO))
    {
        printf("Reference:");
        fflush(stdout);
    }
    sageconv_backward_gemm_tn_v2(l);
    Real *ref_gwr = cache_aligned_alloc(in_dim * out_dim * sizeof(Real));
    Real *ref_gwa = cache_aligned_alloc(in_dim * out_dim * sizeof(Real));
    Real *ref_gi = cache_aligned_alloc(num_nodes * in_dim * sizeof(Real));
    memcpy(ref_gwr, l->grad_Wroot, in_dim * out_dim * sizeof(Real));
    memcpy(ref_gwa, l->grad_Wagg, in_dim * out_dim * sizeof(Real));
    memcpy(ref_gi, l->grad_input, num_nodes * in_dim * sizeof(Real));
    if (isatty(STDOUT_FILENO)) printf(" ok\n");

    // validation
    for (size_t i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++)
    {
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KValidating: %s", funcs[i].name);
            fflush(stdout);
        }

        real_zero_out(l->grad_Wroot, in_dim * out_dim);
        real_zero_out(l->grad_Wagg, in_dim * out_dim);
        real_zero_out(l->grad_input, num_nodes * in_dim);
        real_zero_out(l->grad_scatter, num_nodes * in_dim);

        funcs[i].func(l);
        if(!is_valid(l->grad_Wroot, ref_gwr, in_dim * out_dim))
        {
            printf("\r\033[K");
            ERROR("grad_Wroot doesn't match the reference (%s)", funcs[i].name);
        }
        if(!is_valid(l->grad_Wagg, ref_gwa, in_dim * out_dim))
        {
            printf("\r\033[K");
            ERROR("grad_Wagg doesn't match the reference (%s)", funcs[i].name);
        }
        if(!is_valid(l->grad_input, ref_gi, num_nodes * in_dim))
        {
            printf("\r\033[K");
            ERROR("grad_input doesn't match the reference (%s)", funcs[i].name);
        }
    }
    if (isatty(STDOUT_FILENO)) printf("\r\033[KValidating: ok\n");
    free(ref_gwr);
    free(ref_gwa);
    free(ref_gi);
#else
    printf("Skiping validation\n");
#endif // SKIP_VALID

    for (size_t i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++)
    {
    #if !defined(SKIP_WARMUP)
        // warm up
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KWarmup: %s", funcs[i].name);
            fflush(stdout);
        }

        for (size_t j = 0; j < 10; j++)
        {
            if (isatty(STDOUT_FILENO))
            {
                printf(".");
                fflush(stdout);
            }

            real_zero_out(l->grad_Wroot, in_dim * out_dim);
            real_zero_out(l->grad_Wagg, in_dim * out_dim);
            real_zero_out(l->grad_input, num_nodes * in_dim);
            real_zero_out(l->grad_scatter, num_nodes * in_dim);
            funcs[i].func(l);
        }
#endif // SKIP_WARMUP

        double min_time = DBL_MAX;
        uint64_t bytes = 0, l3_local = 0, l3_remote = 0; // 0 initialized to silent -Wmaybe-uninitialized
        // 4 GEMMs (2*N*I*O each) + inv_degree scale (N*I) + scatter (E*I)
        uint64_t flops = 8UL * num_nodes * in_dim * out_dim + num_nodes * in_dim + l->num_edges * in_dim;

        // Run
        double sum_time = 0.0;
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KRun: %s", funcs[i].name);
            fflush(stdout);
        }
        for (size_t j = 0; j < NTIMES; j++)
        {
            real_zero_out(l->grad_Wroot, in_dim * out_dim);
            real_zero_out(l->grad_Wagg, in_dim * out_dim);
            real_zero_out(l->grad_input, num_nodes * in_dim);

            timer_enable();
            cache_counter_start_all(thread_counters);
            double t0 = omp_get_wtime();
            funcs[i].func(l);
            double elapsed = omp_get_wtime()-t0;
            cache_counter_stop_all(thread_counters);
            timer_record(funcs[i].name, elapsed, NULL);
            timer_disable();

            sum_time += elapsed;

            if (isatty(STDOUT_FILENO))
            {
                printf("\r\033[KRun: %s", funcs[i].name);
                for (size_t k = 0; k <= j; k++)
                    putchar('.');
                printf("%.2fs", sum_time / (j + 1));
                fflush(stdout);
            }

            if (elapsed < min_time)
            {
                min_time = elapsed;
                bytes = 0, l3_local = 0, l3_remote = 0;
                for (int tid = 0; tid < omp_get_max_threads(); tid++)
                {
                    bytes += cache_counter_get_bytes_loaded(&thread_counters[tid]);
                    long long local = 0, remote = 0;
                    cache_counter_get_cache_misses(&thread_counters[tid], &local, &remote);
                    l3_local += (uint64_t)local;
                    l3_remote += (uint64_t)remote;
                }
            }
        }
        timer_enable();
        timer_record_counters(funcs[i].name, flops, l3_local, l3_remote, bytes);
        timer_disable();
    }

    if (isatty(STDOUT_FILENO))
    {
        printf("\r\033[K");
        fflush(stdout);
    }

    timer_print();
    // timer_export_csv("stdout");
    cache_counter_close_all(thread_counters);

    printf("MIN_TIME=%f\n", timer_get_time("sageconv_backward_gemm_tn_v2", TIMER_MIN_TIME));

    timer_reset();
    free(input);
    sage_layer_free(&l);
}

int main(void)
{
    srand(0);

    bool to_symmetric = true;
    Dataset *ds = dataset_load("arxiv", "./data", EDGE_COO, to_symmetric);
    Dataset *ds_train = dataset_split(ds, SPLIT_TRAIN);
    ds = ds_train;
    printf("num nodes: %u\n", ds->num_nodes);

    static const size_t dims[] = { DIMS };
    for (size_t i = 0; i < sizeof(dims)/sizeof(dims[0]); i++)
    {
        printf("in_dim: %zu, out_dim: %zu\n", dims[i], dims[i]);
        benchmark(dims[i], dims[i], ds);
    }
}
