#include <cblas.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cache_counter.h"
#include "core.h"
#include "vreg.h"
#include "dataset.h"
#include "dataset_info.h"
#include "layers.h"
#include "sageconv_backward_fused.h"
#include "sageconv_backward_gemm_tn.h"
#include "sageconv_backward_outer_tn.h"
#include "timer.h"

#define NOB_IMPLEMENTATION
#include "../nob.h"

// Default flag values
#define MAX_DIMS 16
#define DEFAULT_NTIMES  100
#define DEFAULT_DIMS    {256, 512, 1024}
#define DEFAULT_NDIMS   3
#define DEFAULT_DATASET "arxiv"
#define DEFAULT_DATADIR "~/D1/paragnn-dataset"
#define DEFAULT_CSV     "stdout"

static int64_t      ntimes;
static int64_t      dims[MAX_DIMS];
static int          n_dims;
static FILE        *csv_fd;
static DatasetKind  dataset;
static char        *datadir;

void relu_v1(ReluLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t dim       = l->dim;

    const Real *restrict input  = l->input;
    Real       *restrict output = l->output;

    int64_t n = num_nodes * dim;
#pragma omp parallel for simd
    for (int64_t i = 0; i < n; i++)
    {
        output[i] = (input[i] > REAL(0.0)) ? input[i] : REAL(0.0);
    }
}

static inline VReal vmax(VReal a, VReal b)
{
    VReal r;
    for (int i = 0; i < VLEN; i++)
        r[i] = real_fmax(a[i], b[i]);
    return r;
}

static inline void vstream(Real *p, VReal v)
{
#if defined(__AVX512F__)
#if defined(USE_DOUBLE)
    __builtin_ia32_movntpd512((double *)p, (v8d)v);
#else
    __builtin_ia32_movntps512(p, (v16f)v);
#endif
#elif defined(__AVX2__)
#if defined(USE_DOUBLE)
    __builtin_ia32_movntpd256((double *)p, (v4d)v);
#else
    __builtin_ia32_movntps256(p, (v8f)v);
#endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    // ASIMD (ThunderX2, Kunpeng-920): no NT store exposed by GCC
    // SVE2  (Neoverse-V2): has STNT1 but GCC won't emit it through
    //       vector extensions since they're fixed-width, not sizeless
    vstore(p, v);
#else
    vstore(p, v);
#endif
}

static inline void vsfence(void)
{
#if defined(__SSE2__) || defined(__AVX2__) || defined(__AVX512F__)
    __builtin_ia32_sfence();
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    // ARM store ordering doesn't need a fence for regular stores,
    // and we don't emit NT stores on ARM, so this is a no-op.
    // If vstream ever gets real NT stores, switch to:
    //   __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

void relu_v2(ReluLayer *const l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t dim = l->dim;
    const Real *restrict input = l->input;
    Real *restrict output = l->output;

    int64_t n = num_nodes * dim;
    const VReal zero = bcast(REAL(0.0));

#pragma omp parallel
    {
        int tid  = omp_get_thread_num();
        int nt   = omp_get_num_threads();
        int64_t chunk = (n + nt - 1) / nt;
        chunk = (chunk + VLEN - 1) & ~(int64_t)(VLEN - 1);
        int64_t start = tid * chunk;
        int64_t end   = start + chunk;
        if (end > n) end = n;

        int64_t i = start;
        for (; i <= end - VLEN; i += VLEN)
        {
            VReal v = vload(&input[i]);
            vstream(&output[i], vmax(v, zero));
        }
        for (; i < end; i++)
        {
            output[i] = input[i] > REAL(0.0) ? input[i] : REAL(0.0);
        }
    }
    vsfence();
}

void relu_v3(ReluLayer *const l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t dim = l->dim;

    const Real *restrict input = l->input;
    Real *restrict output = l->output;
    input  = __builtin_assume_aligned(input, 32);
    output = __builtin_assume_aligned(output, 32);

    const int64_t n = num_nodes * dim;
    const int64_t stride = 4 * VLEN;
    const int64_t n_vec = (n / stride) * stride;

#pragma omp parallel
    {
        const VReal zero = (VReal){};

#pragma omp for
        for (int64_t i = 0; i < n_vec; i += stride)
        {
            VReal v0 = vload(&input[i + 0*VLEN]);
            VReal v1 = vload(&input[i + 1*VLEN]);
            VReal v2 = vload(&input[i + 2*VLEN]);
            VReal v3 = vload(&input[i + 3*VLEN]);
            vstream(&output[i + 0*VLEN], vmax(v0, zero));
            vstream(&output[i + 1*VLEN], vmax(v1, zero));
            vstream(&output[i + 2*VLEN], vmax(v2, zero));
            vstream(&output[i + 3*VLEN], vmax(v3, zero));
        }

#pragma omp for
        for (int64_t i = n_vec; i < n; i++)
        {
            output[i] = input[i] > REAL(0.0) ? input[i] : REAL(0.0);
        }
    }
    vsfence();
}

void relu_v4(ReluLayer *const l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t dim = l->dim;

    const Real *restrict input = l->input;
    Real *restrict output = l->output;
    input  = __builtin_assume_aligned(input, 32);
    output = __builtin_assume_aligned(output, 32);

    const int64_t n = num_nodes * dim;
    const int64_t stride = 4 * VLEN;
    const int64_t n_vec = (n / stride) * stride;

#pragma omp parallel
    {
        const VReal zero = (VReal){};
        int tid  = omp_get_thread_num();
        int nt   = omp_get_num_threads();
        int64_t chunk = (n + nt - 1) / nt;
        chunk = (chunk + VLEN - 1) & ~(int64_t)(VLEN - 1);
        int64_t start = tid * chunk;
        int64_t end   = start + chunk;

        int64_t i = start;
        for (; i <= end - 4*VLEN; i += 4*VLEN)
        {
            VReal v0 = vload(&input[i + 0*VLEN]);
            VReal v1 = vload(&input[i + 1*VLEN]);
            VReal v2 = vload(&input[i + 2*VLEN]);
            VReal v3 = vload(&input[i + 3*VLEN]);
            vstream(&output[i + 0*VLEN], vmax(v0, zero));
            vstream(&output[i + 1*VLEN], vmax(v1, zero));
            vstream(&output[i + 2*VLEN], vmax(v2, zero));
            vstream(&output[i + 3*VLEN], vmax(v3, zero));
        }
        for (; i <= end - VLEN; i += VLEN)
        {
            VReal v = vload(&input[i]);
            vstream(&output[i], vmax(v, zero));
        }
    }
    vsfence();
}

static inline bool real_eq(Real a, Real b)
{
    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
#ifdef USE_DOUBLE
    const Real abs_tol = 1e-9;
    const Real rel_tol = DBL_EPSILON;
#else
    const Real abs_tol = 1e-5f;
    const Real rel_tol = 1e-4f; // Ideally this should have been FLT_EPSILON, but with -ffast-math, this becomes to tight of a tolerance
#endif

    Real diff = real_fabs(a - b);
    if (diff <= abs_tol)
    {
        return true;
    }

    Real largest = real_fmax(real_fabs(a), real_fabs(b));
    if (diff <= largest * rel_tol)
    {
        return true;
    }

    return false;
}

static bool is_valid(Real *x, Real *y, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        if (!real_eq(x[i], y[i])) return false;
    }

    return true;
}


static inline void fill_uniform_relu(Real *restrict x, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        x[i] = REAL(2.0) * (Real)rand() / (Real)RAND_MAX - REAL(1.0);
    }
}

typedef void (*fptr)(ReluLayer *const l);

typedef struct {
    fptr func;
    const char *name;
} BenchFunc;

#define BENCH_FUNC(fn) { .func = &(fn), .name = #fn }

void benchmark(int64_t dim, Dataset *ds)
{
    cache_counter_t* thread_counters = cache_counter_init_all();

    Real *input = cache_aligned_alloc(ds->num_nodes * dim * sizeof(Real));
    fill_uniform_relu(input, ds->num_nodes * dim);
    ReluLayer* l = relu_layer_create(ds->num_nodes, dim);
    l->input = input;

    int64_t num_nodes = l->num_nodes;

    BenchFunc funcs[] = {
        BENCH_FUNC(relu_v1),
        BENCH_FUNC(relu_v2),
        BENCH_FUNC(relu_v3),
        BENCH_FUNC(relu_v4),
    };

#if !defined(SKIP_VALID)
    // compute reference
    if (isatty(STDOUT_FILENO))
    {
        printf("Reference:");
        fflush(stdout);
    }
    relu_v1(l);
    Real *ref_output = cache_aligned_alloc(num_nodes * dim * sizeof(Real));
    memcpy(ref_output, l->output, num_nodes * dim * sizeof(Real));
    if (isatty(STDOUT_FILENO)) printf(" ok\n");

    // validation
    for (size_t i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++)
    {
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KValidating: %s", funcs[i].name);
            fflush(stdout);
        }

        funcs[i].func(l);
        if(!is_valid(l->output, ref_output, num_nodes * dim))
        {
            printf("\r\033[K");
            ERROR("output doesn't match the reference (%s)", funcs[i].name);
        }
    }
    if (isatty(STDOUT_FILENO)) printf("\r\033[KValidating: ok\n");
    free(ref_output);
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

        for (int64_t j = 0; j < 10; j++)
        {
            if (isatty(STDOUT_FILENO))
            {
                printf(".");
                fflush(stdout);
            }

            funcs[i].func(l);
        }
#endif // SKIP_WARMUP

        double min_time = DBL_MAX;
        uint64_t bytes = 0, l3_local = 0, l3_remote = 0; // 0 initialized to silent -Wmaybe-uninitialized
        uint64_t flops = num_nodes * dim;

        // Run
        double sum_time = 0.0;
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KRun: %s", funcs[i].name);
            fflush(stdout);
        }
        for (int64_t j = 0; j < ntimes; j++)
        {
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
                for (int64_t k = 0; k <= j; k++)
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

    timer_reset();
    free(input);
    relu_layer_free(&l);
}

int main(int argc, char** argv)
{
    srand(0);

    ntimes  = DEFAULT_NTIMES;
    {
        int64_t default_dims[] = DEFAULT_DIMS;
        n_dims = DEFAULT_NDIMS;
        memcpy(dims, default_dims, sizeof(default_dims));
    }

    dataset = str_to_dataset_kind(DEFAULT_DATASET);
    datadir = DEFAULT_DATADIR;
    datadir = nob_expand_path(datadir);

    // openblas_set_num_threads(omp_get_max_threads());
    int openblas_num_threads = openblas_get_num_threads();
    int omp_num_threads = omp_get_max_threads();
    if (openblas_num_threads == 1)
    {
        fprintf(stderr, "Error: OpenBLAS thread count is 1. Set OPENBLAS_NUM_THREADS or call openblas_set_num_threads()\n");
        return 1;
    }

    printf("OpenBLAS config: %s\n", openblas_get_config());
    printf("Using %d threads(omp) and %d threads(openblas)\n", omp_num_threads, openblas_num_threads);

    Dataset *ds = dataset_load(dataset, datadir, EDGE_CSX);
    Dataset *ds_train = dataset_split(ds, SPLIT_TRAIN);
    ds = ds_train;
    printf("num nodes: %ld\n", ds->num_nodes);

    for (int64_t i = 0; i < n_dims; i++)
    {
        printf("dim: %zu\n", dims[i]);
        benchmark(dims[i], ds);
    }

    free(datadir);
}
