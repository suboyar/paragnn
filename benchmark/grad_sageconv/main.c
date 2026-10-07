#include <error.h>
#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cblas.h>
#include <omp.h>

#include "../membw.h"
#include "core.h"
#include "ds.h"
#include "dsinfo.h"
#include "kernels.h"
#include "outer_tn_params.h"
#include "timer.h"
#include "vreg.h"

// Default flag values
#define DEFAULT_NTIMES       1000
#define DEFAULT_DATASET     "ogbn-arxiv"
#define DEFAULT_ROOT        "~/D1/paragnn-dataset"
#define DEFAULT_CSV          false
#define DEFAULT_OUTPUT_TIMER stdout
#define DEFAULT_OUTPUT_STAT  stdout

static int64_t      ntimes;
static DatasetKind  datasetkind;
static char        *root;
static bool         export_csv;
static FILE        *timer_fd;
static FILE        *stat_fd;

#define GRAD_SAGECONV()                                     \
    do {                                                    \
    funcs[i].func(l->in_dim, l->out_dim, l->node_count,     \
                  l->x_in,    l->in_dim,                    \
                  l->dx_out,  l->out_dim,                   \
                  l->dW_self, l->W_stride);                 \
                                                            \
    funcs[i].func(l->in_dim, l->out_dim, l->node_count,     \
                  l->x_neigh,  l->in_dim,                   \
                  l->dx_out,   l->out_dim,                  \
                  l->dW_neigh, l->W_stride);                \
                                                            \
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,    \
                l->node_count, l->in_dim, l->out_dim,       \
                1.0,                                        \
                l->dx_out, l->out_dim,                      \
                l->W_self, l->W_stride,                     \
                0.0,                                        \
                l->dx_in,  l->in_dim);                      \
                                                            \
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,    \
                l->node_count, l->in_dim, l->out_dim,       \
                1.0,                                        \
                l->dx_out,     l->out_dim,                  \
                l->W_neigh,    l->W_stride,                 \
                0.0,                                        \
                l->dx_scatter, l->in_dim);                  \
                                                            \
    scale_by_inv_degree_cs(l);                              \
    scatter_cs(l);                                          \
    } while (0)


typedef struct {
    KernelFunc func;
    TouchFunc func_touch;
    const char *name;
    double flops_per_sec;
    double bw;
    double ai;
    int64_t llc_load_miss;
    int64_t llc_store_miss;
    int64_t l3_local_miss;
    int64_t l3_remote_miss;
    uint64_t bytes_loaded;
} BenchKernel;
#define BENCH_KERNEL(fn) { .func = &(fn), .func_touch = &(fn##_touch), .name = #fn, 0}

typedef struct {
    void (*func)(SageLayer*);
    TouchFunc func_touch;
    const char *name;
    double flops_per_sec;
    double bw;
    double ai;
    int64_t llc_load_miss;
    int64_t llc_store_miss;
    int64_t l3_local_miss;
    int64_t l3_remote_miss;
    uint64_t bytes_loaded;
} BenchGradSageconv;
#define BENCH_GRAD_SAGECONV(fn) { .func = &(grad_sageconv_##fn), .func_touch = &(fn##_touch), .name = #fn, 0}

static void stat_print(BenchKernel *funcs, size_t func_count)
{
    int name_col_width = 30; // Match default minimum of timer_print
    for (size_t i = 0; i < func_count; i++) {
        if (funcs[i].name) {
            int len = (int)strlen(funcs[i].name);
            if (len > name_col_width) {
                name_col_width = len;
            }
        }
    }

    char fixed_cols[256];
    snprintf(fixed_cols, sizeof(fixed_cols),
             "%-10s %-10s %-8s %-12s %-12s %-12s %-12s %-12s",
             "GFLOP/s", "MB/s", "AI", "LLC Ld", "LLC St",
             "L3 Loc", "L3 Rem", "Bytes");

    char heading[512];
    snprintf(heading, sizeof(heading), "%-*s %s", name_col_width, "name", fixed_cols);

    printf("%s\n", heading);

    for (size_t i = 0; i < strlen(heading); i++) printf("-");
    printf("\n");

    for(size_t i = 0; i < func_count; i++)
    {
        printf("%-*s %-10.2f %-10.2f %-8.2f %-12ld %-12ld %-12ld %-12ld %-12lu\n",
               name_col_width, funcs[i].name,
               funcs[i].flops_per_sec / 1e9,
               funcs[i].bw / 1e6,
               funcs[i].ai,
               funcs[i].llc_load_miss,
               funcs[i].llc_store_miss,
               funcs[i].l3_local_miss,
               funcs[i].l3_remote_miss,
               funcs[i].bytes_loaded);
    }
}

static void stat_print_csv(FILE *fd, BenchKernel *funcs, size_t func_count)
{
    if (fd == stdout) fprintf(fd, "\n--- CSV_OUTPUT_BEGIN ---\n");
    printf("name,GFLOP/s,MB/s,AI,LLC Ld,LLC St,L3 Loc,L3 Rem,Bytes\n");
    for(size_t i = 0; i < func_count; i++)
    {
        printf("%s,%f,%f,%f,%ld,%ld,%ld,%ld,%lu\n",
               funcs[i].name,
               funcs[i].flops_per_sec / 1e9,
               funcs[i].bw / 1e6,
               funcs[i].ai,
               funcs[i].llc_load_miss,
               funcs[i].llc_store_miss,
               funcs[i].l3_local_miss,
               funcs[i].l3_remote_miss,
               funcs[i].bytes_loaded);
    }
    if (fd == stdout) fprintf(fd, "--- CSV_OUTPUT_END ---\n");
}

static void scale_by_inv_degree_cs(SageLayer *l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict ptr;
    if (l->flow == SOURCE_TO_TARGET)
    {
        ptr = l->graph->ptr_csc;
    }
    else
    {
        ptr = l->graph->ptr_csr;
    }

    Real *restrict dx_scatter = l->dx_scatter;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real scale = 0.0;
        int64_t degree = ptr[n+1] - ptr[n];
        if (degree != 0) scale = REAL(1.0) / degree;
        Real *dx_scatter_ptr = &dx_scatter[n*in_dim];
#pragma omp simd
        for (int64_t d = 0; d < in_dim; d++)
            dx_scatter_ptr[d] *= scale;
    }
}

static void scatter_cs(SageLayer *l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict ptr, *restrict idx;
    if (l->flow == SOURCE_TO_TARGET)
    {
        ptr = l->graph->ptr_csr;
        idx = l->graph->idx_csr;
    }
    else // flow == TARGET_TO_SOURCE
    {
        ptr = l->graph->ptr_csc;
        idx = l->graph->idx_csc;
    }

    const Real *restrict dx_scatter = l->dx_scatter;
    Real       *restrict dx_in   = l->dx_in;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real *dx_in_ptr = &dx_in[n*in_dim];
        for (int64_t e = ptr[n]; e < ptr[n+1]; e++)
        {
            const Real *dx_scatter_ptr = &dx_scatter[idx[e]*in_dim];
            for (int64_t d = 0; d < l->in_dim; d++)
                dx_in_ptr[d] += dx_scatter_ptr[d];
        }
    }
}

static inline void fill_uniform(Real *restrict x, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        x[i] = (Real)i / (Real)(n - 1);
    }
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
    if (diff <= abs_tol) return true;

    Real largest = real_fmax(real_fabs(a), real_fabs(b));
    if (diff <= largest * rel_tol) return true;

    return false;
}

static char mismatch_buf[1024];
static bool is_valid_2d(Real *x, Real *y, int64_t rows, int64_t cols, int64_t ld)
{
    for (int64_t i = 0; i < rows; i++)
    {
        for (int64_t j = 0; j < cols; j++)
        {
            if (!real_eq(x[i * ld + j], y[i * ld + j]))
            {
                snprintf(mismatch_buf, 1024, "Mismatch at [%ld, %ld]: %f != %f", i, j, x[i * ld + j], y[i * ld + j]);
                return false;
            }
        }
    }
    return true;
}

#if defined(__x86_64__)
#include <immintrin.h>
void flush_memory_region(void *ptr, size_t size)
{
    uintptr_t start = (uintptr_t)ptr & ~63ULL;
    uintptr_t end = (uintptr_t)ptr + size;

#pragma omp parallel for
    for (uintptr_t p = start; p < end; p += 64)
        _mm_clflushopt((void *)p);
    _mm_sfence();
}
#elif defined(__aarch64__)
void flush_memory_region(void *ptr, size_t size)
{
    uintptr_t start = (uintptr_t)ptr & ~63ULL;
    uintptr_t end = (uintptr_t)ptr + size;

    // Data Cache Clean and Invalidate by Virtual Address to Point of Coherency
    for (uintptr_t p = start; p < end; p += 64)
        __asm__ volatile("dc civac, %0" : : "r"(p) : "memory");
}
#endif

static void benchmark_kernel(Dataset *ds, int64_t in_dim, int64_t out_dim)
{
    membw_init_all();
    int is_tty = isatty(STDOUT_FILENO);

    int64_t node_count = ds->node_count;

    Real *A, *B, *C, *C_ref;
    int64_t lda = in_dim, ldb = out_dim, ldc = out_dim;

    BenchKernel funcs[] = {
        // BENCH_KERNEL(naive),
        BENCH_KERNEL(cblas_gemm),
        BENCH_KERNEL(outer_tn_v1),
        BENCH_KERNEL(outer_tn_v2),
        BENCH_KERNEL(outer_tn_v3),
    };
    size_t func_count = sizeof(funcs)/sizeof(funcs[0]);

#if defined(SKIP_VALID)
#else
    printf("Computing reference...\n");
    A = ALLOC_OR_DIE(alloc_shared(node_count * lda * sizeof(Real)));
    fill_uniform(A, node_count * lda);
    B = ALLOC_OR_DIE(alloc_shared(node_count * ldb * sizeof(Real)));
    fill_uniform(B, node_count * ldb);
    C_ref = ALLOC_OR_DIE(alloc_shared(in_dim * ldc * sizeof(Real)));
    memset(C_ref, 0, in_dim * ldc * sizeof(Real));
    cblas_gemm(in_dim, out_dim, node_count, A, lda, B, ldb, C_ref, ldc);

    for (size_t i = 0; i < func_count; i++)
    {
        if (funcs[i].func == outer_tn_v3) ldc = ((out_dim + N_VEC - 1) / N_VEC) * N_VEC;
        else ldc = out_dim;
        C = ALLOC_OR_DIE(alloc_shared(in_dim * ldc * sizeof(Real)));
        memset(C, 0, in_dim * ldc * sizeof(Real));
        printf("Validating: %s...\n", funcs[i].name);
        funcs[i].func(in_dim, out_dim, node_count, A, lda, B, ldb, C, ldc);
        if(!is_valid_2d(C, C_ref, in_dim, out_dim, ldc))
            ERROR("%s doesn't match the reference: %s", funcs[i].name, mismatch_buf);
        free(C);
    }
    free(A);
    free(B);
    free(C_ref);
#endif // SKIP_VALID

    for (size_t i = 0; i < func_count; i++)
    {
        if (funcs[i].func == outer_tn_v3) ldc = ((out_dim + N_VEC - 1) / N_VEC) * N_VEC;
        else ldc = out_dim;

        int do_interleave = 0;
#if defined(FIRST_TOUCH)
        do_interleave = (funcs[i].func == cblas_gemm);
#else
        do_interleave = 1;
#endif // FIRST_TOUCH

// TODO: Fix segfault when RUN_COMPLETE_GRAD_SAGECONV is enabled
#if defined(RUN_COMPLETE_GRAD_SAGECONV)
        SageLayer *l = ALLOC_OR_DIE(malloc(sizeof(*l)));
        *l = (SageLayer) {
            .node_count = ds->node_count,
            .edge_count = ds->edge_count,
            .graph      = ds->graph,
            .in_dim     = in_dim,
            .out_dim    = out_dim,
            .flow       = SOURCE_TO_TARGET,
            .W_stride   = ldc,
        };

        l->W_self     = ALLOC_OR_DIE(alloc_shared(in_dim * ldc * sizeof(Real)));
        l->W_neigh    = ALLOC_OR_DIE(alloc_shared(in_dim * ldc * sizeof(Real)));
        memset(l->W_self, 0, in_dim * ldc * sizeof(Real));
        memset(l->W_neigh, 0, in_dim * ldc * sizeof(Real));

        if (do_interleave)
        {
            l->x_in       = ALLOC_OR_DIE(alloc_shared(in_dim * ldc * sizeof(Real)));
            l->dx_out     = ALLOC_OR_DIE(alloc_shared(out_dim * ldc * sizeof(Real)));
            l->dW_self    = ALLOC_OR_DIE(alloc_shared(in_dim * ldc * sizeof(Real)));
            l->dW_neigh   = ALLOC_OR_DIE(alloc_shared(in_dim * ldc * sizeof(Real)));
            memset(l->x_in, 0, in_dim * ldc * sizeof(Real));
            memset(l->dx_out, 0, out_dim * ldc * sizeof(Real));
            memset(l->dW_self, 0, in_dim * ldc * sizeof(Real));
            memset(l->dW_neigh, 0, in_dim * ldc * sizeof(Real));
        }
        else
        {
            l->x_in       = ALLOC_OR_DIE(alloc_local(in_dim * ldc * sizeof(Real)));
            l->dx_out     = ALLOC_OR_DIE(alloc_local(out_dim * ldc * sizeof(Real)));
            l->dW_self    = ALLOC_OR_DIE(alloc_local(in_dim * ldc * sizeof(Real)));
            l->dW_neigh   = ALLOC_OR_DIE(alloc_local(in_dim * ldc * sizeof(Real)));
            funcs[i].func_touch(l->in_dim, l->out_dim, l->node_count,
                                l->x_in,    l->in_dim,
                                l->dx_out,  l->out_dim,
                                l->dW_self, l->W_stride);
            funcs[i].func_touch(l->in_dim, l->out_dim, l->node_count,
                                l->x_neigh,  l->in_dim,
                                l->dx_out,   l->out_dim,
                                l->dW_neigh, l->W_stride);
        }
#else
        const size_t sz_A = node_count * lda * sizeof(Real);
        const size_t sz_B = node_count * ldb * sizeof(Real);
        const size_t sz_C = in_dim * ldc * sizeof(Real);

        if (do_interleave)
        {
            printf("Performing NUMA interleave: %s...\n", funcs[i].name);
            A = ALLOC_OR_DIE(alloc_shared(sz_A));
            B = ALLOC_OR_DIE(alloc_shared(sz_B));
            C = ALLOC_OR_DIE(alloc_shared(sz_C));
            memset(A, 0, sz_A);
            memset(B, 0, sz_B);
            memset(C, 0, sz_C);
        }
        else
        {
            printf("Performing NUMA first touch: %s...\n", funcs[i].name);
            A = ALLOC_OR_DIE(alloc_local(sz_A));
            B = ALLOC_OR_DIE(alloc_local(sz_B));
            C = ALLOC_OR_DIE(alloc_local(sz_C));
            funcs[i].func_touch(in_dim, out_dim, node_count, A, lda, B, ldb, C, ldc);
        }
#endif // RUN_COMPLETE_GRAD_SAGECONV

#if defined(SKIP_WARMUP)
#else
        printf("Warming up: %s...\n", funcs[i].name);
        const int warmup_count = 100;
        for (int j = 0; j < warmup_count; j++)
        {
#if defined(RUN_COMPLETE_GRAD_SAGECONV)
            GRAD_SAGECONV();
#else
            funcs[i].func(in_dim, out_dim, node_count, A, lda, B, ldb, C, ldc);
#endif // RUN_COMPLETE_GRAD_SAGECONV
        }
#endif // SKIP_WARMUP

        double min_time = DBL_MAX;
        double time = 0.0;
        if (!is_tty) printf("Running: %s...\n", funcs[i].name);
        for (int64_t j = 0; j < ntimes; j++)
        {
            if (is_tty)
            {
                printf("\r\033[KRunning: %s (%ld/%ld) [time %.5fs]",
                       funcs[i].name, j+1, ntimes, time);
                fflush(stdout);
            }

#if defined(FLUSH_MEMORY)
            flush_memory_region(A, node_count * lda * sizeof(Real));
            flush_memory_region(B, node_count * ldb * sizeof(Real));
            flush_memory_region(C, in_dim * ldc * sizeof(Real));
#endif // FLUSH_MEMORY

            timer_enable();
            membw_start_all();
            time = omp_get_wtime();

#if defined(RUN_COMPLETE_GRAD_SAGECONV)
            GRAD_SAGECONV();
#else
            funcs[i].func(in_dim, out_dim, node_count, A, lda, B, ldb, C, ldc);
#endif // RUN_COMPLETE_GRAD_SAGECONV
            time = omp_get_wtime() - time;
            membw_stop_all();
            timer_record(funcs[i].name, time, NULL);
            timer_disable();

            if (time < min_time)
            {
                uint64_t flop = (2 * in_dim * out_dim * node_count);
                min_time = time;
                funcs[i].flops_per_sec = ((double)flop)/time;
                funcs[i].bw = membw_get_bw_all(time);
                funcs[i].ai = flop / membw_get_bytes_loaded_all();
                funcs[i].llc_load_miss = membw_get_llc_load_miss_all();
                funcs[i].llc_store_miss = membw_get_llc_store_miss_all();
                funcs[i].l3_local_miss = membw_get_l3_local_cache_miss_all();
                funcs[i].l3_remote_miss = membw_get_l3_remote_cache_miss_all();
                funcs[i].bytes_loaded = membw_get_bytes_loaded_all();
            }
        }

        if (is_tty) printf("\n");
#if defined(RUN_COMPLETE_GRAD_SAGECONV)
        free(l->x_in);
        free(l->dx_out);
        free(l->W_self);
        free(l->W_neigh);
        free(l->dW_self);
        free(l->dW_neigh);
        free(l);
#else
        free(A);
        free(B);
        free(C);
#endif
    }

    timer_print();
    printf("\n");
    stat_print(funcs, func_count);

    if (export_csv)
    {
        timer_export_csv(timer_fd);
        stat_print_csv(stat_fd, funcs, func_count);
    }

    membw_close_all();
    timer_reset();
}

enum {
    // w/ shorthands
    OPT_HELP = 'h',
    // w/o shorthands
    OPT_DIMS = 256,  // above ASCII
    OPT_NTIMES,
    OPT_DATASET,
    OPT_ROOT,
    OPT_CSV,
    OPT_OUTPUT_TIMER,
    OPT_OUTPUT_STAT,
};

static struct option long_options[] = {
    {"ntimes",        required_argument, NULL, OPT_NTIMES},
    {"dataset",       required_argument, NULL, OPT_DATASET},
    {"datadir",       required_argument, NULL, OPT_ROOT},
    {"csv",           no_argument,       NULL, OPT_CSV},
    {"output-timer",  required_argument, NULL, OPT_OUTPUT_TIMER},
    {"output-stat", required_argument, NULL, OPT_OUTPUT_STAT},
    {"help",          no_argument,       NULL, OPT_HELP},
    {0,               0,                 0,    0}
};

static void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "\n"
            "OPTIONS:\n"
            "  --ntimes N        Number of iterations                         [" XSTR(DEFAULT_NTIMES) "]\n"
            "  --dataset NAME    Dataset name                                 [" DEFAULT_DATASET "]\n"
            "  --root PATH       Root path of dataset directory               [" DEFAULT_ROOT "]\n"
            "  --csv             Enable CSV output                            [" XSTR(DEFAULT_CSV) "]\n"
            "  --output-timer    Output file of timing (stdout,stderr,path)   [" XSTR(DEFAULT_CSV) "]\n"
            "  --output-stat     Output file of stat (stdout,stderr,path)     [" XSTR(DEFAULT_CSV) "]\n"
            "  -h, --help        Show this help\n",
            progname);
}

int main(int argc, char** argv)
{
    srand(0);

    ntimes      = DEFAULT_NTIMES;
    datasetkind = str_to_dataset_kind(DEFAULT_DATASET);
    root     = DEFAULT_ROOT;
    export_csv  = DEFAULT_CSV;
    timer_fd    = DEFAULT_OUTPUT_TIMER;
    stat_fd     = DEFAULT_OUTPUT_STAT;

    int opt;
    while ((opt = getopt_long(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case OPT_NTIMES: ntimes = strtoll(optarg, NULL, 10); break;
            case OPT_DATASET:
            {
                datasetkind = str_to_dataset_kind(optarg);
                if (datasetkind == DATASET_INVALID)
                {
                    ERROR("Given dataset is not valid: %s", optarg);
                    usage(argv[0]);
                    return 1;
                }
                break;
            }
            case OPT_ROOT: root = expand_path(optarg); break;
            case OPT_CSV: export_csv = true; break;
            case OPT_OUTPUT_TIMER:
            {
                if (strcmp("stdout", optarg) == 0) timer_fd = stdout;
                else if (strcmp("stderr", optarg) == 0) timer_fd = stderr;
                else
                {
                    timer_fd = fopen(optarg, "w+");
                    if (!timer_fd)
                    {
                        ERROR("Could not open file %s for csv export: %s", optarg, strerror(errno));
                        usage(argv[0]);
                        return 1;
                    }
                }
                break;
            }
            case OPT_OUTPUT_STAT:
            {
                if (strcmp("stdout", optarg) == 0) stat_fd = stdout;
                else if (strcmp("stderr", optarg) == 0) stat_fd = stderr;
                else
                {
                    stat_fd = fopen(optarg, "w+");
                    if (!stat_fd)
                    {
                        ERROR("Could not open file %s for csv export: %s", optarg, strerror(errno));
                        usage(argv[0]);
                        return 1;
                    }
                }
                break;
            }
            case OPT_HELP:
                usage(argv[0]);
                return 0;
            default:
                usage(argv[0]);
                return 1;
        }
    }

    root = expand_path(root);

    int openblas_num_threads = openblas_get_num_threads();
    int omp_num_threads = omp_get_max_threads();
    if (openblas_num_threads == 1)
    {
        fprintf(stderr,
                "Error: OpenBLAS thread count is 1. Set OPENBLAS_NUM_THREADS (for non-OpenMP), "
                "OMP_NUM_THREADS (for OpenMP builds) or call openblas_set_num_threads()\n");
    }

    omp_set_dynamic(0);
    omp_set_num_threads(omp_num_threads);

    printf("BLAS Config: %s\n"
           "Environment: %d OMP threads, %d OpenBLAS threads, %d NUMA node(s)\n"
           "Kernel     : KC=%d, MR=%d, NR=%d\n",
           openblas_get_config(), omp_num_threads, openblas_num_threads, get_active_numa_nodes(), KC, MR, NR);

    Dataset *ds = dataset_load(datasetkind, root, SPARSE_FORMAT_CS);
    benchmark_kernel(ds, 256, 256);

    if (timer_fd != stdout && timer_fd != stderr)
        fclose(timer_fd);
    if (stat_fd != stdout && stat_fd != stderr)
        fclose(stat_fd);

    free(root);

    fflush(stdout);
    return 0;
}
