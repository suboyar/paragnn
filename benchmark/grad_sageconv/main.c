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
#define DEFAULT_OUTPUT_PERF  stdout

static int64_t      ntimes;
static DatasetKind  datasetkind;
static char        *root;
static bool         export_csv;
static FILE        *timer_fd;
static FILE        *perf_fd;

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

static void perf_print(BenchKernel *funcs, size_t func_count)
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

static void perf_print_csv(FILE *fd, BenchKernel *funcs, size_t func_count)
{
    if (fd == stdout) fprintf(fd, "\n--- CSV_OUTPUT_BEGIN ---\n");
    fprintf(fd, "name,GFLOP/s,MB/s,AI,LLC Ld,LLC St,L3 Loc,L3 Rem,Bytes\n");
    for(size_t i = 0; i < func_count; i++)
    {
        fprintf(fd, "%s,%f,%f,%f,%ld,%ld,%ld,%ld,%lu\n",
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

#define ARRAY_LEN(x) sizeof((x))/sizeof((x)[0])
static void benchmark_kernel(Dataset *ds, int64_t in_dim, int64_t out_dim)
{
    membw_init_all();
    int is_tty = isatty(STDOUT_FILENO);

    int64_t node_count = ds->node_count;

    BenchKernel funcs[] = {
        // BENCH_KERNEL(naive),
        BENCH_KERNEL(cblas_gemm),
        BENCH_KERNEL(outer_tn_v1),
        BENCH_KERNEL(outer_tn_v2),
        BENCH_KERNEL(outer_tn_v3),
    };

#if defined(SKIP_VALID)
#else
    printf("Computing reference...\n");
    Real *A_ref = ALLOC_OR_DIE(alloc_shared(node_count * in_dim * sizeof(Real)));
    fill_uniform(A_ref, node_count * in_dim);
    Real *B_ref = ALLOC_OR_DIE(alloc_shared(node_count * out_dim * sizeof(Real)));
    fill_uniform(B_ref, node_count * out_dim);
    Real *C_ref = ALLOC_OR_DIE(alloc_shared(in_dim * out_dim * sizeof(Real)));
    memset(C_ref, 0, in_dim * out_dim * sizeof(Real));
    cblas_gemm(in_dim, out_dim, node_count, A_ref, in_dim, B_ref, out_dim, C_ref, out_dim);

    for (size_t i = 0; i < ARRAY_LEN(funcs); i++)
    {
        int64_t c_stride = (funcs[i].func == outer_tn_v3) ? ((out_dim + N_VEC - 1) / N_VEC) * N_VEC : out_dim;
        Real *C = ALLOC_OR_DIE(alloc_shared(in_dim * out_dim * sizeof(Real)));
        memset(C, 0, in_dim * out_dim * sizeof(Real));
        printf("Validating: %s...\n", funcs[i].name);
        funcs[i].func(in_dim, out_dim, node_count, A_ref, in_dim, B_ref, out_dim, C, c_stride);
        if(!is_valid_2d(C, C_ref, in_dim, out_dim, c_stride))
            ERROR("%s doesn't match the reference: %s", funcs[i].name, mismatch_buf);
        free(C);
    }
    free(A_ref);
    free(B_ref);
    free(C_ref);
#endif // SKIP_VALID

    for (size_t i = 0; i < ARRAY_LEN(funcs); i++)
    {
        int64_t c_stride = (funcs[i].func == outer_tn_v3) ? ((out_dim + N_VEC - 1) / N_VEC) * N_VEC : out_dim;

        int do_interleave = 0;
#if defined(FIRST_TOUCH)
        do_interleave = (funcs[i].func == cblas_gemm);
#else
        do_interleave = 1;
#endif // FIRST_TOUCH

#if defined(RUN_COMPLETE_GRAD_SAGECONV)
        SageLayer *l = ALLOC_OR_DIE(malloc(sizeof(*l)));
        *l = (SageLayer) {
            .node_count = ds->node_count,
            .edge_count = ds->edge_count,
            .graph      = ds->graph,
            .in_dim     = in_dim,
            .out_dim    = out_dim,
            .flow       = SOURCE_TO_TARGET,
            .W_stride   = c_stride,
        };

        l->W_self     = ALLOC_OR_DIE(alloc_shared(in_dim * l->W_stride * sizeof(Real)));
        l->W_neigh    = ALLOC_OR_DIE(alloc_shared(in_dim * l->W_stride * sizeof(Real)));
        l->dx_in      = ALLOC_OR_DIE(alloc_shared(node_count * in_dim * sizeof(Real)));
        l->x_neigh    = ALLOC_OR_DIE(alloc_shared(node_count * in_dim * sizeof(Real)));
        l->dx_scatter = ALLOC_OR_DIE(alloc_shared(node_count * in_dim * sizeof(Real)));
        memset(l->W_self,     0, in_dim * l->W_stride * sizeof(Real));
        memset(l->W_neigh,    0, in_dim * l->W_stride * sizeof(Real));
        memset(l->dx_in,      0, node_count * in_dim * sizeof(Real));
        memset(l->x_neigh,    0, node_count * in_dim * sizeof(Real));
        memset(l->dx_scatter, 0, node_count * in_dim * sizeof(Real));

        if (do_interleave)
        {
            l->x_in       = ALLOC_OR_DIE(alloc_shared(node_count * in_dim * sizeof(Real)));
            l->dx_out     = ALLOC_OR_DIE(alloc_shared(node_count * out_dim * sizeof(Real)));
            l->dW_self    = ALLOC_OR_DIE(alloc_shared(in_dim * l->W_stride * sizeof(Real)));
            l->dW_neigh   = ALLOC_OR_DIE(alloc_shared(in_dim * l->W_stride * sizeof(Real)));
            memset(l->x_in,     0, node_count * in_dim * sizeof(Real));
            memset(l->dx_out,   0, node_count * out_dim * sizeof(Real));
            memset(l->dW_self,  0, in_dim * l->W_stride * sizeof(Real));
            memset(l->dW_neigh, 0, in_dim * l->W_stride * sizeof(Real));
        }
        else
        {
            l->x_in       = ALLOC_OR_DIE(alloc_local(node_count * in_dim * sizeof(Real)));
            l->dx_out     = ALLOC_OR_DIE(alloc_local(node_count * out_dim * sizeof(Real)));
            l->dW_self    = ALLOC_OR_DIE(alloc_local(in_dim * l->W_stride * sizeof(Real)));
            l->dW_neigh   = ALLOC_OR_DIE(alloc_local(in_dim * l->W_stride * sizeof(Real)));
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
        Real *A, *B, *C;
        const size_t sz_A = node_count * in_dim * sizeof(Real);
        const size_t sz_B = node_count * out_dim * sizeof(Real);
        const size_t sz_C = in_dim * c_stride * sizeof(Real);

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
            funcs[i].func_touch(in_dim, out_dim, node_count, A, in_dim, B, out_dim, C, c_stride);
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
            funcs[i].func(in_dim, out_dim, node_count, A, in_dim, B, out_dim, C, c_stride);
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

#if defined(FLUSH_MEMORY) && !defined(RUN_COMPLETE_GRAD_SAGECONV)
            flush_memory_region(A, node_count * in_dim * sizeof(Real));
            flush_memory_region(B, node_count * out_dim * sizeof(Real));
            flush_memory_region(C, in_dim * c_stride * sizeof(Real));
#endif // FLUSH_MEMORY

            timer_enable();
            membw_start_all();
            time = omp_get_wtime();

#if defined(RUN_COMPLETE_GRAD_SAGECONV)
            GRAD_SAGECONV();
#else
            funcs[i].func(in_dim, out_dim, node_count, A, in_dim, B, out_dim, C, c_stride);
#endif // RUN_COMPLETE_GRAD_SAGECONV
            time = omp_get_wtime() - time;
            membw_stop_all();
            timer_record(funcs[i].name, time, NULL);
            timer_disable();

            if (time < min_time)
            {
#if defined(RUN_COMPLETE_GRAD_SAGECONV)
                /*
                 * funcs[i].func (dW_self): 2 * in_dim * out_dim * node_count
                 * funcs[i].func (dW_neigh): 2 * in_dim * out_dim * node_count
                 * cblas_rgemm (dx_in): 2 * in_dim * out_dim * node_count
                 * cblas_rgemm (dx_scatter): 2 * in_dim * out_dim * node_count
                 * scale_by_inv_degree_cs: node_count * (in_dim + 1) [1 division + in_dim multiplications per node]
                 * scatter_cs: ds->edge_count * in_dim
                 */
                uint64_t flop = (8 * in_dim * out_dim * node_count) +
                                (node_count * (in_dim + 1)) +
                                (ds->edge_count * in_dim);
#else
                uint64_t flop = (2 * in_dim * out_dim * node_count);
#endif
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
        free(l->dx_in);
        free(l->x_neigh);
        free(l->dx_scatter);
        free(l);
#else
        free(A);
        free(B);
        free(C);
#endif
    }

    timer_print();
    printf("\n");
    perf_print(funcs, ARRAY_LEN(funcs));

    if (export_csv)
    {
        timer_export_csv(timer_fd);
        perf_print_csv(perf_fd, funcs, ARRAY_LEN(funcs));
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
    OPT_OUTPUT_PERF,
};

static struct option long_options[] = {
    {"ntimes",        required_argument, NULL, OPT_NTIMES},
    {"dataset",       required_argument, NULL, OPT_DATASET},
    {"datadir",       required_argument, NULL, OPT_ROOT},
    {"csv",           no_argument,       NULL, OPT_CSV},
    {"output-timer",  required_argument, NULL, OPT_OUTPUT_TIMER},
    {"output-perf", required_argument, NULL, OPT_OUTPUT_PERF},
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
            "  --output-perf     Output file of perf (stdout,stderr,path)     [" XSTR(DEFAULT_CSV) "]\n"
            "  -h, --help        Show this help\n",
            progname);
}


void print_config(void)
{
    char *t_path = export_csv ? fd_to_path(timer_fd) : strdup("none");
    char *s_path = export_csv ? fd_to_path(perf_fd)  : strdup("none");
    const char *partition = getenv("SLURM_JOB_PARTITION");
    printf("mode=%s precision=%s ntimes=%ld data=%s validation=%s warmup=%s allocation=%s flush-memory=%s export-csv=%s\n"
           "Files: timer=%s perf=%s\n"
           "Environment: partition=%s, %d OMP threads, %d OpenBLAS threads, %d NUMA node(s)\n"
           "BLAS Config: %s\n"
           "Kernel     : KC=%d, MR=%d, NR=%d\n",
           IS_DEFINED(RUN_COMPLETE_GRAD_SAGECONV) ? "grad_sageconv" : "kernel",
           sizeof(Real) == sizeof(double) ? "fp64" : "fp32",
           ntimes, ds_infos[datasetkind].name,
           IS_DEFINED(SKIP_VALID)   ? "no" : "yes",
           IS_DEFINED(SKIP_WARMUP)  ? "no" : "yes",
           IS_DEFINED(FIRST_TOUCH)  ? "first-touch" : "interleave",
           IS_DEFINED(FLUSH_MEMORY) ? "yes" : "no",
           export_csv ? "yes" : "no",
           t_path, s_path,
           partition ? partition : "none", omp_get_max_threads(), openblas_get_num_threads(), get_active_numa_nodes(),
           openblas_get_config(),
           KC, MR, NR);
    free(t_path);
    free(s_path);
}

int main(int argc, char** argv)
{
    srand(0);

    ntimes      = DEFAULT_NTIMES;
    datasetkind = str_to_dataset_kind(DEFAULT_DATASET);
    root     = DEFAULT_ROOT;
    export_csv  = DEFAULT_CSV;
    timer_fd    = DEFAULT_OUTPUT_TIMER;
    perf_fd     = DEFAULT_OUTPUT_PERF;

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
                    char *full_path = expand_path(optarg);
                    timer_fd = fopen(full_path, "w+");
                    if (!timer_fd)
                    {
                        ERROR("Could not open file %s for csv export: %s", optarg, strerror(errno));
                        usage(argv[0]);
                        return 1;
                    }
                    free(full_path);
                }
                break;
            }
            case OPT_OUTPUT_PERF:
            {
                if (strcmp("stdout", optarg) == 0) perf_fd = stdout;
                else if (strcmp("stderr", optarg) == 0) perf_fd = stderr;
                else
                {
                    char *full_path = expand_path(optarg);
                    perf_fd = fopen(full_path, "w+");
                    if (!perf_fd)
                    {
                        ERROR("Could not open file %s for csv export: %s", optarg, strerror(errno));
                        usage(argv[0]);
                        return 1;
                    }
                    free(full_path);
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

    print_config();

    Dataset *ds = dataset_load(datasetkind, root, SPARSE_FORMAT_CS);
    benchmark_kernel(ds, 256, 256);

    if (timer_fd != stdout && timer_fd != stderr)
        fclose(timer_fd);
    if (perf_fd != stdout && perf_fd != stderr)
        fclose(perf_fd);

    free(root);

    fflush(stdout);
    return 0;
}
