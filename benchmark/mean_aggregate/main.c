#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <error.h>
#include <getopt.h>
#include <omp.h>

#include "../membw.h"
#include "core.h"
#include "ds.h"
#include "flow.h"
#include "kernels.h"
#include "sparsegraph.h"
#include "timer.h"


// Default flag values
#define DEFAULT_NTIMES       1000
#define DEFAULT_DATASET      "ogbn-arxiv"
#define DEFAULT_ROOT         "~/D1/paragnn-dataset"
#define DEFAULT_CSV          false
#define DEFAULT_OUTPUT_TIMER stdout
#define DEFAULT_OUTPUT_PERF  stdout

static int64_t     ntimes;
static DatasetKind datasetkind;
static char       *root;
static bool        export_csv;
static FILE       *timer_fd;
static FILE       *perf_fd;

static FlowDirection flow = SOURCE_TO_TARGET;

typedef int64_t (*FlopsFunc)(SparseGraph *, int64_t);

typedef struct {
    KernelFunc func;
    TouchFunc func_touch;
    FlopsFunc flops_func;
    SparseFormat format;
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

#define BENCH_KERNEL(fn, flops_fn, fmt) \
    { .func = &(fn), .func_touch = &(fn##_touch), .flops_func = &(flops_fn), .format = (fmt), .name = #fn }

#define ARRAY_LEN(x) (sizeof((x)) / sizeof((x)[0]))

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

    for (size_t i = 0; i < func_count; i++)
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
    for (size_t i = 0; i < func_count; i++)
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

static int64_t count_active_nodes(SparseGraph *graph)
{
    const int64_t *indptr = (flow == SOURCE_TO_TARGET) ? graph->ptr_csc : graph->ptr_csr;
    int64_t active_nodes = 0;

#pragma omp parallel for reduction(+:active_nodes)
    for (int64_t i = 0; i < graph->node_count; i++)
    {
        if (indptr[i + 1] > indptr[i]) active_nodes++;
    }
    return active_nodes;
}

static int64_t flops_coo(SparseGraph *graph, int64_t dim)
{
    return 2 * graph->edge_count * dim;
}

static int64_t flops_cs(SparseGraph *graph, int64_t dim)
{
    int64_t edge_count = graph->edge_count;
#if CACHE_INV_DEGREE
    return 2 * edge_count * dim;
#else
    int64_t active_nodes = count_active_nodes(graph);
    return 2 * edge_count * dim + active_nodes;
#endif
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
#ifdef USE_DOUBLE
    const Real abs_tol = 1e-9;
    const Real rel_tol = DBL_EPSILON;
#else
    const Real abs_tol = 1e-5f;
    const Real rel_tol = 1e-4f;
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
    if (ld < 0) ld = cols;

    for (int64_t i = 0; i < rows; i++)
    {
        for (int64_t j = 0; j < cols; j++)
        {
            if (!real_eq(x[i * ld + j], y[i * ld + j]))
            {
                snprintf(mismatch_buf, sizeof(mismatch_buf),
                         "Mismatch at [%ld, %ld]: %f != %f",
                         i, j, x[i * ld + j], y[i * ld + j]);
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

    for (uintptr_t p = start; p < end; p += 64)
        __asm__ volatile("dc civac, %0" : : "r"(p) : "memory");
}
#endif

#define SAGECONV()                                              \
    do {                                                        \
        cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  \
                    l->node_count, l->out_dim, l->in_dim,       \
                    1.0,                                        \
                    l->x_in,   l->in_dim,                       \
                    l->W_self, l->W_stride,                     \
                    0.0,                                        \
                    l->x_out,  l->out_dim);                     \
                                                                \
        funcs[i].func(l->x_in, l->x_neigh, l->in_dim,           \
                      l->graph, l->flow);                       \
                                                                \
        cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  \
                    l->node_count, l->out_dim, l->in_dim,       \
                    1.0,                                        \
                    l->x_neigh, l->in_dim,                      \
                    l->W_neigh, l->W_stride,                    \
                    1.0,                                        \
                    l->x_out,   l->out_dim);                    \
    } while (0)

static void benchmark_kernel(Dataset *ds, int64_t in_dim, int64_t out_dim)
{
    membw_init_all();
    int is_tty = isatty(STDOUT_FILENO);

    int64_t node_count = ds->node_count;

    BenchKernel funcs[] = {
        BENCH_KERNEL(coo_v1, flops_coo, SPARSE_FORMAT_COO),
        BENCH_KERNEL(cs_v1,  flops_cs,  SPARSE_FORMAT_CS),
        BENCH_KERNEL(cs_v2,  flops_cs,  SPARSE_FORMAT_CS),
        BENCH_KERNEL(cs_v3,  flops_cs,  SPARSE_FORMAT_CS),
        BENCH_KERNEL(cs_v4,  flops_cs,  SPARSE_FORMAT_CS),
    };

#if !defined(SKIP_VALID)
    printf("Computing reference...\n");
    const size_t sz_in = node_count * in_dim * sizeof(Real);
    Real *x_in_ref  = ALLOC_OR_DIE(alloc_shared(sz_in));
    Real *x_neigh_ref = ALLOC_OR_DIE(alloc_shared(sz_in));
    fill_uniform(x_in_ref, node_count * in_dim);
    memset(x_neigh_ref, 0, sz_in);

    coo_v1(x_in_ref, x_neigh_ref, in_dim, ds->graph, flow);

    Real *x_in_neigh = ALLOC_OR_DIE(alloc_shared(sz_in));
    for (size_t i = 0; i < ARRAY_LEN(funcs); i++)
    {
        memset(x_in_neigh, 0, sz_in);
        printf("Validating: %s...\n", funcs[i].name);
        funcs[i].func(x_in_ref, x_in_neigh, in_dim, ds->graph, flow);
        if (!is_valid_2d(x_in_neigh, x_neigh_ref, node_count, in_dim, in_dim))
            ERROR("%s doesn't match the reference: %s", funcs[i].name, mismatch_buf);
    }
    free(x_in_neigh);
    free(x_in_ref);
    free(x_neigh_ref);
#endif // SKIP_VALID

    for (size_t i = 0; i < ARRAY_LEN(funcs); i++)
    {
        int do_interleave = 0;
#if defined(FIRST_TOUCH)
        do_interleave = 0;
#else
        do_interleave = 1;
#endif // FIRST_TOUCH

#if defined(RUN_COMPLETE_SAGECONV)
        SageLayer *l = ALLOC_OR_DIE(malloc(sizeof(*l)));
        *l = (SageLayer) {
            .node_count = ds->node_count,
            .edge_count = ds->edge_count,
            .graph      = ds->graph,
            .in_dim     = in_dim,
            .out_dim    = out_dim,
            .flow       = flow,
            .W_stride   = out_dim,
        };

        l->W_self  = ALLOC_OR_DIE(alloc_shared(in_dim * l->W_stride * sizeof(Real)));
        l->W_neigh = ALLOC_OR_DIE(alloc_shared(in_dim * l->W_stride * sizeof(Real)));
        l->x_out   = ALLOC_OR_DIE(alloc_shared(node_count * out_dim * sizeof(Real)));
        memset(l->W_self,  0, in_dim * l->W_stride * sizeof(Real));
        memset(l->W_neigh, 0, in_dim * l->W_stride * sizeof(Real));
        memset(l->x_out,   0, node_count * out_dim * sizeof(Real));

        if (do_interleave)
        {
            l->x_in    = ALLOC_OR_DIE(alloc_shared(node_count * in_dim * sizeof(Real)));
            l->x_neigh = ALLOC_OR_DIE(alloc_shared(node_count * in_dim * sizeof(Real)));
            memset(l->x_in,    0, node_count * in_dim * sizeof(Real));
            memset(l->x_neigh, 0, node_count * in_dim * sizeof(Real));
        }
        else
        {
            l->x_in    = ALLOC_OR_DIE(alloc_local(node_count * in_dim * sizeof(Real)));
            l->x_neigh = ALLOC_OR_DIE(alloc_local(node_count * in_dim * sizeof(Real)));
            funcs[i].func(l->x_in, l->x_neigh, l->in_dim, l->graph, l->flow);
        }
#else
        (void)out_dim;
        Real *x_in, *x_neigh;
        const size_t sz_in  = node_count * in_dim * sizeof(Real);
        if (do_interleave)
        {
            printf("Performing NUMA interleave: %s...\n", funcs[i].name);
            x_in  = ALLOC_OR_DIE(alloc_shared(sz_in));
            x_neigh = ALLOC_OR_DIE(alloc_shared(sz_in));
            memset(x_in,  0, sz_in);
            memset(x_neigh, 0, sz_in);
        }
        else
        {
            printf("Performing NUMA first touch: %s...\n", funcs[i].name);
            x_in  = ALLOC_OR_DIE(alloc_local(sz_in));
            x_neigh = ALLOC_OR_DIE(alloc_local(sz_in));
            funcs[i].func_touch(x_in, x_neigh, in_dim, ds->graph, flow);
        }
#endif // RUN_COMPLETE_SAGECONV

#if !defined(SKIP_WARMUP)
        printf("Warming up: %s...\n", funcs[i].name);
        const int warmup_count = 100;
        for (int j = 0; j < warmup_count; j++)
        {
#if defined(RUN_COMPLETE_SAGECONV)
            SAGECONV();
#else
            funcs[i].func(x_in, x_neigh, in_dim, ds->graph, flow);
#endif // RUN_COMPLETE_SAGECONV
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
                       funcs[i].name, j + 1, ntimes, time);
                fflush(stdout);
            }

#if defined(FLUSH_MEMORY) && !defined(RUN_COMPLETE_SAGECONV)
            flush_memory_region(in_states,  node_count * in_dim * sizeof(Real));
            flush_memory_region(agg_states, node_count * in_dim * sizeof(Real));
#endif // FLUSH_MEMORY

            timer_enable();
            membw_start_all();
            time = omp_get_wtime();

#if defined(RUN_COMPLETE_SAGECONV)
            SAGECONV();
#else
            funcs[i].func(x_in, x_neigh, in_dim, ds->graph, flow);
#endif // RUN_COMPLETE_SAGECONV
            time = omp_get_wtime() - time;
            membw_stop_all();
            timer_record(funcs[i].name, time, NULL);
            timer_disable();

            if (time < min_time)
            {
                uint64_t aggr_flop = (uint64_t)funcs[i].flops_func(ds->graph, in_dim);
#if defined(RUN_COMPLETE_SAGECONV)
                /*
                 * cblas_rgemm (x_in * W_self): 2 * node_count * in_dim * out_dim
                 * funcs[i].func (mean_aggr):   aggr_flop
                 * cblas_rgemm (x_neigh * W_neigh): 2 * node_count * in_dim * out_dim
                 */
                uint64_t flop = (4 * node_count * in_dim * out_dim) + aggr_flop;
#else
                uint64_t flop = aggr_flop;
#endif
                min_time = time;
                funcs[i].flops_per_sec  = ((double)flop) / time;
                funcs[i].bw             = membw_get_bw_all(time);
                funcs[i].ai             = ((double)flop) / membw_get_bytes_loaded_all();
                funcs[i].llc_load_miss  = membw_get_llc_load_miss_all();
                funcs[i].llc_store_miss = membw_get_llc_store_miss_all();
                funcs[i].l3_local_miss  = membw_get_l3_local_cache_miss_all();
                funcs[i].l3_remote_miss = membw_get_l3_remote_cache_miss_all();
                funcs[i].bytes_loaded   = membw_get_bytes_loaded_all();
            }
        }

        if (is_tty) printf("\n");
#if defined(RUN_COMPLETE_SAGECONV)
        free(l->x_in);
        free(l->x_neigh);
        free(l->x_out);
        free(l->W_self);
        free(l->W_neigh);
        free(l);
#else
        free(x_in);
        free(x_neigh);
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
    OPT_HELP = 'h',
    OPT_DIMS = 256,
    OPT_NTIMES,
    OPT_DATASET,
    OPT_ROOT,
    OPT_CSV,
    OPT_OUTPUT_TIMER,
    OPT_OUTPUT_PERF,
};

static struct option long_options[] = {
    {"ntimes",       required_argument, NULL, OPT_NTIMES},
    {"dataset",      required_argument, NULL, OPT_DATASET},
    {"datadir",      required_argument, NULL, OPT_ROOT},
    {"csv",          no_argument,       NULL, OPT_CSV},
    {"output-timer", required_argument, NULL, OPT_OUTPUT_TIMER},
    {"output-perf",  required_argument, NULL, OPT_OUTPUT_PERF},
    {"help",         no_argument,       NULL, OPT_HELP},
    {0,              0,                 0,    0}
};

static void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "\n"
            "OPTIONS:\n"
            "  --ntimes N        Number of iterations                         [" XSTR(DEFAULT_NTIMES) "]\n"
            "  --dataset NAME    Dataset name                                 [" DEFAULT_DATASET "]\n"
            "  --datadir PATH    Root path of dataset directory               [" DEFAULT_ROOT "]\n"
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
           "USE_OLD_LOAD_CS: %s\n",
           IS_DEFINED(RUN_COMPLETE_SAGECONV) ? "sageconv" : "kernel",
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
           IS_DEFINED(USE_OLD_LOAD_CS) ? "yes" : "no"
    );
    free(t_path);
    free(s_path);
}

int main(int argc, char **argv)
{
    srand(0);

    ntimes      = DEFAULT_NTIMES;
    datasetkind = str_to_dataset_kind(DEFAULT_DATASET);
    root        = DEFAULT_ROOT;
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
            case OPT_ROOT: root = optarg; break;
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

    Dataset *ds = dataset_load(datasetkind, root, SPARSE_FORMAT_COO | SPARSE_FORMAT_CS);
    int64_t in_dim = 256;
    printf("in_dim=%ld\n", in_dim);
    benchmark_kernel(ds, in_dim, 256);

    if (timer_fd != stdout && timer_fd != stderr)
        fclose(timer_fd);
    if (perf_fd != stdout && perf_fd != stderr)
        fclose(perf_fd);

    free(root);

    fflush(stdout);
    return 0;
}
