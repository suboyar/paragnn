#include <error.h>
#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
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
#define DEFAULT_NTIMES  1000
#define DEFAULT_DATASET "arxiv"
#define DEFAULT_ROOTDIR "/global/D1/homes/sboyar/paragnn-dataset"
#define DEFAULT_CSV     stdout
#define DEFAULT_SPLIT   "train"

static int64_t      ntimes;
static FILE        *csv_fd;
static DatasetKind  dataset;
static char        *rootdir;
static Split        split;

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

static void validate(int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc, BenchKernel *funcs, size_t func_count)
{
    printf("Computing reference...\n");
    Real *A = ALLOC_OR_DIE(cache_aligned_alloc(K * lda * sizeof(Real)));
    fill_uniform(A, K * lda);
    Real *B = ALLOC_OR_DIE(cache_aligned_alloc(K * ldb * sizeof(Real)));
    fill_uniform(B, K * ldb);
    Real *C_ref = ALLOC_OR_DIE(cache_aligned_alloc(M * ldc * sizeof(Real)));
    memset(C_ref, 0, M * ldc * sizeof(Real));

    cblas_gemm(M, N, K,
               A,     lda,
               B,     ldb,
               C_ref, ldc);

    Real *C = ALLOC_OR_DIE(cache_aligned_alloc(M * ldc * sizeof(Real)));
    for (size_t i = 0; i < func_count; i++)
    {
        printf("Validating: %s...\n", funcs[i].name, i+1, func_count);
        memset(C, 0, M * ldc * sizeof(Real));
        funcs[i].func(M, N, K,
                      A, lda,
                      B, ldb,
                      C, ldc);

        if (isatty(STDOUT_FILENO)) printf("\r\033[K");
        if(!is_valid_2d(C, C_ref, M, N, ldc))
            ERROR("%s doesn't match the reference: %s", funcs[i].name, mismatch_buf);
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
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

static void benchmark_kernel(int64_t M, int64_t N, int64_t K)
{
    membw_init_all();
    int is_tty = isatty(STDOUT_FILENO);

    Real *A, *B, *C;
    int64_t lda = M, ldb = N, ldc = ((N + N_VEC - 1) / N_VEC) * N_VEC;

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
    validate(M, N, K, lda, ldb, ldc, funcs, func_count);
#endif // SKIP_VALID

    const size_t sz_A = K * lda * sizeof(Real);
    const size_t sz_B = K * ldb * sizeof(Real);
    const size_t sz_C = M * ldc * sizeof(Real);

    for (size_t i = 0; i < func_count; i++)
    {
        int do_interleave = 0;
#if defined(INTERLEAVE_MEMORY)
        do_interleave = 1;
#else
        do_interleave = (funcs[i].func == cblas_gemm);
#endif

        if (do_interleave)
        {
            printf("Performing NUMA interleave: %s...\n", funcs[i].name);
            A = ALLOC_OR_DIE(interleaved_aligned_alloc(sz_A));
            B = ALLOC_OR_DIE(interleaved_aligned_alloc(sz_B));
            C = ALLOC_OR_DIE(interleaved_aligned_alloc(sz_C));
            memset(A, 0, sz_A);
            memset(B, 0, sz_B);
            memset(C, 0, sz_C);
        }
        else
        {
            printf("Performing NUMA first touch: %s...\n", funcs[i].name);
            A = ALLOC_OR_DIE(cache_aligned_alloc(sz_A));
            B = ALLOC_OR_DIE(cache_aligned_alloc(sz_B));
            C = ALLOC_OR_DIE(cache_aligned_alloc(sz_C));
            funcs[i].func_touch(M, N, K, A, lda, B, ldb, C, ldc);
        }
#endif

#if defined(SKIP_WARMUP)
#else
        printf("Warming up: %s...\n", funcs[i].name);
        const int warmup_count = 100;
        for (int j = 0; j < warmup_count; j++)
            funcs[i].func(M, N, K, A, lda, B, ldb, C, ldc);

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

#if defined(DONT_FLUSH_MEMORY)
#else
            flush_memory_region(A, K * lda * sizeof(Real));
            flush_memory_region(B, K * ldb * sizeof(Real));
            flush_memory_region(C, M * ldc * sizeof(Real));
#endif // NO_FIRST_TOUCH

            timer_enable();
            membw_start_all();
            time = omp_get_wtime();

            funcs[i].func(M, N, K, A, lda, B, ldb, C, ldc);

            time = omp_get_wtime() - time;
            membw_stop_all();
            timer_record(funcs[i].name, time, NULL);
            timer_disable();

            if (time < min_time)
            {
                uint64_t flop = (2 * M * N * K);
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
        free(A);
        free(B);
        free(C);
    }

#if defined(EXPORT_CSV)
    timer_export_csv(csv_fd);
    stat_print_csv(csv_fd, funcs, func_count);
#else
    timer_print();
    printf("\n");
    stat_print(funcs, func_count);
#endif // EXPORT_CSV

    membw_close_all();
    timer_reset();
}

static void benchmark_grad_sageconv(SparseGraph *graph, int64_t in_dim, int64_t out_dim)
{
    membw_init_all();
    int is_tty = isatty(STDOUT_FILENO);

    Real *A, *B, *C;
    SageLayer* l = sage_layer_alloc(graph->node_count, graph->edge_count, graph, in_dim, out_dim, SOURCE_TO_TARGET);

    BenchGradSageconv funcs[] = {
        BENCH_GRAD_SAGECONV(naive),
        BENCH_GRAD_SAGECONV(cblas_gemm),
        BENCH_GRAD_SAGECONV(outer_tn_v1),
        BENCH_GRAD_SAGECONV(outer_tn_v2),
        BENCH_GRAD_SAGECONV(outer_tn_v3),
    };
    size_t func_count = sizeof(funcs)/sizeof(funcs[0]);

    int64_t out_dim_pad = ((out_dim + N_VEC - 1) / N_VEC) * N_VEC;
    Sage_Layer_Alloc *l = (SageLayer) {
        .num_nodes    = node_count,
        .num_edges    = edge_count,
        .graph        = graph,
        .in_dim       = in_dim,
        .out_dim      = out_dim,
        .flow         = flow,
        .input        = NULL,
        .grad_output  = NULL,
        .Wagg         = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .Wroot        = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .grad_Wagg    = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .grad_Wroot   = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .output       = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->out_dim * sizeof(Real)));
        .agg          = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real)));
        .grad_input   = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real)));
        .grad_scatter = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real)));
    }

    for (size_t i = 0; i < func_count; i++)
    {
        if (funcs[i].func == grad_sageconv_cblas_gemm)
        {
            printf("Performing NUMA interleave: %s...\n", funcs[i].name);
            A = ALLOC_OR_DIE(interleaved_aligned_alloc(K * lda * sizeof(Real)));
            B = ALLOC_OR_DIE(interleaved_aligned_alloc(K * ldb * sizeof(Real)));
            C = ALLOC_OR_DIE(interleaved_aligned_alloc(M * ldc * sizeof(Real)));
            memset(A, 0, K * lda * sizeof(Real));
            memset(B, 0, K * ldb * sizeof(Real));
            memset(C, 0, M * ldc * sizeof(Real));
        }
        else
        {
            printf("Performing NUMA first touch: %s...\n", funcs[i].name);
            A = ALLOC_OR_DIE(cache_aligned_alloc(K * lda * sizeof(Real)));
            B = ALLOC_OR_DIE(cache_aligned_alloc(K * ldb * sizeof(Real)));
            C = ALLOC_OR_DIE(cache_aligned_alloc(M * ldc * sizeof(Real)));
            funcs[i].func_touch(M, N, K, A, lda, B, ldb, C, ldc);
        }

#if defined(SKIP_WARMUP)
#else
        printf("Warming up: %s...\n", funcs[i].name);
        const int warmup_count = 100;
        for (int j = 0; j < warmup_count; j++)
        {
            funcs[i].func(M, N, K, A, lda, B, ldb, C, ldc);
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

#if defined(DONT_FLUSH_MEMORY)
#else
            flush_memory_region(A, K * lda * sizeof(Real));
            flush_memory_region(B, K * ldb * sizeof(Real));
            flush_memory_region(C, M * ldc * sizeof(Real));
#endif // NO_FIRST_TOUCH

            timer_enable();
            membw_start_all();
            time = omp_get_wtime();

            funcs[i].func(M, N, K, A, lda, B, ldb, C, ldc);

            time = omp_get_wtime() - time;
            membw_stop_all();
            timer_record(funcs[i].name, time, NULL);
            timer_disable();

            if (time < min_time)
            {
                uint64_t flop = (2 * M * N * K);
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
        if (funcs[i].func == cblas_gemm)
        {
            munmap(A, K * lda * sizeof(Real));
            munmap(B, K * ldb * sizeof(Real));
            munmap(C, M * ldc * sizeof(Real));
        }
        else
        {
            free(A);
            free(B);
            free(C);
        }
    }

#if defined(EXPORT_CSV)
    timer_export_csv(csv_fd);
    stat_print_csv(csv_fd, funcs, func_count);
#else
    timer_print();
    printf("\n");
    stat_print(funcs, func_count);
#endif // EXPORT_CSV

    membw_close_all();
    timer_reset();
}

static int64_t get_node_count(Split splitkind)
{

    Dataset *ds;
    if (splitkind <= SPLIT_INVALID || splitkind >= SPLIT_COUNT) ERROR("Invlid split value: %d\n", splitkind);
    ds = dataset_alloc(DATASET_ARXIV, rootdir, SPARSE_CS, splitkind);
    int64_t node_count = ds->node_count;
    dataset_free(&ds);
    return node_count;
}

enum {
    // w/ shorthands
    OPT_NTIMES = 'n',
    OPT_HELP = 'h',
    // w/o shorthands
    OPT_DIMS = 256,  // above ASCII
    OPT_DATASET,
    OPT_ROOT,
    OPT_SPLIT,
    OPT_CSV,
};

static struct option long_options[] = {
    {"ntimes",  required_argument, NULL, OPT_NTIMES},
    {"dataset", required_argument, NULL, OPT_DATASET},
    {"datadir", required_argument, NULL, OPT_ROOT},
    {"split",     required_argument, NULL, OPT_SPLIT},
    {"csv",     required_argument, NULL, OPT_CSV},
    {"help",    no_argument,       NULL, OPT_HELP},
    {0,         0,                 0,    0}
};

static void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "\n"
            "OPTIONS:\n"
            "  -n, -ntimes N    Number of iterations                     [" XSTR(DEFAULT_NTIMES) "]\n"
            "  -dataset NAME    Dataset name                             [" DEFAULT_DATASET "]\n"
            "  -root PATH       Root path of dataset directory           [" DEFAULT_ROOTDIR "]\n"
            "  -split SPLIT     Split dataset by (none,train,valid,test) [" DEFAULT_SPLIT "]\n"
            "  -csv FILE        CSV output (stdout/stderr/path)          [" XSTR(DEFAULT_CSV) "]\n"
            "  -h, -help        Show this help\n",
            progname);
}

int main(int argc, char** argv)
{
    srand(0);

    ntimes  = DEFAULT_NTIMES;
    csv_fd  = DEFAULT_CSV;
    dataset = str_to_dataset_kind(DEFAULT_DATASET);
    rootdir = DEFAULT_ROOTDIR;
    split = DEFAULT_SPLIT;

    int opt;
    while ((opt = getopt_long_only(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case OPT_NTIMES: ntimes = strtoll(optarg, NULL, 10); break;
            case OPT_DATASET:
            {
                dataset = str_to_dataset_kind(optarg);
                if (dataset == DATASET_INVALID)
                {
                    ERROR("Given dataset is not valid: %s", optarg);
                    usage(argv[0]);
                    return 1;
                }
                break;
            }
            case OPT_ROOT: rootdir = expand_path(optarg); break;
            case OPT_SPLIT:
            {
                if (strcmp(optarg, "none")  == 0)
                    split = SPLIT_NONE;
                else if (strcmp(optarg, "train")  == 0)
                    split = SPLIT_TRAIN;
                else if (strcmp(optarg, "valid")  == 0)
                    split = SPLIT_VALID;
                else if (strcmp(optarg, "test")  == 0)
                    split = SPLIT_VALID;
                else ERROR("Split %s not valid, should be either (none,train,valid,test)",  optarg);
            }
            case OPT_CSV:
            {
                if (strcmp("stdout", optarg) == 0) csv_fd = stdout;
                else if (strcmp("stderr", optarg) == 0) csv_fd = stderr;
                else
                {
                    csv_fd = fopen(optarg, "w+");
                    if (!csv_fd)
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

    Dataset *ds = dataset_alloc(dataset, rootdir, SPARSE_CS, splitkind);

    int openblas_num_threads = openblas_get_num_threads();
    int omp_num_threads = omp_get_max_threads();
    if (openblas_num_threads == 1)
    {
        fprintf(stderr,
                "Error: OpenBLAS thread count is 1. Set OPENBLAS_NUM_THREADS (for non-OpenMP), "
                "OMP_NUM_THREADS (for OpenMP builds) or call openblas_set_num_threads()\n");
        return 1;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(omp_num_threads);

    printf("BLAS Config: %s\n"
           "Environment: %d OMP threads, %d OpenBLAS threads, %d NUMA node(s)\n"
           "Kernel     : KC=%d, MR=%d, NR=%d\n",
           openblas_get_config(), omp_num_threads, openblas_num_threads, get_active_sockets(), KC, MR, NR);

    // benchmark_kernel(256, 256, node_counts[SPLIT_TRAIN]);
    // benchmark_grad_sageconv(256, 256, node_counts[SPLIT_TRAIN]);

    benchmark_kernel(100, 100, 100);
    benchmark_grad_sageconv(100, 100, 100);
}
