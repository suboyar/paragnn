#include <error.h>
#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "core.h"
#include "ds.h"
#include "dsinfo.h"
#include "kernels.h"
#include "layers.h"
#include "../membw.h"
#include "timer.h"

// Default flag values
#define MAX_DIMS 16
#define DEFAULT_NTIMES  100
#define DEFAULT_DIMS    {256}
#define DEFAULT_NDIMS   1
#define DEFAULT_DATASET "arxiv"
#define DEFAULT_DATADIR "~/D1/paragnn-dataset"
#define DEFAULT_CSV     "stdout"

static int64_t      ntimes;
static int64_t      dims[MAX_DIMS];
static int          n_dims;
static FILE        *csv_fd;
static DatasetKind  dataset;
static char        *datadir;

typedef struct {
    KernelFunc func;
    TouchFunc func_touch;
    const char *name;
    double flops_per_sec;
    double bw;
    double ai;
    uint64_t llc_load_miss;
    uint64_t llc_store_miss;
    uint64_t l3_local_miss;
    uint64_t l3_remote_miss;
    uint64_t bytes_loaded;
} BenchKernel;
#define BENCH_FUNC(fn) { .func = &(fn), .func_touch = &(fn##_touch), .name = #fn, 0}

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
        printf("%-*s %-10.2f %-10.2f %-8.2f %-12lu %-12lu %-12lu %-12lu %-12lu\n",
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

static void validate(int64_t in_dim, int64_t out_dim, Dataset *ds, BenchKernel *funcs, size_t func_count)
{
    SageLayer *l = sage_layer_alloc(ds->node_count, ds->edge_count, ds->graph, in_dim, out_dim, SOURCE_TO_TARGET);

    Real *input = cache_aligned_alloc(ds->node_count * l->in_dim * sizeof(Real));
    fill_uniform(input, ds->node_count * l->in_dim);
    l->input = input;

    Real *grad_output = cache_aligned_alloc(ds->node_count * l->out_dim * sizeof(Real));
    fill_uniform(grad_output, ds->node_count * l->out_dim);
    l->grad_output = grad_output;

    // compute reference
    printf("Reference:");
    fflush(stdout);

    Real *ref_grad_Wroot = cache_aligned_alloc(l->in_dim * l->ldW * sizeof(Real));
    cblas_gemm(l->in_dim, l->out_dim, l->num_nodes,
               l->input,       l->in_dim,
               l->grad_output, l->out_dim,
               ref_grad_Wroot,  l->ldW);

    printf(" ok\n");
    fflush(stdout);

    for (size_t i = 0; i < func_count; i++)
    {
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KValidating: %s", funcs[i].name, i+1, func_count);
            fflush(stdout);
        }

        real_zero_out(l->grad_Wroot, l->in_dim * l->ldW);

        funcs[i].func(l->in_dim, l->out_dim, l->num_nodes,
                            l->input,       l->in_dim,
                            l->grad_output, l->out_dim,
                            l->grad_Wroot,  l->ldW);

        if (isatty(STDOUT_FILENO)) printf("\r\033[K");
        if(!is_valid_2d(l->grad_Wroot, ref_grad_Wroot, l->in_dim, l->out_dim, l->ldW))
        {
            printf("Validating: %s (fail)\n", funcs[i].name, i+1, func_count);
            fflush(stdout);
            printf("%s\n", mismatch_buf);
            ERROR("grad_Wroot doesn't match the reference (%s)", funcs[i].name);
        }
        printf("Validating: %s (ok)\n", funcs[i].name);
        fflush(stdout);
    }


    free(ref_grad_Wroot);
    free(input);
    sage_layer_free(&l);
}

#define CACHE_FLUSH_SIZE (512 * 1024 * 1024)

static void flush_cache(void)
{
    static volatile char *flush_buffer = NULL;

    if (!flush_buffer) {
        flush_buffer = malloc(CACHE_FLUSH_SIZE);
        if (!flush_buffer) {
            fprintf(stderr, "Failed to allocate cache flush buffer\n");
            return;
        }
    }
    #pragma omp parallel for
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64) {
        flush_buffer[i] = (char)i;
    }
}

static void benchmark_kernel(int64_t in_dim, int64_t out_dim)
{
    membw_init_all();
    Dataset *ds = dataset_alloc(DATASET_ARXIV, "/global/D1/homes/sboyar/paragnn-dataset", SPARSE_CS, SPLIT_NONE);

    BenchKernel funcs[] = {
        // BENCH_FUNC(naive),
        BENCH_FUNC(cblas_gemm),
        BENCH_FUNC(outer_tn_v1),
        BENCH_FUNC(outer_tn_v2),
        BENCH_FUNC(outer_tn_v3),
    };
    size_t func_count = sizeof(funcs)/sizeof(funcs[0]);

#if !defined(SKIP_VALID)
    validate(in_dim, out_dim, ds, funcs, func_count);
#endif // SKIP_VALID

    SageLayer *l = NULL;
#if !defined(MANUAL_FIRST_TOUCH) // numactl --interleave=all
    l = sage_layer_alloc(ds->node_count, ds->edge_count, ds->graph, in_dim, out_dim, SOURCE_TO_TARGET);
    l->input = cache_aligned_alloc(ds->node_count * l->in_dim * sizeof(Real));
    l->grad_output = cache_aligned_alloc(ds->node_count * l->out_dim * sizeof(Real));
    sage_layer_reset_parameters(l);
#endif

    for (size_t i = 0; i < func_count; i++)
    {
#if defined(MANUAL_FIRST_TOUCH)
       printf("Performing NUMA first touch: %s...\n", funcs[i].name);
        if (l != NULL)
        {
            free(l->input);
            sage_layer_free(&l);
        }
        l = sage_layer_alloc(ds->node_count, ds->edge_count, ds->graph, in_dim, out_dim, SOURCE_TO_TARGET);
        l->input = cache_aligned_alloc(ds->node_count * l->in_dim * sizeof(Real));
        l->grad_output = cache_aligned_alloc(ds->node_count * l->out_dim * sizeof(Real));
        funcs[i].func_touch(l->in_dim, l->out_dim, l->num_nodes,
                            l->input,       l->in_dim,
                            l->grad_output, l->out_dim,
                            l->grad_Wroot,  l->ldW);
#else
        memset(l->input, 0, ds->node_count * l->in_dim * sizeof(Real));
        memset(l->grad_output, 0, ds->node_count * l->out_dim * sizeof(Real));
#endif
        sage_layer_reset_parameters(l);

#if !defined(SKIP_WARMUP)
        const int warmup_count = 10;
        for (int j = 0; j < warmup_count; j++)
        {
            if (isatty(STDOUT_FILENO))
            {
                printf("\r\033[KWarmup: %s (%d/%d)", funcs[i].name, j+1, warmup_count);
                fflush(stdout);
            }
            funcs[i].func(l->in_dim, l->out_dim, l->num_nodes,
                          l->input,       l->in_dim,
                          l->grad_output, l->out_dim,
                          l->grad_Wroot,  l->ldW);
        }

        if (isatty(STDOUT_FILENO)) printf("\r\033[K");
        printf("Warmup: %s (ok)\n", funcs[i].name);
        fflush(stdout);

#endif // SKIP_WARMUP

        double min_time = DBL_MAX;
        if (!isatty(STDOUT_FILENO))
        {
            printf("Run: %s", funcs[i].name);
            fflush(stdout);
        }

        double sum_time = 0.0;
        for (int64_t j = 0; j < ntimes; j++)
        {
            flush_cache();

            timer_enable();
            membw_start_all();
            double start_time = omp_get_wtime();

            funcs[i].func(l->in_dim, l->out_dim, l->num_nodes,
                          l->input,       l->in_dim,
                          l->grad_output, l->out_dim,
                          l->grad_Wroot,  l->ldW);

            double elapsed_time = omp_get_wtime()-start_time;
            membw_stop_all();
            timer_record(funcs[i].name, elapsed_time, NULL);
            timer_disable();

            sum_time += elapsed_time;

            if (isatty(STDOUT_FILENO))
            {
                printf("\r\033[KRun: %s (%d/%d) [%.5fs]", \
                       funcs[i].name, j+1, ntimes, sum_time / (j + 1));
                fflush(stdout);
            }

            if (elapsed_time < min_time)
            {
                uint64_t flop = (2 * l->num_nodes * in_dim * out_dim);
                min_time = elapsed_time;
                funcs[i].flops_per_sec = ((double)flop)/elapsed_time;
                funcs[i].bw = membw_get_bw_all(elapsed_time);
                funcs[i].ai = flop / membw_get_bytes_loaded_all();
                funcs[i].llc_load_miss = membw_get_llc_load_miss_all();
                funcs[i].llc_store_miss = membw_get_llc_store_miss_all();
                funcs[i].l3_local_miss = membw_get_l3_local_cache_miss_all();
                funcs[i].l3_remote_miss = membw_get_l3_remote_cache_miss_all();
                funcs[i].bytes_loaded = membw_get_bytes_loaded_all();
            }
        }

        if (isatty(STDOUT_FILENO)) printf("\r\033[KRun: %s", funcs[i].name);
        printf(" (ok)\n");
        fflush(stdout);
        // timer_enable();
        // timer_record_counters(funcs[i].name, flops, l3_local, l3_remote, bytes);
        // timer_disable();
    }

    timer_print();
    printf("\n");
    stat_print(funcs, func_count);
    // timer_export_csv("stdout");

    membw_close_all();
    timer_reset();
    free(l->input);
    sage_layer_free(&l);
    dataset_free(&ds);
}

enum {
    // w/ shorthands
    OPT_NTIMES = 'n',
    OPT_HELP = 'h',
    // w/o shorthands
    OPT_DIMS = 256,  // above ASCII
    OPT_LAYERS,
    OPT_DATASET,
    OPT_DATADIR,
    OPT_CSV,
};

static struct option long_options[] = {
    {"ntimes",  required_argument, NULL, OPT_NTIMES},
    {"dims",    required_argument, NULL, OPT_DIMS},
    {"dataset", required_argument, NULL, OPT_DATASET},
    {"datadir", required_argument, NULL, OPT_DATADIR},
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
            "  -n, -ntimes N       Number of iterations            [" STRINGIFY(DEFAULT_NTIMES) "]\n"
            "  -dims D1[,D2,...]   Dimensions to benchmark         [256,512,1024]\n"
            "  -dataset NAME       Dataset name                    [" DEFAULT_DATASET "]\n"
            "  -datadir PATH       Dataset directory               [" DEFAULT_DATADIR "]\n"
            "  -csv FILE           CSV output (stdout/stderr/path) [" DEFAULT_CSV "]\n"
            "  -h, -help           Show this help\n",
            progname);
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
    csv_fd  = stdout;
    dataset = str_to_dataset_kind(DEFAULT_DATASET);
    datadir = DEFAULT_DATADIR;

    int opt;
    while ((opt = getopt_long_only(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case OPT_NTIMES: ntimes = strtoll(optarg, NULL, 10); break;
        case OPT_DIMS:
        {
            char *s = optarg;
            char *end;
            n_dims = 0;
            while (*s && n_dims < MAX_DIMS)
            {
                dims[n_dims++] = strtoll(s, &end, 10);
                if (*end == ',') s = end + 1;
                else if (*end == '\0') break;
                else
                {
                    ERROR("Invalid -dims value: %s", optarg);
                    usage(argv[0]);
                    return 1;
                }
            }
            if (n_dims == 0)
            {
                ERROR("-dims requires at least one value");
                usage(argv[0]);
                return 1;
            }
            break;
        }
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
        case OPT_DATADIR: datadir = optarg; break;
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
    datadir = expand_path(datadir);

    // openblas_set_num_threads(omp_get_max_threads());
    int openblas_num_threads = openblas_get_num_threads();
    int omp_num_threads = omp_get_max_threads();
    if (openblas_num_threads == 1)
    {
        fprintf(stderr,
                "Error: OpenBLAS thread count is 1. "
                "Set OPENBLAS_NUM_THREADS (for non-OpenMP), "
                "OMP_NUM_THREADS (for OpenMP builds) or "
                "call openblas_set_num_threads()\n");
        return 1;
    }

    printf("OpenBLAS config: %s\n", openblas_get_config());
    printf("Using %d threads(omp), %d threads(openblas), and %d NUMA node(s)\n",
           omp_num_threads, openblas_num_threads, get_active_sockets());

    for (int64_t i = 0; i < n_dims; i++)
    {
        printf("in_dim: %zu, out_dim: %zu\n", dims[i], dims[i]);
        benchmark_kernel(dims[i], dims[i]);
    }

    free(datadir);
}
