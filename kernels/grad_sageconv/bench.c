#include <error.h>
#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include <cblas.h>

#include "outer_tn/outer_tn_kernel.h"
#include "grad_mean_aggregate.h"
#include "../cache_counter.h"
#include "core.h"
#include "ds.h"
#include "dsinfo.h"
#include "layers.h"
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

#define OUTER_TN_SIGNATUR int64_t, int64_t, int64_t,                \
        const Real*, int64_t,                                       \
        const Real*, int64_t,                                       \
        Real*, int64_t

void outer_tn_v1(OUTER_TN_SIGNATUR);
void outer_tn_v2(OUTER_TN_SIGNATUR);
void outer_tn_v3(OUTER_TN_SIGNATUR);
void outer_tn_v4(OUTER_TN_SIGNATUR);
void outer_tn_v5(OUTER_TN_SIGNATUR);
void outer_tn_v6(OUTER_TN_SIGNATUR);
void outer_tn_v7(OUTER_TN_SIGNATUR);

typedef void (*outer_fn)(OUTER_TN_SIGNATUR);
typedef void (*fptr)(SageLayer *const l);

typedef struct {
    fptr func;
    const char *name;
} BenchKernel;

#define BENCH_FUNC(fn) { .func = &(fn), .name = #fn }

void cblas_gemm(SageLayer *l)
{
    cblas_rgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                l->in_dim, l->out_dim, l->num_nodes,
                1.0,
                l->input,       l->in_dim,
                l->grad_output, l->out_dim,
                0.0,
                l->grad_Wroot,  l->out_dim);
}

static void grad_sageconv_impl(SageLayer *l, outer_fn kernel)
{
    // grad_Wroot = input^T @ grad_output
    kernel(l->in_dim, l->out_dim, l->num_nodes,
           l->input,       l->in_dim,
           l->grad_output, l->out_dim,
           l->grad_Wroot,  l->out_dim);

    // grad_Wagg = agg^T @ grad_output
    kernel(l->in_dim, l->out_dim, l->num_nodes,
           l->agg,         l->in_dim,
           l->grad_output, l->out_dim,
           l->grad_Wagg,   l->out_dim);

    // grad_input = grad_output @ Wroot^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output, l->out_dim,
                l->Wroot,       l->out_dim,
                0.0,
                l->grad_input,  l->in_dim);

    // grad_scatter = grad_output @ Wagg^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output,  l->out_dim,
                l->Wagg,         l->out_dim,
                0.0,
                l->grad_scatter, l->in_dim);

    grad_mean_aggregate(l);
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

static bool is_valid(Real *x, Real *y, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
        if (!real_eq(x[i], y[i])) return false;
    return true;
}

void validate(int64_t in_dim, int64_t out_dim, Dataset *ds, BenchKernel *funcs, size_t func_count)
{
    SageLayer *l = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, in_dim, out_dim, SOURCE_TO_TARGET);

    Real *input = cache_aligned_alloc(ds->num_nodes * l->in_dim * sizeof(Real));
    fill_uniform(input, ds->num_nodes * l->in_dim);
    l->input = input;

    Real *grad_output = cache_aligned_alloc(ds->num_nodes * l->out_dim * sizeof(Real));
    fill_uniform(grad_output, ds->num_nodes * l->out_dim);
    l->grad_output = grad_output;

    // compute reference
    printf("Reference:");
    fflush(stdout);
    Real *ref_gwr = cache_aligned_alloc(l->in_dim * l->out_dim * sizeof(Real));
    Real *ref_gwa = cache_aligned_alloc(l->in_dim * l->out_dim * sizeof(Real));
    // grad_Wroot = input^T @ grad_output
    cblas_rgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                l->in_dim, l->out_dim, l->num_nodes,
                1.0,
                l->input,       l->in_dim,
                l->grad_output, l->out_dim,
                0.0,
                ref_gwr,  l->out_dim);
    // grad_Wagg = agg^T @ grad_output
    cblas_rgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                l->in_dim, l->out_dim, l->num_nodes,
                1.0,
                l->agg,         l->in_dim,
                l->grad_output, l->out_dim,
                0.0,
                ref_gwa,   l->out_dim);
    printf(" ok\n");

    for (size_t i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++)
    {
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KValidating: %s", funcs[i].name);
            fflush(stdout);
        }

        real_zero_out(l->grad_Wroot, l->in_dim * l->out_dim);
        real_zero_out(l->grad_Wagg, l->in_dim * l->out_dim);
        real_zero_out(l->grad_input, l->num_nodes * l->in_dim);
        real_zero_out(l->grad_scatter, l->num_nodes * l->in_dim);

        funcs[i].func(l);
        if(!is_valid(l->grad_Wroot, ref_gwr, l->in_dim * l->out_dim))
        {
            if (isatty(STDOUT_FILENO)) printf("\r\033[K");
            ERROR("grad_Wroot doesn't match the reference (%s)", funcs[i].name);
        }
        if(!is_valid(l->grad_Wagg, ref_gwa, l->in_dim * l->out_dim))
        {
            if (isatty(STDOUT_FILENO)) printf("\r\033[K");
            ERROR("grad_Wagg doesn't match the reference (%s)", funcs[i].name);
        }

        if (isatty(STDOUT_FILENO))
        {
            printf(".");
            fflush(stdout);
        };
    }

    if (isatty(STDOUT_FILENO)) printf("\r\033[KValidating: ok\n");

    free(ref_gwr);
    free(ref_gwa);
    free(input);
    sage_layer_free(&l);
}

void benchmark_kernel(int64_t in_dim, int64_t out_dim, Dataset *ds)
{
    cache_counter_t* thread_counters = cache_counter_init_all();

    BenchKernel funcs[] = {
        BENCH_FUNC(outer_tn_kernel_v1),
        BENCH_FUNC(cblas_gemm),
        BENCH_FUNC(outer_tn_kernel_v2),
        BENCH_FUNC(outer_tn_kernel_v3),
        BENCH_FUNC(outer_tn_kernel_v4),
        BENCH_FUNC(outer_tn_kernel_v5),
        BENCH_FUNC(outer_tn_kernel_v6),
        BENCH_FUNC(outer_tn_kernel_v7),
    };
    size_t func_count = sizeof(funcs)/sizeof(funcs[0]);

#if !defined(SKIP_VALID)
    validate(in_dim, out_dim, ds, funcs, sizeof(funcs)/sizeof(funcs[0]));
#endif // SKIP_VALID

    // NUMA first touch
    SageLayer *layers[func_count];
    for (size_t i = 0; i < func_count; i++)
    {
        SageLayer *l = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, in_dim, out_dim, SOURCE_TO_TARGET);

        // Real *input = cache_aligned_alloc(ds->num_nodes * l->in_dim * sizeof(Real));
        // fill_uniform(input, ds->num_nodes * l->in_dim);
        // l->input = input;

        // Real *grad_output = cache_aligned_alloc(ds->num_nodes * l->out_dim * sizeof(Real));
        // fill_uniform(grad_output, ds->num_nodes * l->out_dim);
        // l->grad_output = grad_output;

        l->input = cache_aligned_alloc(ds->num_nodes * l->in_dim * sizeof(Real));
        l->grad_output = cache_aligned_alloc(ds->num_nodes * l->out_dim * sizeof(Real));

        layers[i] = l;

        funcs[i].func(l);
    }

    for (size_t i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++)
    {
        SageLayer *l = layers[i];
#if !defined(SKIP_WARMUP)
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

            real_zero_out(l->grad_Wroot, in_dim * out_dim);
            real_zero_out(l->grad_Wagg, in_dim * out_dim);
            real_zero_out(l->grad_input, l->num_nodes * in_dim);
            real_zero_out(l->grad_scatter, l->num_nodes * in_dim);
            funcs[i].func(l);
        }
#endif // SKIP_WARMUP

        double min_time = DBL_MAX;
        uint64_t bytes = 0, l3_local = 0, l3_remote = 0;
        uint64_t flops = 2 * l->num_nodes * in_dim * out_dim;

        // Run
        double sum_time = 0.0;
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KRun: %s", funcs[i].name);
            fflush(stdout);
        }

        for (int64_t j = 0; j < ntimes; j++)
        {
            real_zero_out(l->grad_Wroot, in_dim * out_dim);
            real_zero_out(l->grad_Wagg, in_dim * out_dim);
            real_zero_out(l->grad_input, l->num_nodes * in_dim);

            timer_enable();
            cache_counter_start_all(thread_counters);
            double start_time = omp_get_wtime();

            funcs[i].func(l);

            double elapsed_time = omp_get_wtime()-start_time;
            cache_counter_stop_all(thread_counters);
            timer_record(funcs[i].name, elapsed_time, NULL);
            timer_disable();

            sum_time += elapsed_time;

            if (isatty(STDOUT_FILENO))
            {
                printf("\r\033[KRun: %s", funcs[i].name);
                for (int64_t k = 0; k <= j; k++)
                    putchar('.');
                printf("%.2fs", sum_time / (j + 1));
                fflush(stdout);
            }

            if (elapsed_time < min_time)
            {
                min_time = elapsed_time;
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
    for (size_t i = 0; i < func_count; i++)
    {
        SageLayer *l = layers[i];
        free(l->input);
        sage_layer_free(&l);
    }
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

void usage(const char *progname)
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
        printf("in_dim: %zu, out_dim: %zu\n", dims[i], dims[i]);
        benchmark_kernel(dims[i], dims[i], ds);
    }

    free(datadir);
}
