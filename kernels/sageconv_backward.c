#include <cblas.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cache_counter.h"
#include "core.h"
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


static inline void fill_uniform(Real *restrict x, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        x[i] = (Real)rand() / (Real)RAND_MAX;
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

typedef void (*fptr)(SageLayer *const l);

typedef struct {
    fptr func;
    const char *name;
} BenchFunc;

#define BENCH_FUNC(fn) { .func = &(fn), .name = #fn }

void benchmark(int64_t in_dim, int64_t out_dim, Dataset *ds)
{
    cache_counter_t* thread_counters = cache_counter_init_all();

    Real *input = cache_aligned_alloc(ds->num_nodes * in_dim * sizeof(Real));
    fill_uniform(input, ds->num_nodes * in_dim);
    Real *grad_output = cache_aligned_alloc(ds->num_nodes * out_dim * sizeof(Real));
    fill_uniform(grad_output, ds->num_nodes * out_dim);

    SageLayer *l = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, in_dim, out_dim, SOURCE_TO_TARGET);
    l->input = input;
    l->grad_output = grad_output;

    int64_t num_nodes = l->num_nodes;

    BenchFunc funcs[] = {
        // BENCH_FUNC(sageconv_backward_gemm_tn_v1),
        // BENCH_FUNC(sageconv_backward_gemm_tn_v2),
        // BENCH_FUNC(sageconv_backward_gemm_tn_v3),
        // BENCH_FUNC(sageconv_backward_gemm_tn_v4),
        // BENCH_FUNC(sageconv_backward_gemm_tn_blas),

        // BENCH_FUNC(sageconv_backward_fused_v1),
        // BENCH_FUNC(sageconv_backward_fused_v2),

        // BENCH_FUNC(sageconv_backward_outer_v1),
        // BENCH_FUNC(sageconv_backward_outer_v2),
        // BENCH_FUNC(sageconv_backward_outer_v3),
        // BENCH_FUNC(sageconv_backward_outer_v4),
        BENCH_FUNC(sageconv_backward_outer_v5),
        // BENCH_FUNC(sageconv_backward_outer_v6),
    };

#if !defined(SKIP_VALID)
    // compute reference
    if (isatty(STDOUT_FILENO))
    {
        printf("Reference:");
        fflush(stdout);
    }
    sageconv_backward_gemm_tn_blas(l);
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

        for (int64_t j = 0; j < 10; j++)
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
        for (int64_t j = 0; j < ntimes; j++)
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
    sage_layer_free(&l);
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
    {"ntimes",     required_argument, NULL, OPT_NTIMES},
    {"dims",       required_argument, NULL, OPT_DIMS},
    {"dataset",    required_argument, NULL, OPT_DATASET},
    {"datadir",    required_argument, NULL, OPT_DATADIR},
    {"csv",        required_argument, NULL, OPT_CSV},
    {"help",          no_argument,       NULL, OPT_HELP},
    {0,            0,                 0,    0}
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
                if (*end == ',')
                {
                    s = end + 1;
                }
                else if (*end == '\0')
                {
                    break;
                }
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
            if (strcmp("stdout", optarg) == 0)
            {
                csv_fd = stdout;
            }
            else if (strcmp("stderr", optarg) == 0)
            {
                csv_fd = stderr;
            }
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
        printf("in_dim: %zu, out_dim: %zu\n", dims[i], dims[i]);
        benchmark(dims[i], dims[i], ds);
    }

    free(datadir);
}
