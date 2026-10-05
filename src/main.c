#include <errno.h>
#include <float.h>
#include <math.h>
#include <getopt.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#include <omp.h>
#include <cblas.h>

#include "core.h"
#include "dsinfo.h"
#include "timer.h"
#include "layers.h"
#include "nn.h"
#include "sageconv.h"
#include "ds.h"
#include "optim.h"

// Default values
#if defined(BENCHMARK_MODE)
#define DEFAULT_EPOCHS      10
#else
#define DEFAULT_EPOCHS      1000
#endif
#define DEFAULT_LAYERS      4
#define DEFAULT_CHANNELS    256
#define DEFAULT_LR          0.01
#define DEFAULT_DATASET     "ogbn-arxiv"
#define DEFAULT_ROOT        "~/D1/paragnn-dataset"
#define DEFAULT_CSV         false
#define DEFAULT_OUTPUT      stdout
#define DEFAULT_QUICK       false

static DatasetKind  datasetkind;
static char        *root;
static uint64_t     epochs;
static uint64_t     layers;
static uint64_t     channels;
static Real         lr;
static bool         export_csv;
static FILE        *output_fd;
static bool         quick;

void print_config(void)
{
#if defined(BENCHMARK_MODE)
    const char *mode = "benchmark";
#else
    const char *mode = "convergence";
#endif

#if defined(SAGECONV_NAIVE_IMPL)
    const char *impl = "naive";
#elif defined(SAGECONV_BLAS_IMPL)
    const char *impl = "blas";
#else
    const char *impl = "tuned";
#endif

#if defined(SPARSE_COO)
    const char *sparse_format_name = "SPARSE_FORMAT_COO";
#else
    const char *sparse_format_name = "SPARSE_FORMAT_CS";
#endif


    printf("mode=%s impl=%s prec=%s epochs=%zu lr=%g layers=%zu hidden=%zu sparse=%s data=%s "
           "omp=%d blas=%d partition=%s\n"
           "openblas: %s\n",
           mode, impl,
           sizeof(Real) == sizeof(double) ? "dp" : "sp",
           epochs, lr, layers, channels, sparse_format_name, ds_infos[datasetkind].name,
           omp_get_max_threads(), openblas_get_num_threads(),
           getenv("SLURM_JOB_PARTITION"),
           openblas_get_config());
}

static void inference(SageNet *net)
{
    TIMER_FUNC();

    for (int64_t i = 0; i < net->layer_count; i++)
    {
        Layer layer = net->layers[i];
        switch(layer.type)
        {
        case LAYER_SAGE:
            sageconv((SageLayer*)layer.ctx);
            break;
        case LAYER_RELU:
            relu((ReluLayer*)layer.ctx);
            break;
        case LAYER_L2NORM:
            l2norm((L2NormLayer*)layer.ctx);
            break;
        case LAYER_LOGSOFTMAX:
            logsoftmax((LogSoftmaxLayer*)layer.ctx);
            break;
        default:
            ERROR("Unknown layer type %d", layer.type);
        }
    }
}

static void train(SageNet *net, int64_t *y, Optim *optim)
{
    TIMER_FUNC();

    for (size_t i = net->layer_count; i-- > 0; ) {
        Layer layer = net->layers[i];
        switch(layer.type)
        {
        case LAYER_SAGE:
            grad_sageconv((SageLayer*)layer.ctx);
            break;
        case LAYER_RELU:
            grad_relu((ReluLayer*)layer.ctx);
            break;
        case LAYER_L2NORM:
            grad_l2norm((L2NormLayer*)layer.ctx);
            break;
        case LAYER_LOGSOFTMAX:
            grad_logsoftmax_nll((LogSoftmaxLayer*)layer.ctx, y);
            break;
        default:
            ERROR("Unknown layer type %d", layer.type);
        }
    }

    optim_update(optim, net);
    for (size_t i = net->layer_count; i-- > 0; )
    {
        Layer layer = net->layers[i];
        if (layer.type == LAYER_SAGE)
            sage_layer_reset_gradient((SageLayer*)layer.ctx);
    }
}

enum {
    // w/ shorthands
    OPT_HELP = 'h',
    // w/o shorthands
    OPT_EPOCHS = 256,  // above ASCII
    OPT_LAYERS,
    OPT_CHANNELS,
    OPT_LR,
    OPT_DATASET,
    OPT_ROOT,
    OPT_CSV,
    OPT_OUTPUT,
    OPT_QUICK,
};

static struct option long_options[] = {
    {"epochs",     required_argument, NULL, OPT_EPOCHS},
    {"layers",     required_argument, NULL, OPT_LAYERS},
    {"channels",   required_argument, NULL, OPT_CHANNELS},
    {"lr",         required_argument, NULL, OPT_LR},
    {"dataset",    required_argument, NULL, OPT_DATASET},
    {"root",       required_argument, NULL, OPT_ROOT},
    {"csv",        no_argument,       NULL, OPT_CSV},
    {"output",     required_argument, NULL, OPT_OUTPUT},
    {"quick",      no_argument,       NULL, OPT_QUICK},
    {"help",       no_argument,       NULL, OPT_HELP},
    {0,            0,                 0,    0}
};

static void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "\n"
            "OPTIONS:\n"
            "  --epochs N       Number of epochs                                         [" XSTR(DEFAULT_EPOCHS) "]\n"
            "  --layers N       Number of layers                                         [" XSTR(DEFAULT_LAYERS) "]\n"
            "  --channels N     Number of channels                                       [" XSTR(DEFAULT_CHANNELS) "]\n"
            "  --lr F           Learning rate                                            [" XSTR(DEFAULT_LR) "]\n"
            "  --dataset NAME   Dataset name (ogbn-arxiv,ogbn-products,ogbn-papers100M)  [" DEFAULT_DATASET "]\n"
            "  --root PATH      Dataset directory                                        [" DEFAULT_ROOT "]\n"
            "  --csv            Enable CSV output                                        [" XSTR(DEFAULT_CSV) "]\n"
            "  --output FILE    Output file (stdout,stderr,path)                         [" XSTR(DEFAULT_OUTPUT) "]\n"
            "  --quick          Quick mode                                               [" XSTR(DEFAULT_QUICK) "]\n"
            "  -h, --help       Show this help\n",
            progname);
}


int main(int argc, char** argv)
{
    srand(0);

    epochs = DEFAULT_EPOCHS;
    layers = DEFAULT_LAYERS;
    channels = DEFAULT_CHANNELS;
    lr = REAL(DEFAULT_LR);
    export_csv = DEFAULT_CSV;
    output_fd = DEFAULT_OUTPUT;
    datasetkind = str_to_dataset_kind(DEFAULT_DATASET);
    root = DEFAULT_ROOT;
    quick = DEFAULT_QUICK;

    int opt;
    while ((opt = getopt_long(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case OPT_EPOCHS: epochs = strtoull(optarg, NULL, 10); break;
            case OPT_LAYERS: layers = strtoull(optarg, NULL, 10); break;
            case OPT_CHANNELS: channels = strtoull(optarg, NULL, 10); break;
            case OPT_LR: lr = REAL(strtod(optarg, NULL)); break;
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
            case OPT_OUTPUT:
            {
                if (strcmp("stdout", optarg) == 0)
                    output_fd = stdout;
                else if (strcmp("stderr", optarg) == 0)
                    output_fd = stderr;
                else
                {
                    output_fd = fopen(optarg, "w+");
                    if (!output_fd)
                    {
                        ERROR("Could not open file %s for csv export: %s", optarg, strerror(errno));
                        usage(argv[0]);
                        return 1;
                    }
                }
                break;
            }
            case OPT_QUICK: quick = true; break;
            case OPT_HELP:
                usage(argv[0]);
                return 0;
            default:
                usage(argv[0]);
                return 1;
        }
    }

    root = expand_path(root);

    if (layers < 2)
    {
        ERROR("Number of layers must be at least 2");
        return 1;
    }

    if (quick)
    {
        if (epochs == DEFAULT_EPOCHS)     epochs   = 10;
        if (layers == DEFAULT_LAYERS)     layers   = 2;
        if (channels == DEFAULT_CHANNELS) channels = 4;
    }

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

    print_config();

#if defined(SPARSE_COO)
    SparseFormat sparse_format = SPARSE_FORMAT_COO;
#else
    SparseFormat sparse_format = SPARSE_FORMAT_CS;
#endif
    Dataset *ds = dataset_load(datasetkind, root, sparse_format);

    int64_t feature_count = ds->feature_count;
    int64_t class_count = ds->class_count;
    uint64_t num_entries = (layers - 1) * 3 + 2;
    LayerConf arch[num_entries];
    size_t n = 0;

    // First layer
    arch[n++] = SAGE(feature_count, channels);
    arch[n++] = RELU(channels);
    arch[n++] = L2NORM(channels);

    // Intermediate layers
    for (size_t i = 1; i < layers - 1; i++)
    {
        arch[n++] = SAGE(channels, channels);
        arch[n++] = RELU(channels);
        arch[n++] = L2NORM(channels);
    }

    // Last layers
    arch[n++] = SAGE(channels, class_count);
    arch[n++] = LOGSOFTMAX(class_count);

    SageNet *net = SAGE_NET_ALLOC(arch, ds, SOURCE_TO_TARGET);
    sage_net_reset_parameters(net);
    sage_net_info(net);

    LogSoftmaxLayer *log_prob_layer = (LogSoftmaxLayer *)net->layers[net->layer_count - 1].ctx;

    Optim *optim = optim_create(OPTIM_ADAM, net, lr);

    printf("GraphSAGE starting...\n");
#if defined(BENCHMARK_MODE)
    timer_enable();

    __attribute__((unused)) volatile Real _unused;
    for (size_t epoch = 1; epoch <= epochs; epoch++)
    {
        TIMER_BLOCK("epoch", {
                inference(net);
                _unused = nll_loss(log_prob_layer, ds->y_train);
                _unused = accuracy(log_prob_layer, ds->y_train);
                _unused = accuracy(log_prob_layer, ds->y_valid);
                _unused = accuracy(log_prob_layer, ds->y_test);
                train(net, ds->y_train, optim);
            });
    }

    if (export_csv)
        timer_export_csv(output_fd);
    else
        timer_print();
#else
    timer_disable();

    Real *loss_hist = ALLOC_OR_DIE(calloc(epochs, sizeof(Real)));
    Real *train_hist = ALLOC_OR_DIE(calloc(epochs, sizeof(Real)));
    Real *valid_hist = ALLOC_OR_DIE(calloc(epochs, sizeof(Real)));
    Real *test_hist = ALLOC_OR_DIE(calloc(epochs, sizeof(Real)));

    for (size_t ep = 1; ep <= epochs; ep++)
    {
        inference(net);

        loss_hist[ep-1] = nll_loss(log_prob_layer, ds->y_train);
        train_hist[ep-1] = accuracy(log_prob_layer, ds->y_train);
        valid_hist[ep-1] = accuracy(log_prob_layer, ds->y_valid);
        test_hist[ep-1]  = accuracy(log_prob_layer, ds->y_test);

        train(net, ds->y_train, optim);

        if (ep % 10 == 0)
            printf("Epoch: %zu/%zu, Loss: %f, Train: %.2f%%, Valid: %.2f%%, Test: %.2f%%\n",
                   ep, epochs, loss_hist[ep-1],
                   100*train_hist[ep-1], 100*valid_hist[ep-1], 100*test_hist[ep-1]);
    }

    if (export_csv)
    {
        if (output_fd == stdout) fprintf(output_fd, "\n--- CSV_OUTPUT_BEGIN ---\n");
        fprintf(output_fd, "epoch,loss,train,valid,test\n");
        for (size_t e = 1; e <= epochs; e++)
        {
            fprintf(output_fd, "%zu,%f,%f,%f,%f\n",
                    e, loss_hist[e-1], 100*train_hist[e-1], 100*valid_hist[e-1], 100*test_hist[e-1]);
        }
        if (output_fd == stdout) fprintf(output_fd, "--- CSV_OUTPUT_END ---\n");
    }
    free(loss_hist);
    free(train_hist);
    free(valid_hist);
    free(test_hist);
#endif

    if (output_fd != stdout && output_fd != stderr)
        fclose(output_fd);
    optim_free(&optim);
    sage_net_free(&net);
    dataset_free(&ds);
    free(root);
    return 0;
}

// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
// #define DO_NOT_OPTIMIZE(expr) __asm__ __volatile__("" : : "g"(expr) : "memory")
