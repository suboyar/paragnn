#define _GNU_SOURCE
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
#include "dataset_info.h"
#include "timer.h"
#include "layers.h"
#include "nn.h"
#include "sageconv.h"
#include "dataset.h"
#include "optim.h"

#define NOB_IMPLEMENTATION
#include "../nob.h"

// Default values
#define DEFAULT_EPOCHS      100
#define DEFAULT_LAYERS      4
#define DEFAULT_CHANNELS    256
#define DEFAULT_LR          REAL(0.01)
#define DEFAULT_DATASET     "arxiv"
#define DEFAULT_DATADIR     "~/D1/paragnn-dataset"
#define DEFAULT_CSV         "stdout"

static uint64_t     epochs      = DEFAULT_EPOCHS;
static uint64_t     layers      = DEFAULT_LAYERS;
static uint64_t     channels    = DEFAULT_CHANNELS;
static Real         lr          = DEFAULT_LR;
static EdgeFormat   edge_format = EDGE_CSX;
static bool         quick       = false;
static bool         early_stop  = false;
static bool         loss_track  = false;
static FILE        *csv_fd      = NULL; // Set in main, since stdout ins't compile-time constant
static DatasetKind  dataset     = DATASET_ARXIV;
static char        *datadir     = NULL; // Set in main, since it might need to be expanded

void print_config(void)
{
#if defined(SAGECONV_NAIVE_IMPL)
    const char *impl = "naive";
#elif defined(SAGECONV_BLAS_IMPL)
    const char *impl = "blas";
#else
    const char *impl = "tuned";
#endif

    printf("impl=%s prec=%s epochs=%zu lr=%g layers=%zu hidden=%zu data=%s "
           "omp=%d blas=%d partition=%s\n"
           "openblas: %s\n",
           impl,
           sizeof(Real) == sizeof(double) ? "dp" : "sp",
           epochs, lr, layers, channels, ds_infos[dataset].name,
           omp_get_max_threads(), openblas_get_num_threads(),
           getenv("SLURM_JOB_PARTITION"),
           openblas_get_config());
}

// TODO: Check out getrusage from <sys/resource.h>
void print_memory_usage(void)
{
    FILE* file = fopen("/proc/self/status", "r");
    char line[128];

    if (file) {
        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                printf("Memory usage: %s", line);
                break;
            }
        }
        fclose(file);
    }
}

void inference(SageNet *net)
{
    TIMER_FUNC();

    for (int64_t i = 0; i < net->num_layers; i++)
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
        case LAYER_LINEAR:
            linear((LinearLayer*)layer.ctx);
            break;
        default:
            ERROR("Unknown layer type %d", layer.type);
        }
    }
}

void train(SageNet *net, Dataset *ds, Optim *optim, OptimKind kind)
{
    TIMER_FUNC();

    for (size_t i = net->num_layers; i-- > 0; ) {
        Layer layer = net->layers[i];
        switch(layer.type)
        {
        case LAYER_SAGE:
            grad_sageconv((SageLayer*)layer.ctx);
            // sage_layer_update_weights((SageLayer*)layer.ctx, (float)lr);
            break;
        case LAYER_RELU:
            grad_relu((ReluLayer*)layer.ctx);
            break;
        case LAYER_L2NORM:
            grad_l2norm((L2NormLayer*)layer.ctx);
            break;
        case LAYER_LOGSOFTMAX:
            grad_cross_entropy((LogSoftmaxLayer*)layer.ctx, ds->labels);
            break;
        case LAYER_LINEAR:
            grad_linear((LinearLayer*)layer.ctx);
            // linear_layer_update_weights((LinearLayer*)layer.ctx, (float)lr);
            break;
        default:
            ERROR("Unknown layer type %d", layer.type);
        }
    }

    optim_update(optim, kind, net);
}

void zero_grad(SageNet *net)
{
    for (size_t i = net->num_layers; i-- > 0; ) {
        Layer layer = net->layers[i];
        switch (layer.type)
        {
        case LAYER_SAGE:
            sage_layer_zero_gradients((SageLayer*)layer.ctx);
            break;
        case LAYER_RELU:
            relu_layer_zero_gradients((ReluLayer*)layer.ctx);
            break;
        case LAYER_L2NORM:
            normalize_layer_zero_gradients((L2NormLayer*)layer.ctx);
            break;
        case LAYER_LINEAR:
            linear_layer_zero_gradients((LinearLayer*)layer.ctx);
            break;
        case LAYER_LOGSOFTMAX:
            logsoft_layer_zero_gradients((LogSoftmaxLayer*)layer.ctx);
            break;
        default:
            ERROR("Unknown layer type %d", layer.type);
        }
    }
}

bool is_double_eq(double a, double b)
{
    const double abs_tol = 1e-9;
    const double rel_tol = 1e-6;

    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

    double abs_diff = fabs(a - b);
    double abs_max = fmax(fabs(a), fabs(b));

    if (abs_diff > abs_tol && abs_diff > rel_tol * abs_max) {
        return false;
    }

    return true;
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
    OPT_DATADIR,
    OPT_CSV,
    OPT_COO,
    OPT_QUICK,
    OPT_EARLYSTOP,
    OPT_LOSSTRACK,
};

static struct option long_options[] = {
    {"epochs",     required_argument, NULL, OPT_EPOCHS},
    {"layers",     required_argument, NULL, OPT_LAYERS},
    {"channels",   required_argument, NULL, OPT_CHANNELS},
    {"lr",         required_argument, NULL, OPT_LR},
    {"dataset",    required_argument, NULL, OPT_DATASET},
    {"datadir",    required_argument, NULL, OPT_DATADIR},
    {"csv",        required_argument, NULL, OPT_CSV},
    {"coo",        no_argument,       NULL, OPT_COO},
    {"quick",      no_argument,       NULL, OPT_QUICK},
    {"earlystop",  no_argument,       NULL, OPT_EARLYSTOP},
    {"losstrack",  no_argument,       NULL, OPT_LOSSTRACK},
    {0,            0,                 0,    0}
};

void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "\n"
            "OPTIONS:\n"
            "  -epochs N       Number of epochs                [" STRINGIFY(DEFAULT_EPOCHS) "]\n"
            "  -layers N       Number of layers                [" STRINGIFY(DEFAULT_LAYERS) "]\n"
            "  -channels N     Number of channels              [" STRINGIFY(DEFAULT_CHANNELS) "]\n"
            "  -lr F           Learning rate                   [" STRINGIFY(DEFAULT_LR) "]\n"
            "  -dataset NAME   Dataset name                    [" DEFAULT_DATASET "]\n"
            "  -datadir PATH   Dataset directory               [" DEFAULT_DATADIR "]\n"
            "  -csv FILE       CSV output (stdout/stderr/path) [" DEFAULT_CSV "]\n"
            "  -coo            Use COO edge format             [off]\n"
            "  -quick          Quick mode                      [off]\n"
            "  -earlystop      Enable early stopping           [off]\n"
            "  -losstrack      Track loss of each epoch        [off]\n"
            "  -h, -help       Show this help\n",
            progname);
}

int main(int argc, char** argv)
{

    srand(0);

    int opt;
    while ((opt = getopt_long_only(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case OPT_EPOCHS: epochs = strtoull(optarg, NULL, 10); break;
        case OPT_LAYERS: layers = strtoull(optarg, NULL, 10); break;
        case OPT_CHANNELS: channels = strtoull(optarg, NULL, 10); break;
        case OPT_LR: lr = strtoull(optarg, NULL, 10); break;
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
        case OPT_QUICK:      quick       = true;     break;
        case OPT_EARLYSTOP:  early_stop  = true;     break;
        case OPT_LOSSTRACK:  loss_track  = true;     break;
        case OPT_COO:        edge_format = EDGE_COO; break;
        case OPT_HELP:
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (!datadir)
    {
        datadir = DEFAULT_DATADIR;
    }
    datadir = nob_expand_path(datadir);

    if (quick)
    {
        if (epochs == DEFAULT_EPOCHS)     epochs   = 10;
        if (layers == DEFAULT_LAYERS)     layers   = 2;
        if (channels == DEFAULT_CHANNELS) channels = 4;
    }

    FlowDirection flow = SOURCE_TO_TARGET;

    openblas_set_num_threads(omp_get_max_threads());
    print_config();

    Dataset *ds = dataset_load(dataset, datadir, edge_format);
    Dataset *ds_train = dataset_split(ds, SPLIT_TRAIN);
    Dataset *ds_valid = dataset_split(ds, SPLIT_VALID);
    Dataset *ds_test = dataset_split(ds, SPLIT_TEST);
    dataset_free(&ds);

    int64_t num_features = ds_train->num_features;
    int64_t num_classes = ds_train->num_classes;
    uint64_t num_entries = (layers - 1) * 3 + 2;
    LayerConf arch[num_entries];
    size_t n = 0;

    // First layer
    arch[n++] = SAGE(num_features, channels);
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
    arch[n++] = SAGE(channels, num_classes);
    arch[n++] = LOGSOFTMAX(num_classes);

    SageNet *net = SAGE_NET_CREATE(arch, ds_train, flow);
    sage_net_info(net);

    LogSoftmaxLayer *log_prob_layer = (LogSoftmaxLayer *)net->layers[net->num_layers - 1].ctx;

    OptimKind optim_kind = OPTIM_ADAM;
    Optim *optim = optim_create(optim_kind, net, lr);

    timer_enable();
    Real old_loss = REAL_MAX;
    Real *loss_hist = NULL;
    size_t loss_hist_len = 0;
    if (loss_track)
    {
        loss_hist = malloc(epochs * sizeof(*loss_hist));
    }
    for (size_t epoch = 1; epoch <= epochs; epoch++)
    {
        TIMER_BLOCK("epoch", {
                inference(net);
                Real loss = nll_loss(log_prob_layer, ds_train->labels);
                if (early_stop && old_loss < loss)
                {
                    printf("Early stopping at epoch %zu/%zu: loss increased (%.6f -> %.6f)\n",
                           epoch, epochs, old_loss, loss);
                    break;
                }
                if (loss_track)
                {
                    loss_hist[loss_hist_len++] = loss;
                }
                old_loss = loss;
                Real train_acc = accuracy(log_prob_layer, ds_train->labels);
                train(net, ds_train, optim, optim_kind);
                printf("Epoch: %zu/%zu, Loss: %f, Train: %.2f%%\n",
                       epoch, epochs, loss, 100*train_acc);
            });
    }

    timer_disable();
    Real val_acc, test_acc;
    TIMER_BLOCK("valid-inference", {
            sage_net_bind(net, ds_valid);
            inference(net);
            val_acc = accuracy(log_prob_layer, ds_valid->labels);
        });
    TIMER_BLOCK("test-inference", {
            sage_net_bind(net, ds_test);
            inference(net);
            test_acc = accuracy(log_prob_layer, ds_test->labels);
        });
    printf("Valid: %.2f%%, Test: %.2f%%\n", 100*val_acc, 100*test_acc);

    timer_print();
    timer_export_csv(csv_fd);
    if (loss_track)
    {
        printf("Loss history:\n");
        for (size_t i = 0; i < loss_hist_len; i++)
        {
            printf("%f%s", loss_hist[i], i < loss_hist_len-1 ? "," : "\n");
        }
    }

    optim_free(&optim, optim_kind);
    sage_net_free(&net);
    dataset_free(&ds_test);
    dataset_free(&ds_valid);
    dataset_free(&ds_train);
    free(datadir);
    return 0;
}

// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
