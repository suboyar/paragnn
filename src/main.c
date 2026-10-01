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
#define DEFAULT_DATASET     "arxiv"
#define DEFAULT_DATADIR     "~/D1/paragnn-dataset"
#define DEFAULT_CSV         stdout

static uint64_t     epochs;
static uint64_t     layers;
static uint64_t     channels;
static Real         lr;
static bool         quick;
static bool         early_stop;
static bool         loss_track;
static FILE        *csv_fd;
static DatasetKind  datasetkind;
static char        *datadir;

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

static void train(SageNet *net, Dataset *ds, Optim *optim, OptimKind kind)
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
            grad_logsoftmax_nll((LogSoftmaxLayer*)layer.ctx, ds->y);
            break;
        default:
            ERROR("Unknown layer type %d", layer.type);
        }
    }

    optim_update(optim, kind, net);
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
    OPT_DATADIR,
    OPT_CSV,
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
    {"quick",      no_argument,       NULL, OPT_QUICK},
    {"earlystop",  no_argument,       NULL, OPT_EARLYSTOP},
    {"losstrack",  no_argument,       NULL, OPT_LOSSTRACK},
    {0,            0,                 0,    0}
};

static void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "\n"
            "OPTIONS:\n"
            "  -epochs N       Number of epochs                [" XSTR(DEFAULT_EPOCHS) "]\n"
            "  -layers N       Number of layers                [" XSTR(DEFAULT_LAYERS) "]\n"
            "  -channels N     Number of channels              [" XSTR(DEFAULT_CHANNELS) "]\n"
            "  -lr F           Learning rate                   [" XSTR(DEFAULT_LR) "]\n"
            "  -dataset NAME   Dataset name                    [" DEFAULT_DATASET "]\n"
            "  -datadir PATH   Dataset directory               [" DEFAULT_DATADIR "]\n"
            "  -csv FILE       CSV output (stdout/stderr/path) [" XSTR(DEFAULT_CSV) "]\n"
            "  -quick          Quick mode                      [off]\n"
            "  -earlystop      Enable early stopping           [off]\n"
            "  -losstrack      Track loss of each epoch        [off]\n"
            "  -h, -help       Show this help\n",
            progname);
}

int main(int argc, char** argv)
{
    srand(0);

    epochs = DEFAULT_EPOCHS;
    layers = DEFAULT_LAYERS;
    channels = DEFAULT_CHANNELS;
    lr = REAL(DEFAULT_LR);
    csv_fd = DEFAULT_CSV;
    datasetkind = DATASET_ARXIV;
    datadir = DEFAULT_DATADIR;
    quick = false;
    early_stop = false;
    loss_track = false;

    int opt;
    while ((opt = getopt_long_only(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case OPT_EPOCHS: epochs = strtoull(optarg, NULL, 10); break;
        case OPT_LAYERS: layers = strtoull(optarg, NULL, 10); break;
        case OPT_CHANNELS: channels = strtoull(optarg, NULL, 10); break;
        case OPT_LR: lr = strtof(optarg, NULL); break;
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
        case OPT_HELP:
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    datadir = expand_path(datadir);

    if (quick)
    {
        if (epochs == DEFAULT_EPOCHS)     epochs   = 10;
        if (layers == DEFAULT_LAYERS)     layers   = 2;
        if (channels == DEFAULT_CHANNELS) channels = 4;
    }

    openblas_set_num_threads(omp_get_max_threads());
    print_config();

#if defined(SPARSE_COO)
    SparseFormat sparse_format = SPARSE_FORMAT_COO;
#else
    SparseFormat sparse_format = SPARSE_FORMAT_CS;
#endif
    Dataset *ds_train = dataset_load(datasetkind, datadir, sparse_format, SPLIT_TRAIN);
    Dataset *ds_valid = dataset_load(datasetkind, datadir, sparse_format, SPLIT_VALID);
    Dataset *ds_test  = dataset_load(datasetkind, datadir, sparse_format, SPLIT_TEST);

    int64_t feature_count = ds_train->feature_count;
    int64_t class_count = ds_train->class_count;
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

    FlowDirection flow = SOURCE_TO_TARGET;
    SageNet *net = SAGE_NET_ALLOC(arch, ds_train, flow);
    sage_net_reset_parameters(net);
    sage_net_info(net);

    LogSoftmaxLayer *log_prob_layer = (LogSoftmaxLayer *)net->layers[net->layer_count - 1].ctx;

    OptimKind optim_kind = OPTIM_ADAM;
    Optim *optim = optim_create(optim_kind, net, lr);

    printf("GraphSAGE starting...\n");

#if defined(BENCHMARK_MODE)
    timer_enable();

    for (size_t epoch = 1; epoch <= epochs; epoch++)
    {
        TIMER_BLOCK("epoch", {
                inference(net);
                accuracy(log_prob_layer, ds_train->y);
                nll_loss(log_prob_layer, ds_train->y);
                train(net, ds_train, optim, optim_kind);
            });
    }

    timer_print();
    timer_export_csv(csv_fd);
#else
    timer_disable();
    Real *loss_hist = malloc(epochs * sizeof(Real));
    Real *train_hist = malloc(epochs * sizeof(Real));
    Real *valid_hist = malloc(epochs * sizeof(Real));
    Real *test_hist = malloc(epochs * sizeof(Real));

    for (size_t epoch = 1; epoch <= epochs; epoch++)
    {
        sage_net_bind(net, ds_train);
        inference(net);

        train_hist[epoch-1] = accuracy(log_prob_layer, ds_train->y);
        loss_hist[epoch-1] = nll_loss(log_prob_layer, ds_train->y);

        train(net, ds_train, optim, optim_kind);

        sage_net_bind(net, ds_valid);
        inference(net);
        valid_hist[epoch-1] = accuracy(log_prob_layer, ds_valid->y);

        sage_net_bind(net, ds_test);
        inference(net);
        test_hist[epoch-1] = accuracy(log_prob_layer, ds_test->y);

        printf("Epoch: %zu/%zu, Loss: %f, Train: %.2f%%, Valid: %.2f%%, Test: %.2f%%\n",
               epoch, epochs, loss_hist[epoch-1],
               100*train_hist[epoch-1], 100*valid_hist[epoch-1], 100*test_hist[epoch-1]);
    }

    if (csv_fd)
    {
        if (csv_fd == stdout) fprintf(csv_fd, "\n--- CSV_OUTPUT_BEGIN ---\n");
        fprintf(csv_fd, "epoch,loss,train,valid,test\n");
        for (size_t e = 1; e <= epochs; e++)
        {
            fprintf(csv_fd, "%zu,%f,%f,%f,%f\n",
                    e, loss_hist[e-1], 100*train_hist[e-1], 100*valid_hist[e-1], 100*test_hist[e-1]);
        }
        if (csv_fd == stdout) fprintf(csv_fd, "--- CSV_OUTPUT_END ---\n");
    }
    free(loss_hist);
    free(train_hist);
    free(valid_hist);
    free(test_hist);
#endif


    optim_free(&optim, optim_kind);
    sage_net_free(&net);
    dataset_free(&ds_test);
    dataset_free(&ds_valid);
    dataset_free(&ds_train);
    free(datadir);
    return 0;
}

// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
