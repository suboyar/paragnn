#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <omp.h>
#include <cblas.h>

#include "core.h"
#include "timer.h"
#include "layers.h"
#include "gnn.h"
#include "dataset.h"
#include "optim.h"

#define NOB_IMPLEMENTATION
#include "../nob.h"

#define FLAG_IMPLEMENTATION
const int flag_push_dash_DASH_BACK;
#include "../flag.h"

static bool quick = false;
static bool early_stop = false;
static size_t epochs = 1000;
static size_t layers = 4;
static size_t channels = 256;
static double lr = 0.01;
static char *edge_format_str = "cco";
static char *csv_name = NULL;

void usage(FILE *stream)
{
    fprintf(stream, "Usage: ./paragnn [OPTIONS]\n");
    fprintf(stream, "OPTIONS:\n");
    flag_print_options(stream);
}

void print_config(void)
{
    const char *dataset = "ogb-arxiv";
    const char *partition = getenv("SLURM_JOB_PARTITION");
    const int omp_threads = omp_get_max_threads();

    printf("Config: epochs=%zu, lr=%g, layers=%zu, hidden=%zu, data=%s\n",
           epochs, lr, layers, channels, dataset);
#if defined(USE_CBLAS)
    printf("Runtime: threads(OMP=%d, BLAS=%d), partition=%s\n",
           omp_threads, openblas_get_num_threads(), partition);
    printf("OpenBLAS: %s\n", openblas_get_config());
#else
    printf("Runtime: threads(OMP=%d, BLAS=off), partition=%s\n",
           omp_threads, partition);
#endif
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

    for (size_t i = 0; i < net->num_layers; i++)
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
            normalize((L2NormLayer*)layer.ctx);
            break;
        case LAYER_LOGSOFT:
            logsoft((LogSoftLayer*)layer.ctx);
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
            sageconv_backward((SageLayer*)layer.ctx);
            // sage_layer_update_weights((SageLayer*)layer.ctx, (float)lr);
            break;
        case LAYER_RELU:
            relu_backward((ReluLayer*)layer.ctx);
            break;
        case LAYER_L2NORM:
            normalize_backward((L2NormLayer*)layer.ctx);
            break;
        case LAYER_LOGSOFT:
            cross_entropy_backward((LogSoftLayer*)layer.ctx, ds->labels);
            break;
        case LAYER_LINEAR:
            linear_backward((LinearLayer*)layer.ctx);
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
        case LAYER_LOGSOFT:
            logsoft_layer_zero_gradients((LogSoftLayer*)layer.ctx);
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

int main(int argc, char** argv)
{
    // srand(time(NULL));
    srand(0);
    nob_minimal_log_level = NOB_WARNING;
    flag_bool_var(&quick, "quick", false, "Use small model config for fast iteration");
    flag_bool_var(&early_stop, "early-stop", false, "Stop training when loss stops decreasing");
    flag_str_var(&csv_name, "csv", NULL, "Output path for timing CSV (use 'stdout' for console)");
    flag_size_var(&epochs, "epochs", 1000, "Number of training epochs");
    flag_size_var(&layers, "layers", 4, "Number of layers in the SageNet model");
    flag_size_var(&channels, "channels", 256, "Number of hidden channels per layer");
    flag_double_var(&lr, "lr", 0.01f, "Learning rate for training");
    flag_str_var(&edge_format_str, "edges", "coo", "Edge storage format (coo, csr, csc)");

    if (!flag_parse(argc, argv)) {
        usage(stderr);
        flag_print_error(stderr);
        exit(1);
    }

    EdgeFormat edge_format = parse_edge_format(edge_format_str);
    if (quick)
    {
        epochs   = 5;
        layers   = 2;
        channels = 4;
    }

    openblas_set_num_threads(omp_get_max_threads());
    print_config();

    bool to_symmetric = true;
    Dataset *ds = dataset_load_arxiv(edge_format, to_symmetric);

    Dataset *ds_train = dataset_split(ds, SPLIT_TRAIN);
    Dataset *ds_valid = dataset_split(ds, SPLIT_VALID);
    Dataset *ds_test = dataset_split(ds, SPLIT_TEST);
    dataset_free(&ds);

    size_t num_features = ds_train->num_features;
    size_t num_classes = ds_train->num_classes;
    size_t num_entries = (layers - 1) * 3 + 2;
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
    arch[n++] = LOGSOFT(num_classes);

    SageNet *net = SAGE_NET_CREATE(arch, ds_train);
    sage_net_info(net);

    LogSoftLayer *log_prob_layer = (LogSoftLayer *)net->layers[net->num_layers - 1].ctx;

    OptimKind optim_kind = OPTIM_ADAM;
    Optim *optim = optim_create(optim_kind, net, lr);

    timer_enable();
    double old_loss = DBL_MAX;
    for (size_t epoch = 1; epoch <= epochs; epoch++)
    {
        TIMER_BLOCK("epoch", {
                inference(net);
                float loss = nll_loss(log_prob_layer, ds_train->labels);
                if (early_stop && old_loss < loss)
                {
                    printf("Early stopping at epoch %zu/%zu: loss increased (%.6f -> %.6f)\n",
                           epoch, epochs, old_loss, loss);
                    break;
                }
                old_loss = loss;
                float train_acc = accuracy(log_prob_layer, ds_train->labels);
                train(net, ds_train, optim, optim_kind);
                printf("Epoch: %zu/%zu, Loss: %f, Train: %.2f%%\n",
                       epoch, epochs, loss, 100*train_acc);
            });
    }

    timer_disable();
    double val_acc, test_acc;
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
    timer_export_csv(csv_name);

    optim_free(&optim, optim_kind);
    sage_net_free(&net);
    dataset_free(&ds_test);
    dataset_free(&ds_valid);
    dataset_free(&ds_train);
    return 0;
}

// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
