#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <omp.h>

#include "core.h"
#include "timer.h"
#include "matrix.h"
#include "layers.h"
#include "gnn.h"
#include "dataset.h"

#define NOB_IMPLEMENTATION
#include "../nob.h"

#define FLAG_IMPLEMENTATION
const int flag_push_dash_DASH_BACK;
#include "../flag.h"

#ifdef NDEBUG
static size_t epochs = 1000;
static size_t layers = 4;
static size_t channels = 256;
static float lr = 0.01f;
#else
static size_t epochs = 50;
static size_t layers = 2;
static size_t channels = 4;
static float lr = 0.01f;
#endif // NDEBUG

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
    printf("Config: epochs=%zu, lr=%g, layers=%zu, hidden=%zu, data=%s, partition=%s, threads=%d\n",
           epochs, lr, layers, channels, dataset, getenv("SLURM_JOB_PARTITION"), omp_get_max_threads());
}

// TODO: Check out getrusage from <sys/resource.h>
void print_memory_usage()
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

void inference(SageNet *net, Slice slice)
{
    TIMER_FUNC();

    for (size_t i = 0; i < net->enc_depth; i++) {
        sageconv(net->enc_sage[i], slice.node, slice.edge);
        relu(net->enc_relu[i], slice.node);
        normalize(net->enc_norm[i], slice.node);
    }

    sageconv(net->cls_sage, slice.node, slice.edge);

#ifdef USE_PREDICTION_HEAD
    linear(net->linear, slice.node);
#endif

    logsoft(net->logsoft, slice.node);

}

void train(SageNet *net, Slice slice)
{
    TIMER_FUNC();

    cross_entropy_backward(net->logsoft, slice.node);

#ifdef USE_PREDICTION_HEAD
    linear_backward(net->linear, slice.node);
    linear_layer_update_weights(net->linear, LEARNING_RATE);
#endif

    sageconv_backward(net->cls_sage, slice.node, slice.edge);
    sage_layer_update_weights(net->cls_sage, lr);

    for (size_t i = net->enc_depth-1; i < net->enc_depth; i--) {
        normalize_backward(net->enc_norm[i], slice.node);
        relu_backward(net->enc_relu[i], slice.node);
        sageconv_backward(net->enc_sage[i], slice.node, slice.edge);
        sage_layer_update_weights(net->enc_sage[i], lr);
    }

    // Reset grads
    sage_net_zero_gradients(net);
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

    flag_str_var(&csv_name, "csv", NULL, "Output path for timing CSV (use 'stdout' for console)");
    flag_size_var(&epochs, "epochs", 1000, "Number of training epochs");
    flag_size_var(&layers, "layers", 4, "Number of layers in the SageNet model");
    flag_size_var(&channels, "channels", 256, "Number of hidden channels per layer");
    flag_float_var(&lr, "lr", 0.01f, "Learning rate for training");

    if (!flag_parse(argc, argv)) {
        usage(stderr);
        flag_print_error(stderr);
        exit(1);
    }

    print_config();

    Dataset *data = load_arxiv_dataset();

    SageNet *net = sage_net_create(layers, channels, data);
    sage_net_info(net);

    TIMER_BLOCK("run_time", {
            for (size_t epoch = 1; epoch <= epochs; epoch++) {
                TIMER_BLOCK("epoch", {
                        inference(net, data->full);
                        double loss = nll_loss(net->logsoft->output, data->labels, data->train.node);
                        double train_acc = accuracy(net->logsoft->output, data->labels, data->num_classes, data->train.node);
                        double valid_acc = accuracy(net->logsoft->output, data->labels, data->num_classes, data->valid.node);
                        double test_acc = accuracy(net->logsoft->output, data->labels, data->num_classes, data->test.node);
                        train(net, data->train);
                        printf("[Epoch %zu] Loss: %f, Train: %.2f%%, Valid: %.2f%%, Test: %.2f%%\n",
                               epoch, loss, 100*train_acc, 100*valid_acc, 100*test_acc);
                    });
            }
        });

    timer_print();
    printf("Total run time: %f\n", timer_get_time("run_time", TIMER_TOTAL_TIME));

    timer_export_csv(csv_name);

    sage_net_destroy(net);
    destroy_dataset(data);
    return 0;
}

// TODO: Use CRS format for edges
// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
