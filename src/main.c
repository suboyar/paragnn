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

    for (size_t i = 0; i < net->num_layers; i++) {
        net->layers[i].forward(&net->layers[i], slice);
    }
}

void train(SageNet *net, Slice slice)
{
    TIMER_FUNC();
    for (size_t i = net->num_layers; i-- > 0; ) {
        net->layers[i].backward(&net->layers[i], slice);
        if (net->layers[i].update)
            net->layers[i].update(&net->layers[i], lr);
    }

    // Reset grads
    for (size_t i = 0; i < net->num_layers; i++) {
        if (net->layers[i].zero_grad)
            net->layers[i].zero_grad(&net->layers[i]);
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

    openblas_set_num_threads(omp_get_max_threads());
    print_config();

    Dataset *data = load_arxiv_dataset();
    size_t num_features = data->num_features;
    size_t num_classes = data->num_classes;
    size_t num_entries = (layers - 1) * 3 + 2;
    LayerConf arch[num_entries];
    size_t n = 0;

    // First layer
    arch[n++] = SAGE(num_features, channels);
    arch[n++] = RELU(channels);
    arch[n++] = L2NORM(channels);

    // Intermediate layers
    for (size_t i = 1; i < layers - 1; i++) {
        arch[n++] = SAGE(channels, channels);
        arch[n++] = RELU(channels);
        arch[n++] = L2NORM(channels);
    }

    // Last layers
    arch[n++] = SAGE(channels, num_classes);
    arch[n++] = LOGSOFT(num_classes);

    SageNet *net = SAGE_NET_CREATE(arch, data);
    sage_net_info(net);

    size_t runs = 1;
#ifdef FULL_INFERENCE
    Slice inference_slice = data->full;
    double test_accs[runs];
#else
    Slice inference_slice = data->train;
#endif // FULL_INFERENCE

    Matrix *output = *net->layers[net->num_layers - 1].output_ptr;

    for (size_t run = 1; run <= runs; run++) {
#ifdef FULL_INFERENCE
        double best_valid_acc = 0.0;
        double test_at_best_valid = 0.0;
#endif

        TIMER_BLOCK("run", {
            for (size_t epoch = 1; epoch <= epochs; epoch++) {
                TIMER_BLOCK("epoch", {
                    inference(net, inference_slice);
                    double loss = nll_loss(output, data->labels, data->train.node);
                    double train_acc = accuracy(output, data->labels, data->num_classes, data->train.node);
#ifdef FULL_INFERENCE
                    double valid_acc = accuracy(output, data->labels, data->num_classes, data->valid.node);
                    double test_acc = accuracy(output, data->labels, data->num_classes, data->test.node);
#endif
                    train(net, data->train);

                    printf("(Run %zu) Epoch: %zu/%zu, Loss: %f, Train: %.2f%%",
                           run, epoch, epochs, loss, 100*train_acc);
#ifdef FULL_INFERENCE
                    printf(", Valid: %.2f%%, Test: %.2f%%", 100*valid_acc, 100*test_acc);

                    if (valid_acc > best_valid_acc) {
                        best_valid_acc = valid_acc;
                        test_at_best_valid = test_acc;
                    }
#endif
                    printf("\n");
                });
            }
        });

#ifdef FULL_INFERENCE
        test_accs[run - 1] = test_at_best_valid;
        printf("Run %zu: Test accuracy at best valid = %.2f%%\n\n", run, 100*test_at_best_valid);
#endif
        sage_net_reset(net);
    }

#ifdef FULL_INFERENCE
    double mean = 0.0;
    for (size_t i = 0; i < runs; i++) mean += test_accs[i];
    mean /= runs;
    double var = 0.0;
    for (size_t i = 0; i < runs; i++) var += (test_accs[i] - mean) * (test_accs[i] - mean);
    double std = sqrt(var / runs);
    printf("Test accuracy: %.2f%% ± %.2f%%\n", 100*mean, 100*std);
#endif

    timer_print();
    printf("Total run time: %f\n", timer_get_time("run", TIMER_TOTAL_TIME));
    timer_export_csv(csv_name);

    sage_net_destroy(net);
    destroy_dataset(data);
    return 0;
}

// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
