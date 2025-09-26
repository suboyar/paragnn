#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <zlib.h>

#include "core.h"
#include "benchmark.h"
#include "matrix.h"
#include "layers.h"
#include "gnn.h"
#include "graph.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

#define EPOCH 2
#define LEARNING_RATE 0.1f

size_t K;
size_t sample_size;

void reset_grad(matrix_t *grad)
{
    memset(grad->data, 0, grad->capacity);
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

void benchmark_layers(K_SageLayers *k_sagelayers, LinearLayer *linearlayer, LogSoftLayer *logsoftlayer, graph_t *g)
{
    for (size_t k = 0; k < k_sagelayers->k_layers; k++) {
        BENCH_START("sagelayer");
        sageconv(k_sagelayers->sagelayer[k], g);
        BENCH_STOP("sagelayer");

        BENCH_START("relu");
        relu(k_sagelayers->relulayer[k]);
        BENCH_STOP("relu");

        BENCH_START("normalize");
        normalize(k_sagelayers->normalizelayer[k]);
        BENCH_STOP("normalize");
    }

    BENCH_START("linear");
    linear(linearlayer);
    BENCH_STOP("linear");

    BENCH_START("logsoft");
    logsoft(logsoftlayer);
    BENCH_STOP("logsoft");

    BENCH_START("nll_loss");
    double loss = nll_loss(logsoftlayer->output, g->y) / BATCH_DIM(g->y);
    BENCH_STOP("nll_loss");
    printf("[epoch %zu] loss: %f\n", (size_t)EPOCH, loss);

    BENCH_START("cross_entropy_backward");
    cross_entropy_backward(logsoftlayer, g->y);
    BENCH_STOP("cross_entropy_backward");

    BENCH_START("linear_backward");
    linear_backward(linearlayer);
    BENCH_STOP("linear_backward");

    for (size_t k = k_sagelayers->k_layers-1; k < k_sagelayers->k_layers; k--) {
        BENCH_START("normalize_backward");
        normalize_backward(k_sagelayers->normalizelayer[k]);
        BENCH_STOP("normalize_backward");

        BENCH_START("relu_backward");
        relu_backward(k_sagelayers->relulayer[k]);
        BENCH_STOP("relu_backward");

        BENCH_START("sageconv_backward");
        sageconv_backward(k_sagelayers->sagelayer[k], g);
        BENCH_STOP("sageconv_backward");
    }

    BENCH_START("update_linear_weights");
    update_linear_weights(linearlayer, LEARNING_RATE);
    BENCH_STOP("update_linear_weights");

    for (size_t k = k_sagelayers->k_layers-1; k < k_sagelayers->k_layers; k--) {
        BENCH_START("update_sageconv_weights");
        update_sageconv_weights(k_sagelayers->sagelayer[k], LEARNING_RATE);
        BENCH_STOP("update_sageconv_weights");
    }
}

int main(void)
{
    srand(0);
    nob_minimal_log_level = NOB_WARNING;

    // graph_t *g = load_graph();
    graph_t *train, *valid, *test;
    load_split_graph(&train, &valid, &test);

    size_t hidden_dim = 256;

    size_t batch_size   = train->num_nodes;
    size_t num_classes  = train->num_label_classes;

    K_SageLayers *k_sagelayers = init_k_sage_layers(1, hidden_dim, train);
    LinearLayer *linearlayer = init_linear_layer(batch_size, hidden_dim, num_classes);
    LogSoftLayer *logsoftlayer = init_logsoft_layer(batch_size, num_classes);

    CONNECT_SAGE_LAYERS(k_sagelayers);
    CONNECT_LAYER(LAST_SAGE_LAYER(k_sagelayers), linearlayer);
    CONNECT_LAYER(linearlayer, logsoftlayer);


    for (size_t epoch = 1; epoch <= EPOCH; epoch++) {
        benchmark_layers(k_sagelayers, linearlayer, logsoftlayer, train);
    }

    printf("Benchmark results (%zu epochs):\n", (size_t)EPOCH);
    benchmark_print();

    destroy_logsoft_layer(logsoftlayer);
    destroy_linear_layer(linearlayer);
    destroy_k_sage_layers(k_sagelayers);

    destroy_graph(train);
    destroy_graph(valid);
    destroy_graph(test);

    return 0;
}

// TODO: Finish benchmarking tool (Friday)
// TODO: Implement orchestration tool within nob.h (Saturday)
// TODO: Perform benchmarking and gather results (Sunday)
// TODO: Find out how you should run each of the splits
// TODO: Use CRS format for edges
// TODO: Xavier Initialization for weight matrices
