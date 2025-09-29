#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "core.h"
#include "perf.h"
#include "matrix.h"
#include "layers.h"
#include "gnn.h"
#include "graph.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

#ifndef BENC_CSV_FILE
#define BENC_CSV_FILE stdout
#endif // BENC_CSV_FILE

#ifndef EPOCH
#define EPOCH 10               // Goal is to have 500 epochs
#endif

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01f
#endif

#ifndef NUM_LAYERS
#define NUM_LAYERS 3
#endif

#ifndef HIDDEN_DIM
#    ifdef USE_OGB_ARXIV
#        define HIDDEN_DIM 256
#    else
#        define HIDDEN_DIM 5
#    endif // USE_OGB_ARXIV
#endif // HIDDEN_DIM

// #define USE_PREDICTION_HEAD

void print_config()
{
    printf("==============================CONFIG============================\n");
    printf("Epoch: %zu\n", (size_t)EPOCH);
    printf("Learning rate: %f\n", (float)LEARNING_RATE);
    printf("Layers: %zu\n", (size_t)NUM_LAYERS);
    printf("Hidden dim: %zu\n", (size_t)HIDDEN_DIM);
#ifndef USE_OGB_ARXIV
    printf("Graph Dataset: dev\n");
#else
    printf("Graph Dataset: ogb-arxiv\n");
#endif
#ifndef USE_PREDICTION_HEAD
    printf("Prediction Head: Disabled\n");
#else
    printf("Prediction Head: Enabled\n");
#endif
    printf("Slurm Partition: %s\n", getenv("SLURM_JOB_PARTITION"));
    printf("================================================================\n\n");
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

void inference(SageNet *sage_net, LinearLayer *linearlayer, LogSoftLayer *logsoftlayer, graph_t *g)
{
    for (size_t i = 0; i < sage_net->num_layers; i++) {
        sageconv(sage_net->sagelayer[i], g);
        relu(sage_net->relulayer[i]);
        normalize(sage_net->normalizelayer[i]);
    }

#ifdef USE_PREDICTION_HEAD
    linear(linearlayer);
#else
    (void) linearlayer;
#endif

    logsoft(logsoftlayer);
}

void train(SageNet *sage_net, LinearLayer *linearlayer, LogSoftLayer *logsoftlayer, graph_t *g)
{
    cross_entropy_backward(logsoftlayer, g->y);

#ifdef USE_PREDICTION_HEAD
    linear_backward(linearlayer);
#else
    (void) linearlayer;
#endif

    for (size_t i = sage_net->num_layers-1; i < sage_net->num_layers; i--) {
        normalize_backward(sage_net->normalizelayer[i]);
        relu_backward(sage_net->relulayer[i]);
        sageconv_backward(sage_net->sagelayer[i], g);
    }

#ifdef USE_PREDICTION_HEAD
    update_linear_weights(linearlayer, LEARNING_RATE);
#endif

    for (size_t i = sage_net->num_layers-1; i < sage_net->num_layers; i--) {
        update_sageconv_weights(sage_net->sagelayer[i], LEARNING_RATE);

    }
}

int main(void)
{
    srand(0);
    nob_minimal_log_level = NOB_WARNING;

    print_config();

    // graph_t *train_graph = load_graph();
    graph_t *train_graph, *valid_graph, *test_graph;
    load_split_graph(&train_graph, &valid_graph, &test_graph);

    size_t batch_size   = train_graph->num_nodes;
    size_t num_classes  = train_graph->num_label_classes;

#ifdef USE_PREDICTION_HEAD
    SageNet *sage_net = init_sage_net(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, train_graph);
    LinearLayer *linearlayer = init_linear_layer(batch_size, HIDDEN_DIM, num_classes);
    LogSoftLayer *logsoftlayer = init_logsoft_layer(batch_size, num_classes);

    CONNECT_SAGE_NET(sage_net);
    CONNECT_LAYER(SAGE_NET_OUTPUT(sage_net), linearlayer);
    CONNECT_LAYER(linearlayer, logsoftlayer);
#else
    SageNet *sage_net = init_sage_net(NUM_LAYERS, HIDDEN_DIM, num_classes, train_graph);
    LinearLayer *linearlayer = NULL;
    LogSoftLayer *logsoftlayer = init_logsoft_layer(batch_size, num_classes);

    CONNECT_SAGE_NET(sage_net);
    CONNECT_LAYER(SAGE_NET_OUTPUT(sage_net), logsoftlayer);
#endif // USE_PREDICTION_HEAD


    for (size_t epoch = 1; epoch <= EPOCH; epoch++) {

        BENCH_CALL("inference", inference, sage_net, linearlayer, logsoftlayer, train_graph);

        double loss = BENCH_EXPR("nll_loss", nll_loss(logsoftlayer->output, train_graph->y) / BATCH_DIM(train_graph->y));
        double acc = BENCH_EXPR("accuracy", accuracy(logsoftlayer->output, train_graph->y));
        // double loss = nll_loss(logsoftlayer->output, train_graph->y) / BATCH_DIM(train_graph->y);
        printf("[epoch %zu] loss: %f, accuracy: %f\n", epoch, loss, acc);

        BENCH_CALL("train", train, sage_net, linearlayer, logsoftlayer, train_graph);
    }

    BENCH_PRINT();

    destroy_sage_net(sage_net);
#ifdef USE_PREDICTION_HEAD
    destroy_linear_layer(linearlayer);
#endif
    destroy_graph(train_graph);
    destroy_graph(valid_graph);
    destroy_graph(test_graph);

    return 0;
}

// TODO: Finish benchmarking tool (Friday)
// TODO: Implement orchestration tool within nob.h (Saturday)
// TODO: Perform benchmarking and gather results (Sunday)
// TODO: Find out how you should run each of the splits
// TODO: Use CRS format for edges
// TODO: Xavier Initialization for weight matrices
