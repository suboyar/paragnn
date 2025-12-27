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
#include "perf.h"
#include "matrix.h"
#include "layers.h"
#include "gnn.h"
#include "graph.h"

#define NOB_IMPLEMENTATION
#include "../nob.h"

#define FLAG_IMPLEMENTATION
#define FLAG_PUSH_DASH_DASH_BACK
#include "../flag.h"

#ifndef EPOCH
#    ifdef USE_OGB_ARXIV
#        define EPOCH 100               // Goal is to have 500 epochs
#    else
#        define EPOCH 10
#    endif // USE_OGB_ARXIV
#endif // EPOCH

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

FileHandler csv_out = {0};

void usage(FILE *stream)
{
    fprintf(stream, "Usage: ./paragnn [OPTIONS]\n");
    fprintf(stream, "OPTIONS:\n");
    flag_print_options(stream);
}

void print_cpu_affinity()
{
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(mask), &mask) == -1) {
        perror("sched_getaffinity");
        return;
    }

    int num_cpus = CPU_COUNT(&mask);
    printf("CPUs in affinity mask: %d [", num_cpus);

    int first = 1;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            if (!first) printf(",");
            printf("%d", i);
            first = 0;
        }
    }
    printf("]\n");
}

void print_config()
{
    printf("==============================CONFIG============================\n");
    printf("Epoch: %zu\n", (size_t)EPOCH);
    printf("Learning rate: %f\n", (float)LEARNING_RATE);
    printf("Layers: %zu\n", (size_t)NUM_LAYERS);
    printf("Hidden dim: %zu\n", (size_t)HIDDEN_DIM);
    printf("CSV file: %s\n", csv_out.filename);
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
    printf("OpenMP Threads: %d\n", omp_get_max_threads());
    printf("Slurm Partition: %s\n", getenv("SLURM_JOB_PARTITION"));
    print_cpu_affinity();
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
    PERF_FUNC_START();

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

    PERF_FUNC_END();
}

void train(SageNet *sage_net, LinearLayer *linearlayer, LogSoftLayer *logsoftlayer, graph_t *g)
{
    PERF_FUNC_START();

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

    // Update weights
    for (size_t i = sage_net->num_layers-1; i < sage_net->num_layers; i--) {
        sage_layer_update_weights(sage_net->sagelayer[i], LEARNING_RATE);
    }

#ifdef USE_PREDICTION_HEAD
    linear_layer_update_weights(linearlayer, LEARNING_RATE);
#endif

    // Reset grads
    sage_net_zero_gradients(sage_net);

#ifdef USE_PREDICTION_HEAD
    linear_layer_zero_gradients(linearlayer);
#endif

    logsoft_layer_zero_gradients(logsoftlayer);

    PERF_FUNC_END();
}

int main(int argc, char** argv)
{
    srand(0);
    nob_minimal_log_level = NOB_WARNING;

    csv_out.fp       = stdout;
    csv_out.filename = "stdout";
    flag_str_var(&csv_out.filename, "csv", csv_out.filename, "Name of the csv file to output to");

    if (!flag_parse(argc, argv)) {
        usage(stderr);
        flag_print_error(stderr);
        exit(1);
    }

    if (strcmp(csv_out.filename, "stdout") != 0) {
        csv_out.fp = fopen(csv_out.filename, "w");
        if (csv_out.fp == NULL) {
            ERROR("Could not open file %s: %s", csv_out.filename, strerror(errno));
        }
    }

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

    PERF_START("run_time");
    for (size_t epoch = 1; epoch <= EPOCH; epoch++) {
        PERF_START("epoch");
        inference(sage_net, linearlayer, logsoftlayer, train_graph);

        double loss = nll_loss(logsoftlayer->output, train_graph->y) / train_graph->y->batch;
        double acc = accuracy(logsoftlayer->output, train_graph->y);
        printf("[epoch %zu] loss: %f, accuracy: %f\n", epoch, loss, acc);

        train(sage_net, linearlayer, logsoftlayer, train_graph);
        PERF_END("epoch");
    }
    PERF_END("run_time");
    
    PERF_PRINT();
    if (csv_out.fp != NULL) PERF_CSV(csv_out.fp);

    double total_time;
    PERF_GET_METRIC("run_time", TOTAL_TIME, &total_time);
    printf("Total run time: %f\n", total_time);

    destroy_sage_net(sage_net);
#ifdef USE_PREDICTION_HEAD
    destroy_linear_layer(linearlayer);
#endif
    destroy_logsoft_layer(logsoftlayer);

    destroy_graph(train_graph);
    destroy_graph(valid_graph);
    destroy_graph(test_graph);

    return 0;
}

// TODO: Use CRS format for edges
// TODO: Xavier Initialization for weight matrices
// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
// TODO: Reset all matrices after train(), and remove memset(agg, 0, ...) in aggregate()
