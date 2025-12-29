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

void print_config(void)
{
    const char *dataset =
#ifdef USE_OGB_ARXIV
        "ogb-arxiv";
#else
        "dev";
#endif

    printf("Config: epochs=%zu, lr=%.4f, layers=%zu, hidden=%zu, data=%s, partition=%s, threads=%d\n",
           (size_t)EPOCH, (float)LEARNING_RATE, (size_t)NUM_LAYERS,
           (size_t)HIDDEN_DIM, dataset, getenv("SLURM_JOB_PARTITION"), omp_get_max_threads());
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

void inference(SageNet *net, graph_t *g)
{
    TIMER_FUNC();

    for (size_t i = 0; i < net->enc_depth; i++) {
        sageconv(net->enc_sage[i], g);
        relu(net->enc_relu[i]);
        normalize(net->enc_norm[i]);
    }

    sageconv(net->cls_sage, g);

#ifdef USE_PREDICTION_HEAD
    linear(net->linear);
#endif

    logsoft(net->logsoft);

}

void train(SageNet *net, graph_t *g)
{
    TIMER_FUNC();

   cross_entropy_backward(net->logsoft, g->y);

#ifdef USE_PREDICTION_HEAD
    linear_backward(net->linear);
    linear_layer_update_weights(net->linear, LEARNING_RATE);
#endif

    sageconv_backward(net->cls_sage, g);
    sage_layer_update_weights(net->cls_sage, LEARNING_RATE);

    for (size_t i = net->enc_depth-1; i < net->enc_depth; i--) {
        normalize_backward(net->enc_norm[i]);
        relu_backward(net->enc_relu[i]);
        sageconv_backward(net->enc_sage[i], g);
        sage_layer_update_weights(net->enc_sage[i], LEARNING_RATE);
    }

    // Reset grads
    sage_net_zero_gradients(net);
}

int main(int argc, char** argv)
{
    srand(0);
    nob_minimal_log_level = NOB_WARNING;

    flag_str_var(&csv_out.filename, "csv", csv_out.filename, "Name of the csv file to output to");

    if (!flag_parse(argc, argv)) {
        usage(stderr);
        flag_print_error(stderr);
        exit(1);
    }

    print_config();

    // graph_t *train_graph = load_graph();
    graph_t *train_graph, *valid_graph, *test_graph;
    load_split_graph(&train_graph, &valid_graph, &test_graph);

    SageNet *net = sage_net_create(NUM_LAYERS, HIDDEN_DIM, train_graph);
    sage_net_info(net);

    TIMER_BLOCK("run_time", {
            for (size_t epoch = 1; epoch <= EPOCH; epoch++) {
                TIMER_BLOCK("epoch", {
                        inference(net, train_graph);

                        double loss = nll_loss(net->logsoft->output, train_graph->y) / train_graph->y->batch;
                        double acc = accuracy(net->logsoft->output, train_graph->y);
                        printf("[epoch %zu] loss: %f, accuracy: %f\n", epoch, loss, acc);

                        train(net, train_graph);
                    });
            }
        });

    timer_print();
    if (csv_out.fp != NULL) timer_export_csv(csv_out.fp);
    printf("Total run time: %f\n", timer_get_time("run_time", TIMER_TOTAL_TIME));

    sage_net_destroy(net);
    return 0;
}

// TODO: Use CRS format for edges
// TODO: Xavier Initialization for weight matrices
// TODO: Fast exp: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
