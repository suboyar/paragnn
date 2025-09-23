#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <zlib.h>

#include "core.h"
#include "matrix.h"
#include "layers.h"
#include "gnn.h"
#include "graph.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

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

int main(void)
{
    srand(0);
    // nob_minimal_log_level = NOB_NO_LOGS;

    graph_t *g = load_graph();

    size_t hidden_dim = 256;
    float lr = 0.1;

    size_t batch_size   = g->num_nodes;
    size_t num_features = g->num_node_features;
    size_t num_classes  = g->num_label_classes;

    K_SageLayers *k_sagelayers = init_k_sage_layers(2, hidden_dim, g);
    LinearLayer *linearlayer = init_linear_layer(batch_size, hidden_dim, num_classes);
    LogSoftLayer *logsoftlayer = init_logsoft_layer(batch_size, num_classes);

    CONNECT_SAGE_LAYERS(k_sagelayers);
    CONNECT_LAYER(LAST_SAGE_LAYER(k_sagelayers), linearlayer);
    CONNECT_LAYER(linearlayer, logsoftlayer);

    const size_t MAX_EPOCH = 0;
    for (size_t epoch = 1; epoch <= MAX_EPOCH; epoch++) {
        for (size_t k = 0; k < k_sagelayers->k_layers; k++) {
            sageconv(k_sagelayers->sagelayer[k], g);
            // MAT_PRINT(sagelayer->output);
            relu(k_sagelayers->relulayer[k]);
            // MAT_PRINT(relulayer->output);
            normalize(k_sagelayers->normalizelayer[k]);
            // MAT_PRINT(normalizelayer->output);
        }

        linear(linearlayer);
        // MAT_PRINT(linearlayer->output);

        logsoft(logsoftlayer);
        // MAT_PRINT(logsoftlayer->output);

        double loss = nll_loss(logsoftlayer->output, g->y) / BATCH_DIM(g->y);
        printf("[epoch %zu] loss: %f\n", epoch, loss);

        cross_entropy_backward(logsoftlayer, g->y);
        linear_backward(linearlayer);
        for (size_t k = k_sagelayers->k_layers-1; k < k_sagelayers->k_layers; k--) {
            normalize_backward(k_sagelayers->normalizelayer[k]);
            relu_backward(k_sagelayers->relulayer[k]);
            sageconv_backward(k_sagelayers->sagelayer[k], g);
        }

        update_linear_weights(linearlayer, lr);
        for (size_t k = k_sagelayers->k_layers-1; k < k_sagelayers->k_layers; k--) {
            update_sageconv_weights(k_sagelayers->sagelayer[k], lr);
        }

    }

    destroy_logsoft_layer(logsoftlayer);
    destroy_linear_layer(linearlayer);
    destroy_k_sage_layers(k_sagelayers);
    destroy_graph(g);

    return 0;
}

// TODO: Use CRS format for edges
// TODO: Split up dataset according to DATASET_PATH/split/time/{test.csv.gz,train.csv.gz,valid.csv.gz} which are indexes
// TODO: Xavier Initialization for weight matrices
// TODO: Make a global memory pool that for internal use
