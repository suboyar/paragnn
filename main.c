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
#include "print.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

#define USE_SIMPLE_GRAPH 1

#if USE_SIMPLE_GRAPH
#include "simple_graph.h"
#define load_data load_simple_data
#else
#include "arxiv.h"
#define load_data load_arxiv_data
#endif

#define ERROR(fmt, ...) do { \
    fprintf(stderr, "%s:%d: ERROR: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    abort(); \
} while(0)

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
    // srand(time(NULL));

    nob_minimal_log_level = NOB_NO_LOGS;

    graph_t g = {0};
    // BREAKPOINT();
    load_data(&g);

    // printf("After loading dataset:\n");
    // print_memory_usage();

    size_t hidden_dim = 5;
    float lr = 0.1;

#ifdef NEWWAY

    MAT_PRINT(g.y);
    MAT_SPEC(g.y);

    size_t n_nodes = g.num_nodes;
    SageLayer *sagelayer = init_sage_layer(n_nodes, g.num_node_features, hidden_dim);
    ReluLayer *relulayer = init_relu_layer(n_nodes, hidden_dim);
    NormalizeLayer *normalizelayer = init_l2norm_layer(n_nodes, hidden_dim);

    LinearLayer *linearlayer = init_linear_layer(n_nodes, hidden_dim, g.num_label_classes);
    LogSoftLayer *logsoftlayer = init_logsoft_layer(n_nodes, g.num_label_classes);


    sagelayer->input = g.x;
    CONNECT_LAYER(sagelayer, relulayer);
    CONNECT_LAYER(relulayer, normalizelayer);
    CONNECT_LAYER(normalizelayer, linearlayer);
    CONNECT_LAYER(linearlayer, logsoftlayer);

    sage_layer_info(sagelayer);
    relu_layer_info(relulayer);
    normalize_layer_info(normalizelayer);
    linear_layer_info(linearlayer);
    logsoft_layer_info(logsoftlayer);
    printf("\n");

    const size_t MAX_EPOCH = 10;

    for (size_t epoch = 1; epoch <= MAX_EPOCH; epoch++) {
        // MAT_PRINT(sagelayer->Wagg);
        // MAT_PRINT(sagelayer->Wroot);
        // MAT_PRINT(linearlayer->W);

        // MAT_PRINT(g.x);
        sageconv(sagelayer, &g);
        // MAT_PRINT(sagelayer->output);

        relu(relulayer);
        // MAT_PRINT(relulayer->output);

        normalize(normalizelayer);
        // MAT_PRINT(normalizelayer->output);

        linear(linearlayer);
        // MAT_PRINT(linearlayer->output);

        logsoft(logsoftlayer);
        // MAT_PRINT(logsoftlayer->output);

        double loss = nll_loss(logsoftlayer->output, g.y) / BATCH_DIM(g.y);
        printf("[epoch %zu] loss: %f\n", epoch, loss);

        cross_entropy_backward(logsoftlayer, g.y);
        linear_backward(linearlayer);
        normalize_backward(normalizelayer);
        relu_backward(relulayer);
        sageconv_backward(sagelayer);

        update_linear_weights(linearlayer, lr);
        update_sageconv_weights(sagelayer, lr);

    }
#else

    size_t input_dim = g.num_node_features;  // 128
    size_t output_dim = g.num_label_classes; // 40

    K = 1;

    // Weights will be transposed when feed through linear transformation, hence
    // the reverse shape
    matrix_t* W1l = mat_create(hidden_dim, g.num_node_features); // For center vertex
    matrix_t* W1r = mat_create(hidden_dim, g.num_node_features); // For aggregated vertices
    mat_rand(W1l, -1.0, 1.0);
    mat_rand(W1r, -1.0, 1.0);
    matrix_t* W2 = mat_create(g.num_label_classes, hidden_dim);
    mat_rand(W2, -1.0, 1.0);

    matrix_t* x = g.x;
    matrix_t* y = g.y;
    matrix_t* bias = NULL;
    matrix_t* agg = mat_create(g.num_nodes, g.num_node_features);
    matrix_t* h1 = mat_create(g.num_nodes, hidden_dim);
    matrix_t* h1_relu = mat_create(g.num_nodes, hidden_dim);
    matrix_t* h1_l2 = mat_create(g.num_nodes, hidden_dim);
    matrix_t* logits = mat_create(g.num_nodes, g.num_label_classes);
    matrix_t* yhat = mat_create(g.num_nodes, g.num_label_classes);

    matrix_t* grad_logits = mat_create(g.num_nodes, g.num_label_classes);
    matrix_t* grad_W2 = mat_create(g.num_label_classes, hidden_dim); // grad_out
    matrix_t* grad_bias = grad_logits; // dC/dBias = dC/dLogits
    matrix_t* grad_h1 = mat_create(g.num_nodes, hidden_dim);
    matrix_t* grad_h1_relu = mat_create(g.num_nodes, hidden_dim);
    matrix_t* grad_h1_l2 = mat_create(g.num_nodes, hidden_dim);
    // This is the transposed shape of the weight matrices
    matrix_t* grad_W1l = mat_create(g.num_node_features, hidden_dim);
    matrix_t* grad_W1r = mat_create(g.num_node_features, hidden_dim);


    // printf("After initializing matrices:\n");
    // print_memory_usage();

    size_t max_epoch = 100;
    for (size_t epoch = 1; epoch <= max_epoch; epoch++) {
        MAT_PRINT(W1l);
        MAT_PRINT(W1r);
        MAT_PRINT(W2);

        MAT_PRINT(g.x);
        sage_conv(g.x, W1l, W1r, agg, h1, &g);
        MAT_PRINT(h1);

        relu(h1, h1_relu);
        MAT_PRINT(h1_relu);
        l2_normalization(h1_relu, h1_l2, &g);
        MAT_PRINT(h1_l2);

        linear_layer(h1_l2, W2, bias, logits);
        MAT_PRINT(logits);

        log_softmax(logits, yhat);
        MAT_PRINT(yhat);

        double loss = nll_loss(yhat, y);
        printf("Loss: %f\n", loss);

        break;

        cross_entropy_backward(grad_logits, yhat, y);
        linear_weight_backward(grad_logits, h1_l2, grad_W2);

        linear_h_backward(grad_logits, W2, grad_h1);
        l2_normalization_backward(grad_h1, h1_relu, h1_l2, grad_h1_l2);
        relu_backward(grad_h1_l2, h1, grad_h1_relu);
        sage_conv_backward(grad_h1_relu, h1_relu, x, agg, grad_W1l, grad_W1r, &g);


        update_weights(W2, grad_W2, g.num_nodes);
        update_sage_weights(W1l, grad_W1l, g.num_nodes);
        update_sage_weights(W1r, grad_W1r, g.num_nodes);



    }

#endif

#ifndef NEWWAY
    mat_destroy(W1l);
    mat_destroy(W1r);
    mat_destroy(W2);
    mat_destroy(h1);
    mat_destroy(logits);
    mat_destroy(yhat);
    mat_destroy(g.x);
    mat_destroy(g.y);
    free(g.node_year);
    free(g.edge_index);
#endif

    return 0;
}

// TODO: Implement gradient descent training
// TODO: Configurable layer dimensions
// TODO: Use CRS format for edges
// TODO: Clean up all allocated memory
// TODO: Add bias
// TODO: Split up dataset according to DATASET_PATH/split/time/{test.csv.gz,train.csv.gz,valid.csv.gz} which are indexes
// TODO: Xavier Initialization for weight matrices
// TODO: Make a global memory pool that for internal use
