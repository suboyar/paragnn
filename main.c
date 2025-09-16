#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <zlib.h>

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
    load_data(&g);

    // printf("After loading dataset:\n");
    // print_memory_usage();

    size_t input_dim = g.num_node_features;  // 128
    size_t output_dim = g.num_label_classes; // 40
    size_t hidden_layer_size = 5;
    K = 1;

    // Weights will be transposed when feed through linear transformation, hence
    // the reverse shape
    matrix_t* W1l = mat_create(hidden_layer_size, g.num_node_features); // For center vertex
    matrix_t* W1r = mat_create(hidden_layer_size, g.num_node_features); // For aggregated vertices
    mat_rand(W1l, -1.0, 1.0);
    mat_rand(W1r, -1.0, 1.0);
    matrix_t* W2 = mat_create(g.num_label_classes, hidden_layer_size);
    mat_rand(W2, -1.0, 1.0);

    // printf("After initializing only weights:\n");
    // print_memory_usage();



    matrix_t* x = g.x;
    matrix_t* y = g.y;
    matrix_t* bias = NULL;
    matrix_t* agg = mat_create(g.num_nodes, g.num_node_features);
    matrix_t* h1 = mat_create(g.num_nodes, hidden_layer_size);
    matrix_t* h1_relu = mat_create(g.num_nodes, hidden_layer_size);
    matrix_t* h1_l2 = mat_create(g.num_nodes, hidden_layer_size);
    matrix_t* logits = mat_create(g.num_nodes, g.num_label_classes);
    matrix_t* yhat = mat_create(g.num_nodes, g.num_label_classes);

    matrix_t* grad_logits = mat_create(g.num_nodes, g.num_label_classes);
    matrix_t* grad_W2 = mat_create(g.num_label_classes, hidden_layer_size); // grad_out
    matrix_t* grad_bias = grad_logits; // dC/dBias = dC/dLogits
    matrix_t* grad_h1 = mat_create(g.num_nodes, hidden_layer_size);
    matrix_t* grad_h1_relu = mat_create(g.num_nodes, hidden_layer_size);
    matrix_t* grad_h1_l2 = mat_create(g.num_nodes, hidden_layer_size);
    // This is the transposed shape of the weight matrices
    matrix_t* grad_W1l = mat_create(g.num_node_features, hidden_layer_size);
    matrix_t* grad_W1r = mat_create(g.num_node_features, hidden_layer_size);

    // printf("After initializing matrices:\n");
    // print_memory_usage();

    size_t max_epoch = 100;
    for (size_t epoch = 1; epoch <= max_epoch; epoch++) {
        sage_conv(x, W1l, W1r, agg, h1, &g);
        relu(h1, h1_relu);
        l2_normalization(h1_relu, h1_l2, &g);

        linear_layer(h1_l2, W2, bias, logits);
        log_softmax(logits, yhat);

        double loss = nll_loss(yhat, y);
        printf("Loss: %f\n", loss);

        cross_entropy_backward(grad_logits, yhat, y);
        linear_weight_backward(grad_logits, h1_l2, grad_W2);

        linear_h_backward(grad_logits, W2, grad_h1);
        l2_normalization_backward(grad_h1, h1_relu, h1_l2, grad_h1_l2);
        relu_backward(grad_h1_l2, h1, grad_h1_relu);
        sage_conv_backward(grad_h1_relu, h1_relu, x, agg, grad_W1l, grad_W1r, &g);
        // break;

        update_weights(W2, grad_W2, g.num_nodes);
        update_sage_weights(W1l, grad_W1l, g.num_nodes);
        update_sage_weights(W1r, grad_W1r, g.num_nodes);

        reset_grad(grad_logits);
        reset_grad(grad_W2);
        reset_grad(grad_bias);
        reset_grad(grad_h1);
        reset_grad(grad_h1_relu);
        reset_grad(grad_h1_l2);
        reset_grad(grad_W1l);
        reset_grad(grad_W1r);
    }

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
