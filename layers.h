#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "matrix.h"

#define CONNECT_LAYER(l1, l2) do {              \
        (l2)->input      = (l1)->output;        \
        (l1)->grad_input = (l2)->grad_output;   \
    } while(0);

typedef struct {
    matrix_t *input;            // Points to previous layer's output
    matrix_t *output;
    matrix_t *agg;
    matrix_t *Wagg, *Wroot;
    matrix_t *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    matrix_t *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    matrix_t *grad_Wagg, *grad_Wroot;
    size_t sample_size;
    size_t agg_size;
} SageLayer;

typedef struct {
    matrix_t *input;            // Points to previous layer's output
    matrix_t *output;
    matrix_t *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    matrix_t *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
} ReluLayer;

typedef struct {
    matrix_t *input;            // Points to previous layer's output
    matrix_t *output;
    matrix_t *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    matrix_t *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
} L2NormLayer;

typedef struct {
    matrix_t *input;            // Points to previous layer's output
    matrix_t *output;
    matrix_t *W;
    matrix_t *bias;
    matrix_t *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    matrix_t *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    matrix_t *grad_W;
    matrix_t *grad_bias;
} LinearLayer;

typedef struct {
    matrix_t *input;            // Points to previous layer's output
    matrix_t *output;
    // We use cross-entropy derivative since we we'll be using (LogSoftmax+NLLLoss)
    matrix_t *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
} LogSoftLayer;


// Init helpers
SageLayer* init_sage_layer(size_t n_nodes, size_t in_dim, size_t out_dim);
ReluLayer* init_relu_layer(size_t n_nodes, size_t dim);
L2NormLayer* init_l2norm_layer(size_t n_nodes, size_t dim);
LinearLayer* init_linear_layer(size_t n_nodes, size_t in_dim, size_t out_dim);
LogSoftLayer* init_logsoft_layer(size_t n_nodes, size_t out_dim);

// Inspect helpers

// TODO: Finish this info print
#define SAGE_LAYER_INFO(l) do {                                         \
        printf("\nSAGE LAYER\n");                                       \
        printf("========================================\n");           \
        printf("output = Wroot * input + Wagg  * agg\n");               \
        printf("%-6s = %-5s * %-5s + %-4s * %s\n",                      \
               mat_shape((l)->output), mat_shape((l)->Wroot),           \
               mat_shape((l)->input), mat_shape((l)->Wagg),             \
               mat_shape((l)->agg));                                    \
        printf("----------------------------------------\n");           \
        printf("grad_out = output\n");                                  \
        printf("%-6s = %-5s \n",                                        \
            mat_shape((l)->grad_output), mat_shape((l)->output));       \
        printf("grad_input = input\n");                                 \
        printf("%-6s = %-5s \n",                                        \
            mat_shape((l)->grad_input), mat_shape((l)->input));         \
        printf("grad_agg = agg\n");                                     \
        printf("%-6s = %-5s \n",                                        \
               mat_shape((l)->grad_Wagg), mat_shape((l)->agg));          \
        printf("========================================\n");           \
    } while(0);


#endif // LAYERS_H
