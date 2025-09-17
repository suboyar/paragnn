#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "matrix.h"


#ifdef ROW_MAJOR
    #define BATCH_DIM(m)     ((m)->height)
    #define NODE_DIM(m)      ((m)->width)
#else
    #define BATCH_DIM(m)     ((m)->width)
    #define NODE_DIM(m)      ((m)->height)
#endif

#define CONNECT_LAYER(l1, l2) do {              \
        (l2)->input       = (l1)->output;       \
        (l1)->grad_output = (l2)->grad_input;   \
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
} NormalizeLayer;

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
    matrix_t *grad_input;
    matrix_t *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
} LogSoftLayer;


// Init helpers
SageLayer* init_sage_layer(size_t n_nodes, size_t in_dim, size_t out_dim);
ReluLayer* init_relu_layer(size_t n_nodes, size_t dim);
NormalizeLayer* init_l2norm_layer(size_t n_nodes, size_t dim);
LinearLayer* init_linear_layer(size_t n_nodes, size_t in_dim, size_t out_dim);
LogSoftLayer* init_logsoft_layer(size_t n_nodes, size_t out_dim);

// Inspect helpers

#define SAGE_LAYER_INFO(l) do {                                         \
        printf("\nSAGE LAYER\n");                                       \
        printf("========================================\n");           \
        printf("output = Wroot * input + Wagg  * agg\n");               \
        printf("%-6s = %-5s * %-5s + %-4s * %s\n",                      \
               mat_shape((l)->output), mat_shape((l)->Wroot),           \
               mat_shape((l)->input), mat_shape((l)->Wagg),             \
               mat_shape((l)->agg));                                    \
        printf("----------------------------------------\n");           \
        printf("grad_output = output\n");                               \
        printf("   %-8s = %s\n",                                        \
            mat_shape((l)->grad_output), mat_shape((l)->output));       \
        printf("grad_input  = input\n");                                \
        printf("   %-7s  = %s\n",                                       \
               mat_shape((l)->grad_input), mat_shape((l)->input));      \
        printf("grad_Wagg   = Wagg\n");                                 \
        printf("   %-6s   = %s\n",                                      \
               mat_shape((l)->grad_Wagg), mat_shape((l)->Wagg));        \
        printf("grad_Wroot  = Wroot\n");                                \
        printf("   %-7s  = %s\n",                                       \
               mat_shape((l)->grad_Wroot), mat_shape((l)->Wroot));      \
        printf("========================================\n");           \
    } while(0);

#define RELU_LAYER_INFO(l) do {                                         \
        printf("\nRELU LAYER\n");                                       \
        printf("========================================\n");           \
        printf("output = relu(input)\n");                               \
        printf("%-6s = relu(%-5s)\n",                                   \
               mat_shape((l)->output), mat_shape((l)->input));          \
        printf("----------------------------------------\n");           \
        printf("grad_output = output\n");                               \
        printf("   %-8s = %s\n",                                        \
               mat_shape((l)->grad_output), mat_shape((l)->output));    \
        printf("grad_input  = input\n");                                \
        printf("   %-7s  = %s\n",                                       \
               mat_shape((l)->grad_input), mat_shape((l)->input));      \
        printf("========================================\n");           \
    } while(0)

#define NORMALIZE_LAYER_INFO(l) do {                                    \
        printf("\nNORMALIZE LAYER\n");                                  \
        printf("========================================\n");           \
        printf("output = l2norm(input)\n");                             \
        printf("%-6s = l2norm(%-5s)\n",                                 \
               mat_shape((l)->output), mat_shape((l)->input));          \
        printf("----------------------------------------\n");           \
        printf("grad_output = output\n");                               \
        printf("   %-8s = %s\n",                                        \
               mat_shape((l)->grad_output), mat_shape((l)->output));    \
        printf("grad_input  = input\n");                                \
        printf("   %-7s  = %s\n",                                       \
               mat_shape((l)->grad_input), mat_shape((l)->input));      \
        printf("========================================\n");           \
    } while(0)

#define LINEAR_LAYER_INFO(l) do {                                       \
        printf("\nLINEAR LAYER\n");                                     \
        printf("========================================\n");           \
        printf("output =   W   * input + bias\n");                          \
        printf("%-6s = %-5s * %-5s + %s\n",                             \
               mat_shape((l)->output), mat_shape((l)->W),               \
               mat_shape((l)->input), mat_shape((l)->bias));            \
        printf("----------------------------------------\n");           \
        printf("grad_output = output\n");                               \
        printf("   %-8s = %s\n",                                        \
               mat_shape((l)->grad_output), mat_shape((l)->output));    \
        printf("grad_input  = input\n");                                \
        printf("   %-7s  = %s\n",                                       \
               mat_shape((l)->grad_input), mat_shape((l)->input));      \
        printf("grad_W      =   W\n");                                    \
        printf("   %-6s   = %s\n",                                      \
               mat_shape((l)->grad_W), mat_shape((l)->W));              \
        printf("grad_bias   = bias\n");                                 \
        printf("   %-6s   = %s\n",                                      \
               mat_shape((l)->grad_bias), mat_shape((l)->bias));        \
        printf("========================================\n");           \
    } while(0)

#define LOGSOFT_LAYER_INFO(l) do {                                      \
        printf("\nLOGSOFT LAYER\n");                                    \
        printf("========================================\n");           \
        printf("output = log_softmax(input)\n");                        \
        printf("%-6s = log_softmax(%-5s)\n",                            \
               mat_shape((l)->output), mat_shape((l)->input));          \
        printf("----------------------------------------\n");           \
        printf("grad_input  = input\n");                               \
        printf("   %-8s = %s\n",                                        \
               mat_shape((l)->grad_input), mat_shape((l)->input));    \
        printf("grad_output = output\n");                               \
        printf("   %-8s = %s\n",                                        \
               mat_shape((l)->grad_output), mat_shape((l)->output));    \
        printf("========================================\n");           \
    } while(0)

#endif // LAYERS_H
