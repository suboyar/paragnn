#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "matrix.h"
#include "graph.h"

#define CONNECT_LAYER(l1, l2) do {              \
        (l2)->input       = (l1)->output;       \
        (l1)->grad_output = (l2)->grad_input;   \
    } while(0);

#define CONNECT_SAGE_LAYERS(K_SAGE_LAYERS) do {                                                   \
        for (size_t k = 0; k < (K_SAGE_LAYERS)->k_layers; k++) {                                  \
            CONNECT_LAYER((K_SAGE_LAYERS)->sagelayer[k], (K_SAGE_LAYERS)->relulayer[k]);          \
            CONNECT_LAYER((K_SAGE_LAYERS)->relulayer[k], (K_SAGE_LAYERS)->normalizelayer[k]);     \
            if (k < (K_SAGE_LAYERS)->k_layers - 1) {                                              \
                CONNECT_LAYER((K_SAGE_LAYERS)->normalizelayer[k], (K_SAGE_LAYERS)->sagelayer[k+1]); \
            }                                                                                     \
        }                                                                                         \
    } while(0);

#define LAST_SAGE_LAYER(K_SAGE_LAYERS) (K_SAGE_LAYERS)->normalizelayer[(K_SAGE_LAYERS)->k_layers-1]


// TODO: add references to graph struct
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
    matrix_t *recip_mag;        // 1/||x||_2
} NormalizeLayer;

typedef struct {
    SageLayer      **sagelayer;
    ReluLayer      **relulayer;
    NormalizeLayer **normalizelayer;
    size_t k_layers;
} K_SageLayers;

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
K_SageLayers* init_k_sage_layers(size_t k_layers, size_t hidden_dim, graph_t *g);
SageLayer* init_sage_layer(size_t n_nodes, size_t in_dim, size_t out_dim);
ReluLayer* init_relu_layer(size_t n_nodes, size_t dim);
NormalizeLayer* init_l2norm_layer(size_t n_nodes, size_t dim);
LinearLayer* init_linear_layer(size_t n_nodes, size_t in_dim, size_t out_dim);
LogSoftLayer* init_logsoft_layer(size_t n_nodes, size_t out_dim);

// Free up layers
void free_k_sage_layers(K_SageLayers *k_sagelayers);
void free_sage_layer(SageLayer* l);
void free_relu_layer(ReluLayer* l);
void free_l2norm_layer(NormalizeLayer* l);
void free_linear_layer(LinearLayer *linearlayer);
void free_logsoft_layer(LogSoftLayer *logsoftlayer);

// Inspect helpers
void sage_layer_info(const SageLayer* const l);
void relu_layer_info(const ReluLayer* const l);
void normalize_layer_info(const NormalizeLayer* const l);
void k_sage_layers_info(const K_SageLayers* const l);
void linear_layer_info(const LinearLayer* const l);
void logsoft_layer_info(const LogSoftLayer* const l);




#endif // LAYERS_H
