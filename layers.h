#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "matrix.h"

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
    matrix_t *recip_mag;        // 1/||x||_2
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
void sage_layer_info(const SageLayer* const l);
void relu_layer_info(const ReluLayer* const l);
void normalize_layer_info(const NormalizeLayer* const l);
void linear_layer_info(const LinearLayer* const l);
void logsoft_layer_info(const LogSoftLayer* const l);




#endif // LAYERS_H
