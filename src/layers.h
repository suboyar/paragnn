#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "matrix.h"
#include "graph.h"

typedef struct {
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *agg;
    Matrix *Wagg, *Wroot;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Matrix *grad_Wagg, *grad_Wroot;
    double *mean_scale;       // Scaling factors for mean aggregation (1/neighbor_count)
} SageLayer;

typedef struct {
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
} ReluLayer;

typedef struct {
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Matrix *recip_mag;        // 1/||x||_2
} L2NormLayer;

typedef struct {
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *W;
    Matrix *bias;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Matrix *grad_W;
    Matrix *grad_bias;
} LinearLayer;

typedef struct {
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    // We use cross-entropy derivative since we are be using (LogSoftmax+NLLLoss)
    Matrix *grad_input;
} LogSoftLayer;

typedef struct {
    // Encoder layers
    SageLayer    **enc_sage;
    ReluLayer    **enc_relu;
    L2NormLayer  **enc_norm;
    size_t         enc_depth;
    // Final layer
    SageLayer     *cls_sage;
#ifdef USE_PREDICTION_HEAD
    // Prediction head
    LinearLayer   *linear;
#endif
    // Output layer
    LogSoftLayer  *logsoft;
    size_t         num_layers;
} SageNet;

SageLayer* sage_layer_create(size_t batch_size, size_t in_dim, size_t out_dim);
ReluLayer* relu_layer_create(size_t batch_size, size_t dim);
L2NormLayer* l2norm_layer_create(size_t batch_size, size_t dim);
LinearLayer* linear_layer_create(size_t batch_size, size_t in_dim, size_t out_dim);
LogSoftLayer* logsoft_layer_create(size_t batch_size, size_t dim);
SageNet* sage_net_create(size_t num_layers, size_t hidden_dim, graph_t *g);

void sage_layer_destroy(SageLayer* l);
void relu_layer_destroy(ReluLayer* l);
void l2norm_layer_destroy(L2NormLayer* l);
void linear_layer_destroy(LinearLayer *l);
void logsoft_layer_destroy(LogSoftLayer *l);
void sage_net_destroy(SageNet *n);

void linear_layer_update_weights(LinearLayer* const l, float lr);
void sage_layer_update_weights(SageLayer* const l, float lr);

void sage_layer_zero_gradients(SageLayer* l);
void relu_layer_zero_gradients(ReluLayer* l);
void normalize_layer_zero_gradients(L2NormLayer* l);
void linear_layer_zero_gradients(LinearLayer* l);
void logsoft_layer_zero_gradients(LogSoftLayer* l);
void sage_net_zero_gradients(SageNet* net);

void sage_net_info(const SageNet *net);

#endif // LAYERS_H
