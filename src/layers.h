#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "matrix.h"
#include "dataset.h"

typedef enum {
    LAYER_SAGE,
    LAYER_RELU,
    LAYER_L2NORM,
    LAYER_LOGSOFT,
    LAYER_LINEAR,
} LayerType;

typedef struct {
    LayerType type;
    size_t in_dim;
    size_t out_dim;  // ignored for relu, l2norm, logsoft
} LayerConf;

#define SAGE(in, out)   (LayerConf){ LAYER_SAGE,    (in), (out) }
#define RELU(dim)       (LayerConf){ LAYER_RELU,    (dim), 0    }
#define L2NORM(dim)     (LayerConf){ LAYER_L2NORM,  (dim), 0    }
#define LINEAR(in, out) (LayerConf){ LAYER_LINEAR,  (in), (out) }
#define LOGSOFT(dim)    (LayerConf){ LAYER_LOGSOFT,  (dim), 0    }

typedef struct Layer {
    LayerType type;
    void *ctx;              // Points to SageLayer, ReluLayer, etc.

    // Used  when connecting layers to each other
    Matrix **input_ptr;
    Matrix **output_ptr;
    Matrix **grad_input_ptr;
    Matrix **grad_output_ptr;
} Layer;

typedef struct {
    Layer *layers;
    size_t num_layers;
} SageNet;

typedef struct {
    uint32_t num_nodes;
    uint32_t num_edges;
    Edges edges;
    size_t in_dim;
    size_t out_dim;
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *agg;
    Matrix *Wagg, *Wroot;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Matrix *grad_Wagg, *grad_Wroot;
    double *mean_scale;       // Scaling factors for mean aggregation (1/neighbor_count)

    const char *timer_dWroot;
    const char *timer_dWagg;
    const char *timer_dinput;
    const char *timer_dneigh;
} SageLayer;

typedef struct {
    uint32_t num_nodes;
    size_t dim;
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
} ReluLayer;

typedef struct {
    uint32_t num_nodes;
    size_t dim;
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Matrix *recip_mag;        // 1/||x||_2
} L2NormLayer;

typedef struct {
    uint32_t num_nodes;
    size_t in_dim;
    size_t out_dim;
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
    uint32_t num_nodes;
    uint32_t num_classes;
    size_t dim;
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    // We use cross-entropy derivative since we are be using (LogSoftmax+NLLLoss)
    Matrix *grad_input;
} LogSoftLayer;

#define SAGE_NET_CREATE(conf, d) sage_net_create((conf), sizeof(conf)/sizeof(conf[0]), (d))
SageNet* sage_net_create(LayerConf *conf, size_t count, Dataset *ds);
SageLayer* sage_layer_create(uint32_t num_nodes, uint32_t num_edges, Edges edges, size_t in_dim, size_t out_dim);
ReluLayer* relu_layer_create(uint32_t num_nodes, size_t dim);
L2NormLayer* l2norm_layer_create(uint32_t num_nodes, size_t dim);
LinearLayer* linear_layer_create(uint32_t num_nodes, size_t in_dim, size_t out_dim);
LogSoftLayer* logsoft_layer_create(uint32_t num_nodes, uint32_t num_classes, size_t dim);

void sage_net_reset(const SageNet *net, Dataset *ds);
void sage_layer_reset(const SageLayer *l, Dataset *ds);
void relu_layer_reset(const ReluLayer *l);
void l2norm_layer_reset(const L2NormLayer *l);
void linear_layer_reset(const LinearLayer *l);
void logsoft_layer_reset(const LogSoftLayer *l);

void sage_net_free(SageNet *net);
void sage_layer_free(SageLayer* l);
void relu_layer_free(ReluLayer* l);
void l2norm_layer_free(L2NormLayer* l);
void linear_layer_free(LinearLayer *l);
void logsoft_layer_free(LogSoftLayer *l);

void sage_net_info(const SageNet *net);

#endif // LAYERS_H
