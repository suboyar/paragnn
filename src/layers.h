#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "core.h"
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
    Real **input_ptr;
    Real **output_ptr;
    Real **grad_input_ptr;
    Real **grad_output_ptr;
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
    Real *input;            // Points to previous layer's output
    Real *output;
    Real *agg;
    Real *Wagg, *Wroot;
    Real *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Real *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Real *grad_Wagg, *grad_Wroot;
    double *mean_scale;       // Scaling factors for mean aggregation (1/neighbor_count)

    const char *timer_dWroot;
    const char *timer_dWagg;
    const char *timer_dinput;
    const char *timer_dneigh;
} SageLayer;

typedef struct {
    uint32_t num_nodes;
    size_t dim;
    Real *input;            // Points to previous layer's output
    Real *output;
    Real *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Real *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
} ReluLayer;

typedef struct {
    uint32_t num_nodes;
    size_t dim;
    Real *input;            // Points to previous layer's output
    Real *output;
    Real *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Real *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Real *recip_mag;        // 1/||x||_2
} L2NormLayer;

typedef struct {
    uint32_t num_nodes;
    size_t in_dim;
    size_t out_dim;
    Real *input;            // Points to previous layer's output
    Real *output;
    Real *W;
    Real *bias;
    Real *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Real *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Real *grad_W;
    Real *grad_bias;
} LinearLayer;

typedef struct {
    uint32_t num_nodes;
    uint32_t num_classes;
    size_t dim;
    Real *input;            // Points to previous layer's output
    Real *output;
    // We use cross-entropy derivative since we are be using (LogSoftmax+NLLLoss)
    Real *grad_input;
} LogSoftLayer;

#define SAGE_NET_CREATE(conf, d) sage_net_create((conf), sizeof(conf)/sizeof(conf[0]), (d))
SageNet* sage_net_create(LayerConf *conf, size_t count, Dataset *ds);
SageLayer* sage_layer_create(uint32_t num_nodes, uint32_t num_edges, Edges edges, size_t in_dim, size_t out_dim);
ReluLayer* relu_layer_create(uint32_t num_nodes, size_t dim);
L2NormLayer* l2norm_layer_create(uint32_t num_nodes, size_t dim);
LinearLayer* linear_layer_create(uint32_t num_nodes, size_t in_dim, size_t out_dim);
LogSoftLayer* logsoft_layer_create(uint32_t num_nodes, uint32_t num_classes, size_t dim);

void sage_net_bind(SageNet *net, Dataset *ds, bool no_grad);
void sage_layer_bind(SageLayer *layer, uint32_t num_nodes, uint32_t num_edges, Edges edges, bool no_grad);
void relu_layer_bind(ReluLayer *layer, uint32_t num_nodes, bool no_grad);
void l2norm_layer_bind(L2NormLayer *layer, uint32_t num_nodes, bool no_grad);
void linear_layer_bind(LinearLayer *layer, uint32_t num_nodes, bool no_grad);
void logsoft_layer_bind(LogSoftLayer *layer, uint32_t num_nodes, bool no_grad);

void sage_net_free(SageNet **net);
void sage_layer_free(SageLayer **l);
void relu_layer_free(ReluLayer **l);
void l2norm_layer_free(L2NormLayer **l);
void linear_layer_free(LinearLayer **l);
void logsoft_layer_free(LogSoftLayer **l);

void sage_net_info(const SageNet *net);

#endif // LAYERS_H
