#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "core.h"
#include "dataset.h"
#include "edges.h"
#include "flow.h"

typedef enum {
    LAYER_SAGE,
    LAYER_RELU,
    LAYER_L2NORM,
    LAYER_LOGSOFTMAX,
    LAYER_LINEAR,
} LayerType;

typedef struct {
    LayerType type;
    int64_t in_dim;
    int64_t out_dim;  // ignored for relu, l2norm, logsoft
} LayerConf;

#define SAGE(in, out)   (LayerConf){ LAYER_SAGE,        (in), (out) }
#define RELU(dim)       (LayerConf){ LAYER_RELU,       (dim),   0   }
#define L2NORM(dim)     (LayerConf){ LAYER_L2NORM,     (dim),   0   }
#define LINEAR(in, out) (LayerConf){ LAYER_LINEAR,      (in), (out) }
#define LOGSOFTMAX(dim) (LayerConf){ LAYER_LOGSOFTMAX, (dim),   0   }

typedef struct Layer {
    LayerType type;
    void *ctx;              // Points to SageLayer, ReluLayer, etc.
} Layer;

typedef struct {
    Layer *layers;
    int64_t num_layers;
} SageNet;

typedef struct {
    int64_t        num_nodes;
    int64_t        num_edges;
    Edges          edges;
    int64_t        in_dim;
    int64_t        out_dim;
    FlowDirection  flow;
    Real          *input;            // Points to previous layer's output
    Real          *output;
    Real          *agg;
    Real          *Wagg, *Wroot;
    Real          *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Real          *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Real          *grad_Wagg, *grad_Wroot;

    // scratch buffer that layers can share between them so only they only needs to be allocated once
    Real     *tls_dW; // layout: [dWroot_row0][dWagg_row0][dWroot_row1][dWagg_row1]...
    Real     *grad_scatter;
} SageLayer;

typedef struct {
    int64_t num_nodes;
    int64_t dim;
    Real *input;            // Points to previous layer's output
    Real *output;
    Real *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Real *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
} ReluLayer;

typedef struct {
    int64_t num_nodes;
    int64_t dim;
    Real *input;            // Points to previous layer's output
    Real *output;
    Real *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Real *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Real *recip_mag;        // 1/||x||_2
} L2NormLayer;

typedef struct {
    int64_t num_nodes;
    int64_t in_dim;
    int64_t out_dim;
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
    int64_t num_nodes;
    int64_t dim; // if last layer, should be number of classes
    Real *input;            // Points to previous layer's output
    Real *output;
    // We use cross-entropy derivative since we are be using (LogSoftmax+NLLLoss)
    Real *grad_input;
} LogSoftmaxLayer;

#define SAGE_NET_CREATE(conf, d, flow) sage_net_create((conf), sizeof(conf)/sizeof(conf[0]), (d), (flow))
SageNet* sage_net_create(LayerConf *conf, int64_t count, Dataset *ds, FlowDirection flow);
SageLayer* sage_layer_create(int64_t num_nodes, int64_t num_edges, Edges edges, int64_t in_dim, int64_t out_dim, FlowDirection flow);
ReluLayer* relu_layer_create(int64_t num_nodes, int64_t dim);
L2NormLayer* l2norm_layer_create(int64_t num_nodes, int64_t dim);
LinearLayer* linear_layer_create(int64_t num_nodes, int64_t in_dim, int64_t out_dim);
LogSoftmaxLayer* logsoft_layer_create(int64_t num_nodes, int64_t dim);

void sage_net_bind(SageNet *net, Dataset *ds);
void sage_layer_bind(SageLayer *l, int64_t num_nodes, int64_t num_edges, Edges edges);
void relu_layer_bind(ReluLayer *l, int64_t num_nodes);
void l2norm_layer_bind(L2NormLayer *l, int64_t num_nodes);
void linear_layer_bind(LinearLayer *l, int64_t num_nodes);
void logsoft_layer_bind(LogSoftmaxLayer *l, int64_t num_nodes);

void sage_net_free(SageNet **net);
void sage_layer_free(SageLayer **l);
void relu_layer_free(ReluLayer **l);
void l2norm_layer_free(L2NormLayer **l);
void linear_layer_free(LinearLayer **l);
void logsoft_layer_free(LogSoftmaxLayer **l);

void sage_net_info(const SageNet *net);

#endif // LAYERS_H
