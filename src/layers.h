#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>

#include "core.h"
#include "ds.h"
#include "sparsegraph.h"
#include "flow.h"

typedef enum {
    LAYER_SAGE,
    LAYER_RELU,
    LAYER_L2NORM,
    LAYER_LOGSOFTMAX,
} LayerType;

typedef struct {
    LayerType type;
    int64_t in_dim;
    int64_t out_dim;  // ignored for relu, l2norm, logsoft
} LayerConf;

#define SAGE(in, out)   (LayerConf){ LAYER_SAGE,        (in), (out) }
#define RELU(dim)       (LayerConf){ LAYER_RELU,       (dim),   0   }
#define L2NORM(dim)     (LayerConf){ LAYER_L2NORM,     (dim),   0   }
#define LOGSOFTMAX(dim) (LayerConf){ LAYER_LOGSOFTMAX, (dim),   0   }

typedef struct Layer {
    LayerType type;
    void *ctx;              // Points to SageLayer, ReluLayer, etc.
} Layer;

typedef struct {
    Layer *layers;
    int64_t layer_count;
} SageNet;

typedef struct {
    int64_t        node_count;
    int64_t        edge_count;
    SparseGraph   *graph;
    int64_t        in_dim;
    int64_t        out_dim;
    FlowDirection  flow;
    Real          *x_in;        // Incoming node features (from upstream)
    Real          *x_out;       // Outgoing node features (from downstream)
    Real          *dx_in;       // Gradients w.r.t x_in (from upstream)
    Real          *dx_out;      // Gradients w.r.t x_out (from downstream)
    Real          *x_neigh;
    Real          *W_neigh;
    Real          *W_self;
    Real          *dW_neigh;
    Real          *dW_self;
    Real          *dx_scatter;
    // Physical row stride for gradW matrices (padded to N_VEC for non-temporal stores)
    int64_t        W_stride;
} SageLayer;

typedef struct {
    int64_t  node_count;
    int64_t  dim;
    Real    *x;                 // Forward buffer (mutated in-place)
    Real    *dx;                // Backward buffer (mutated in-place);
} ReluLayer;

typedef struct {
    int64_t  node_count;
    int64_t  dim;
    Real    *x;                 // Forward buffer (mutated in-place)
    Real    *dx;                // Backward buffer (mutated in-place);
    Real    *inv_norm;          // 1/||x||_2
} L2NormLayer;

 typedef struct {
    int64_t  node_count;
    int64_t  dim;               // Should be number of classes
    Real    *x;                 // Forward buffer (mutated in-place)
    Real    *dx;                // Backward buffer (mutated in-place);
} LogSoftmaxLayer;

SageNet* sage_net_alloc(LayerConf *conf, int64_t count, Dataset *ds, FlowDirection flow);
#define SAGE_NET_ALLOC(conf, d, flow) sage_net_alloc((conf), sizeof(conf)/sizeof(conf[0]), (d), (flow))
SageLayer* sage_layer_alloc(int64_t node_count, int64_t edge_count, SparseGraph *graph, int64_t in_dim, int64_t out_dim, FlowDirection flow);
ReluLayer* relu_layer_alloc(int64_t node_count, int64_t dim);
L2NormLayer* l2norm_layer_alloc(int64_t node_count, int64_t dim);
LogSoftmaxLayer* logsoft_layer_alloc(int64_t node_count, int64_t dim);

void sage_net_reset_parameters(SageNet *net);
void sage_layer_reset_parameters(SageLayer *l);
void sage_layer_reset_gradient(SageLayer *l);

void sage_net_bind(SageNet *net, Dataset *ds);
void sage_layer_bind(SageLayer *l, int64_t node_count, int64_t edge_count, SparseGraph *graph);
void relu_layer_bind(ReluLayer *l, int64_t node_count);
void l2norm_layer_bind(L2NormLayer *l, int64_t node_count);
void logsoft_layer_bind(LogSoftmaxLayer *l, int64_t node_count);

void sage_net_free(SageNet **net);
void sage_layer_free(SageLayer **l);
void relu_layer_free(ReluLayer **l);
void l2norm_layer_free(L2NormLayer **l);
void logsoft_layer_free(LogSoftmaxLayer **l);

void sage_net_info(const SageNet *net);

#endif // LAYERS_H
