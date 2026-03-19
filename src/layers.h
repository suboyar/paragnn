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

#define SAGE_NET_CREATE(conf, d) sage_net_create((conf), sizeof(conf)/sizeof(conf[0]), (d))

typedef struct Layer Layer;
struct Layer {
    LayerType type;
    void *impl;              // Points to SageLayer, ReluLayer, etc.

    Matrix **input_ptr;
    Matrix **output_ptr;
    Matrix **grad_input_ptr;
    Matrix **grad_output_ptr;

    // vtable
    void (*forward)(Layer *self, Slice slice);
    void (*backward)(Layer *self, Slice slice);
    void (*update)(Layer *self, double lr);   // NULL for layers with no weights
    void (*zero_grad)(Layer *self);           // NULL for layers with no weights
    void (*reset)(Layer *self);
    void (*destroy)(Layer *self);
};

typedef struct {
    Dataset *data;
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
    Dataset *data;
    size_t dim;
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
} ReluLayer;

typedef struct {
    Dataset *data;
    size_t dim;
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    Matrix *grad_output;      // Gradients w.r.t. this layer's output (from downstream)
    Matrix *grad_input;       // Gradients w.r.t. this layer's input (to upstream)
    Matrix *recip_mag;        // 1/||x||_2
} L2NormLayer;

typedef struct {
    Dataset *data;
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
    Dataset *data;
    size_t dim;
    Matrix *input;            // Points to previous layer's output
    Matrix *output;
    // We use cross-entropy derivative since we are be using (LogSoftmax+NLLLoss)
    Matrix *grad_input;
} LogSoftLayer;

typedef struct {
    Layer *layers;
    size_t num_layers;
} SageNet;

SageLayer* sage_layer_create(size_t batch_size, size_t in_dim, size_t out_dim, Dataset *data);
ReluLayer* relu_layer_create(size_t batch_size, size_t dim, Dataset *data);
L2NormLayer* l2norm_layer_create(size_t batch_size, size_t dim, Dataset *data);
LinearLayer* linear_layer_create(size_t batch_size, size_t in_dim, size_t out_dim, Dataset *data);
LogSoftLayer* logsoft_layer_create(size_t batch_size, size_t dim, Dataset *data);
SageNet* sage_net_create(LayerConf *conf, size_t count, Dataset *data);

void sage_net_reset();

void sage_layer_destroy(SageLayer* l);
void relu_layer_destroy(ReluLayer* l);
void l2norm_layer_destroy(L2NormLayer* l);
void linear_layer_destroy(LinearLayer *l);
void logsoft_layer_destroy(LogSoftLayer *l);
void sage_net_destroy(SageNet *n);

void sage_net_info(const SageNet *net);

#endif // LAYERS_H
