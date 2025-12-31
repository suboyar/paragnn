#include <stdlib.h>

#include "core.h"
#include "layers.h"
#include "matrix.h"
#include "graph.h"
#include "timer.h"
#include "linalg/linalg.h"

#define PIPE(src, dst) do {              \
        (dst)->input       = (src)->output;       \
        (src)->grad_output = (dst)->grad_input;   \
    } while(0);

SageLayer* sage_layer_create(size_t batch_size, size_t in_dim, size_t out_dim)
{
    SageLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate SageLayer");

    *layer = (SageLayer){
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(batch_size, out_dim),
        .agg         = matrix_create(batch_size, in_dim),
        .Wagg        = matrix_create(in_dim, out_dim),
        .Wroot       = matrix_create(in_dim, out_dim),
        .grad_input  = matrix_create(batch_size, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_Wagg   = matrix_create(in_dim, out_dim),
        .grad_Wroot  = matrix_create(in_dim, out_dim),
    };
    layer->mean_scale = malloc(batch_size * sizeof(*layer->mean_scale));

    // Initialize weights randomly
    matrix_fill_xavier_uniform(layer->Wroot, in_dim, out_dim);
    matrix_fill_xavier_uniform(layer->Wagg, in_dim, out_dim);

    return layer;
}

ReluLayer* relu_layer_create(size_t batch_size, size_t dim)
{
    ReluLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate ReluLayer");

    *layer = (ReluLayer) {
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
    };

    return layer;
}

L2NormLayer* l2norm_layer_create(size_t batch_size, size_t dim)
{
    L2NormLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate L2NormLayer");

    *layer = (L2NormLayer) {
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
        .recip_mag   = matrix_create(batch_size, 1)
    };

    return layer;
}

LinearLayer* linear_layer_create(size_t batch_size, size_t in_dim, size_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LinearLayer");

    *layer = (LinearLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(batch_size, out_dim),
        .W           = matrix_create(in_dim, out_dim),
        .bias        = matrix_create(1, out_dim),
        .grad_input  = matrix_create(batch_size, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_W      = matrix_create(in_dim, out_dim),
        .grad_bias   = matrix_create(1, out_dim),
    };

    matrix_fill_xavier_uniform(layer->W, in_dim, out_dim);
    matrix_fill_xavier_uniform(layer->bias, 1, out_dim);

    return layer;
}


LogSoftLayer* logsoft_layer_create(size_t batch_size, size_t dim)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LogSoftLayer");

    *layer = (LogSoftLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
    };

    return layer;
}

SageNet* sage_net_create(size_t num_layers, size_t hidden_dim, graph_t *g)
{
    if (num_layers < 1) ERROR("Number of layers needed for SageNet is >=1");

    size_t num_classes = g->num_label_classes;
    size_t batch_size  = g->num_nodes;
    size_t in_features = g->num_node_features;

    SageNet *net = malloc(sizeof(*net));
    if (!net) ERROR("Could not allocate SageNet");

    net->num_layers = num_layers;
    net->enc_depth = num_layers - 1;
    if (net->enc_depth > 0) {
        net->enc_sage = malloc(sizeof(*net->enc_sage) * net->enc_depth);
        if (!net->enc_sage) ERROR("Could not allocate net->enc_sage");
        net->enc_relu = malloc(sizeof(*net->enc_relu) * net->enc_depth);
        if (!net->enc_relu) ERROR("Could not allocate net->enc_relu");
        net->enc_norm = malloc(sizeof(*net->enc_norm) * net->enc_depth);
        if (!net->enc_norm) ERROR("Could not allocate net->norm");
    } else {
        net->enc_sage = NULL;
        net->enc_relu = NULL;
        net->enc_norm = NULL;
    }

    for (size_t i = 0; i < net->enc_depth; i++) {
        size_t layer_in  = (i == 0) ? in_features : hidden_dim;
        size_t layer_out = hidden_dim;

        net->enc_sage[i] = sage_layer_create(batch_size, layer_in, layer_out);
        net->enc_relu[i] = relu_layer_create(batch_size, layer_out);
        net->enc_norm[i] = l2norm_layer_create(batch_size, layer_out);
    }

    size_t final_in = (num_layers == 1) ? in_features : hidden_dim;

#ifndef USE_PREDICTION_HEAD
    net->cls_sage = sage_layer_create(batch_size, final_in, num_classes);
#else
    net->cls_sage = sage_layer_create(batch_size, final_in, hidden_dim);
    LinearLayer *linearlayer = init_linear_layer(batch_size, hidden_dim, num_classes);
#endif

    net->logsoft = logsoft_layer_create(batch_size, num_classes);

    if (net->enc_depth > 0) {
        net->enc_sage[0]->input = g->x;
    } else {
        net->cls_sage->input = g->x;
    }

    // Connect each layer
    for (size_t i = 0; i < net->enc_depth; i++) {
        PIPE(net->enc_sage[i], net->enc_relu[i]);
        PIPE(net->enc_relu[i], net->enc_norm[i]);
        if (i < net->enc_depth - 1) PIPE(net->enc_norm[i], net->enc_sage[i+1]);
    }

    if (net->enc_depth > 0) PIPE(net->enc_norm[net->enc_depth - 1], net->cls_sage);

#ifndef USE_PREDICTION_HEAD
    PIPE(net->cls_sage, net->logsoft);
#else
    PIPE(net->cls_sage, net->linear);
    PIPE(net->linear, net->logsoft);
#endif

    return net;
}


// We don't free matrices that are references from other layers, i.e input and grad_output
void sage_layer_destroy(SageLayer* l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->agg) matrix_destroy(l->agg);
    if (l->Wagg) matrix_destroy(l->Wagg);
    if (l->Wroot) matrix_destroy(l->Wroot);
    if (l->grad_input) matrix_destroy(l->grad_input);
    if (l->grad_Wagg) matrix_destroy(l->grad_Wagg);
    if (l->grad_Wroot) matrix_destroy(l->grad_Wroot);
    if (l->mean_scale) free(l->mean_scale);

    free(l);
}

void relu_layer_destroy(ReluLayer* l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

void l2norm_layer_destroy(L2NormLayer* l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);
    if (l->recip_mag) matrix_destroy(l->recip_mag);

    free(l);
}

void linear_layer_destroy(LinearLayer *l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->W) matrix_destroy(l->W);
    if (l->bias) matrix_destroy(l->bias);
    if (l->grad_input) matrix_destroy(l->grad_input);
    if (l->grad_W) matrix_destroy(l->grad_W);
    if (l->grad_bias) matrix_destroy(l->grad_bias);

    free(l);
}

void logsoft_layer_destroy(LogSoftLayer *l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

void sage_net_destroy(SageNet *net)
{
    if (!net) return;

    if (net->enc_depth > 0) {
        for (size_t i = 0; i < net->enc_depth; i++) {
            l2norm_layer_destroy(net->enc_norm[i]);
            relu_layer_destroy(net->enc_relu[i]);
            sage_layer_destroy(net->enc_sage[i]);
        }

        if (net->enc_sage) free(net->enc_sage);
        if (net->enc_relu) free(net->enc_relu);
        if (net->enc_norm) free(net->enc_norm);
    }

    if (net->cls_sage) free(net->cls_sage);
    if (net->logsoft) free(net->logsoft);
    free(net);
}

void sage_net_info(const SageNet *net)
{
    printf("SageNet(\n");

    for (size_t i = 0; i < net->enc_depth; i++) {
        SageLayer *s = net->enc_sage[i];
        printf("  (sage_%zu): SageConv(%zu, %zu)\n", i, s->Wroot->M, s->Wroot->N);
        printf("  (relu_%zu): ReLU()\n", i);
        printf("  (norm_%zu): L2Norm()\n", i);
    }

    SageLayer *s = net->cls_sage;
    printf("  (sage_%zu): SageConv(%zu, %zu)\n", net->num_layers, s->Wroot->M, s->Wroot->N);
    printf("  (logsoft): LogSoftmax(dim=1)\n");

    printf(")\n");
}
