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

SageLayer* sage_layer_create(size_t batch_size, size_t in_dim, size_t out_dim, Dataset *data)
{
    SageLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate SageLayer");

    *layer = (SageLayer){
        .data        = data,
        .in_dim      = in_dim,
        .out_dim     = out_dim,
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

ReluLayer* relu_layer_create(size_t batch_size, size_t dim, Dataset *data)
{
    ReluLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate ReluLayer");

    *layer = (ReluLayer) {
        .data        = data,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
    };

    return layer;
}

L2NormLayer* l2norm_layer_create(size_t batch_size, size_t dim, Dataset *data)
{
    L2NormLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate L2NormLayer");

    *layer = (L2NormLayer) {
        .data        = data,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
        .recip_mag   = matrix_create(batch_size, 1)
    };

    return layer;
}

LinearLayer* linear_layer_create(size_t batch_size, size_t in_dim, size_t out_dim, Dataset *data)
{
    LinearLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LinearLayer");

    *layer = (LinearLayer) {
        .data        = data,
        .in_dim      = in_dim,
        .out_dim     = out_dim,
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


LogSoftLayer* logsoft_layer_create(size_t batch_size, size_t dim, Dataset *data)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LogSoftLayer");

    *layer = (LogSoftLayer) {
        .data        = data,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
    };

    return layer;
}

SageNet* sage_net_create(size_t num_layers, size_t hidden_dim, Dataset *data)
{
    if (num_layers < 1) ERROR("Number of layers needed for SageNet is >=1");

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
        size_t layer_in  = (i == 0) ? data->num_features : hidden_dim;
        size_t layer_out = hidden_dim;

        net->enc_sage[i] = sage_layer_create(data->num_inputs, layer_in, layer_out, data);
        net->enc_relu[i] = relu_layer_create(data->num_inputs, layer_out, data);
        net->enc_norm[i] = l2norm_layer_create(data->num_inputs, layer_out, data);

#ifdef VERBOSE_TIMERS
        net->enc_sage[i]->timer_dWroot = nob_temp_sprintf("L%zu_dWr:%zu->%zu", i, layer_in, layer_out);
        net->enc_sage[i]->timer_dWagg  = nob_temp_sprintf("L%zu_dWa:%zu->%zu", i, layer_in, layer_out);
        net->enc_sage[i]->timer_dinput = nob_temp_sprintf("L%zu_dIn:%zu", i, layer_in);
        net->enc_sage[i]->timer_dneigh = nob_temp_sprintf("L%zu_dNe:%zu", i, layer_in);
#else
        net->enc_sage[i]->timer_dWroot = "dWroot";
        net->enc_sage[i]->timer_dWagg  = "dWagg";
        net->enc_sage[i]->timer_dinput = "dinput";
        net->enc_sage[i]->timer_dneigh = "dneigh";
#endif
    }

    size_t final_in = (num_layers == 1) ? data->num_features : hidden_dim;

#ifndef USE_PREDICTION_HEAD
    size_t final_out = data->num_classes;
    net->cls_sage = sage_layer_create(data->num_inputs, final_in, data->num_classes, data);
#else
    size_t final_out = hidden_dim;
    net->cls_sage = sage_layer_create(batch_size, final_in, hidden_dim);
    LinearLayer *linearlayer = init_linear_layer(batch_size, hidden_dim, num_classes, Dataset *data);
#endif

#ifdef VERBOSE_TIMERS
    net->cls_sage->timer_dWroot = nob_temp_sprintf("Lc_dWr:%zu->%zu", final_in, final_out);
    net->cls_sage->timer_dWagg  = nob_temp_sprintf("Lc_dWa:%zu->%zu", final_in, final_out);
    net->cls_sage->timer_dinput = nob_temp_sprintf("Lc_dIn:%zu", final_in);
    net->cls_sage->timer_dneigh = nob_temp_sprintf("Lc_dNe:%zu", final_in);
#else
    (void)final_out;
    net->cls_sage->timer_dWroot = "dWroot";
    net->cls_sage->timer_dWagg  = "dWagg";
    net->cls_sage->timer_dinput = "dinput";
    net->cls_sage->timer_dneigh = "dneigh";
#endif

    net->logsoft = logsoft_layer_create(data->num_inputs, data->num_classes, data);

    Matrix *input = matrix_create(data->num_inputs, data->num_features);
#pragma omp parallel for
    for (size_t i = 0; i < data->num_inputs*data->num_features; i++) {
        input->data[i] = data->inputs[i];
    }

    if (net->enc_depth > 0) {
        net->enc_sage[0]->input = input;
    } else {
        net->cls_sage->input = input;
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

    if (net->enc_depth > 0) matrix_destroy(net->enc_sage[0]->input);
    else matrix_destroy(net->cls_sage->input);

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

    if (net->cls_sage) sage_layer_destroy(net->cls_sage);
    if (net->logsoft) logsoft_layer_destroy(net->logsoft);
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
