#include <stdlib.h>

#include "core.h"
#include "layers.h"
#include "gnn.h"
#include "matrix.h"
#include "dataset.h"
#include "timer.h"
#include "linalg/linalg.h"

SageLayer* sage_layer_create(uint32_t num_nodes, size_t in_dim, size_t out_dim)
{
    SageLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate SageLayer");

    *layer = (SageLayer){
        .in_dim      = in_dim,
        .out_dim     = out_dim,
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(num_nodes, out_dim),
        .agg         = matrix_create(num_nodes, in_dim),
        .Wagg        = matrix_create(in_dim, out_dim),
        .Wroot       = matrix_create(in_dim, out_dim),
        .grad_input  = matrix_create(num_nodes, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_Wagg   = matrix_create(in_dim, out_dim),
        .grad_Wroot  = matrix_create(in_dim, out_dim),
    };
    layer->mean_scale = malloc(num_nodes * sizeof(*layer->mean_scale));

    // Initialize weights randomly
    matrix_fill_xavier_uniform(layer->Wroot, in_dim, out_dim);
    matrix_fill_xavier_uniform(layer->Wagg, in_dim, out_dim);

    return layer;
}

ReluLayer* relu_layer_create(uint32_t num_nodes, size_t dim)
{
    ReluLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate ReluLayer");

    *layer = (ReluLayer) {
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(num_nodes, dim),
        .grad_input  = matrix_create(num_nodes, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
    };

    return layer;

}

L2NormLayer* l2norm_layer_create(uint32_t num_nodes, size_t dim)
{
    L2NormLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate L2NormLayer");

    *layer = (L2NormLayer) {
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(num_nodes, dim),
        .grad_input  = matrix_create(num_nodes, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, num_nodes),
        .recip_mag   = matrix_create(num_nodes, 1)
    };

    return layer;
}

LinearLayer* linear_layer_create(uint32_t batch_size, size_t in_dim, size_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LinearLayer");

    *layer = (LinearLayer) {
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


LogSoftLayer* logsoft_layer_create(uint32_t batch_size, size_t dim)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LogSoftLayer");

    *layer = (LogSoftLayer) {
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
    };

    return layer;
}

SageNet* sage_net_create(LayerConf *conf, size_t count, Dataset *ds)
{
    SageNet *net = malloc(sizeof(*net));
    if (!net) ERROR("Could not allocate SageNet");

    net->num_layers = count;
    net->layers = malloc(count * sizeof(*net->layers));
    if (!net->layers) ERROR("Could not allocate layers");

    for (size_t i = 0; i < count; i++)
    {
        void *ctx = NULL;
        switch (conf[i].type)
        {
        case LAYER_SAGE:
            ctx = sage_layer_create(ds->num_nodes, conf[i].in_dim, conf[i].out_dim);

#ifdef VERBOSE_TIMERS
            ((SageLayer*)ctx)->timer_dWroot = nob_temp_sprintf("L%zu_dWr:%zu->%zu", i, conf[i].in_dim, conf[i].out_dim);
            ((SageLayer*)ctx)->timer_dWagg  = nob_temp_sprintf("L%zu_dWa:%zu->%zu", i, conf[i].in_dim, conf[i].out_dim);
            ((SageLayer*)ctx)->timer_dinput = nob_temp_sprintf("L%zu_dIn:%zu", i, conf[i].in_dim);
            ((SageLayer*)ctx)->timer_dneigh = nob_temp_sprintf("L%zu_dNe:%zu", i, conf[i].in_dim);
#else
            ((SageLayer*)ctx)->timer_dWroot = "dWroot";
            ((SageLayer*)ctx)->timer_dWagg  = "dWagg";
            ((SageLayer*)ctx)->timer_dinput = "dinput";
            ((SageLayer*)ctx)->timer_dneigh = "dneigh";
#endif
            net->layers[i] = (Layer){
				.type            = LAYER_SAGE,
                .ctx             = ctx,
                .input_ptr       = &((SageLayer*)ctx)->input,
                .grad_output_ptr = &((SageLayer*)ctx)->grad_output,
                .output_ptr      = &((SageLayer*)ctx)->output,
                .grad_input_ptr  = &((SageLayer*)ctx)->grad_input,
            };
            break;
        case LAYER_RELU:
            ctx = (void*)relu_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_RELU,
                .ctx             = ctx,
                .input_ptr       = &((ReluLayer*)ctx)->input,
                .grad_output_ptr = &((ReluLayer*)ctx)->grad_output,
                .output_ptr      = &((ReluLayer*)ctx)->output,
                .grad_input_ptr  = &((ReluLayer*)ctx)->grad_input,
            };
            break;
        case LAYER_L2NORM:
            ctx = (void*)l2norm_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_L2NORM,
                .ctx             = ctx,
                .input_ptr       = &((L2NormLayer*)ctx)->input,
                .grad_output_ptr = &((L2NormLayer*)ctx)->grad_output,
                .output_ptr      = &((L2NormLayer*)ctx)->output,
                .grad_input_ptr  = &((L2NormLayer*)ctx)->grad_input,
            };
            break;
        case LAYER_LINEAR:
            ctx = (void*)linear_layer_create(ds->num_nodes, conf[i].in_dim, conf[i].out_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_LINEAR,
                .ctx             = ctx,
                .input_ptr       = &((LinearLayer*)ctx)->input,
                .grad_output_ptr = &((LinearLayer*)ctx)->grad_output,
                .output_ptr      = &((LinearLayer*)ctx)->output,
                .grad_input_ptr  = &((LinearLayer*)ctx)->grad_input,
            };
            break;
        case LAYER_LOGSOFT:
            ctx = (void*)logsoft_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_LOGSOFT,
                .ctx             = ctx,
                .input_ptr       = &((LogSoftLayer*)ctx)->input,
                .grad_output_ptr = NULL,
                .output_ptr      = &((LogSoftLayer*)ctx)->output,
                .grad_input_ptr  = &((LogSoftLayer*)ctx)->grad_input,
            };
            break;
        default:
            ERROR("Unknown layer type %d", conf[i].type);
        }
    }

    // Set first layer's input to the dataset features
    Matrix *input = matrix_create(ds->num_nodes, ds->num_features);
#pragma omp parallel for
    for (size_t i = 0; i < ds->num_nodes * ds->num_features; i++)
    {
        input->data[i] = ds->nodes[i];
    }
    *(net->layers[0].input_ptr) = input;

// Wire everything up
    for (size_t i = 0; i < count - 1; i++)
    {
        *(net->layers[i + 1].input_ptr)   = *(net->layers[i].output_ptr);
        if (net->layers[i].grad_output_ptr)
        {
            *(net->layers[i].grad_output_ptr) = *(net->layers[i + 1].grad_input_ptr);
        }
    }

    return net;
}

void sage_layer_reset(const SageLayer *l, Dataset *ds)
{
    matrix_zero(l->output);
    matrix_zero(l->agg);
    matrix_zero(l->Wagg);
    matrix_zero(l->Wroot);
    matrix_zero(l->grad_input);
    matrix_zero(l->grad_Wagg);
    matrix_zero(l->grad_Wroot);
    memset(l->mean_scale, 0, ds->num_nodes*sizeof(*l->mean_scale));

    matrix_fill_xavier_uniform(l->Wroot, l->in_dim, l->out_dim);
    matrix_fill_xavier_uniform(l->Wagg, l->in_dim, l->out_dim);
}

void relu_layer_reset(const ReluLayer *l, Dataset *ds)
{
    matrix_zero(l->output);
    matrix_zero(l->grad_input);
}

void l2norm_layer_reset(const L2NormLayer *l, Dataset *ds)
{
    matrix_zero(l->output);
    matrix_zero(l->grad_input);
    matrix_zero(l->recip_mag);
}

void linear_layer_reset(const LinearLayer *l, Dataset *ds)
{
    matrix_zero(l->output);
    matrix_zero(l->W);
    matrix_zero(l->bias);
    matrix_zero(l->grad_input);
    matrix_zero(l->grad_W);
    matrix_zero(l->grad_bias);

    matrix_fill_xavier_uniform(l->W, l->in_dim, l->out_dim);
    matrix_fill_xavier_uniform(l->bias, 1, l->out_dim);
}

void logsoft_layer_reset(const LogSoftLayer *l, Dataset *ds)
{
    matrix_zero(l->output);
    matrix_zero(l->grad_input);
}

// void sage_net_reset(const SageNet *net, Dataset *ds)
// {
//     for (size_t i = 0; i < net->num_layers; i++)
// 	{
//         net->layers[i].reset(&net->layers[i], ds);
//     }
// }

// We don't free matrices that are references from other layers, i.e input and grad_output
void sage_layer_free(SageLayer *l, Dataset *ds)
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

void relu_layer_free(ReluLayer *l, Dataset *ds)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

void l2norm_layer_free(L2NormLayer *l, Dataset *ds)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);
    if (l->recip_mag) matrix_destroy(l->recip_mag);

    free(l);
}

void linear_layer_free(LinearLayer *l, Dataset *ds)
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

void logsoft_layer_free(LogSoftLayer *l, Dataset *ds)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

void sage_net_free(SageNet *net, Dataset *ds)
{
    if (!net) return;

    for (size_t i = 0; i < net->num_layers; i++)
	{
        Layer layer = net->layers[i];
        LayerType type = layer.type;
        void *ctx = layer.ctx;
        switch(type)
        {
        case LAYER_SAGE:
            sage_layer_free((SageLayer*)ctx, ds);
            break;
        case LAYER_RELU:
            relu_layer_free((ReluLayer*)ctx, ds);
            break;
        case LAYER_L2NORM:
            l2norm_layer_free((L2NormLayer*)ctx, ds);
            break;
        case LAYER_LOGSOFT:
            logsoft_layer_free((LogSoftLayer*)ctx, ds);
            break;
        case LAYER_LINEAR:
            linear_layer_free((LinearLayer*)ctx, ds);
            break;
        default:
            ERROR("Unknown layer type %d", type);
        }
    }

    free(net->layers);
    free(net);
}

void sage_net_info(const SageNet *net)
{
    printf("SageNet(\n");

    for (size_t i = 0; i < net->num_layers; i++)
    {
        Layer *l = &net->layers[i];
        switch (l->type)
        {
        case LAYER_SAGE:
            printf("  (%zu) SageConv(%zu, %zu)\n", i,
                   ((SageLayer *)l->ctx)->in_dim, ((SageLayer *)l->ctx)->out_dim);
            break;
        case LAYER_RELU:
            printf("  (%zu) ReLU()\n", i);
            break;
        case LAYER_L2NORM:
            printf("  (%zu) L2Norm()\n", i);
            break;
        case LAYER_LINEAR:
            printf("  (%zu) Linear(%zu, %zu)\n", i,
                   ((LinearLayer *)l->ctx)->in_dim, ((LinearLayer *)l->ctx)->out_dim);
            break;
        case LAYER_LOGSOFT:
            printf("  (%zu) LogSoftmax(dim=1)\n", i);
            break;
        }
    }

    printf(")\n");
}
