#include <stdlib.h>

#include "core.h"
#include "layers.h"
#include "gnn.h"
#include "dataset.h"
#include "timer.h"
#include "linalg/linalg.h"

static inline void fill_xavier_uniform(Real *x, size_t in, size_t out)
{
    const Real limit = real_sqrt(6.0 / (in + out));
    const Real recip_rand_max = 1.0 / RAND_MAX;

    // OpenMP can't be used here as rand() isn't thread-safe, variants that might
    // be of interest are srand48_r or random_r. This can be looked more closely if this
    // function ever takes more too much time.
    for (size_t i = 0; i < in * out; i++)
    {
        x[i] = limit * (2 * (Real)rand() * recip_rand_max - 1.0);
    }
}

SageLayer* sage_layer_create(uint32_t num_nodes, uint32_t num_edges, Edges edges, size_t in_dim, size_t out_dim)
{
    SageLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate SageLayer");
    *layer = (SageLayer) {
        .num_nodes    = num_nodes,
        .num_edges    = num_edges,
        .edges        = edges,
        .in_dim       = in_dim,
        .out_dim      = out_dim,
        .input        = NULL,   // Set later when connecting layer
        .output       = malloc(num_nodes * out_dim * sizeof(Real)),
        .agg          = malloc(num_nodes * in_dim * sizeof(Real)),
        .Wagg         = malloc(in_dim * out_dim * sizeof(Real)),
        .Wroot        = malloc(in_dim * out_dim * sizeof(Real)),
        .grad_input   = malloc(num_nodes * in_dim * sizeof(Real)),
        .grad_output  = NULL,   // Set later when connecting layer
        .grad_Wagg    = malloc(in_dim * out_dim * sizeof(Real)),
        .grad_Wroot   = malloc(in_dim * out_dim * sizeof(Real)),
        .mean_scale   = malloc(num_nodes * sizeof(Real)),
    };

    if (!layer->output || !layer->agg || !layer->Wagg || !layer->Wroot ||
        !layer->grad_input || !layer->grad_Wagg || !layer->grad_Wroot || !layer->mean_scale)
    {
        ERROR("Could not allocate SageLayer buffers");
    }

    // Initialize weights randomly
    fill_xavier_uniform(layer->Wroot, in_dim, out_dim);
    fill_xavier_uniform(layer->Wagg, in_dim, out_dim);

    return layer;
}

ReluLayer* relu_layer_create(uint32_t num_nodes, size_t dim)
{
    ReluLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate ReluLayer");

    *layer = (ReluLayer) {
        .num_nodes   = num_nodes,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers,
        .output      = malloc(num_nodes * dim * sizeof(Real)),
        .grad_input  = malloc(num_nodes * dim * sizeof(Real)),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, num_nodes),
    };

    return layer;

}

L2NormLayer* l2norm_layer_create(uint32_t num_nodes, size_t dim)
{
    L2NormLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate L2NormLayer");

    *layer = (L2NormLayer) {
        .num_nodes   = num_nodes,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers,
        .output      = malloc(num_nodes * dim * sizeof(Real)),
        .grad_input  = malloc(num_nodes * dim * sizeof(Real)),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, num_nodes),
        .recip_mag   = malloc(num_nodes * sizeof(Real)),
    };

    return layer;
}

LinearLayer* linear_layer_create(uint32_t num_nodes, size_t in_dim, size_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LinearLayer");

    *layer = (LinearLayer) {
        .num_nodes   = num_nodes,
        .in_dim      = in_dim,
        .out_dim     = out_dim,
        .input       = NULL, // Set later when connecting layers
        .output      = malloc(num_nodes * out_dim * sizeof(Real)),
        .W           = malloc(in_dim * out_dim * sizeof(Real)),
        .bias        = malloc(out_dim * sizeof(Real)),
        .grad_input  = malloc(num_nodes * in_dim * sizeof(Real)),
        .grad_output = NULL, // Set later when connecting layers
        .grad_W      = malloc(in_dim * out_dim * sizeof(Real)),
        .grad_bias   = malloc(out_dim * sizeof(Real)),
    };

    fill_xavier_uniform(layer->W, in_dim, out_dim);
    fill_xavier_uniform(layer->bias, 1, out_dim);

    return layer;
}

LogSoftLayer* logsoft_layer_create(uint32_t num_nodes, uint32_t num_classes, size_t dim)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LogSoftLayer");

    *layer = (LogSoftLayer) {
        .num_nodes   = num_nodes,
        .num_classes = num_classes,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers
        .output      = malloc(num_nodes * dim * sizeof(Real)),
        .grad_input  = malloc(num_nodes * dim * sizeof(Real)),
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
            ctx = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, conf[i].in_dim, conf[i].out_dim);

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
            ctx = (void*)logsoft_layer_create(ds->num_nodes, ds->num_classes, conf[i].in_dim);
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
    Real *input = malloc(ds->num_nodes * ds->num_features * sizeof(Real));
#pragma omp parallel for
    for (size_t i = 0; i < ds->num_nodes * ds->num_features; i++)
    {
        input[i] = ds->nodes[i];
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

void sage_layer_bind(SageLayer *layer, uint32_t num_nodes, uint32_t num_edges, Edges edges, bool no_grad)
{
    if (layer->num_nodes != num_nodes)
    {
        free(layer->output);
        free(layer->agg);
        if (!no_grad) free(layer->grad_input);
        if (!no_grad) free(layer->grad_Wroot);
        if (!no_grad) free(layer->grad_Wagg);

        free(layer->mean_scale);

        layer->output                   = malloc(num_nodes * layer->out_dim * sizeof(Real));
        layer->agg                      = malloc(num_nodes * layer->in_dim * sizeof(Real));
        if (!no_grad) layer->grad_input = malloc(num_nodes * layer->in_dim * sizeof(Real));
        if (!no_grad) layer->grad_Wagg  = malloc(layer->in_dim * layer->out_dim * sizeof(Real));
        if (!no_grad) layer->grad_Wroot = malloc(layer->in_dim * layer->out_dim * sizeof(Real));
        layer->mean_scale               = malloc(num_nodes * sizeof(Real));
    }
    layer->num_nodes = num_nodes;
    layer->num_edges = num_edges;
    layer->edges     = edges;
}

void relu_layer_bind(ReluLayer *layer, uint32_t num_nodes, bool no_grad)
{
    if (layer->num_nodes != num_nodes)
    {
        free(layer->output);
        if (!no_grad) free(layer->grad_input);

        layer->output                   = malloc(num_nodes * layer->dim * sizeof(Real));
        if (!no_grad) layer->grad_input = malloc(num_nodes * layer->dim * sizeof(Real));
    }
    layer->num_nodes = num_nodes;
}

void l2norm_layer_bind(L2NormLayer *layer, uint32_t num_nodes, bool no_grad)
{
    if (layer->num_nodes != num_nodes)
    {
        free(layer->output);
        if (!no_grad) free(layer->grad_input);
        free(layer->recip_mag);

        layer->output                   = malloc(num_nodes * layer->dim * sizeof(Real));
        if (!no_grad) layer->grad_input = malloc(num_nodes * layer->dim * sizeof(Real));
        layer->recip_mag                = malloc(num_nodes * sizeof(Real));
    }
    layer->num_nodes = num_nodes;
}

void linear_layer_bind(LinearLayer *layer, uint32_t num_nodes, bool no_grad)
{
    if (layer->num_nodes != num_nodes)
    {
        free(layer->output);
        if (!no_grad) free(layer->grad_input);
        if (!no_grad) free(layer->grad_W);
        if (!no_grad) free(layer->grad_bias);

        layer->output                   = malloc(num_nodes * layer->out_dim * sizeof(Real));
        if (!no_grad) layer->grad_input = malloc(num_nodes * layer->in_dim * sizeof(Real));
        if (!no_grad) layer->grad_W     = malloc(layer->in_dim * layer->out_dim * sizeof(Real));
        if (!no_grad) layer->grad_bias  = malloc(layer->out_dim * sizeof(Real));
    }
    layer->num_nodes = num_nodes;
}

void logsoft_layer_bind(LogSoftLayer *layer, uint32_t num_nodes, bool no_grad)
{
    if (layer->num_nodes != num_nodes)
    {
        free(layer->output);
        if (!no_grad) free(layer->grad_input);

        layer->output                   = malloc(num_nodes * layer->dim * sizeof(Real));
        if (!no_grad) layer->grad_input = malloc(num_nodes * layer->dim * sizeof(Real));
    }
    layer->num_nodes = num_nodes;
}

void sage_net_bind(SageNet *net, Dataset *ds, bool no_grad)
{
    for (size_t i = 0; i < net->num_layers; i++)
    {
        Layer *layer = &net->layers[i];
        switch (layer->type)
        {
        case LAYER_SAGE:
            sage_layer_bind(layer->ctx, ds->num_nodes, ds->num_edges, ds->edges, no_grad);
            break;
        case LAYER_RELU:
            relu_layer_bind(layer->ctx, ds->num_nodes, no_grad);
            break;
        case LAYER_L2NORM:
            l2norm_layer_bind(layer->ctx, ds->num_nodes, no_grad);
            break;
        case LAYER_LINEAR:
            linear_layer_bind(layer->ctx, ds->num_nodes, no_grad);
            break;
        case LAYER_LOGSOFT:
            logsoft_layer_bind(layer->ctx, ds->num_nodes, no_grad);
            break;
        default:
            ERROR("Unknown layer type %d", layer->type);
        }
    }

    // Re-set input features
    Real *input = malloc(ds->num_nodes * ds->num_features * sizeof(Real));
#pragma omp parallel for
    for (size_t i = 0; i < ds->num_nodes * ds->num_features; i++)
    {
        input[i] = ds->nodes[i];
    }

    // TODO: just use the the pointer of input from dataset
    free(*(net->layers[0].input_ptr));
    *(net->layers[0].input_ptr) = input;

    // Re-wire inter-layer pointers (outputs changed address)
    for (size_t i = 0; i < net->num_layers - 1; i++)
    {
        *(net->layers[i + 1].input_ptr) = *(net->layers[i].output_ptr);
        if (net->layers[i].grad_output_ptr)
        {
            *(net->layers[i].grad_output_ptr) = *(net->layers[i + 1].grad_input_ptr);
        }
    }
}


// We don't free matrices that are references from other layers, i.e input and grad_output
void sage_layer_free(SageLayer **l)
{
    if (!(*l)) return;

    free((*l)->output);
    free((*l)->agg);
    free((*l)->Wagg);
    free((*l)->Wroot);
    free((*l)->grad_input);
    free((*l)->grad_Wagg);
    free((*l)->grad_Wroot);
    free((*l)->mean_scale);
    free((*l));
    *l = NULL;
}

void relu_layer_free(ReluLayer **l)
{
    if (!*l) return;

    free((*l)->output);
    free((*l)->grad_input);
    free(*l);
    *l = NULL;
}

void l2norm_layer_free(L2NormLayer **l)
{
    if (!*l) return;

    free((*l)->output);
    free((*l)->grad_input);
    free((*l)->recip_mag);
    free((*l));
    *l = NULL;
}

void linear_layer_free(LinearLayer **l)
{
    if (!*l) return;

    free((*l)->output);
    free((*l)->W);
    free((*l)->bias);
    free((*l)->grad_input);
    free((*l)->grad_W);
    free((*l)->grad_bias);
    free(*l);
    *l = NULL;
}

void logsoft_layer_free(LogSoftLayer **l)
{
    if (!*l) return;

    free((*l)->output);
    free((*l)->grad_input);
    free(*l);
    *l = NULL;
}

void sage_net_free(SageNet **net)
{
    if (!(*net)) return;

    for (size_t i = 0; i < (*net)->num_layers; i++)
	{
        Layer layer = (*net)->layers[i];
        switch(layer.type)
        {
        case LAYER_SAGE:
            sage_layer_free((SageLayer**)&layer.ctx);
            break;
        case LAYER_RELU:
            relu_layer_free((ReluLayer**)&layer.ctx);
            break;
        case LAYER_L2NORM:
            l2norm_layer_free((L2NormLayer**)&layer.ctx);
            break;
        case LAYER_LOGSOFT:
            logsoft_layer_free((LogSoftLayer**)&layer.ctx);
            break;
        case LAYER_LINEAR:
            linear_layer_free((LinearLayer**)&layer.ctx);
            break;
        default:
            ERROR("Unknown layer type %d", layer.type);
        }
    }

    free((*net)->layers);
    free(*net);
    *net = NULL;
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
