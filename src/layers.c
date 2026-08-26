#include "layers.h"

#include <stdlib.h>

#include "core.h"
#include "ds.h"
#include "timer.h"
#include "vreg.h"

static inline void fill_xavier_uniform(Real *x, int64_t in, int64_t ld, int64_t out)
{
    const Real limit = real_sqrt(REAL(6.0) / (in + out));
    const Real recip_rand_max = REAL(1.0) / REAL(RAND_MAX);

#pragma omp parallel for
    for (int64_t i = 0; i < in; i++)
    {
        Real *xi = &x[i * ld];
        for (int64_t j = 0; j < ld; j++)
        {
            xi[j] = REAL(0.0);
        }
    }

    // OpenMP can't be used here as rand() isn't thread-safe, variants that
    // might be of interest are srand48_r or random_r. This can be looked into
    // more closely if this function ever uses to much time.
    for (int64_t i = 0; i < in * out; i++)
    {
        x[i] = limit * (REAL(2.0) * REAL(rand()) * recip_rand_max - REAL(1.0));
    }
}

// SAGE LAYER

static void sage_alloc_node_buffers(SageLayer *l, uint32_t num_nodes)
{
    l->output       = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->out_dim * sizeof(Real)));
    l->agg          = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real)));
    l->grad_input   = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real)));
    l->grad_scatter = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real)));
}

SageLayer* sage_layer_alloc(int64_t num_nodes, int64_t num_edges, SparseGraph *graph, int64_t in_dim, int64_t out_dim, FlowDirection flow)
{
    SageLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));
    int64_t out_dim_pad = ((out_dim + N_VEC - 1) / N_VEC) * N_VEC;
    *layer = (SageLayer) {
        .num_nodes    = num_nodes,
        .num_edges    = num_edges,
        .graph        = graph,
        .in_dim       = in_dim,
        .out_dim      = out_dim,
        .flow         = flow,
        .input        = NULL,   // Set later when connecting layer
        .grad_output  = NULL,   // Set later when connecting layer
        // This allows us to perfrom non-temporal store even if out_dim isn't a
        // mutliple of N_VEC, e.g for last layer. Currently Wagg and Wroot
        // doesn't need non-temporal store, but this makes performing weight
        // optimization much simpler (optim.c).
        .Wagg         = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .Wroot        = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .grad_Wagg    = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .grad_Wroot   = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim_pad * sizeof(Real))),
        .ldW          = out_dim_pad,
    };
    sage_alloc_node_buffers(layer, num_nodes);
    return layer;
}

void sage_layer_reset_parameters(SageLayer *l)
{
    fill_xavier_uniform(l->Wroot, l->in_dim, l->ldW, l->out_dim);
    fill_xavier_uniform(l->Wagg, l->in_dim, l->ldW, l->out_dim);
}

static void sage_free_node_buffers(SageLayer *l)
{
    free(l->output);
    free(l->agg);
    free(l->grad_input);
    free(l->grad_scatter);
}

void sage_layer_free(SageLayer **l)
{
    if (!(*l)) return;
    sage_free_node_buffers(*l);
    free((*l)->Wagg);
    free((*l)->Wroot);
    free((*l)->grad_Wroot);
    free((*l)->grad_Wagg);
    free((*l));
    *l = NULL;
}

void sage_layer_bind(SageLayer *l, int64_t num_nodes, int64_t num_edges, SparseGraph *graph)
{
    if (l->num_nodes < num_nodes)
    {
        sage_free_node_buffers(l);
        sage_alloc_node_buffers(l, num_nodes);
    }
    l->num_nodes = num_nodes;
    l->num_edges = num_edges;
    l->graph     = graph;
}

// RELU LAYER

static void relu_alloc_node_buffers(ReluLayer *l, int64_t num_nodes)
{
    l->output      = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->dim * sizeof(Real)));
    l->grad_input  = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->dim * sizeof(Real)));
}

ReluLayer* relu_layer_alloc(int64_t num_nodes, int64_t dim)
{
    ReluLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));
    *layer = (ReluLayer) {
        .num_nodes   = num_nodes,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers
        .grad_output = NULL, // Set later when connecting layers
    };
    relu_alloc_node_buffers(layer, num_nodes);
    return layer;
}

static void relu_free_node_buffers(ReluLayer *l)
{
    free(l->output);
    free(l->grad_input);
}

void relu_layer_free(ReluLayer **l)
{
    if (!(*l)) return;
    relu_free_node_buffers(*l);
    free((*l));
    *l = NULL;
}

void relu_layer_bind(ReluLayer *l, int64_t num_nodes)
{
    if (l->num_nodes < num_nodes)
    {
        relu_free_node_buffers(l);
        relu_alloc_node_buffers(l, num_nodes);
    }
    l->num_nodes = num_nodes;
}

// L2-NORMALIZE LAYER

static void l2norm_alloc_node_buffers(L2NormLayer *l, int64_t num_nodes)
{
    l->output      = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->dim * sizeof(Real)));
    l->grad_input  = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->dim * sizeof(Real)));
    l->recip_mag   = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * sizeof(Real)));
}

L2NormLayer* l2norm_layer_alloc(int64_t num_nodes, int64_t dim)
{
    L2NormLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));
    *layer = (L2NormLayer) {
        .num_nodes   = num_nodes,
        .dim         = dim,
        .input       = NULL, // Set later when connecting layers
        .grad_output = NULL, // Set later when connecting layers
    };
    l2norm_alloc_node_buffers(layer, num_nodes);
    return layer;
}

static void l2norm_free_node_buffers(L2NormLayer *l)
{
    free(l->output);
    free(l->grad_input);
    free(l->recip_mag);
}

void l2norm_layer_free(L2NormLayer **l)
{
    if (!(*l)) return;
    l2norm_free_node_buffers(*l);
    free((*l));
    *l = NULL;
}

void l2norm_layer_bind(L2NormLayer *l, int64_t num_nodes)
{
    if (l->num_nodes < num_nodes)
    {
        l2norm_free_node_buffers(l);
        l2norm_alloc_node_buffers(l, num_nodes);
    }
    l->num_nodes = num_nodes;
}

// LINEAR LAYER

static void linear_alloc_node_buffers(LinearLayer *l, int64_t num_nodes)
{
    l->output      = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->out_dim * sizeof(Real)));
    l->grad_input  = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real)));
    l->grad_W      = ALLOC_OR_DIE(cache_aligned_alloc(l->in_dim * l->out_dim * sizeof(Real)));
    l->grad_bias   = ALLOC_OR_DIE(cache_aligned_alloc(l->out_dim * sizeof(Real)));
}

LinearLayer* linear_layer_alloc(int64_t num_nodes, int64_t in_dim, int64_t out_dim)
{
    LinearLayer *l = ALLOC_OR_DIE(malloc(sizeof(*l)));
    *l = (LinearLayer) {
        .num_nodes   = num_nodes,
        .in_dim      = in_dim,
        .out_dim     = out_dim,
        .input       = NULL, // Set later when connecting layers
        .grad_output = NULL, // Set later when connecting layers
        .W           = ALLOC_OR_DIE(cache_aligned_alloc(in_dim * out_dim * sizeof(Real))),
        .bias        = ALLOC_OR_DIE(cache_aligned_alloc(out_dim * sizeof(Real))),
    };
    linear_alloc_node_buffers(l, num_nodes);
    return l;
}

void linear_layer_reset_parameters(LinearLayer* l)
{
    fill_xavier_uniform(l->W, l->in_dim, l->out_dim, l->out_dim);
    fill_xavier_uniform(l->bias, 1, l->out_dim, l->out_dim);
}


static void linear_free_node_buffers(LinearLayer *l)
{
    free(l->output);
    free(l->grad_input);
    free(l->grad_W);
    free(l->grad_bias);
}

void linear_layer_free(LinearLayer **l)
{
    if (!*l) return;
    linear_free_node_buffers(*l);
    free((*l)->W);
    free((*l)->bias);
    free(*l);
    *l = NULL;
}

void linear_layer_bind(LinearLayer *l, int64_t num_nodes)
{
    if (l->num_nodes < num_nodes)
    {
        linear_free_node_buffers(l);
        linear_alloc_node_buffers(l, num_nodes);
    }
    l->num_nodes = num_nodes;
}

// LOGSOFTMAX LAYER
static void logsoft_alloc_node_buffers(LogSoftmaxLayer *l, int64_t num_nodes)
{
    l->output     = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->dim * sizeof(Real)));
    l->grad_input = ALLOC_OR_DIE(cache_aligned_alloc(num_nodes * l->dim * sizeof(Real)));
}

LogSoftmaxLayer* logsoft_layer_alloc(int64_t num_nodes, int64_t dim)
{
    LogSoftmaxLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));

    *layer = (LogSoftmaxLayer) {
        .num_nodes   = num_nodes,
        .dim         = dim,  // if last layer, should be number of classes
        .input       = NULL, // Set later when connecting layers
    };

    logsoft_alloc_node_buffers(layer, num_nodes);

    return layer;
}

static void logsoft_free_node_buffers(LogSoftmaxLayer *l)
{
    free(l->output);
    free(l->grad_input);
}

void logsoft_layer_free(LogSoftmaxLayer **l)
{
    if (!*l) return;
    logsoft_free_node_buffers(*l);
    free(*l);
    *l = NULL;
}

void logsoft_layer_bind(LogSoftmaxLayer *l, int64_t num_nodes)
{
    if (l->num_nodes < num_nodes)
    {
        logsoft_free_node_buffers(l);
        logsoft_alloc_node_buffers(l, num_nodes);
    }
    l->num_nodes = num_nodes;
}

// SAGENET

static void wireup_network(SageNet *net, int64_t num_layers, Dataset *ds)
{
    // Set first layer's input to the dataset features
    ((SageLayer*)net->layers[0].ctx)->input = ds->nodes;

    // Wire everything up
    for (int64_t i = 0; i < num_layers - 1; i++)
    {
        Layer layer = net->layers[i];
        Real *out, **go;
        switch (layer.type)
        {
            case LAYER_SAGE:
                out = ((SageLayer*)layer.ctx)->output;
                go  = &((SageLayer*)layer.ctx)->grad_output;
                break;
            case LAYER_RELU:
                out = ((ReluLayer*)layer.ctx)->output;
                go  = &((ReluLayer*)layer.ctx)->grad_output;
                break;
            case LAYER_L2NORM:
                out = ((L2NormLayer*)layer.ctx)->output;
                go  = &((L2NormLayer*)layer.ctx)->grad_output;
                break;
            case LAYER_LOGSOFTMAX:
                out = ((LogSoftmaxLayer*)layer.ctx)->output;
                go  = NULL;
                break;
            case LAYER_LINEAR:
                out = ((LinearLayer*)layer.ctx)->output;
                go  = &((LinearLayer*)layer.ctx)->grad_output;
                break;
            default:
                ERROR("Unknown layer type %d", layer.type);
        }

        Layer next_layer = net->layers[i+1];
        Real **next_in, *next_gi;
        switch (next_layer.type)
        {
            case LAYER_SAGE:
                next_in = &((SageLayer*)next_layer.ctx)->input;
                next_gi = ((SageLayer*)next_layer.ctx)->grad_input;
                break;
            case LAYER_RELU:
                next_in = &((ReluLayer*)next_layer.ctx)->input;
                next_gi = ((ReluLayer*)next_layer.ctx)->grad_input;
                break;
            case LAYER_L2NORM:
                next_in = &((L2NormLayer*)next_layer.ctx)->input;
                next_gi = ((L2NormLayer*)next_layer.ctx)->grad_input;
                break;
            case LAYER_LOGSOFTMAX:
                next_in = &((LogSoftmaxLayer*)next_layer.ctx)->input;
                next_gi = ((LogSoftmaxLayer*)next_layer.ctx)->grad_input;
                break;
            case LAYER_LINEAR:
                next_in = &((LinearLayer*)next_layer.ctx)->input;
                next_gi = ((LinearLayer*)next_layer.ctx)->grad_input;
                break;
            default:
                ERROR("Unknown layer type %d", next_layer.type);
        }

        *next_in = out;
        if (go) *go = next_gi;
    }
}


SageNet* sage_net_alloc(LayerConf *conf, int64_t count, Dataset *ds, FlowDirection flow)
{
    SageNet *net = ALLOC_OR_DIE(malloc(sizeof(*net)));

    net->num_layers = count;
    // NOTE: we calloc here to make gcc not complain
    net->layers = ALLOC_OR_DIE(calloc(count,sizeof(*net->layers)));

    for (int64_t i = 0; i < count; i++)
    {
        void *ctx = NULL;
        switch (conf[i].type)
        {
            case LAYER_SAGE:
                ctx = sage_layer_alloc(ds->node_count, ds->edge_count, ds->graph, conf[i].in_dim, conf[i].out_dim, flow);
                net->layers[i] = (Layer){
                    .type            = LAYER_SAGE,
                    .ctx             = ctx,
                };
                break;
            case LAYER_RELU:
                ctx = (void*)relu_layer_alloc(ds->node_count, conf[i].in_dim);
                net->layers[i] = (Layer){
                    .type            = LAYER_RELU,
                    .ctx             = ctx,
                };
                break;
            case LAYER_L2NORM:
                ctx = (void*)l2norm_layer_alloc(ds->node_count, conf[i].in_dim);
                net->layers[i] = (Layer){
                    .type            = LAYER_L2NORM,
                    .ctx             = ctx,
                };
                break;
            case LAYER_LINEAR:
                ctx = (void*)linear_layer_alloc(ds->node_count, conf[i].in_dim, conf[i].out_dim);
                net->layers[i] = (Layer){
                    .type            = LAYER_LINEAR,
                    .ctx             = ctx,
                };
                break;
            case LAYER_LOGSOFTMAX:
                ctx = (void*)logsoft_layer_alloc(ds->node_count, conf[i].in_dim);
                net->layers[i] = (Layer){
                    .type            = LAYER_LOGSOFTMAX,
                    .ctx             = ctx,
                };
                break;
            default:
                ERROR("Unknown layer type %d", conf[i].type);
        }
    }

    wireup_network(net, count, ds);
    return net;
}

void sage_net_bind(SageNet *net, Dataset *ds)
{
    for (int64_t i = 0; i < net->num_layers; i++)
    {
        Layer *layer = &net->layers[i];
        switch (layer->type)
        {
            case LAYER_SAGE:
                sage_layer_bind(layer->ctx, ds->node_count, ds->edge_count, ds->graph);
                break;
            case LAYER_RELU:
                relu_layer_bind(layer->ctx, ds->node_count);
                break;
            case LAYER_L2NORM:
                l2norm_layer_bind(layer->ctx, ds->node_count);
                break;
            case LAYER_LINEAR:
                linear_layer_bind(layer->ctx, ds->node_count);
                break;
            case LAYER_LOGSOFTMAX:
                logsoft_layer_bind(layer->ctx, ds->node_count);
                break;
            default:
                ERROR("Unknown layer type %d", layer->type);
        }
    }

    wireup_network(net, net->num_layers, ds);
}

void sage_net_reset_parameters(SageNet *net)
{
    for (int64_t i = 0; i < net->num_layers; i++)
    {
        Layer *layer = &net->layers[i];
        switch (layer->type)
        {
            case LAYER_SAGE:
                sage_layer_reset_parameters((SageLayer*)layer->ctx);
                break;
            case LAYER_LINEAR:
                linear_layer_reset_parameters((LinearLayer*)layer->ctx);
                break;
            case LAYER_RELU:
            case LAYER_L2NORM:
            case LAYER_LOGSOFTMAX:
                break;
            default:
                ERROR("Unknown layer type %d", layer->type);
        }
    }

}

void sage_net_free(SageNet **net)
{
    if (!(*net)) return;

    for (int64_t i = 0; i < (*net)->num_layers; i++)
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
            case LAYER_LOGSOFTMAX:
                logsoft_layer_free((LogSoftmaxLayer**)&layer.ctx);
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

    for (int64_t i = 0; i < net->num_layers; i++)
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
            case LAYER_LOGSOFTMAX:
                printf("  (%zu) LogSoftmax(dim=1)\n", i);
                break;
        }
    }

    printf(")\n");
}
