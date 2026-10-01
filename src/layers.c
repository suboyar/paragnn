#include "layers.h"

#include <stdlib.h>

#include "core.h"
#include "ds.h"
#include "timer.h"
#include "vreg.h"

static inline void fill_xavier_uniform(Real *x, int64_t in, int64_t out, int64_t stride)
{
    const Real limit = real_sqrt(REAL(6.0) / (in + out));
    const Real recip_rand_max = REAL(1.0) / REAL(RAND_MAX);


    // OpenMP can't be used here as rand() isn't thread-safe, variants that
    // might be of interest are srand48_r or random_r. This can be looked into
    // more closely if this function ever uses to much time.
    for (int64_t i = 0; i < in; i++)
    {
        Real *x_ptr = &x[i * stride];
        for (int64_t j = 0; j < out; j++)
            x_ptr[j] = limit * (REAL(2.0) * REAL(rand()) * recip_rand_max - REAL(1.0));
    }
}

// SAGE LAYER

static void sage_alloc_node_buffers(SageLayer *l, uint32_t node_count)
{
    l->x_out      = ALLOC_OR_DIE(alloc_shared(node_count * l->out_dim * sizeof(Real)));
    l->dx_in      = ALLOC_OR_DIE(alloc_shared(node_count * l->in_dim * sizeof(Real)));
    l->x_neigh    = ALLOC_OR_DIE(alloc_shared(node_count * l->in_dim * sizeof(Real)));
    l->dx_scatter = ALLOC_OR_DIE(alloc_shared(node_count * l->in_dim * sizeof(Real)));
}

static void sage_free_node_buffers(SageLayer *l)
{
    free(l->x_out);
    free(l->x_neigh);
    free(l->dx_in);
    free(l->dx_scatter);
}

static void sage_layer_init(SageLayer* l)
{
    real_zero_out(l->x_out, l->node_count * l->out_dim);
    real_zero_out(l->dx_in, l->node_count * l->in_dim);
    real_zero_out(l->x_neigh, l->node_count * l->in_dim);
    real_zero_out(l->dx_scatter, l->node_count * l->in_dim);
    real_zero_out(l->W_self, l->in_dim * l->W_stride);
    real_zero_out(l->W_neigh, l->in_dim * l->W_stride);
    real_zero_out(l->dW_self, l->in_dim * l->W_stride);
    real_zero_out(l->dW_neigh, l->in_dim * l->W_stride);
}

SageLayer* sage_layer_alloc(int64_t node_count, int64_t edge_count, SparseGraph *graph, int64_t in_dim, int64_t out_dim, FlowDirection flow)
{
    // This allows us to perfrom non-temporal store even if out_dim isn't a
    // mutliple of N_VEC, e.g for last layer. Currently W_self and W_neigh
    // doesn't need non-temporal store, but this makes performing weight
    // optimization much simpler (optim.c).
    int64_t W_stride = ((out_dim + N_VEC - 1) / N_VEC) * N_VEC;

    SageLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));
    *layer          = (SageLayer) {
        .node_count = node_count,
        .edge_count = edge_count,
        .graph      = graph,
        .in_dim     = in_dim,
        .out_dim    = out_dim,
        .flow       = flow,
        .x_in       = NULL,     // Set later when connecting layer
        .dx_out     = NULL,     // Set later when connecting layer
        .W_self     = ALLOC_OR_DIE(alloc_shared(in_dim * W_stride * sizeof(Real))),
        .W_neigh    = ALLOC_OR_DIE(alloc_shared(in_dim * W_stride * sizeof(Real))),
        .dW_self    = ALLOC_OR_DIE(alloc_shared(in_dim * W_stride * sizeof(Real))),
        .dW_neigh   = ALLOC_OR_DIE(alloc_shared(in_dim * W_stride * sizeof(Real))),
        .W_stride   = W_stride,
    };
    sage_alloc_node_buffers(layer, node_count);
    sage_layer_init(layer);
    return layer;
}

void sage_layer_reset_parameters(SageLayer *l)
{
    fill_xavier_uniform(l->W_self, l->in_dim, l->out_dim, l->W_stride);
    fill_xavier_uniform(l->W_neigh, l->in_dim, l->out_dim, l->W_stride);
}

void sage_layer_reset_gradient(SageLayer *l)
{
    real_zero_out(l->dx_in, l->node_count * l->in_dim);
    real_zero_out(l->dx_scatter, l->node_count * l->in_dim);
    real_zero_out(l->dW_self, l->in_dim * l->W_stride);
    real_zero_out(l->dW_neigh, l->in_dim * l->W_stride);
}

void sage_layer_free(SageLayer **l)
{
    if (!(*l)) return;
    sage_free_node_buffers(*l);
    free((*l)->W_neigh);
    free((*l)->W_self);
    free((*l)->dW_self);
    free((*l)->dW_neigh);
    free((*l));
    *l = NULL;
}

void sage_layer_bind(SageLayer *l, int64_t node_count, int64_t edge_count, SparseGraph *graph)
{
    if (l->node_count < node_count)
    {
        sage_free_node_buffers(l);
        sage_alloc_node_buffers(l, node_count);
    }
    l->node_count = node_count;
    l->edge_count = edge_count;
    l->graph     = graph;
}

// RELU LAYER

static void relu_alloc_node_buffers(ReluLayer *l, int64_t node_count)
{
    (void)(l);
    (void)(node_count);
}
static void relu_free_node_buffers(ReluLayer *l)
{
    (void)(l);
}

ReluLayer* relu_layer_alloc(int64_t node_count, int64_t dim)
{
    ReluLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));
    *layer = (ReluLayer) {
        .node_count = node_count,
        .dim        = dim,
        .x          = NULL,
        .dx         = NULL,
    };
    return layer;
}

void relu_layer_free(ReluLayer **l)
{
    if (!(*l)) return;
    relu_free_node_buffers(*l);
    free((*l));
    *l = NULL;
}

void relu_layer_bind(ReluLayer *l, int64_t node_count)
{
    if (l->node_count < node_count)
    {
        relu_free_node_buffers(l);
        relu_alloc_node_buffers(l, node_count);
    }
    l->node_count = node_count;
}

// L2-NORMALIZE LAYER

static void l2norm_alloc_node_buffers(L2NormLayer *l, int64_t node_count)
{
    l->inv_norm = ALLOC_OR_DIE(alloc_local(node_count * sizeof(Real)));
}

static void l2norm_free_node_buffers(L2NormLayer *l)
{
    free(l->inv_norm);
}

void l2norm_layer_init(L2NormLayer *l)
{
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < l->node_count; n++)
        l->inv_norm[0] = 0.0;
}

L2NormLayer* l2norm_layer_alloc(int64_t node_count, int64_t dim)
{
    L2NormLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));
    *layer = (L2NormLayer) {
        .node_count = node_count,
        .dim        = dim,
        .x          = NULL,     // Set later when connecting layers
        .dx         = NULL,     // Set later when connecting layers
    };
    l2norm_alloc_node_buffers(layer, node_count);
    l2norm_layer_init(layer);
    return layer;
}

void l2norm_layer_free(L2NormLayer **l)
{
    if (!(*l)) return;
    l2norm_free_node_buffers(*l);
    free((*l));
    *l = NULL;
}

void l2norm_layer_bind(L2NormLayer *l, int64_t node_count)
{
    if (l->node_count < node_count)
    {
        l2norm_free_node_buffers(l);
        l2norm_alloc_node_buffers(l, node_count);
    }
    l->node_count = node_count;
}

// LOGSOFTMAX LAYER
static void logsoft_alloc_node_buffers(LogSoftmaxLayer *l, int64_t node_count)
{
    l->dx = ALLOC_OR_DIE(alloc_shared(node_count * l->dim * sizeof(Real)));
}

static void logsoft_free_node_buffers(LogSoftmaxLayer *l)
{
    free(l->dx);
}

LogSoftmaxLayer* logsoft_layer_alloc(int64_t node_count, int64_t dim)
{
    LogSoftmaxLayer *layer = ALLOC_OR_DIE(malloc(sizeof(*layer)));
    *layer = (LogSoftmaxLayer) {
        .node_count = node_count,
        .dim        = dim,
        .x          = NULL
    };
    logsoft_alloc_node_buffers(layer, node_count);
    real_zero_out(layer->dx, node_count * dim);
    return layer;
}

void logsoft_layer_free(LogSoftmaxLayer **l)
{
    if (!*l) return;
    logsoft_free_node_buffers(*l);
    free(*l);
    *l = NULL;
}

void logsoft_layer_bind(LogSoftmaxLayer *l, int64_t node_count)
{
    if (l->node_count < node_count)
    {
        logsoft_free_node_buffers(l);
        logsoft_alloc_node_buffers(l, node_count);
    }
    l->node_count = node_count;
}

// SAGENET

static void wireup_network(SageNet *net, int64_t layer_count, Dataset *ds)
{
    // Set first layer's input to the dataset features
    ((SageLayer*)net->layers[0].ctx)->x_in = ds->x;

    Real *current_x = ds->x;
    for (int64_t i = 0; i < layer_count; i++)
    {
        Layer layer = net->layers[i];
        switch (layer.type)
        {
            case LAYER_SAGE:
                ((SageLayer*)layer.ctx)->x_in = current_x;
                current_x = ((SageLayer*)layer.ctx)->x_out;
                break;
            case LAYER_RELU:
                ((ReluLayer*)layer.ctx)->x = current_x;
                break;
            case LAYER_L2NORM:
                ((L2NormLayer*)layer.ctx)->x = current_x;
                break;
            case LAYER_LOGSOFTMAX:
                ((LogSoftmaxLayer*)layer.ctx)->x = current_x;
                break;
            default:
                ERROR("Unknown layer type %d", layer.type);
        }
    }

    Real *current_dx = NULL;
    for (int64_t i = layer_count - 1; i >= 0; i--)
    {
        Layer layer = net->layers[i];
        switch (layer.type)
        {
            case LAYER_SAGE:
                ((SageLayer*)layer.ctx)->dx_out = current_dx;
                current_dx = ((SageLayer*)layer.ctx)->dx_in;
                break;
            case LAYER_RELU:
                ((ReluLayer*)layer.ctx)->dx = current_dx;
                break;
            case LAYER_L2NORM:
                ((L2NormLayer*)layer.ctx)->dx = current_dx;
                break;
            case LAYER_LOGSOFTMAX:
                current_dx = ((LogSoftmaxLayer*)layer.ctx)->dx;
                break;
            default:
                ERROR("Unknown layer type %d", layer.type);
        }
    }
}


SageNet* sage_net_alloc(LayerConf *conf, int64_t count, Dataset *ds, FlowDirection flow)
{
    SageNet *net = ALLOC_OR_DIE(malloc(sizeof(*net)));

    net->layer_count = count;
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
    for (int64_t i = 0; i < net->layer_count; i++)
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
            case LAYER_LOGSOFTMAX:
                logsoft_layer_bind(layer->ctx, ds->node_count);
                break;
            default:
                ERROR("Unknown layer type %d", layer->type);
        }
    }

    wireup_network(net, net->layer_count, ds);
}

void sage_net_reset_parameters(SageNet *net)
{
    for (int64_t i = 0; i < net->layer_count; i++)
    {
        Layer *layer = &net->layers[i];
        switch (layer->type)
        {
            case LAYER_SAGE:
                sage_layer_reset_parameters((SageLayer*)layer->ctx);
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

    for (int64_t i = 0; i < (*net)->layer_count; i++)
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

    for (int64_t i = 0; i < net->layer_count; i++)
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
            case LAYER_LOGSOFTMAX:
                printf("  (%zu) LogSoftmax()\n", i);
                break;
        }
    }

    printf(")\n");
}
