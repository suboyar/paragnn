#include <stdlib.h>

#include "core.h"
#include "layers.h"
#include "dataset.h"
#include "timer.h"

static inline void fill_xavier_uniform(Real *x, int64_t in, int64_t out)
{
    const Real limit = real_sqrt(REAL(6.0) / (in + out));
    const Real recip_rand_max = REAL(1.0) / REAL(RAND_MAX);

    // OpenMP can't be used here as rand() isn't thread-safe, variants that might
    // be of interest are srand48_r or random_r. This can be looked more closely if this
    // function ever takes more too much time.
    for (int64_t i = 0; i < in * out; i++)
    {
        x[i] = limit * (REAL(2.0) * REAL(rand()) * recip_rand_max - REAL(1.0));
    }
}

// SAGE LAYER

static void sage_alloc_node_buffers(SageLayer *l, uint32_t num_nodes)
{
    l->output       = cache_aligned_alloc(num_nodes * l->out_dim * sizeof(Real));
    l->agg          = cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real));
    l->grad_input   = cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real));
    l->grad_Wagg    = cache_aligned_alloc(l->in_dim * l->out_dim * sizeof(Real));
    l->grad_Wroot   = cache_aligned_alloc(l->in_dim * l->out_dim * sizeof(Real));
    // l->tls_dW       = cache_aligned_alloc(2 * nthreads * l->in_dim * l->out_dim * sizeof(Real));
    l->grad_scatter = cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real));

    if (!l->output || !l->agg || !l->grad_input || !l->grad_Wagg || !l->grad_Wroot || !l->grad_scatter)
    {
        ERROR("Could not allocate SageLayer buffers");
    }
}

SageLayer* sage_layer_create(int64_t num_nodes, int64_t num_edges, Edges edges, int64_t in_dim, int64_t out_dim, FlowDirection flow)
{
    SageLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate SageLayer");
    *layer = (SageLayer) {
        .num_nodes    = num_nodes,
        .num_edges    = num_edges,
        .edges        = edges,
        .in_dim       = in_dim,
        .out_dim      = out_dim,
        .flow         = flow,
        .input        = NULL,   // Set later when connecting layer
        .grad_output  = NULL,   // Set later when connecting layer
        .Wagg         = cache_aligned_alloc(in_dim * out_dim * sizeof(Real)),
        .Wroot        = cache_aligned_alloc(in_dim * out_dim * sizeof(Real)),
    };

    if (!layer->Wagg || !layer->Wroot)
    {
        ERROR("Could not allocate SageLayer parameters");
    }

    sage_alloc_node_buffers(layer, num_nodes);

    // Initialize weights randomly
    fill_xavier_uniform(layer->Wroot, in_dim, out_dim);
    fill_xavier_uniform(layer->Wagg, in_dim, out_dim);

    return layer;
}

static void sage_free_node_buffers(SageLayer *l)
{
    free(l->output);       l->output       = NULL;
    free(l->agg);          l->agg          = NULL;
    free(l->grad_input);   l->grad_input   = NULL;
    free(l->grad_Wagg);    l->grad_Wagg    = NULL;
    free(l->grad_Wroot);   l->grad_Wroot   = NULL;
    free(l->grad_scatter); l->grad_scatter = NULL;
}

void sage_layer_free(SageLayer **l)
{
    if (!(*l)) return;

    sage_free_node_buffers(*l);
    free((*l)->Wagg);  (*l)->Wagg  = NULL;
    free((*l)->Wroot); (*l)->Wroot = NULL;
    free((*l));
    *l = NULL;
}

void sage_layer_bind(SageLayer *l, int64_t num_nodes, int64_t num_edges, Edges edges)
{
    if (l->num_nodes < num_nodes)
    {
        sage_free_node_buffers(l);
        sage_alloc_node_buffers(l, num_nodes);
    }
    l->num_nodes = num_nodes;
    l->num_edges = num_edges;
    l->edges     = edges;
}

// RELU LAYER

static void relu_alloc_node_buffers(ReluLayer *l, int64_t num_nodes)
{
    l->output      = cache_aligned_alloc(num_nodes * l->dim * sizeof(Real));
    l->grad_input  = cache_aligned_alloc(num_nodes * l->dim * sizeof(Real));

    if (!l->output || !l->grad_input)
    {
        ERROR("Could not allocate ReluLayer buffers");
    }
}

ReluLayer* relu_layer_create(int64_t num_nodes, int64_t dim)
{
    ReluLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate ReluLayer");

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
    free(l->output);       l->output       = NULL;
    free(l->grad_input);   l->grad_input   = NULL;
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
    l->output      = cache_aligned_alloc(num_nodes * l->dim * sizeof(Real));
    l->grad_input  = cache_aligned_alloc(num_nodes * l->dim * sizeof(Real));
    l->recip_mag   = cache_aligned_alloc(num_nodes * sizeof(Real));
    if (!l->output || !l->grad_input || !l->recip_mag)
    {
        ERROR("Could not allocate L2NormLayer buffers");
    }
}

L2NormLayer* l2norm_layer_create(int64_t num_nodes, int64_t dim)
{
    L2NormLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate L2NormLayer");

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
    free(l->output);     l->output     = NULL;
    free(l->grad_input); l->grad_input = NULL;
    free(l->recip_mag);  l->recip_mag  = NULL;
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
    l->output      = cache_aligned_alloc(num_nodes * l->out_dim * sizeof(Real));
    l->grad_input  = cache_aligned_alloc(num_nodes * l->in_dim * sizeof(Real));
    l->grad_W      = cache_aligned_alloc(l->in_dim * l->out_dim * sizeof(Real));
    l->grad_bias   = cache_aligned_alloc(l->out_dim * sizeof(Real));

    if (!l->output || !l->grad_input || !l->grad_W || l->grad_bias)
    {
        ERROR("Could not allocate LinearLayer buffers");
    }
}

LinearLayer* linear_layer_create(int64_t num_nodes, int64_t in_dim, int64_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LinearLayer");

    *layer = (LinearLayer) {
        .num_nodes   = num_nodes,
        .in_dim      = in_dim,
        .out_dim     = out_dim,
        .input       = NULL, // Set later when connecting layers
        .grad_output = NULL, // Set later when connecting layers
        .W           = cache_aligned_alloc(in_dim * out_dim * sizeof(Real)),
        .bias        = cache_aligned_alloc(out_dim * sizeof(Real)),
    };

    linear_alloc_node_buffers(layer, num_nodes);

    fill_xavier_uniform(layer->W, in_dim, out_dim);
    fill_xavier_uniform(layer->bias, 1, out_dim);

    return layer;
}

static void linear_free_node_buffers(LinearLayer *l)
{
    free(l->output);     l->output     = NULL;
    free(l->grad_input); l->grad_input = NULL;
    free(l->grad_W);     l->grad_W     = NULL;
    free(l->grad_bias);  l->grad_bias  = NULL;
}

void linear_layer_free(LinearLayer **l)
{
    if (!*l) return;

    linear_free_node_buffers(*l);
    free((*l)->W);    (*l)->W    = NULL;
    free((*l)->bias); (*l)->bias = NULL;
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
    l->output      = cache_aligned_alloc(num_nodes * l->dim * sizeof(Real));
    l->grad_input  = cache_aligned_alloc(num_nodes * l->dim * sizeof(Real));

    if (!l->output || !l->grad_input)
    {
        ERROR("Could not allocate L2NormLayer buffers");
    }
}

LogSoftmaxLayer* logsoft_layer_create(int64_t num_nodes, int64_t dim)
{
    LogSoftmaxLayer *layer = malloc(sizeof(*layer));
    if (!layer) ERROR("Could not allocate LogSoftmaxLayer");

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
    free(l->output);     l->output     = NULL;
    free(l->grad_input); l->grad_input = NULL;
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


SageNet* sage_net_create(LayerConf *conf, int64_t count, Dataset *ds, FlowDirection flow)
{
    SageNet *net = malloc(sizeof(*net));
    if (!net) ERROR("Could not allocate SageNet");

    net->num_layers = count;
    // NOTE: we calloc here to make gcc not complain
    net->layers = calloc(count,sizeof(*net->layers));
    if (!net->layers) ERROR("Could not allocate layers");

    for (int64_t i = 0; i < count; i++)
    {
        void *ctx = NULL;
        switch (conf[i].type)
        {
        case LAYER_SAGE:
            ctx = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, conf[i].in_dim, conf[i].out_dim, flow);
            net->layers[i] = (Layer){
				.type            = LAYER_SAGE,
                .ctx             = ctx,
            };
            break;
        case LAYER_RELU:
            ctx = (void*)relu_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_RELU,
                .ctx             = ctx,
            };
            break;
        case LAYER_L2NORM:
            ctx = (void*)l2norm_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_L2NORM,
                .ctx             = ctx,
            };
            break;
        case LAYER_LINEAR:
            ctx = (void*)linear_layer_create(ds->num_nodes, conf[i].in_dim, conf[i].out_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_LINEAR,
                .ctx             = ctx,
            };
            break;
        case LAYER_LOGSOFTMAX:
            ctx = (void*)logsoft_layer_create(ds->num_nodes, conf[i].in_dim);
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
            sage_layer_bind(layer->ctx, ds->num_nodes, ds->num_edges, ds->edges);
            break;
        case LAYER_RELU:
            relu_layer_bind(layer->ctx, ds->num_nodes);
            break;
        case LAYER_L2NORM:
            l2norm_layer_bind(layer->ctx, ds->num_nodes);
            break;
        case LAYER_LINEAR:
            linear_layer_bind(layer->ctx, ds->num_nodes);
            break;
        case LAYER_LOGSOFTMAX:
            logsoft_layer_bind(layer->ctx, ds->num_nodes);
            break;
        default:
            ERROR("Unknown layer type %d", layer->type);
        }
    }

    wireup_network(net, net->num_layers, ds);
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
