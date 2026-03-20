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

// Dispatchers
#define DISPATCH(name, Type, func)                              \
    static void name(Layer *self, Dataset *ds) {                \
        func((Type *)self->impl, ds);                           \
    }

#define DISPATCH_UPDATE(name, Type, func)                       \
    static void name(Layer *self, double lr, Dataset *ds) {     \
        func((Type *)self->impl, (float)lr, ds);                \
    }

// GraphSage
DISPATCH(sage_forward_dispatch,  SageLayer, sageconv)
DISPATCH(sage_backward_dispatch, SageLayer, sageconv_backward)
DISPATCH(sage_zero_grad_dispatch, SageLayer, sage_layer_zero_gradients)
DISPATCH(sage_destroy_dispatch,  SageLayer, sage_layer_destroy)
DISPATCH(sage_reset_dispatch,    SageLayer, sage_layer_reset)
DISPATCH_UPDATE(sage_update_dispatch, SageLayer, sage_layer_update_weights)

// ReLU
DISPATCH(relu_forward_dispatch,  ReluLayer, relu)
DISPATCH(relu_backward_dispatch, ReluLayer, relu_backward)
DISPATCH(relu_destroy_dispatch,  ReluLayer, relu_layer_destroy)
DISPATCH(relu_reset_dispatch,    ReluLayer, relu_layer_reset)

// L2 Normalization
DISPATCH(l2norm_forward_dispatch,  L2NormLayer, normalize)
DISPATCH(l2norm_backward_dispatch, L2NormLayer, normalize_backward)
DISPATCH(l2norm_destroy_dispatch,  L2NormLayer, l2norm_layer_destroy)
DISPATCH(l2norm_reset_dispatch,    L2NormLayer, l2norm_layer_reset)

// Linear projection
DISPATCH(linear_forward_dispatch,  LinearLayer, linear)
DISPATCH(linear_backward_dispatch, LinearLayer, linear_backward)
DISPATCH(linear_zero_grad_dispatch, LinearLayer, linear_layer_zero_gradients)
DISPATCH(linear_destroy_dispatch,  LinearLayer, linear_layer_destroy)
DISPATCH(linear_reset_dispatch,    LinearLayer, linear_layer_reset)
DISPATCH_UPDATE(linear_update_dispatch, LinearLayer, linear_layer_update_weights)

// LogSoftmax
DISPATCH(logsoft_forward_dispatch,  LogSoftLayer, logsoft)
DISPATCH(logsoft_backward_dispatch, LogSoftLayer, cross_entropy_backward)
DISPATCH(logsoft_destroy_dispatch,  LogSoftLayer, logsoft_layer_destroy)
DISPATCH(logsoft_reset_dispatch,    LogSoftLayer, logsoft_layer_reset)

SageNet* sage_net_create(LayerConf *conf, size_t count, Dataset *ds)
{
    SageNet *net = malloc(sizeof(*net));
    if (!net) ERROR("Could not allocate SageNet");

    net->num_layers = count;
    net->layers = malloc(count * sizeof(*net->layers));
    if (!net->layers) ERROR("Could not allocate layers");

    for (size_t i = 0; i < count; i++)
    {
        void *impl = NULL;
        switch (conf[i].type)
        {
        case LAYER_SAGE:
            impl = sage_layer_create(ds->num_nodes, conf[i].in_dim, conf[i].out_dim);

#ifdef VERBOSE_TIMERS
            ((SageLayer*)impl)->timer_dWroot = nob_temp_sprintf("L%zu_dWr:%zu->%zu", i, conf[i].in_dim, conf[i].out_dim);
            ((SageLayer*)impl)->timer_dWagg  = nob_temp_sprintf("L%zu_dWa:%zu->%zu", i, conf[i].in_dim, conf[i].out_dim);
            ((SageLayer*)impl)->timer_dinput = nob_temp_sprintf("L%zu_dIn:%zu", i, conf[i].in_dim);
            ((SageLayer*)impl)->timer_dneigh = nob_temp_sprintf("L%zu_dNe:%zu", i, conf[i].in_dim);
#else
            ((SageLayer*)impl)->timer_dWroot = "dWroot";
            ((SageLayer*)impl)->timer_dWagg  = "dWagg";
            ((SageLayer*)impl)->timer_dinput = "dinput";
            ((SageLayer*)impl)->timer_dneigh = "dneigh";
#endif
            net->layers[i] = (Layer){
				.type            = LAYER_SAGE,
                .impl            = impl,
                .input_ptr       = &((SageLayer*)impl)->input,
                .grad_output_ptr = &((SageLayer*)impl)->grad_output,
                .output_ptr      = &((SageLayer*)impl)->output,
                .grad_input_ptr  = &((SageLayer*)impl)->grad_input,
                .forward         = sage_forward_dispatch,
                .backward        = sage_backward_dispatch,
                .update          = sage_update_dispatch,
                .zero_grad       = sage_zero_grad_dispatch,
                .reset           = sage_reset_dispatch,
                .destroy         = sage_destroy_dispatch,
            };
            break;
        case LAYER_RELU:
            impl = (void*)relu_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_RELU,
                .impl            = impl,
                .input_ptr       = &((ReluLayer*)impl)->input,
                .grad_output_ptr = &((ReluLayer*)impl)->grad_output,
                .output_ptr      = &((ReluLayer*)impl)->output,
                .grad_input_ptr  = &((ReluLayer*)impl)->grad_input,
                .forward         = relu_forward_dispatch,
                .backward        = relu_backward_dispatch,
                .update          = NULL,
                .zero_grad       = NULL,
                .reset           = relu_reset_dispatch,
                .destroy         = relu_destroy_dispatch,
            };
            break;
        case LAYER_L2NORM:
            impl = (void*)l2norm_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_L2NORM,
                .impl            = impl,
                .input_ptr       = &((L2NormLayer*)impl)->input,
                .grad_output_ptr = &((L2NormLayer*)impl)->grad_output,
                .output_ptr      = &((L2NormLayer*)impl)->output,
                .grad_input_ptr  = &((L2NormLayer*)impl)->grad_input,
                .forward         = l2norm_forward_dispatch,
                .backward        = l2norm_backward_dispatch,
                .update          = NULL,
                .zero_grad       = NULL,
                .reset           = l2norm_reset_dispatch,
                .destroy         = l2norm_destroy_dispatch,
            };
            break;
        case LAYER_LINEAR:
            impl = (void*)linear_layer_create(ds->num_nodes, conf[i].in_dim, conf[i].out_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_LINEAR,
                .impl            = impl,
                .input_ptr       = &((LinearLayer*)impl)->input,
                .grad_output_ptr = &((LinearLayer*)impl)->grad_output,
                .output_ptr      = &((LinearLayer*)impl)->output,
                .grad_input_ptr  = &((LinearLayer*)impl)->grad_input,
                .forward         = linear_forward_dispatch,
                .backward        = linear_backward_dispatch,
                .update          = linear_update_dispatch,
                .zero_grad       = linear_zero_grad_dispatch,
                .reset           = linear_reset_dispatch,
                .destroy         = linear_destroy_dispatch,
            };
            break;
        case LAYER_LOGSOFT:
            impl = (void*)logsoft_layer_create(ds->num_nodes, conf[i].in_dim);
            net->layers[i] = (Layer){
				.type            = LAYER_LOGSOFT,
                .impl            = impl,
                .input_ptr       = &((LogSoftLayer*)impl)->input,
                .grad_output_ptr = NULL,
                .output_ptr      = &((LogSoftLayer*)impl)->output,
                .grad_input_ptr  = &((LogSoftLayer*)impl)->grad_input,
                .forward         = logsoft_forward_dispatch,
                .backward        = logsoft_backward_dispatch,
                .update          = NULL,
                .zero_grad       = NULL,
                .reset           = logsoft_reset_dispatch,
                .destroy         = logsoft_destroy_dispatch,
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

void sage_net_reset(const SageNet *net, Dataset *ds)
{
    for (size_t i = 0; i < net->num_layers; i++)
	{
        net->layers[i].reset(&net->layers[i], ds);
    }
}

// We don't free matrices that are references from other layers, i.e input and grad_output
void sage_layer_destroy(SageLayer *l, Dataset *ds)
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

void relu_layer_destroy(ReluLayer *l, Dataset *ds)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

void l2norm_layer_destroy(L2NormLayer *l, Dataset *ds)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);
    if (l->recip_mag) matrix_destroy(l->recip_mag);

    free(l);
}

void linear_layer_destroy(LinearLayer *l, Dataset *ds)
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

void logsoft_layer_destroy(LogSoftLayer *l, Dataset *ds)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

void sage_net_destroy(SageNet *net, Dataset *ds)
{
    if (!net) return;

    // Free the input matrix (owned by the network, not any layer)
    if (net->num_layers > 0 && net->layers[0].input_ptr)
	{
        matrix_destroy(*net->layers[0].input_ptr);
	}

    for (size_t i = 0; i < net->num_layers; i++)
	{
        if (net->layers[i].destroy) net->layers[i].destroy(&net->layers[i], ds);
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
                   ((SageLayer *)l->impl)->in_dim, ((SageLayer *)l->impl)->out_dim);
            break;
        case LAYER_RELU:
            printf("  (%zu) ReLU()\n", i);
            break;
        case LAYER_L2NORM:
            printf("  (%zu) L2Norm()\n", i);
            break;
        case LAYER_LINEAR:
            printf("  (%zu) Linear(%zu, %zu)\n", i,
                   ((LinearLayer *)l->impl)->in_dim, ((LinearLayer *)l->impl)->out_dim);
            break;
        case LAYER_LOGSOFT:
            printf("  (%zu) LogSoftmax(dim=1)\n", i);
            break;
        }
    }

    printf(")\n");
}
