#include <stdlib.h>

#include "layers.h"
#include "matrix.h"

// Init helpers
SageLayer* init_sage_layer(size_t n_nodes, size_t in_dim, size_t out_dim)
{
    SageLayer *layer = malloc(sizeof(*layer));

    *layer = (SageLayer){
        .input       = NULL,    // Set later when connecting layers
        .output      = mat_create(out_dim, n_nodes),
        .agg         = mat_create(in_dim, n_nodes),
        .Wagg        = mat_create(out_dim, in_dim),
        .Wroot       = mat_create(out_dim, in_dim),
        .grad_input  = NULL,    // Set later when connecting layers
        .grad_output = mat_create(out_dim, n_nodes),
        .grad_Wagg   = mat_create(out_dim, in_dim),
        .grad_Wroot  = mat_create(out_dim, in_dim)
    };

    // Initialize weights randomly
    mat_rand(layer->Wagg, -1.0, 1.0);
    mat_rand(layer->Wroot, -1.0, 1.0);

    return layer;
}

ReluLayer* init_relu_layer(size_t n_nodes, size_t dim)
{
    ReluLayer *layer = malloc(sizeof(*layer));

    *layer = (ReluLayer) {
        .input       = NULL,    // Set later when connecting layers,
        .output      = mat_create(dim, n_nodes),
        .grad_input  = NULL,    // Set later when connecting layers,
        .grad_output = mat_create(dim, n_nodes),
    };

    return layer;
}

L2NormLayer* init_l2norm_layer(size_t n_nodes, size_t dim)
{
    L2NormLayer *layer = malloc(sizeof(*layer));

    *layer = (L2NormLayer) {
        .input       = NULL,    // Set later when connecting layers,
        .output      = mat_create(dim, n_nodes),
        .grad_input  = NULL,    // Set later when connecting layers,
        .grad_output = mat_create(dim, n_nodes),
    };

    return layer;
}

LinearLayer* init_linear_layer(size_t n_nodes, size_t in_dim, size_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));

    *layer = (LinearLayer) {
        .input       = NULL,    // Set later when connecting layers
        .output      = mat_create(out_dim, n_nodes),
        .W           = mat_create(out_dim, in_dim),
        .bias        = mat_create(out_dim, 1),
        .grad_output = mat_create(out_dim, n_nodes),
        .grad_input  = NULL,    // Set later when connecting layers
        .grad_W      = mat_create(out_dim, in_dim),
        .grad_bias   = mat_create(out_dim, 1),
    };

    return layer;
}


LogSoftLayer* init_logsoft_layer(size_t n_nodes, size_t out_dim)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));

    *layer = (LogSoftLayer) {
        .input       = NULL,    // Set later when connecting layers
        .output      = mat_create(out_dim, n_nodes),
        .grad_output = mat_create(out_dim, n_nodes),
    };

    return layer;
}
