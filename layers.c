#include <stdlib.h>

#include "layers.h"
#include "matrix.h"

// Init helpers
SageLayer* init_sage_layer(size_t n_nodes, size_t in_dim, size_t out_dim)
{
    SageLayer *layer = malloc(sizeof(*layer));

    *layer = (SageLayer){
        .input       = NULL, // Set later when connecting layers
        .output      = MAT_CREATE(n_nodes, out_dim),
        .agg         = MAT_CREATE(n_nodes, in_dim),
        .Wagg        = MAT_CREATE(in_dim, out_dim),
        .Wroot       = MAT_CREATE(in_dim, out_dim),
        .grad_input  = MAT_CREATE(n_nodes, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_Wagg   = MAT_CREATE(in_dim, out_dim),
        .grad_Wroot  = MAT_CREATE(in_dim, out_dim)
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
        .input       = NULL, // Set later when connecting layers,
        .output      = MAT_CREATE(n_nodes, dim),
        .grad_input  = MAT_CREATE(n_nodes, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, n_nodes),
    };

    return layer;
}

NormalizeLayer* init_l2norm_layer(size_t n_nodes, size_t dim)
{
    NormalizeLayer *layer = malloc(sizeof(*layer));

    *layer = (NormalizeLayer) {
        .input       = NULL, // Set later when connecting layers,
        .output      = MAT_CREATE(n_nodes, dim),
        .grad_input  = MAT_CREATE(n_nodes, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, n_nodes),
    };

    return layer;
}

LinearLayer* init_linear_layer(size_t n_nodes, size_t in_dim, size_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));

    *layer = (LinearLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = MAT_CREATE(n_nodes, out_dim),
        .W           = MAT_CREATE(in_dim, out_dim),
        .bias        = NULL, // MAT_CREATE(1, out_dim),
        .grad_input  = MAT_CREATE(n_nodes, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_W      = MAT_CREATE(in_dim, out_dim),
        .grad_bias   = MAT_CREATE(1, out_dim),
    };

    mat_rand(layer->W, -1.0, 1.0);
    // mat_rand(layer->bias, -1.0, 1.0);

    return layer;
}


LogSoftLayer* init_logsoft_layer(size_t n_nodes, size_t dim)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));

    *layer = (LogSoftLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = MAT_CREATE(n_nodes, dim),
        .grad_input  = MAT_CREATE(n_nodes, dim),
        .grad_output = NULL, // Set later when connecting layers
    };

    return layer;
}
