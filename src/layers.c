#include <stdlib.h>

#include "core.h"
#include "layers.h"
#include "matrix.h"
#include "graph.h"

// Init helpers

SageNet* init_sage_net(size_t k_layers, size_t hidden_dim, size_t output_dim, graph_t *g)
{
    SageNet *net = malloc(sizeof(*net));
    net->num_layers = k_layers;
    net->sagelayer      = malloc(k_layers * sizeof(*net->sagelayer));
    net->relulayer      = malloc(k_layers * sizeof(*net->relulayer));
    net->normalizelayer = malloc(k_layers * sizeof(*net->normalizelayer));

    size_t batch_size = g->num_nodes;
    size_t num_features = g->num_node_features;

    // First layer: num_features -> hidden_dim
    net->sagelayer[0] = init_sage_layer(batch_size, num_features, hidden_dim);
    net->relulayer[0] = init_relu_layer(batch_size, hidden_dim);
    net->normalizelayer[0] = init_l2norm_layer(batch_size, hidden_dim);

    // Middle layers: hidden_dim -> hidden_dim
    for (size_t k = 1; k < k_layers - 1; k++) {
        net->sagelayer[k] = init_sage_layer(batch_size, hidden_dim, hidden_dim);
        net->relulayer[k] = init_relu_layer(batch_size, hidden_dim);
        net->normalizelayer[k] = init_l2norm_layer(batch_size, hidden_dim);
    }

    // Last layer: hidden_dim -> output_dim
    if (k_layers > 1) {
        net->sagelayer[k_layers - 1] = init_sage_layer(batch_size, hidden_dim, output_dim);
        net->relulayer[k_layers - 1] = init_relu_layer(batch_size, output_dim);
        net->normalizelayer[k_layers - 1] = init_l2norm_layer(batch_size, output_dim);
    }

    net->sagelayer[0]->input = g->x;

    return net;
}

SageLayer* init_sage_layer(size_t batch_size, size_t in_dim, size_t out_dim)
{
    SageLayer *layer = malloc(sizeof(*layer));

    *layer = (SageLayer){
        .input       = NULL, // Set later when connecting layers
        .output      = MAT_CREATE(batch_size, out_dim),
        .agg         = MAT_CREATE(batch_size, in_dim),
        .Wagg        = MAT_CREATE(in_dim, out_dim),
        .Wroot       = MAT_CREATE(in_dim, out_dim),
        .grad_input  = MAT_CREATE(batch_size, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_Wagg   = MAT_CREATE(in_dim, out_dim),
        .grad_Wroot  = MAT_CREATE(in_dim, out_dim),
    };
    layer->mean_scale = malloc(batch_size * sizeof(*layer->mean_scale));

    // Initialize weights randomly
    mat_rand(layer->Wagg, -1.0, 1.0);
    mat_rand(layer->Wroot, -1.0, 1.0);

    return layer;
}

ReluLayer* init_relu_layer(size_t batch_size, size_t dim)
{
    ReluLayer *layer = malloc(sizeof(*layer));

    *layer = (ReluLayer) {
        .input       = NULL, // Set later when connecting layers,
        .output      = MAT_CREATE(batch_size, dim),
        .grad_input  = MAT_CREATE(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
    };

    return layer;
}

NormalizeLayer* init_l2norm_layer(size_t batch_size, size_t dim)
{
    NormalizeLayer *layer = malloc(sizeof(*layer));

    *layer = (NormalizeLayer) {
        .input       = NULL, // Set later when connecting layers,
        .output      = MAT_CREATE(batch_size, dim),
        .grad_input  = MAT_CREATE(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
        .recip_mag   = MAT_CREATE(batch_size, 1)
    };

    return layer;
}

LinearLayer* init_linear_layer(size_t batch_size, size_t in_dim, size_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));

    *layer = (LinearLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = MAT_CREATE(batch_size, out_dim),
        .W           = MAT_CREATE(in_dim, out_dim),
        .bias        = MAT_CREATE(1, out_dim),
        .grad_input  = MAT_CREATE(batch_size, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_W      = MAT_CREATE(in_dim, out_dim),
        .grad_bias   = MAT_CREATE(1, out_dim),
    };

    mat_rand(layer->W, -1.0, 1.0);
    mat_rand(layer->bias, -1.0, 1.0);

    return layer;
}


LogSoftLayer* init_logsoft_layer(size_t batch_size, size_t dim)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));

    *layer = (LogSoftLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = MAT_CREATE(batch_size, dim),
        .grad_input  = MAT_CREATE(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers
    };

    return layer;
}

// We don't free matrices that are references from other layers, i.e input and grad_output
void destroy_sage_net(SageNet *n)
{
    for (size_t i = 0; i < n->num_layers; i++) {
        destroy_l2norm_layer(n->normalizelayer[i]);
        destroy_relu_layer(n->relulayer[i]);
        destroy_sage_layer(n->sagelayer[i]);
    }
    free(n->relulayer);
    free(n->normalizelayer);
    free(n->sagelayer);
    free(n);
}

void destroy_sage_layer(SageLayer* l)
{
    mat_destroy(l->output);
    mat_destroy(l->agg);
    mat_destroy(l->Wagg);
    mat_destroy(l->Wroot);
    mat_destroy(l->grad_input);
    mat_destroy(l->grad_Wagg);
    mat_destroy(l->grad_Wroot);
    free(l->mean_scale);
    free(l);
}

void destroy_relu_layer(ReluLayer* l)
{
    mat_destroy(l->output);
    mat_destroy(l->grad_input);
    free(l);
}

void destroy_l2norm_layer(NormalizeLayer* l)
{
    mat_destroy(l->output);
    mat_destroy(l->grad_input);
    mat_destroy(l->recip_mag);
    free(l);
}

void destroy_linear_layer(LinearLayer *l)
{
    mat_destroy(l->output);
    mat_destroy(l->W);
    mat_destroy(l->bias);
    mat_destroy(l->grad_input);
    mat_destroy(l->grad_W);
    mat_destroy(l->grad_bias);
    free(l);
}

void destroy_logsoft_layer(LogSoftLayer *l)
{
    mat_destroy(l->output);
    mat_destroy(l->grad_input);
    free(l);
}


// Inspect helpers
void sage_layer_info(const SageLayer* const l)
{
    printf("\nSAGE LAYER\n");
    printf("========================================\n");
    printf("output = input * Wroot + agg  * Wagg\n");
    printf("%-6s = %-5s * %-5s + %-4s * %s\n",
           mat_shape((l)->output), mat_shape((l)->input),
           mat_shape((l)->Wroot), mat_shape((l)->agg),
           mat_shape((l)->Wagg));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           mat_shape((l)->grad_output), mat_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           mat_shape((l)->grad_input), mat_shape((l)->input));
    printf("grad_Wagg   = Wagg\n");
    printf("   %-6s   = %s\n",
           mat_shape((l)->grad_Wagg), mat_shape((l)->Wagg));
    printf("grad_Wroot  = Wroot\n");
    printf("   %-7s  = %s\n",
           mat_shape((l)->grad_Wroot), mat_shape((l)->Wroot));
    printf("========================================\n");
}

void relu_layer_info(const ReluLayer* const l)
{
    printf("\nRELU LAYER\n");
    printf("========================================\n");
    printf("output = relu(input)\n");
    printf("%-6s = relu(%-5s)\n",
           mat_shape((l)->output), mat_shape((l)->input));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           mat_shape((l)->grad_output), mat_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           mat_shape((l)->grad_input), mat_shape((l)->input));
    printf("========================================\n");
}

void normalize_layer_info(const NormalizeLayer* const l)
{
    printf("\nNORMALIZE LAYER\n");
    printf("========================================\n");
    printf("output = l2norm(input)\n");
    printf("%-6s = l2norm(%-5s)\n",
           mat_shape((l)->output), mat_shape((l)->input));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           mat_shape((l)->grad_output), mat_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           mat_shape((l)->grad_input), mat_shape((l)->input));
    printf("========================================\n");
}

void sage_net_layers_info(const SageNet* const n)
{
    for (size_t i = 0; i < n->num_layers; i++) {
        sage_layer_info(n->sagelayer[i]);
        relu_layer_info(n->relulayer[i]);
        normalize_layer_info(n->normalizelayer[i]);
    }
}

void linear_layer_info(const LinearLayer* const l)
{
    printf("\nLINEAR LAYER\n");
    printf("========================================\n");
    printf("output = input *   W   + bias\n");
    printf("%-6s = %-5s * %-5s + %s\n",
           mat_shape((l)->output), mat_shape((l)->input),
           mat_shape((l)->W), mat_shape((l)->bias));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           mat_shape((l)->grad_output), mat_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           mat_shape((l)->grad_input), mat_shape((l)->input));
    printf("grad_W      =   W\n");
    printf("   %-6s   = %s\n",
           mat_shape((l)->grad_W), mat_shape((l)->W));
    printf("grad_bias   = bias\n");
    printf("   %-6s   = %s\n",
           mat_shape((l)->grad_bias), mat_shape((l)->bias));
    printf("========================================\n");
}

void logsoft_layer_info(const LogSoftLayer* const l)
{
    printf("\nLOGSOFT LAYER\n");
    printf("========================================\n");
    printf("output = log_softmax(input)\n");
    printf("%-6s = log_softmax(%-5s)\n",
           mat_shape((l)->output), mat_shape((l)->input));
    printf("----------------------------------------\n");
    printf("grad_input  = input\n");
    printf("   %-8s = %s\n",
           mat_shape((l)->grad_input), mat_shape((l)->input));
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           mat_shape((l)->grad_output), mat_shape((l)->output));
    printf("========================================\n");
}
