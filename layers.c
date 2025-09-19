#include <stdlib.h>

#include "layers.h"
#include "matrix.h"

// Init helpers
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
        .grad_Wroot  = MAT_CREATE(in_dim, out_dim)
    };

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
        .bias        = NULL, // MAT_CREATE(1, out_dim),
        .grad_input  = MAT_CREATE(batch_size, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_W      = MAT_CREATE(in_dim, out_dim),
        .grad_bias   = NULL, //MAT_CREATE(1, out_dim),
    };

    mat_rand(layer->W, -1.0, 1.0);
    // mat_rand(layer->bias, -1.0, 1.0);

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
