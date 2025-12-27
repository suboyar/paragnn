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
        .output      = matrix_create(batch_size, out_dim),
        .agg         = matrix_create(batch_size, in_dim),
        .Wagg        = matrix_create(in_dim, out_dim),
        .Wroot       = matrix_create(in_dim, out_dim),
        .grad_input  = matrix_create(batch_size, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_Wagg   = matrix_create(in_dim, out_dim),
        .grad_Wroot  = matrix_create(in_dim, out_dim),
    };
    layer->mean_scale = malloc(batch_size * sizeof(*layer->mean_scale));

    // Initialize weights randomly
    matrix_fill_random(layer->Wagg, -1.0, 1.0);
    matrix_fill_random(layer->Wroot, -1.0, 1.0);

    return layer;
}

ReluLayer* init_relu_layer(size_t batch_size, size_t dim)
{
    ReluLayer *layer = malloc(sizeof(*layer));

    *layer = (ReluLayer) {
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
    };

    return layer;
}

NormalizeLayer* init_l2norm_layer(size_t batch_size, size_t dim)
{
    NormalizeLayer *layer = malloc(sizeof(*layer));

    *layer = (NormalizeLayer) {
        .input       = NULL, // Set later when connecting layers,
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
        .grad_output = NULL, // Set later when connecting layers, mat_create(dim, batch_size),
        .recip_mag   = matrix_create(batch_size, 1)
    };

    return layer;
}

LinearLayer* init_linear_layer(size_t batch_size, size_t in_dim, size_t out_dim)
{
    LinearLayer *layer = malloc(sizeof(*layer));

    *layer = (LinearLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(batch_size, out_dim),
        .W           = matrix_create(in_dim, out_dim),
        .bias        = matrix_create(1, out_dim),
        .grad_input  = matrix_create(batch_size, in_dim),
        .grad_output = NULL, // Set later when connecting layers
        .grad_W      = matrix_create(in_dim, out_dim),
        .grad_bias   = matrix_create(1, out_dim),
    };

    matrix_fill_random(layer->W, -1.0, 1.0);
    matrix_fill_random(layer->bias, -1.0, 1.0);

    return layer;
}


LogSoftLayer* init_logsoft_layer(size_t batch_size, size_t dim)
{
    LogSoftLayer *layer = malloc(sizeof(*layer));

    *layer = (LogSoftLayer) {
        .input       = NULL, // Set later when connecting layers
        .output      = matrix_create(batch_size, dim),
        .grad_input  = matrix_create(batch_size, dim),
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

void destroy_relu_layer(ReluLayer* l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

void destroy_l2norm_layer(NormalizeLayer* l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);
    if (l->recip_mag) matrix_destroy(l->recip_mag);

    free(l);
}

void destroy_linear_layer(LinearLayer *l)
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
void destroy_logsoft_layer(LogSoftLayer *l)
{
    if (!l) return;

    if (l->output) matrix_destroy(l->output);
    if (l->grad_input) matrix_destroy(l->grad_input);

    free(l);
}

// Update weights
void linear_layer_update_weights(LinearLayer* const l, float lr)
{
    if(l->W->M != l->grad_W->M) {
        nob_log(NOB_ERROR, "Expected M to be the same");
        abort();
    }

    if(l->W->N != l->grad_W->N) {
        nob_log(NOB_ERROR, "Expected N to be the same");
        abort();
    }

    size_t M = l->W->M;
    size_t N = l->W->N;

    double *restrict A = l->W->data;
    size_t lda = l->W->stride;
    double *restrict B = l->grad_W->data;
    size_t ldb = l->grad_W->stride;

    double batch_recip = (double) 1/l->input->batch;

    // We are doing axpy: https://www.netlib.org/lapack/explore-html/d5/d4b/group__axpy.html
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i*lda+j] -= batch_recip * lr * B[i*ldb+j];
        }
    }

    if (l->grad_bias != NULL) {
        assert(l->bias->N == N);
        double *restrict C = l->bias->data;
        double *restrict D = l->grad_bias->data;
        for (size_t i = 0; i < N; i++) {
            C[i] -= batch_recip * lr * D[i];
        }
    }

    nob_log(NOB_INFO, "update_linear_weights: ok");
}

void sage_layer_update_weights(SageLayer* const l, float lr)
{
    if(l->Wroot->M != l->grad_Wroot->M) {
        nob_log(NOB_ERROR, "Expected M to be the same");
        abort();
    }

    if(l->Wroot->N != l->grad_Wroot->N) {
        nob_log(NOB_ERROR, "Expected N to be the same");
        abort();
    }

    if(l->Wagg->M != l->grad_Wagg->M) {
        nob_log(NOB_ERROR, "Expected M to be the same");
        abort();
    }

    if(l->Wagg->N != l->grad_Wagg->N) {
        nob_log(NOB_ERROR, "Expected N to be the same");
        abort();
    }


    double *restrict A = l->Wroot->data;
    size_t lda = l->Wroot->stride;
    double *restrict B = l->grad_Wroot->data;
    size_t ldb = l->grad_Wroot->stride;

    double *restrict C = l->Wagg->data;
    size_t ldc = l->Wagg->stride;
    double *restrict D = l->grad_Wagg->data;
    size_t ldd = l->grad_Wagg->stride;

    float batch_recip = (float) 1/l->input->batch;
    for (size_t i = 0; i < l->Wroot->M; i++) {
        for (size_t j = 0; j < l->Wroot->N; j++) {
            A[i*lda+j] -= batch_recip * lr * B[i*ldb+j];
            C[i*ldc+j] -= batch_recip * lr * D[i*ldd+j];
        }
    }

    nob_log(NOB_INFO, "update_sageconv_weights: ok");
}

// Reset gradient
void sage_layer_zero_gradients(SageLayer* l)
{
    if (!l) return;

    if (l->grad_output) matrix_zero(l->grad_output);
    if (l->grad_input) matrix_zero(l->grad_input);
    if (l->grad_Wagg) matrix_zero(l->grad_Wagg);
    if (l->grad_Wroot) matrix_zero(l->grad_Wroot);
}

void relu_layer_zero_gradients(ReluLayer* l)
{
    if (!l) return;

    if (l->grad_output) matrix_zero(l->grad_output);
    if (l->grad_input) matrix_zero(l->grad_input);
}

void normalize_layer_zero_gradients(NormalizeLayer* l)
{
    if (!l) return;

    if (l->grad_output) matrix_zero(l->grad_output);
    if (l->grad_input) matrix_zero(l->grad_input);
}

void linear_layer_zero_gradients(LinearLayer* l)
{
    if (!l) return;

    if (l->grad_output) matrix_zero(l->grad_output);
    if (l->grad_input) matrix_zero(l->grad_input);
    if (l->grad_W) matrix_zero(l->grad_W);
    if (l->grad_bias) matrix_zero(l->grad_bias);
}

void logsoft_layer_zero_gradients(LogSoftLayer* l)
{
    if (!l) return;

    if (l->grad_output) matrix_zero(l->grad_output);
    if (l->grad_input) matrix_zero(l->grad_input);
}

// Network-wide gradient reset
void sage_net_zero_gradients(SageNet* net)
{
    if (!net) return;

    for (size_t i = 0; i < net->num_layers; i++) {
        if (net->sagelayer && net->sagelayer[i]) {
            sage_layer_zero_gradients(net->sagelayer[i]);
        }
        if (net->relulayer && net->relulayer[i]) {
            relu_layer_zero_gradients(net->relulayer[i]);
        }
        if (net->normalizelayer && net->normalizelayer[i]) {
            normalize_layer_zero_gradients(net->normalizelayer[i]);
        }
    }
}

// Inspect helpers
void sage_layer_info(const SageLayer* const l)
{
    printf("\nSAGE LAYER\n");
    printf("========================================\n");
    printf("output = input * Wroot + agg  * Wagg\n");
    printf("%-6s = %-5s * %-5s + %-4s * %s\n",
           matrix_shape((l)->output), matrix_shape((l)->input),
           matrix_shape((l)->Wroot), matrix_shape((l)->agg),
           matrix_shape((l)->Wagg));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           matrix_shape((l)->grad_output), matrix_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           matrix_shape((l)->grad_input), matrix_shape((l)->input));
    printf("grad_Wagg   = Wagg\n");
    printf("   %-6s   = %s\n",
           matrix_shape((l)->grad_Wagg), matrix_shape((l)->Wagg));
    printf("grad_Wroot  = Wroot\n");
    printf("   %-7s  = %s\n",
           matrix_shape((l)->grad_Wroot), matrix_shape((l)->Wroot));
    printf("========================================\n");
}

void relu_layer_info(const ReluLayer* const l)
{
    printf("\nRELU LAYER\n");
    printf("========================================\n");
    printf("output = relu(input)\n");
    printf("%-6s = relu(%-5s)\n",
           matrix_shape((l)->output), matrix_shape((l)->input));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           matrix_shape((l)->grad_output), matrix_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           matrix_shape((l)->grad_input), matrix_shape((l)->input));
    printf("========================================\n");
}

void normalize_layer_info(const NormalizeLayer* const l)
{
    printf("\nNORMALIZE LAYER\n");
    printf("========================================\n");
    printf("output = l2norm(input)\n");
    printf("%-6s = l2norm(%-5s)\n",
           matrix_shape((l)->output), matrix_shape((l)->input));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           matrix_shape((l)->grad_output), matrix_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           matrix_shape((l)->grad_input), matrix_shape((l)->input));
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
           matrix_shape((l)->output), matrix_shape((l)->input),
           matrix_shape((l)->W), matrix_shape((l)->bias));
    printf("----------------------------------------\n");
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           matrix_shape((l)->grad_output), matrix_shape((l)->output));
    printf("grad_input  = input\n");
    printf("   %-7s  = %s\n",
           matrix_shape((l)->grad_input), matrix_shape((l)->input));
    printf("grad_W      =   W\n");
    printf("   %-6s   = %s\n",
           matrix_shape((l)->grad_W), matrix_shape((l)->W));
    printf("grad_bias   = bias\n");
    printf("   %-6s   = %s\n",
           matrix_shape((l)->grad_bias), matrix_shape((l)->bias));
    printf("========================================\n");
}

void logsoft_layer_info(const LogSoftLayer* const l)
{
    printf("\nLOGSOFT LAYER\n");
    printf("========================================\n");
    printf("output = log_softmax(input)\n");
    printf("%-6s = log_softmax(%-5s)\n",
           matrix_shape((l)->output), matrix_shape((l)->input));
    printf("----------------------------------------\n");
    printf("grad_input  = input\n");
    printf("   %-8s = %s\n",
           matrix_shape((l)->grad_input), matrix_shape((l)->input));
    printf("grad_output = output\n");
    printf("   %-8s = %s\n",
           matrix_shape((l)->grad_output), matrix_shape((l)->output));
    printf("========================================\n");
}
