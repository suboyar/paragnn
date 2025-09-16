#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "gnn.h"
#include "matrix.h"
#include "graph.h"

#include "nob.h"

// Forward propagation
#define SAMPLE_SIZE 10 // XXX: Placeholder unitl I implement proper neighbor samplig
size_t aggregate(matrix_t *in, size_t v, matrix_t *out, graph_t *g)
{
    size_t adj[SAMPLE_SIZE] = {0};
    size_t neigh_count = 0;

    // NOTE: Collects neighbors from both directions (incoming/outgoing
    // edges). Correct for directed graphs like ogbn-arxiv, but causes
    // double counting for undirected graphs since each edge (i,j) appears
    // as both (i,j) and (j,i).

    // Find neighbors of count sample size
    for (size_t edge = 0; edge < g->num_edges && neigh_count < SAMPLE_SIZE; edge++) {
        size_t u0 = EDGE_AT(g, edge, 0);
        size_t u1 = EDGE_AT(g, edge, 1);
        assert(u0 < g->num_nodes && nob_temp_sprintf("Neighbor u0 points to out-of-bounds value %zu", u0));
        assert(u1 < g->num_nodes && nob_temp_sprintf("Neighbor u1 points to out-of-bounds value %zu", u1));

        if (v == u0) {  // Paper v cites u1
            adj[neigh_count++] = u1;
        } else if (v == u1) {   // Paper u0 cites v
            adj[neigh_count++] = u0;
        }
    }

    if (neigh_count == 0) {
        return 0;
    }

    // Move the first neighbor features to memory space for aggregation
    size_t u = adj[0];
    memcpy(&MAT_ROW(out, v), &MAT_ROW(in, u), g->num_node_features * sizeof(*out->data)); // __memmove_evex_unaligned_erms
    for (size_t i = 1; i < neigh_count; i++) {
        u = adj[i];
        for (size_t f = 0; f < g->num_node_features; f++) {
            MAT_AT(out, v, f) += MAT_AT(in, u, f);
        }
    }

    // Aggregation with mean
    float neigh_count_recp = (float)1/neigh_count;
    for (size_t f = 0; f < g->num_node_features; f++) {
        MAT_AT(out, v, f) *= neigh_count_recp;
    }

    return neigh_count;
}


void sage_conv(matrix_t *in, matrix_t *Wl, matrix_t *Wr, matrix_t *agg, matrix_t *out, graph_t *g)
{
    MAT_ASSERT(in, agg);
    MAT_ASSERT(Wl, Wr);
    MAT_ASSERT(Wl, Wr);
    assert(out->height == in->height);
    assert(out->width == Wl->height);

    for (size_t v = 0; v < g->num_nodes; v++) {
        (void)aggregate(in, v, agg, g);
    }

    dot_ex(agg, Wr, out, false, true);
    dot_agg_ex(in, Wl, out, false, true);

    nob_log(NOB_INFO, "sage_layer: ok");
}

void relu(matrix_t* in, matrix_t* out)
{
    assert(in->height == out->height);
    assert(in->width == out->width);

    for (size_t i = 0; i < in->height; i++) {
        for (size_t j = 0; j < in->width; j++) {
            out->data[IDX(i, j, out->width)] = fmax(0.0, in->data[IDX(i, j, in->width)]);
        }
    }
    nob_log(NOB_INFO, "relu: ok");
}

void l2_normalization(matrix_t *in, matrix_t *out, graph_t *G)
{
    MAT_ASSERT(out, in);

    for (size_t v = 0; v < G->num_nodes; v++) {
        double norm = 0.0;
        for (size_t x = 0; x < in->width; x++) {
            double val = MAT_AT(in, v, x);
            norm += val * val;
        }
        double norm_recp = 1/sqrt(norm);

        if (norm > 1e-8) {
            for (size_t x = 0; x < in->width; x++) {
                MAT_AT(out, v, x) = MAT_AT(in, v, x) * norm_recp;
            }
        }
    }
    nob_log(NOB_INFO, "l2_normalization: ok");
}

// Applies an affine linear transformation to the incoming data: y = x @ A^T + b
void linear_layer(matrix_t* in, matrix_t* weight, matrix_t* bias, matrix_t* out)
{
    // Transposing weight will give correct inner dimensions
    assert(in->height == out->height);
    assert(in->width == weight->width);
    assert(out->width == weight->height);

    dot_ex(in, weight, out, false, true);

    if (bias) {
        assert(bias->height == 1);
        assert(out->width == bias->width);

        for (size_t i = 0; i < out->height; i++) {
            for (size_t j = 0; j < out->width; j++) {
                MAT_AT(out, i, j) += MAT_AT(bias, 0, j);
            }
        }
    }
    nob_log(NOB_INFO, "linear_layer: ok");
}

/*
  https://stackoverflow.com/a/61570752:
*/
void log_softmax(matrix_t* in, matrix_t* out)
{
    MAT_ASSERT(in, out);

    for (size_t i = 0; i < in->height; i++) {
        double max_x = in->data[IDX(i, 0, in->width)];
        for (size_t j = 0; j < in->width; j++) {
            double x = in->data[IDX(i, j, in->width)];
            if (x > max_x) max_x = x;
        }


        double sum = 0.0;
        for (size_t j = 0; j < in->width; j++) {
            double x = in->data[IDX(i, j, in->width)];
            sum += exp(x - max_x);
        }

        double log_sum = log(sum);
        for (size_t j = 0; j < in->width; j++) {
            double x = in->data[IDX(i, j, in->width)];
            out->data[IDX(i, j, out->width)] = x - max_x - log_sum;
        }
    }
    nob_log(NOB_INFO, "log_softmax: ok");
}

#define NLL_LOSS_REDUCTION_MEAN
double nll_loss(matrix_t* pred, matrix_t* target)
{
    MAT_ASSERT(pred, target);

    double* L = malloc(target->height * sizeof(*L));
	for (size_t v = 0; v < target->height; v++) {
        size_t c;
        for (c = 0; c < target->width; c++) {
            double class = MAT_AT(target, v, c);
            if (class == 1.0) {
                break;
            }
        }
        // printf("Node %zu class %zu pred: %f\n", v, c, );
        assert(c < target->width && "No true class was found");
        double logits = MAT_AT(pred, v, c);
        L[v] = -logits;
    }

    size_t N = target->height;
    double sum = 0.0;
    for (size_t l = 0; l < N; l++) {
        sum += L[l];
    }
    free(L);
#ifdef NLL_LOSS_REDUCTION_MEAN
    nob_log(NOB_INFO, "nll_loss(mean): ok");
    return sum / N;
#else

    nob_log(NOB_INFO, "nll_loss(sum): ok");
    return sum;
#endif
}

// Backpropagation

void update_weights(matrix_t* W, matrix_t* grad_W, size_t V)
{
    assert(W->height == grad_W->height);
    assert(W->width == grad_W->width);

    float V_recip = (float) 1/V;
    size_t height = W->height, width = W->width;
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            MAT_AT(W, i, j) -= MAT_AT(grad_W, i, j) * V_recip;
        }
    }
}

void update_sage_weights(matrix_t* W, matrix_t* grad_W, size_t V)
{
    // assert(W->height == grad_W->height);
    // assert(W->width == grad_W->width);

    float V_recip = (float) 1/V;
    size_t height = W->height, width = W->width;
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            MAT_AT(W, i, j) -= MAT_AT(grad_W, j, i) * V_recip;
        }
    }
}

void sage_conv_backward(matrix_t *grad_in, matrix_t *h_relu, matrix_t *h, matrix_t *agg, matrix_t *grad_Wl, matrix_t *grad_Wr, graph_t *g)
{
    // MAT_ASSERT(grad_in, h_relu);
    // MAT_ASSERT(grad_Wl, grad_Wr);
    // assert(grad_in->width == grad_Wl->height);

    size_t height = grad_Wl->height;
    size_t width = grad_Wl->width;

    dot_ex(h, grad_in, grad_Wl, true, false);
    dot_ex(agg, grad_in, grad_Wr, true, false);
}

void relu_backward(matrix_t* grad_in, matrix_t* h, matrix_t* grad_out)
{
    MAT_ASSERT(grad_in, grad_out);
    MAT_ASSERT(grad_in, h);


    // TODO: Check if one of these are better:
    // - grad_input->data[IDX] = (output->data[IDX] > 0.0f) * grad_output->data[IDX];
    // - grad_input->data[IDX] = grad_output->data[IDX] * fmaxf(0.0f, copysignf(1.0f, output->data[IDX]));

    size_t height = grad_in->height;
    size_t width = grad_in->width;

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            MAT_AT(grad_out, i, j) = 0;
            if (MAT_AT(h, i, j) > 0.0) { MAT_AT(grad_out, i, j) = MAT_AT(grad_in, i, j); }
        }
    }

    nob_log(NOB_INFO, "relu_backward: ok");
}

void l2_normalization_backward(matrix_t *grad_in, matrix_t *h_relu, matrix_t *h_l2, matrix_t *grad_out)
{
    MAT_ASSERT(grad_in, h_relu);
    MAT_ASSERT(grad_in, h_l2);
    MAT_ASSERT(grad_in, grad_out);
    size_t V = grad_in->height;
    size_t H = grad_in->width;
    for (size_t v = 0; v < V; v++) {
        double norm = 0.0;
        for (size_t x = 0; x < h_relu->width; x++) {
            double val = MAT_AT(h_relu, v, x);
            norm += val * val;
        }
        double norm_recp = 1/sqrt(norm);

        for (size_t j = 0; j < H; j++) {
            MAT_AT(grad_out, v, j) = 0;
            for (size_t k = 0; k < H; k++) {
                if (j == k) { MAT_AT(grad_out, v, j) = 1; }
                MAT_AT(grad_out, v, j) *= MAT_AT(h_l2, v, k) * MAT_AT(h_l2, v, j);
                MAT_AT(grad_out, v, j) *= MAT_AT(grad_in, v, k);
            }
            MAT_AT(grad_out, v, j) *= norm_recp;
        }
    }
    nob_log(NOB_INFO, "l2_normalization_backward: ok");
}

void linear_weight_backward(matrix_t *grad_in, matrix_t *lin_in, matrix_t *grad_out)
{
    dot_ex(grad_in, lin_in, grad_out, true, false);
    nob_log(NOB_INFO, "linear_weight_backward: ok");
}

void linear_h_backward(matrix_t* grad_in, matrix_t* W, matrix_t* grad_out)
{
    assert(grad_out->height == grad_in->height);
    assert(grad_out->width == W->width);
    assert(grad_in->width == W->height);

    dot(grad_in, W, grad_out);
    nob_log(NOB_INFO, "linear_h_backward: ok");
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
void cross_entropy_backward(matrix_t *grad_out, matrix_t *yhat, matrix_t *y)
{
    MAT_ASSERT(yhat, y);
    MAT_ASSERT(yhat, grad_out);

    size_t height = yhat->height, width = yhat->width;
    for (size_t v = 0; v < height; v++) {
        for (size_t i = 0; i < width; i++) {
            MAT_AT(grad_out, v, i) = exp(MAT_AT(yhat, v, i));
        }

        for (size_t c = 0; c < width; c++) {
            if (MAT_AT(y, v, c) == 1.0) { //  True class
                MAT_AT(grad_out, v, c) = MAT_AT(grad_out, v, c) - 1;
                break;
            }
        }
    }

    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}
