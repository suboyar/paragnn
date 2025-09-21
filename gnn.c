#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "gnn.h"
#include "matrix.h"
#include "graph.h"

#include "nob.h"

// Forward propagation
#ifdef NEWWAY
#define SAMPLE_SIZE 10 // XXX: Placeholder unitl I implement proper neighbor samplig
void aggregate(SageLayer* const l, graph_t* const g)
{
    size_t adj[SAMPLE_SIZE] = {0};
    size_t neigh_count = 0;

    for (size_t v = 0; v < BATCH_DIM(l->input); v++) {
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
            mat_fill(l->agg, 0.0);
            return;
        }

        // Copy the first neighbor features to memory space for aggregation
        size_t u = adj[0];
        for (size_t f = 0; f < NODE_DIM(l->agg); f++) {
            // TODO: memcpy(MAT_ROW(out, v), MAT_ROW(in, u), g->num_node_features * sizeof(*out->data));
            MAT_BOUNDS_CHECK(l->agg, v, f);
            MAT_BOUNDS_CHECK(l->input, u, f);
            MAT_AT(l->agg, v, f) = MAT_AT(l->input, u, f);
        }

        // Add remaining neighbors
        for (size_t i = 1; i < neigh_count; i++) {
            u = adj[i];
            for (size_t f = 0; f < g->num_node_features; f++) {
                MAT_BOUNDS_CHECK(l->agg, v, f);
                MAT_BOUNDS_CHECK(l->input, u, f);
                MAT_AT(l->agg, v, f) += MAT_AT(l->input, u, f);
            }
        }

        // Aggregation with mean
        float neigh_count_recp = (float)1/neigh_count;
        for (size_t f = 0; f < g->num_node_features; f++) {
            MAT_BOUNDS_CHECK(l->agg, v, f);
            MAT_AT(l->agg, v, f) *= neigh_count_recp;
        }
    }
}

void sageconv(SageLayer* const l, graph_t* const g)
{

    MAT_ASSERT(l->input, l->agg);
    MAT_ASSERT(l->Wroot, l->Wagg);
    MAT_ASSERT_DOT(l->input, l->Wroot);
    MAT_ASSERT_DOT(l->agg, l->Wagg);
    MAT_ASSERT_NODE(l->output, l->Wroot);
    MAT_ASSERT_NODE(l->output, l->Wagg);
    MAT_ASSERT_BATCH(l->output, l->input);
    MAT_ASSERT_BATCH(l->output, l->agg);

    aggregate(l, g);

    dot(l->input, l->Wroot, l->output);
    dot_agg(l->agg, l->Wagg, l->output);

    nob_log(NOB_INFO, "sageconv: ok");
}

void relu(ReluLayer* const l)
{
    MAT_ASSERT(l->output, l->input);

    for (size_t i = 0; i < BATCH_DIM(l->input); i++) {
        for (size_t j = 0; j < NODE_DIM(l->input); j++) {
            MAT_BOUNDS_CHECK(l->output, i, j);
            MAT_BOUNDS_CHECK(l->input, i, j);
            MAT_AT(l->output, i, j) = fmax(0.0, MAT_AT(l->input, i, j));
        }
    }
    nob_log(NOB_INFO, "relu: ok");
}

void normalize(NormalizeLayer * const l)
{
    MAT_ASSERT(l->output, l->input);

    for (size_t i = 0; i < BATCH_DIM(l->input); i++) {
        double norm = 0.0;
        for (size_t j = 0; j < NODE_DIM(l->input); j++) {
            MAT_BOUNDS_CHECK(l->input, i, j);
            double val = MAT_AT(l->input, i, j);
            norm += val * val;
        }

        MAT_BOUNDS_CHECK(l->recip_mag, i, 0);
        MAT_AT(l->recip_mag, i, 0) = (double)1/sqrt(norm);

        if (norm > 1e-8) {
            for (size_t j = 0; j < NODE_DIM(l->input); j++) {
                MAT_BOUNDS_CHECK(l->output, i, j);
                MAT_BOUNDS_CHECK(l->input, i, j);
                MAT_AT(l->output, i, j) = MAT_AT(l->input, i, j) * MAT_AT(l->recip_mag, i, 0);
            }
        }
    }
    nob_log(NOB_INFO, "normalize: ok");
}

void linear(LinearLayer* const l)
{
    MAT_ASSERT_DOT(l->input, l->W);
    MAT_ASSERT_NODE(l->output, l->W);
    MAT_ASSERT_BATCH(l->output, l->input);

    dot(l->input, l->W, l->output);

    if (l->bias) {
        MAT_ASSERT_NODE(l->output, l->bias);
        assert(BATCH_DIM(l->bias) == 1);

        for (size_t i = 0; i < BATCH_DIM(l->output); i++) {
            for (size_t j = 0; j < NODE_DIM(l->output); j++) {
                MAT_BOUNDS_CHECK(l->output, i, j);
                MAT_BOUNDS_CHECK(l->bias, 0, j);
                MAT_AT(l->output, i, j) += MAT_AT(l->bias, 0, j);
            }
        }
    }

    nob_log(NOB_INFO, "linear: ok");
}

/*
  Log Sum Exp: https://stackoverflow.com/a/61570752
*/
void logsoft(LogSoftLayer* const l)
{
    MAT_ASSERT(l->output, l->input);

    for (size_t i = 0; i < BATCH_DIM(l->input); i++) {
        MAT_BOUNDS_CHECK(l->input, i, 0);
        double max = MAT_AT(l->input, i, 0);

        for (size_t j = 1; j < NODE_DIM(l->input); j++) {
            MAT_BOUNDS_CHECK(l->input, i, j);
            max = fmax(max, MAT_AT(l->input, i, j));
        }

        double logsumexp = 0.0;
        for (size_t j = 0; j < NODE_DIM(l->input); j++) {
            MAT_BOUNDS_CHECK(l->input, i, j);
            logsumexp += exp(MAT_AT(l->input, i, j) - max);
        }

        logsumexp = log(logsumexp);

        for (size_t j = 0; j < NODE_DIM(l->input); j++) {
            MAT_BOUNDS_CHECK(l->output, i, j);
            MAT_BOUNDS_CHECK(l->input, i, j);
            MAT_AT(l->output, i, j) = MAT_AT(l->input, i, j) - max - logsumexp;
        }
    }

    nob_log(NOB_INFO, "log_softmax: ok");
}

double nll_loss(matrix_t* const yhat, matrix_t* const y)
{
    //TODO: shape assert
    double sum = 0.0;
	for (size_t i = 0; i < BATCH_DIM(yhat); i++) {
        size_t j;
	    for (j = 0; j < NODE_DIM(yhat); j++) {
            MAT_BOUNDS_CHECK(y, i, j);
            double class = MAT_AT(y, i, j);
            if (class == 1.0) { break; }
        }

        assert(j < NODE_DIM(yhat) && "No true class was found");

        MAT_BOUNDS_CHECK(y, i, j);
        double logits = MAT_AT(yhat, i, j);
        sum -= logits;
    }

    return sum;
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
void cross_entropy_backward(LogSoftLayer* const l, matrix_t* const y)
{
    MAT_ASSERT(l->output, y);
    MAT_ASSERT(l->output, l->output);

    for (size_t i = 0; i < BATCH_DIM(l->output); i++) {
        for (size_t j = 0; j < NODE_DIM(l->output); j++) {
            MAT_BOUNDS_CHECK(l->grad_input, i, j);
            MAT_BOUNDS_CHECK(l->output, i, j);
            MAT_AT(l->grad_input, i, j) = exp(MAT_AT(l->output, i, j));
        }

        for (size_t j = 0; j < NODE_DIM(l->output); j++) {
            MAT_BOUNDS_CHECK(l->output, i, j);
            if (MAT_AT(y, i, j) == 1.0) { //  True class
                MAT_BOUNDS_CHECK(l->grad_input, i, j);
                MAT_AT(l->grad_input, i, j) -= 1;
                break;
            }
        }
    }

    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}

void linear_backward(LinearLayer* const l)
{
    // Downstream:
    // Column-major: grad_input = W^T @ grad_output
    // Row-major:    grad_input = grad_output @ W^T
    // Note: Row-major storage causes W to be implicitly transposed
    dot_ex(l->grad_output, l->W, l->grad_input, false, true);

    // Cost of weights:
    // Column-major: grad_W = grad_output @ input^T
    // Row-major:    grad_W = input^T @ grad_output
    // Note: Similar reasoning - Row-major storage causes input to be implicitly transposed
    dot_ex(l->input, l->grad_output, l->grad_W, true, false);

    if (l->grad_bias != NULL) {
        MAT_ASSERT(l->bias, l->grad_bias);
        MAT_ASSERT_NODE(l->grad_output, l->grad_bias);

        // Sum gradients across batch dimension
        for (size_t batch = 0; batch < BATCH_DIM(l->grad_output); batch++) {
            for (size_t i = 0; i < NODE_DIM(l->grad_output); i++) {
                MAT_BOUNDS_CHECK(l->grad_bias, 0, i);
                MAT_BOUNDS_CHECK(l->grad_output, batch, i);

                // Accumulate the bias used by all batch samples
                MAT_AT(l->grad_bias, 0, i) += MAT_AT(l->grad_output, batch, i);
            }
        }

    }

    nob_log(NOB_INFO, "linear_backward: ok");
}

void normalize_backward(NormalizeLayer* const l)
{
    // TODO: add the asserts for matrix dims

    for (size_t i = 0; i < BATCH_DIM(l->input); i++) {
        double grad_local_data[NODE_DIM(l->input) * NODE_DIM(l->input)];
        matrix_t grad_local = {
            .height = NODE_DIM(l->input),
            .width = NODE_DIM(l->input),
            .data = grad_local_data
        };
        for (size_t j = 0; j < NODE_DIM(l->input); j++) {
            for (size_t k = 0; k < NODE_DIM(l->input); k++) {
                MAT_AT(&grad_local, j, k) = MAT_AT(l->output, i, j) * MAT_AT(l->output, i, k);
                if (j == k) {     // Kronecker delta
                    MAT_AT(&grad_local, j, k) = 1 + MAT_AT(&grad_local, j, k);
                }
                MAT_AT(&grad_local, j, k) *= MAT_AT(l->recip_mag, i, 0);
            }
        }

        for (size_t j = 0; j < NODE_DIM(l->grad_input); j++) {
            double sum = 0.0;
            for (size_t k = 0; k < NODE_DIM(l->grad_output); k++) {
                sum += MAT_AT(l->grad_output, i, k) * MAT_AT(&grad_local, k, j);
            }
            MAT_AT(l->grad_input, i, j) = sum;
        }
    }
    nob_log(NOB_INFO, "normalize_backward: ok");
}

void relu_backward(ReluLayer* const l)
{
    MAT_ASSERT(l->output, l->grad_output);
    MAT_ASSERT(l->output, l->grad_input);

    // TODO: grad_input = grad_output * fmaxf(0.0f, copysignf(1.0f, output));
    for (size_t i = 0; i < BATCH_DIM(l->output); i++) {
        for (size_t j = 0; j < NODE_DIM(l->output); j++) {
            MAT_AT(l->grad_input, i, j) = MAT_AT(l->grad_output, i, j);
            if (MAT_AT(l->output, i, j) <= 0.0) {
                MAT_AT(l->grad_input, i, j) = 0;
            }
        }
    }

    nob_log(NOB_INFO, "relu_backward: ok");
}

void sageconv_backward(SageLayer* const l)
{
    // TODO: shape assert

    // Weight gradient

    dot_ex(l->input, l->grad_output, l->grad_Wroot, true, false);
    dot_ex(l->agg,   l->grad_output, l->grad_Wagg,   true, false);

    // TODO: Downstream gradient

    nob_log(NOB_INFO, "sageconv_backward: ok");
}


void update_linear_weights(LinearLayer* const l, float lr)
{
    MAT_ASSERT(l->W, l->grad_W);
    float batch_recip = (float) 1/BATCH_DIM(l->input);

    for (size_t batch = 0; batch < BATCH_DIM(l->input); batch++) {
        for (size_t i = 0; i < l->W->height; i++) {
            for (size_t j = 0; j < l->W->width; j++) {
                MAT_BOUNDS_CHECK(l->W, i, j);
                MAT_BOUNDS_CHECK(l->grad_W, i, j);
                MAT_AT(l->W, i, j) -= batch_recip * lr * MAT_AT(l->grad_W, i, j);
            }
        }

    }


    if (l->grad_bias != NULL) {
        MAT_ASSERT(l->grad_bias, l->bias);
        for (size_t i = 0; i < NODE_DIM(l->bias); i++) {
            MAT_BOUNDS_CHECK(l->bias, 0, i);
            MAT_BOUNDS_CHECK(l->grad_bias, 0, i);
            MAT_AT(l->bias, 0, i) -= batch_recip * lr * MAT_AT(l->grad_bias, 0, i);
        }
    }

    nob_log(NOB_INFO, "update_linear_weights: ok");
}

void update_sageconv_weights(SageLayer* const l, float lr)
{
    MAT_ASSERT(l->Wroot, l->grad_Wroot);
    MAT_ASSERT(l->Wagg,  l->grad_Wagg);
    float batch_recip = (float) 1/BATCH_DIM(l->input);

    for (size_t n = 0; n < BATCH_DIM(l->input); n++) {
        for (size_t i = 0; i < l->Wroot->height; i++) {
            for (size_t j = 0; j < l->Wroot->width; j++) {
                MAT_BOUNDS_CHECK(l->Wroot, i, j);
                MAT_BOUNDS_CHECK(l->grad_Wroot, i, j);
                MAT_AT(l->Wroot, i, j) -= batch_recip * lr * MAT_AT(l->grad_Wroot, i, j);

                MAT_BOUNDS_CHECK(l->Wagg, i, j);
                MAT_BOUNDS_CHECK(l->grad_Wagg, i, j);
                MAT_AT(l->Wagg, i, j) -= batch_recip * lr * MAT_AT(l->grad_Wagg, i, j);
            }
        }
    }
    nob_log(NOB_INFO, "update_sageconv_weights: ok");
}


#else // NEWWAY

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
#endif
