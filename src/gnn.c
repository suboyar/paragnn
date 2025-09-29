#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>

#include "gnn.h"
#include "matrix.h"
#include "graph.h"

#include "nob.h"

// Forward propagation
#define SAMPLE_SIZE 10 // XXX: Placeholder unitl I implement proper neighbor samplig
void aggregate(SageLayer* const l, graph_t* const g)
{
    size_t adj[SAMPLE_SIZE] = {0};
    size_t neigh_count = 0;

    for (size_t v = 0; v < BATCH_DIM(l->input); v++) {
        // NOTE: Collects neighbors from only incoming direction.

        // Find neighbors of count sample size
        for (size_t edge = 0; edge < g->num_edges && neigh_count < SAMPLE_SIZE; edge++) {
            EDGE_BOUNDS_CHECK(g, edge, 0);
            size_t u0 = EDGE_AT(g, edge, 0);
            EDGE_BOUNDS_CHECK(g, edge, 1);
            size_t u1 = EDGE_AT(g, edge, 1);

            if (v == u1) {  // Paper v cites u1
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
            for (size_t f = 0; f < NODE_DIM(l->agg); f++) {
                MAT_BOUNDS_CHECK(l->agg, v, f);
                MAT_BOUNDS_CHECK(l->input, u, f);
                MAT_AT(l->agg, v, f) += MAT_AT(l->input, u, f);
            }
        }

        // Aggregation with mean
        l->mean_scale[v] = (float)1/neigh_count;
        for (size_t f = 0; f < NODE_DIM(l->agg); f++) {
            MAT_BOUNDS_CHECK(l->agg, v, f);
            MAT_AT(l->agg, v, f) *= l->mean_scale[v];
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
                MAT_AT(&grad_local, j, k) = - MAT_AT(l->output, i, j) * MAT_AT(l->output, i, k);
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

void sageconv_backward(SageLayer* const l, graph_t* const g)
{
    // TODO: shape assert

    // Weight gradient
    dot_ex(l->input, l->grad_output, l->grad_Wroot, true, false);
    dot_ex(l->agg,   l->grad_output, l->grad_Wagg,   true, false);

    dot_ex(l->grad_output, l->Wroot, l->grad_input, false, true);

    for (size_t edge = 0; edge < g->num_edges; edge++) {
        size_t u0 = EDGE_AT(g, edge, 0);  // source
        size_t u1 = EDGE_AT(g, edge, 1);  // target

        // u0 gets gradient from u1's computation (outgoing gradient)
        // grad_input[u0] += grad_output[u1] @ Wagg^T
        for (size_t i = 0; i < l->Wagg->height; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < l->grad_output->width; j++) {
                MAT_BOUNDS_CHECK(l->grad_output, u1, j);
                MAT_BOUNDS_CHECK(l->Wagg, i, j);
                sum += MAT_AT(l->grad_output, u1, j) * MAT_AT(l->Wagg, i, j);
            }
            MAT_BOUNDS_CHECK(l->grad_input, u0, i);
            MAT_AT(l->grad_input, u0, i) += sum * l->mean_scale[u1];
        }
    }

    nob_log(NOB_INFO, "sageconv_backward: ok");
}

void update_linear_weights(LinearLayer* const l, float lr)
{
    MAT_ASSERT(l->W, l->grad_W);
    float batch_recip = (float) 1/BATCH_DIM(l->input);

    for (size_t i = 0; i < l->W->height; i++) {
        for (size_t j = 0; j < l->W->width; j++) {
            MAT_BOUNDS_CHECK(l->W, i, j);
            MAT_BOUNDS_CHECK(l->grad_W, i, j);
            MAT_AT(l->W, i, j) -= batch_recip * lr * MAT_AT(l->grad_W, i, j);
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

    nob_log(NOB_INFO, "update_sageconv_weights: ok");
}
