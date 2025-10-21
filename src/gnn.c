#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>

#include "core.h"
#include "gnn.h"
#include "matrix.h"
#include "graph.h"
#include "perf.h"

#include "nob.h"

// Forward propagation
void aggregate(SageLayer* const l, graph_t* const g)
{
    size_t B = BATCH_DIM(l->input);
    size_t E = g->num_edges;
    size_t N = NODE_DIM(l->agg);

    // uint64_t flops = N*(E-1ULL)*B + B + N*B;

#pragma omp parallel
    {
        size_t* adj = malloc(E * sizeof(size_t));

#pragma omp for
        for (size_t v = 0; v < B; v++) {
            size_t neigh_count = 0;

            // NOTE: Collects neighbors from only incoming direction.

            // Find neighbors of count sample size
            for (size_t edge = 0; edge < E; edge++) {
                EDGE_BOUNDS_CHECK(g, edge, 0);
                size_t u0 = EDGE_AT(g, edge, 0);
                EDGE_BOUNDS_CHECK(g, edge, 1);
                size_t u1 = EDGE_AT(g, edge, 1);

                if (v == u1) {  // Paper v cites u1
                    adj[neigh_count++] = u0;
                }
            }

            if (neigh_count == 0) {
                for (size_t f = 0; f < N; f++) {
                    MAT_BOUNDS_CHECK(l->agg, v, f);
                    MAT_AT(l->agg, v, f) = 0;
                }
                continue;
            }
            // Skiped as this isn't worst-case

            // Copy the first neighbor features to memory space for aggregation
            size_t u = adj[0];
            for (size_t f = 0; f < N; f++) {
                // TODO: memcpy(MAT_ROW(out, v), MAT_ROW(in, u), g->num_node_features * sizeof(*out->data));
                MAT_BOUNDS_CHECK(l->agg, v, f);
                MAT_BOUNDS_CHECK(l->input, u, f);
                MAT_AT(l->agg, v, f) = MAT_AT(l->input, u, f);
            }

            // Add remaining neighbors
            for (size_t i = 1; i < neigh_count; i++) {
                u = adj[i];
                for (size_t f = 0; f < N; f++) {
                    MAT_BOUNDS_CHECK(l->agg, v, f);
                    MAT_BOUNDS_CHECK(l->input, u, f);
                    MAT_AT(l->agg, v, f) += MAT_AT(l->input, u, f);
                }
            }

            // Aggregation with mean
            l->mean_scale[v] = (double)1/neigh_count;
            for (size_t f = 0; f < N; f++) {
                MAT_BOUNDS_CHECK(l->agg, v, f);
                MAT_AT(l->agg, v, f) *= l->mean_scale[v];
            }
        }

        free(adj);
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

    PERF_FUNC_START();

    PERF_START("aggregate");
    aggregate(l, g);
    PERF_END("aggregate");

    PERF_START("sage_Wroot");
    dot(l->input, l->Wroot, l->output);
    PERF_END("sage_Wroot");

    PERF_START("sage_Wagg");
    dot_agg(l->agg, l->Wagg, l->output);
    PERF_END("sage_Wagg");

    PERF_FUNC_END();

    nob_log(NOB_INFO, "sageconv: ok");
}

void relu(ReluLayer* const l)
{
    MAT_ASSERT(l->output, l->input);

    size_t B = BATCH_DIM(l->input);
    size_t N = NODE_DIM(l->input);

    // uint64_t flops = 0;

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < N; j++) {
            MAT_BOUNDS_CHECK(l->output, i, j);
            MAT_BOUNDS_CHECK(l->input, i, j);
            MAT_AT(l->output, i, j) = fmax(0.0, MAT_AT(l->input, i, j));
        }
    }

    PERF_FUNC_END();
}

void normalize(NormalizeLayer * const l)
{
    MAT_ASSERT(l->output, l->input);

    size_t B = BATCH_DIM(l->input);
    size_t N = NODE_DIM(l->input);

    // uint64_t flops = B * (3ULL * N + 2);

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        double norm = 0.0;
        for (size_t j = 0; j < N; j++) {
            MAT_BOUNDS_CHECK(l->input, i, j);
            double val = MAT_AT(l->input, i, j);
            norm += val * val;
        }

        MAT_BOUNDS_CHECK(l->recip_mag, i, 0);
        MAT_AT(l->recip_mag, i, 0) = (double)1/sqrt(norm);

        if (norm > 1e-8) {
            for (size_t j = 0; j < N; j++) {
                MAT_BOUNDS_CHECK(l->output, i, j);
                MAT_BOUNDS_CHECK(l->input, i, j);
                MAT_AT(l->output, i, j) = MAT_AT(l->input, i, j) * MAT_AT(l->recip_mag, i, 0);
            }
        }
    }

    PERF_FUNC_END();

    nob_log(NOB_INFO, "normalize: ok");
}

void linear(LinearLayer* const l)
{
    MAT_ASSERT_DOT(l->input, l->W);
    MAT_ASSERT_NODE(l->output, l->W);
    MAT_ASSERT_BATCH(l->output, l->input);

    PERF_FUNC_START();

    dot(l->input, l->W, l->output);

    if (l->bias) {
        MAT_ASSERT_NODE(l->output, l->bias);
        assert(BATCH_DIM(l->bias) == 1);

        size_t B = BATCH_DIM(l->output);
        size_t N = NODE_DIM(l->output);

        uint64_t flops = B * N;
        uint64_t bytes = 3ULL * B * N * sizeof(double);
        PERF_ADD_METRICS(flops, bytes);

#pragma omp parallel for
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < N; j++) {
                MAT_BOUNDS_CHECK(l->output, i, j);
                MAT_BOUNDS_CHECK(l->bias, 0, j);
                MAT_AT(l->output, i, j) += MAT_AT(l->bias, 0, j);
            }
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "linear: ok");
}

/*
  Log Sum Exp: https://stackoverflow.com/a/61570752
*/
void logsoft(LogSoftLayer* const l)
{
    MAT_ASSERT(l->output, l->input);

    size_t B = BATCH_DIM(l->input);
    size_t N = NODE_DIM(l->input);

    // uint64_t flops = B * (1ULL + 5ULL * N);

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        MAT_BOUNDS_CHECK(l->input, i, 0);
        double max = MAT_AT(l->input, i, 0);

        for (size_t j = 1; j < N; j++) {
            MAT_BOUNDS_CHECK(l->input, i, j);
            max = fmax(max, MAT_AT(l->input, i, j));
        }

        double logsumexp = 0.0;
        for (size_t j = 0; j < N; j++) {
            MAT_BOUNDS_CHECK(l->input, i, j);
            logsumexp += exp(MAT_AT(l->input, i, j) - max);
        }

        logsumexp = log(logsumexp);

        for (size_t j = 0; j < N; j++) {
            MAT_BOUNDS_CHECK(l->output, i, j);
            MAT_BOUNDS_CHECK(l->input, i, j);
            MAT_AT(l->output, i, j) = MAT_AT(l->input, i, j) - max - logsumexp;
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "log_softmax: ok");
}

double nll_loss(matrix_t* const yhat, matrix_t* const y)
{
    // TODO: shape assert

    size_t B = BATCH_DIM(yhat);
    size_t N = NODE_DIM(yhat);

    // uint64_t flops = B;

    PERF_FUNC_START();

    double loss = 0.0;
#pragma omp parallel for reduction(+:loss)
	for (size_t i = 0; i < B; i++) {
        size_t j;
	    for (j = 0; j < N; j++) {
            MAT_BOUNDS_CHECK(y, i, j);
            double class = MAT_AT(y, i, j);
            if (class == 1.0) { break; }
        }

        assert(j < N && "No true class was found");

        MAT_BOUNDS_CHECK(y, i, j);
        double logits = MAT_AT(yhat, i, j);

        loss -= logits;
    }

    PERF_FUNC_END();
    return loss;
}

double accuracy(matrix_t* const yhat, matrix_t* const y)
{
    // TODO: shape assert
    MAT_ASSERT(yhat, y);

    size_t B = BATCH_DIM(yhat);
    size_t N = NODE_DIM(yhat);

    // uint64_t flops = B;

    PERF_FUNC_START();

    double acc = 0.0;
#pragma omp parallel for reduction(+:acc)
	for (size_t i = 0; i < B; i++) {
        // Find prediction
        size_t pred_class = 0;
        double best_pred = MAT_AT(yhat, i, 0);
	    for (size_t j = 1; j < N; j++) {
            MAT_BOUNDS_CHECK(yhat, i, j);
            double pred = MAT_AT(yhat, i, j);
            if (best_pred < pred) {
                best_pred = pred;
                pred_class = j;
            }
        }


        // Find the true class
        size_t true_class = 0;
        for (size_t j = 1; j < NODE_DIM(y); j++) {
            MAT_BOUNDS_CHECK(y, i, j);
            if (MAT_AT(y, i, j) == 1.0) {
                true_class = j;
                break;
            }
        }

        // Check if prediction matches true label
        if (pred_class == true_class) {
            acc += 1.0;
        }
    }

    PERF_FUNC_END();
    return acc/BATCH_DIM(y);
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
void cross_entropy_backward(LogSoftLayer* const l, matrix_t* const y)
{
    MAT_ASSERT(l->output, y);
    MAT_ASSERT(l->output, l->output);

    size_t B = BATCH_DIM(l->output);
    size_t N =  NODE_DIM(l->output);

    // uint64_t flops = N * (N + 1ULL);

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < N; j++) {
            MAT_BOUNDS_CHECK(l->grad_input, i, j);
            MAT_BOUNDS_CHECK(l->output, i, j);
            MAT_AT(l->grad_input, i, j) = exp(MAT_AT(l->output, i, j));
        }

        for (size_t j = 0; j < N; j++) {
            MAT_BOUNDS_CHECK(y, i, j);
            if (MAT_AT(y, i, j) == 1.0) { //  True class
                MAT_BOUNDS_CHECK(l->grad_input, i, j);
                MAT_AT(l->grad_input, i, j) -= 1;
                break;
            }
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}

void linear_backward(LinearLayer* const l)
{
    // TODO shape assert

    PERF_FUNC_START();

    // Downstream:
    // Column-major: grad_input = W^T @ grad_output
    // Row-major:    grad_input = grad_output @ W^T
    // Note: Row-major storage causes W to be implicitly transposed
    // printf("linear_grad_input\n");
    PERF_START("linear_grad_input");
    dot_ex(l->grad_output, l->W, l->grad_input, false, true);
    PERF_END("linear_grad_input");

    // Cost of weights:
    // Column-major: grad_W = grad_output @ input^T
    // Row-major:    grad_W = input^T @ grad_output
    // Note: Similar reasoning - Row-major storage causes input to be implicitly transposed
    // printf("linear_grad_W");
    PERF_START("linear_grad_W");
    dot_ex(l->input, l->grad_output, l->grad_W, true, false);
    PERF_END("linear_grad_W");

    if (l->grad_bias != NULL) {
        PERF_START("linear_grad_bias");
        MAT_ASSERT(l->bias, l->grad_bias);
        MAT_ASSERT_NODE(l->grad_output, l->grad_bias);

        // linear_bias_backward: bytes_coeff={'BN': 3}, flops_coeff={'BN': 1}
        size_t B = BATCH_DIM(l->grad_output);
        size_t N = NODE_DIM(l->grad_output);

        // uint64_t flops = B * N;

        // Sum gradients across batch dimension
#pragma omp parallel for
        for (size_t batch = 0; batch < B; batch++) {
            for (size_t i = 0; i < N; i++) {
                MAT_BOUNDS_CHECK(l->grad_bias, 0, i);
                MAT_BOUNDS_CHECK(l->grad_output, batch, i);

                // Accumulate the bias used by all batch samples
                MAT_AT(l->grad_bias, 0, i) += MAT_AT(l->grad_output, batch, i);
            }
        }

        PERF_END("linear_grad_bias");
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "linear_backward: ok");
}

void normalize_backward(NormalizeLayer* const l)
{
    // TODO: add the asserts for matrix dims

    MAT_ASSERT(l->input, l->output);
    MAT_ASSERT(l->input, l->grad_input);
    MAT_ASSERT(l->output, l->grad_output);

    size_t B = BATCH_DIM(l->input);
    size_t N = NODE_DIM(l->input);

    // uint64_t flops = 6ULL * B * N * N;

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        double grad_local_data[N * N];
        matrix_t grad_local = {
            .height = N,
            .width = N,
            .data = grad_local_data
        };
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                MAT_BOUNDS_CHECK(&grad_local, j, k);
                MAT_BOUNDS_CHECK(l->output, i, j);
                MAT_BOUNDS_CHECK(l->output, i, k);
                MAT_BOUNDS_CHECK(l->recip_mag, i, 0);

                MAT_AT(&grad_local, j, k) = - MAT_AT(l->output, i, j) * MAT_AT(l->output, i, k);

                if (j == k) {     // Kronecker delta
                    MAT_AT(&grad_local, j, k) = 1 + MAT_AT(&grad_local, j, k);
                }

                MAT_AT(&grad_local, j, k) *= MAT_AT(l->recip_mag, i, 0);
            }
        }

        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                MAT_BOUNDS_CHECK(l->grad_output, i, k);
                MAT_BOUNDS_CHECK(&grad_local, k, j);

                sum += MAT_AT(l->grad_output, i, k) * MAT_AT(&grad_local, k, j);
            }

            MAT_BOUNDS_CHECK(l->grad_input, i, j);
            MAT_AT(l->grad_input, i, j) = sum;
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "normalize_backward: ok");
}

void relu_backward(ReluLayer* const l)
{
    MAT_ASSERT(l->output, l->grad_output);
    MAT_ASSERT(l->output, l->grad_input);

    size_t B = BATCH_DIM(l->output);
    size_t N = NODE_DIM(l->output);

    // uint64_t flops = 0;

    PERF_FUNC_START();

    // TODO: grad_input = grad_output * fmaxf(0.0f, copysignf(1.0f, output));
#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < N; j++) {
            MAT_AT(l->grad_input, i, j) = MAT_AT(l->grad_output, i, j);
            if (MAT_AT(l->output, i, j) <= 0.0) {
                MAT_AT(l->grad_input, i, j) = 0;
            }
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "relu_backward: ok");
}

void sageconv_backward(SageLayer* const l, graph_t* const g)
{
    size_t E = g->num_edges;
    size_t B = BATCH_DIM(l->Wagg);
    size_t N = NODE_DIM(l->grad_output);

    PERF_FUNC_START();

    PERF_START("sage_grad_Wroot");
    dot_ex(l->input, l->grad_output, l->grad_Wroot, true, false);
    PERF_END("sage_grad_Wroot");

    PERF_START("sage_grad_Wagg");
    dot_ex(l->agg, l->grad_output, l->grad_Wagg, true, false);
    PERF_END("sage_grad_Wagg");

    // printf("sage_grad_input: ");
    PERF_START("sage_grad_input_self");
    dot_ex(l->grad_output, l->Wroot, l->grad_input, false, true);
    PERF_END("sage_grad_input_self");

    // uint64_t neighbor_flops = (2ULL * E * B * N) + (2ULL * E * B);
    PERF_START("sageconv_grad_neighbor");
#pragma omp parallel for
    for (size_t edge = 0; edge < E; edge++) {
        size_t u0 = EDGE_AT(g, edge, 0);  // source
        size_t u1 = EDGE_AT(g, edge, 1);  // target

        // u0 gets gradient from u1's computation (outgoing gradient)
        // grad_input[u0] += grad_output[u1] @ Wagg^T
        for (size_t i = 0; i < B; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < N; j++) {
                MAT_BOUNDS_CHECK(l->grad_output, u1, j);
                MAT_BOUNDS_CHECK(l->Wagg, i, j);
                sum += MAT_AT(l->grad_output, u1, j) * MAT_AT(l->Wagg, i, j);
            }
            MAT_BOUNDS_CHECK(l->grad_input, u0, i);
            MAT_AT(l->grad_input, u0, i) += sum * l->mean_scale[u1];
        }
    }

    PERF_END("sageconv_grad_neighbor");

    PERF_FUNC_END();
    nob_log(NOB_INFO, "sageconv_backward: ok");
}
