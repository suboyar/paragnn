#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>
#include <cblas.h>

#include "core.h"
#include "gnn.h"
#include "matrix.h"
#include "graph.h"
#include "perf.h"
#include "linalg/linalg.h"

#include "../nob.h"

// Forward propagation
void aggregate(SageLayer *const l, graph_t *const g)
{
    size_t B = l->input->batch;
    size_t E = g->num_edges;
    size_t N = l->agg->features;

#pragma omp parallel
    {
        size_t* adj = malloc(E * sizeof(size_t));

#pragma omp for
        for (size_t v = 0; v < B; v++) {
            size_t neigh_count = 0;

            // NOTE: Collects neighbors from only incoming direction.

            // Find neighbors of count sample size
            for (size_t edge = 0; edge < E; edge++) {
                size_t u0 = EDGE_AT(g, edge, 0);
                size_t u1 = EDGE_AT(g, edge, 1);

                if (v == u1) {  // Paper v cites u1
                    adj[neigh_count++] = u0;
                }
            }

            if (neigh_count == 0) {
                for (size_t f = 0; f < N; f++) {
                    MIDX(l->agg, v, f) = 0;
                }
                continue;
            }
            // Skiped as this isn't worst-case

            // Copy the first neighbor features to memory space for aggregation
            size_t u = adj[0];
            for (size_t f = 0; f < N; f++) {
                MIDX(l->agg, v, f) = MIDX(l->input, u, f);
            }

            // Add remaining neighbors
            for (size_t i = 1; i < neigh_count; i++) {
                u = adj[i];
                for (size_t f = 0; f < N; f++) {
                    MIDX(l->agg, v, f) += MIDX(l->input, u, f);
                }
            }

            // Aggregation with mean
            l->mean_scale[v] = (double)1/neigh_count;
            for (size_t f = 0; f < N; f++) {
                MIDX(l->agg, v, f) *= l->mean_scale[v];
            }
        }

        free(adj);
    }
}

void sageconv(SageLayer *const l, graph_t *const g)
{
    PERF_FUNC_START();

    PERF_START("aggregate");
    aggregate(l, g);
    PERF_END("aggregate");

    PERF_START("sage_Wroot");
    matrix_dgemm(LinalgNoTrans,
                 LinalgNoTrans,
                 1.0,
                 l->input,
                 l->Wroot,
                 0.0,
                 l->output);
    PERF_END("sage_Wroot");

    PERF_START("sage_Wagg");
    matrix_dgemm(LinalgNoTrans,
                 LinalgNoTrans,
                 1.0,
                 l->agg,
                 l->Wagg,
                 1.0,
                 l->output);
    PERF_END("sage_Wagg");

    PERF_FUNC_END();

    nob_log(NOB_INFO, "sageconv: ok");
}

void relu(ReluLayer *const l)
{
    PERF_FUNC_START();
    size_t B = l->input->batch;
    size_t F = l->input->features;

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < F; j++) {
            MIDX(l->output, i, j) = fmax(0.0, MIDX(l->input, i, j));
        }
    }

    PERF_FUNC_END();
}

void normalize(NormalizeLayer *const l)
{
    PERF_FUNC_START();

    size_t B = l->input->batch;
    size_t F = l->input->features;


#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        double norm = 0.0;
        for (size_t j = 0; j < F; j++) {
            double val = MIDX(l->input, i, j);
            norm += val * val;
        }

        MIDX(l->recip_mag, i, 0) = (double)1/sqrt(norm);

        if (norm > 1e-8) {
            for (size_t j = 0; j < F; j++) {
                MIDX(l->output, i, j) = MIDX(l->input, i, j) * MIDX(l->recip_mag, i, 0);
            }
        }
    }

    PERF_FUNC_END();

    nob_log(NOB_INFO, "normalize: ok");
}

void linear(LinearLayer *const l)
{
    PERF_FUNC_START();
    matrix_dgemm(LinalgNoTrans,
                 LinalgNoTrans,
                 1.0,
                 l->input,
                 l->W,
                 0.0,
                 l->output);

    if (l->bias) {
        size_t B = l->output->batch;
        size_t F = l->output->features;

#pragma omp parallel for
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < F; j++) {
                MIDX(l->output, i, j) += MIDX(l->bias, 0, j);
            }
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "linear: ok");
}

/*
  Log Sum Exp: https://stackoverflow.com/a/61570752
*/
void logsoft(LogSoftLayer *const l)
{
    size_t B = l->input->batch;
    size_t F = l->input->features;

    // uint64_t flops = B * (1ULL + 5ULL * N);

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        double max = MIDX(l->input, i, 0);

        for (size_t j = 1; j < F; j++) {
            max = fmax(max, MIDX(l->input, i, j));
        }

        double logsumexp = 0.0;
        for (size_t j = 0; j < F; j++) {
            logsumexp += exp(MIDX(l->input, i, j) - max);
        }

        logsumexp = log(logsumexp);

        for (size_t j = 0; j < F; j++) {
            MIDX(l->output, i, j) = MIDX(l->input, i, j) - max - logsumexp;
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "log_softmax: ok");
}

double nll_loss(Matrix *const yhat, Matrix *const y)
{
    size_t B = yhat->batch;
    size_t F = yhat->features;

    PERF_FUNC_START();

    double loss = 0.0;
#pragma omp parallel for reduction(+:loss)
	for (size_t i = 0; i < B; i++) {
        size_t j;
	    for (j = 0; j < F; j++) {
            double class = MIDX(y, i, j);
            if (class == 1.0) { break; }
        }

        assert(j < F && "No true class was found");
        double logits = MIDX(yhat, i, j);
        loss -= logits;
    }

    PERF_FUNC_END();
    return loss;
}

double accuracy(Matrix *const yhat, Matrix *const y)
{
    size_t B = yhat->batch;
    size_t F_yhat = yhat->features;
    size_t F_y = y->features;

    PERF_FUNC_START();

    double acc = 0.0;
#pragma omp parallel for reduction(+:acc)
	for (size_t i = 0; i < B; i++) {
        // Find prediction
        size_t pred_class = 0;
        double best_pred = MIDX(yhat, i, 0);
	    for (size_t j = 1; j < F_yhat; j++) {
            double pred = MIDX(yhat, i, j);
            if (best_pred < pred) {
                best_pred = pred;
                pred_class = j;
            }
        }


        // Find the true class
        size_t true_class = 0;
        for (size_t j = 1; j < F_y; j++) {
            if (MIDX(y, i, j) == 1.0) {
                true_class = j;
                break;
            }
        }

        // Check if prediction matches true label
        if (pred_class == true_class) {
            acc += 1.0;
        }
    }

    double res = (double)acc/y->batch;
    PERF_FUNC_END();
    return res;
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
void cross_entropy_backward(LogSoftLayer *const l, Matrix *const y)
{
    size_t B = l->output->batch;
    size_t F = l->output->features;

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < F; j++) {
            MIDX(l->grad_input, i, j) = exp(MIDX(l->output, i, j));
        }

        for (size_t j = 0; j < F; j++) {
            if (MIDX(y, i, j) == 1.0) { //  True class
                MIDX(l->grad_input, i, j) -= 1;
                break;
            }
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}

void linear_backward(LinearLayer *const l)
{
    // TODO shape assert

    PERF_FUNC_START();

    // Downstream:
    // Column-major: grad_input = W^T @ grad_output
    // Row-major:    grad_input = grad_output @ W^T
    // Note: Row-major storage causes W to be implicitly transposed
    // printf("linear_grad_input\n");
    PERF_START("linear_grad_input");
    matrix_dgemm(LinalgNoTrans,
                 LinalgTrans,
                 1.0,
                 l->grad_output,
                 l->W,
                 0.0,
                 l->grad_input);
    PERF_END("linear_grad_input");

    // Cost of weights:
    // Column-major: grad_W = grad_output @ input^T
    // Row-major:    grad_W = input^T @ grad_output
    // Note: Similar reasoning - Row-major storage causes input to be implicitly transposed
    // printf("linear_grad_W");
    PERF_START("linear_grad_W");
    matrix_dgemm(LinalgTrans,
                 LinalgNoTrans,
                 1.0,
                 l->input,
                 l->grad_output,
                 0.0,
                 l->grad_W);
    PERF_END("linear_grad_W");

    if (l->grad_bias != NULL) {
        PERF_START("linear_grad_bias");

        size_t B = l->grad_output->batch;
        size_t F = l->grad_output->features;

        // Sum gradients across batch dimension
#pragma omp parallel for
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < F; j++) {
                // Accumulate the bias used by all batch samples
                MIDX(l->grad_bias, 0, j) += MIDX(l->grad_output, i, j);
            }
        }

        PERF_END("linear_grad_bias");
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "linear_backward: ok");
}

void normalize_backward(NormalizeLayer *const l)
{
    size_t B = l->input->batch;
    size_t F = l->input->features;

    PERF_FUNC_START();

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        double grad_local_data[F * F];
        Matrix grad_local = {
            .M = F,
            .N = F,
            .stride = F,
            .data = grad_local_data
        };
        for (size_t j = 0; j < F; j++) {
            for (size_t k = 0; k < F; k++) {
                MIDX(&grad_local, j, k) = - MIDX(l->output, i, j) * MIDX(l->output, i, k);

                if (j == k) {     // Kronecker delta
                    MIDX(&grad_local, j, k) = 1 + MIDX(&grad_local, j, k);
                }

                MIDX(&grad_local, j, k) *= MIDX(l->recip_mag, i, 0);
            }
        }

        for (size_t j = 0; j < F; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < F; k++) {
                sum += MIDX(l->grad_output, i, k) * MIDX(&grad_local, k, j);
            }
            MIDX(l->grad_input, i, j) = sum;
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "normalize_backward: ok");
}

void relu_backward(ReluLayer *const l)
{
    PERF_FUNC_START();
    size_t B = l->output->batch;
    size_t F = l->output->features;

    // TODO: grad_input = grad_output * fmaxf(0.0f, copysignf(1.0f, output));
#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < F; j++) {
            MIDX(l->grad_input, i, j) = MIDX(l->grad_output, i, j);
            if (MIDX(l->output, i, j) <= 0.0) {
                MIDX(l->grad_input, i, j) = 0;
            }
        }
    }

    PERF_FUNC_END();
    nob_log(NOB_INFO, "relu_backward: ok");
}

void sageconv_backward(SageLayer *const l, graph_t *const g)
{
    PERF_FUNC_START();

    PERF_START("sage_grad_Wroot");
    matrix_dgemm(LinalgTrans,
                 LinalgNoTrans,
                 1.0,
                 l->input,
                 l->grad_output,
                 0.0,
                 l->grad_Wroot);
    PERF_END("sage_grad_Wroot");

    PERF_START("sage_grad_Wagg");
    matrix_dgemm(LinalgTrans,
                 LinalgNoTrans,
                 1.0,
                 l->agg,
                 l->grad_output,
                 0.0,
                 l->grad_Wagg);
    PERF_END("sage_grad_Wagg");

    PERF_START("sage_grad_input_self");
    matrix_dgemm(LinalgNoTrans,
                 LinalgTrans,
                 1.0,
                 l->grad_output,
                 l->Wroot,
                 0.0,
                 l->grad_input);
    PERF_END("sage_grad_input_self");

    PERF_START("sageconv_grad_neighbor");
    size_t E = g->num_edges;
    size_t B = l->Wagg->batch;
    size_t F = l->grad_output->features;
#pragma omp parallel for
    for (size_t edge = 0; edge < E; edge++) {
        size_t u0 = EDGE_AT(g, edge, 0);  // source
        size_t u1 = EDGE_AT(g, edge, 1);  // target

        // u0 gets gradient from u1's computation (outgoing gradient)
        // grad_input[u0] += grad_output[u1] @ Wagg^T
        for (size_t i = 0; i < B; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < F; j++) {
                sum += MIDX(l->grad_output, u1, j) * MIDX(l->Wagg, i, j);
            }
            MIDX(l->grad_input, u0, i) += sum * l->mean_scale[u1];
        }
    }

    PERF_END("sageconv_grad_neighbor");

    PERF_FUNC_END();
    nob_log(NOB_INFO, "sageconv_backward: ok");
}
