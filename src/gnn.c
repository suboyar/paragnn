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
#include "timer.h"
#include "linalg/linalg.h"

#include "../nob.h"

// Forward propagation
void aggregate(SageLayer *const l, graph_t *const g)
{
    TIMER_FUNC();

    size_t B = l->agg->batch;
    size_t E = g->num_edges;
    size_t F = l->agg->features;

    double* X = l->input->data;
    size_t ldX = l->input->stride;

    int nthreads = omp_get_max_threads();
    double* t_gather = calloc(sizeof(double), nthreads);
    if (!t_gather) ERROR("Could not calloc t_gather");
    double* t_pool = calloc(sizeof(double), nthreads);
    if (!t_pool) ERROR("Could not calloc t_pool");

    memset(l->agg->data, 0, B*F*sizeof(*l->agg->data));
    memset(l->mean_scale, 0, B * sizeof(*l->mean_scale));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t* adj = malloc(E * sizeof(size_t));

#pragma omp for
        for (size_t i = 0; i < B; i++) {
            size_t neigh_count = 0;
            double *Y = l->agg->data + i * l->agg->stride;

            // NOTE: Collects neighbors from only incoming direction.

            double t0 = omp_get_wtime();
            // Find neighbors of count sample size
            for (size_t edge = 0; edge < E; edge++) {
                size_t u1 = EDGE_AT(g, edge, 1);
                if (i == u1) {  // Paper i cites u1
                    adj[neigh_count++] = EDGE_AT(g, edge, 0);
                }
            }
            t_gather[tid] += omp_get_wtime() - t0;

            if (neigh_count == 0) continue;

            double scale = 1.0 / neigh_count;

            double t1 = omp_get_wtime();
            for (size_t j = 0; j < F; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < neigh_count; k++) {
                    sum += X[adj[k] * ldX + j];
                }
                Y[j] = sum * scale;
            }
            t_pool[tid] += omp_get_wtime() - t1;

            l->mean_scale[i] = scale;
        }

        free(adj);
    }

    timer_record_parallel("gather", t_gather, nthreads);
    timer_record_parallel("pool", t_pool, nthreads);

    free(t_gather);
    free(t_pool);
}

void sageconv(SageLayer *const l, graph_t *const g)
{
    TIMER_FUNC();
    TIMER_BLOCK("Wroot", {
            matrix_dgemm(LinalgNoTrans,
                         LinalgNoTrans,
                         1.0,
                         l->input,
                         l->Wroot,
                         0.0,
                         l->output);
        });

    aggregate(l, g);

    TIMER_BLOCK("Wagg", {
            matrix_dgemm(LinalgNoTrans,
                         LinalgNoTrans,
                         1.0,
                         l->agg,
                         l->Wagg,
                         1.0,
                         l->output);
        });

    nob_log(NOB_INFO, "sageconv: ok");

}
void relu(ReluLayer *const l)
{
    TIMER_FUNC();
    size_t B = l->input->batch;
    size_t F = l->input->features;

#pragma omp parallel for
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < F; j++) {
            MIDX(l->output, i, j) = fmax(0.0, MIDX(l->input, i, j));
        }
    }
}

void normalize(L2NormLayer *const l)
{
    TIMER_FUNC();

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

    nob_log(NOB_INFO, "normalize: ok");
}

void linear(LinearLayer *const l)
{
    TIMER_FUNC();

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

    nob_log(NOB_INFO, "linear: ok");
}

/*
  Log Sum Exp: https://stackoverflow.com/a/61570752
*/
void logsoft(LogSoftLayer *const l)
{
    TIMER_FUNC();

    size_t B = l->input->batch;
    size_t F = l->input->features;


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

    nob_log(NOB_INFO, "log_softmax: ok");
}

double nll_loss(Matrix *const yhat, Matrix *const y)
{
    TIMER_FUNC();

    size_t B = yhat->batch;
    size_t F = yhat->features;

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

    return loss;
}

double accuracy(Matrix *const yhat, Matrix *const y)
{
    TIMER_FUNC();

    size_t B = yhat->batch;
    size_t F_yhat = yhat->features;
    size_t F_y = y->features;

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
    return res;
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
void cross_entropy_backward(LogSoftLayer *const l, Matrix *const y)
{
    TIMER_FUNC();

    size_t B = l->output->batch;
    size_t F = l->output->features;

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

    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}

void linear_backward(LinearLayer *const l)
{
    TIMER_FUNC();

    // Downstream:
    // Column-major: grad_input = W^T @ grad_output
    // Row-major:    grad_input = grad_output @ W^T
    // Note: Row-major storage causes W to be implicitly transposed
    // printf("linear_grad_input\n");
    TIMER_BLOCK("grad_input", {
            matrix_dgemm(LinalgNoTrans,
                         LinalgTrans,
                         1.0,
                         l->grad_output,
                         l->W,
                         0.0,
                         l->grad_input);
        });

    // Cost of weights:
    // Column-major: grad_W = grad_output @ input^T
    // Row-major:    grad_W = input^T @ grad_output
    // Note: Similar reasoning - Row-major storage causes input to be implicitly transposed
    // printf("linear_grad_W");
    TIMER_BLOCK("grad_W", {
            matrix_dgemm(LinalgTrans,
                         LinalgNoTrans,
                         1.0,
                         l->input,
                         l->grad_output,
                         0.0,
                         l->grad_W);
        });

    if (l->grad_bias != NULL) {
        size_t B = l->grad_output->batch;
        size_t F = l->grad_output->features;

        // Sum gradients across batch dimension
        double t0 = omp_get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < F; j++) {
                // Accumulate the bias used by all batch samples
#pragma omp atomic
                MIDX(l->grad_bias, 0, j) += MIDX(l->grad_output, i, j);
            }
        }

        timer_record("grad_bias", omp_get_wtime() - t0, NULL);
    }

    nob_log(NOB_INFO, "linear_backward: ok");
}

void normalize_backward(L2NormLayer *const l)
{
    TIMER_FUNC();

    size_t B = l->input->batch;
    size_t F = l->input->features;

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

    nob_log(NOB_INFO, "normalize_backward: ok");
}

void relu_backward(ReluLayer *const l)
{
    TIMER_FUNC();

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

    nob_log(NOB_INFO, "relu_backward: ok");
}

void sageconv_backward(SageLayer *const l, graph_t *const g)
{
    TIMER_FUNC();

    TIMER_BLOCK("grad_Wroot", {
    matrix_dgemm(LinalgTrans,
                 LinalgNoTrans,
                 1.0,
                 l->input,
                 l->grad_output,
                 0.0,
                 l->grad_Wroot);
        });

    TIMER_BLOCK("grad_Wagg", {
    matrix_dgemm(LinalgTrans,
                 LinalgNoTrans,
                 1.0,
                 l->agg,
                 l->grad_output,
                 0.0,
                 l->grad_Wagg);
        });

    TIMER_BLOCK("grad_input", {
    matrix_dgemm(LinalgNoTrans,
                 LinalgTrans,
                 1.0,
                 l->grad_output,
                 l->Wroot,
                 0.0,
                 l->grad_input);
        });

    size_t E = g->num_edges;
    size_t B = l->Wagg->batch;
    size_t F = l->grad_output->features;

    double t0 = omp_get_wtime();
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
#pragma omp atomic
            MIDX(l->grad_input, u0, i) += sum * l->mean_scale[u1];
        }
    }
    timer_record("grad_neigh", omp_get_wtime() - t0, NULL);

    nob_log(NOB_INFO, "sageconv_backward: ok");
}

// Update weights
void sage_layer_update_weights(SageLayer* const l, float lr)
{
    double scale = (double)-lr;

    daxpy(l->Wroot->M*l->Wroot->N, scale,
          l->grad_Wroot->data, 1,
          l->Wroot->data, 1);

    daxpy(l->Wagg->M*l->Wagg->N, scale,
          l->grad_Wagg->data, 1,
          l->Wagg->data, 1);

    nob_log(NOB_INFO, "update_sageconv_weights: ok");
}

// Reset gradient
void sage_layer_zero_gradients(SageLayer* l)
{
    matrix_zero(l->grad_output); // just a call to memset
    matrix_zero(l->grad_input);
    matrix_zero(l->grad_Wagg);
    matrix_zero(l->grad_Wroot);
}

void relu_layer_zero_gradients(ReluLayer* l)
{
    matrix_zero(l->grad_output);
    matrix_zero(l->grad_input);
}

void l2norm_layer_zero_gradients(L2NormLayer* l)
{
    matrix_zero(l->grad_output);
    matrix_zero(l->grad_input);
}

void logsoft_layer_zero_gradients(LogSoftLayer* l)
{
    matrix_zero(l->grad_input);
}

// Network-wide gradient reset
void sage_net_zero_gradients(SageNet* net)
{
    TIMER_FUNC();
    for (size_t i = 0; i < net->enc_depth; i++) {
        sage_layer_zero_gradients(net->enc_sage[i]);
        relu_layer_zero_gradients(net->enc_relu[i]);
        l2norm_layer_zero_gradients(net->enc_norm[i]);
    }

    sage_layer_zero_gradients(net->cls_sage);
    logsoft_layer_zero_gradients(net->logsoft);
}
