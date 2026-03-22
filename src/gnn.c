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
#include "dataset.h"
#include "timer.h"
#include "linalg/linalg.h"

#include "../nob.h"

// Forward propagation
void aggregate(SageLayer *const l)
{
    TIMER_FUNC();

    uint32_t in_dim = l->in_dim;

    double* X = l->input->data;
    size_t ldX = l->input->stride;

    int nthreads = omp_get_max_threads();
    double* t_gather = calloc(nthreads, sizeof(double));
    if (!t_gather) ERROR("Could not calloc t_gather");
    double* t_pool = calloc(nthreads, sizeof(double));
    if (!t_pool) ERROR("Could not calloc t_pool");

    memset(l->agg->data, 0, l->num_nodes * in_dim * sizeof(*l->agg->data));
    memset(l->mean_scale, 0, l->num_nodes * sizeof(*l->mean_scale));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t* adj = malloc(l->num_nodes * sizeof(size_t));

#pragma omp for
        for (size_t i = 0; i < l->num_nodes; i++) {
            size_t neigh_count = 0;
            double *Y = l->agg->data + i * l->agg->stride;

            // NOTE: Collects neighbors from only incoming direction.

            double t0 = omp_get_wtime();
            // Find neighbors of count sample size
            for (size_t e = 0; e < l->num_edges; e++) {

                // The source_to_target flow is the default of torch_geometric.nn.conv.message_passing,
                // and as far as I can tell non of the entries of OGB leaderboard that uses SageCONV
                // changes this to target_to_source. But, many seems to transform ogb-arxiv (directed graph)
                // to have symmetric edges, making it an undirected graph. GraphSAGE paper also uses undirected
                // citation graph dataset for their experiments.

                // paper src cites i (source_to_target)
                if (i == l->edges.dst[e])
                {
                    adj[neigh_count++] = l->edges.src[e];
                }

                // else if (i == src) { // paper i cites dst (target_to_source)
                //     adj[neigh_count++] = dst;
                // }
            }

            t_gather[tid] += omp_get_wtime() - t0;

            if (neigh_count == 0) continue;

            double scale = 1.0 / neigh_count;

            double t1 = omp_get_wtime();
            for (size_t j = 0; j < in_dim; j++) {
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

void sageconv(SageLayer *const l)
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

// #if defined(BASELINE) || defined(AGGREGATE_BASELINE)
//     aggregate_coo(l, nodes, edges);
// #else
//     aggregate_ccs(l, nodes, edges);
// #endif
    aggregate(l);
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

#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++) {
#pragma omp simd
        for (size_t j = 0; j < l->dim; j++) {
            MIDX(l->output, i, j) = fmax(0.0, MIDX(l->input, i, j));
        }
    }
}

void normalize(L2NormLayer *const l)
{
    TIMER_FUNC();

    const double eps = 1e-12;

#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++) {
        double sum_sq = 0.0;
        for (size_t j = 0; j < l->dim; j++) {
            double val = MIDX(l->input, i, j);
            sum_sq += val * val;
        }

        double norm = sqrt(sum_sq);
        double recip_mag = 1.0 / fmax(norm, eps);

        for (size_t j = 0; j < l->dim; j++) {
            MIDX(l->output, i, j) = MIDX(l->input, i, j) * recip_mag;
        }

        MIDX(l->recip_mag, i, 0) = recip_mag;
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
#pragma omp parallel for
        for (size_t i = 0; i < l->num_nodes; i++) {
            for (size_t j = 0; j < l->out_dim; j++) {
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

#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        double max = MIDX(l->input, i, 0);

        for (size_t j = 1; j < l->num_classes; j++)
        {
            max = fmax(max, MIDX(l->input, i, j));
        }

        double logsumexp = 0.0;
        for (size_t j = 0; j < l->num_classes; j++)
        {
            logsumexp += exp(MIDX(l->input, i, j) - max);
        }

        logsumexp = log(logsumexp);

        for (size_t j = 0; j < l->num_classes; j++)
        {
            MIDX(l->output, i, j) = MIDX(l->input, i, j) - max - logsumexp;
        }
    }

    nob_log(NOB_INFO, "log_softmax: ok");
}

double nll_loss(Matrix *const log_probs, uint32_t *labels)
{
    TIMER_FUNC();

    double loss = 0.0;
#pragma omp parallel for reduction(+:loss)
	for (size_t i = 0; i < log_probs->batch; i++) {
        uint32_t target = labels[i];
        double log_prob = MIDX(log_probs, i, target);
        loss -= log_prob;
    }

    return loss/log_probs->batch;
}

double accuracy(Matrix *const log_probs, uint32_t *labels, uint32_t num_classes)
{
    TIMER_FUNC();

    uint64_t correct = 0.0;
#pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < log_probs->batch; i++) {
        uint32_t target = labels[i];
        uint32_t pred_class = 0;
        for (size_t j = 1; j < num_classes; j++) {
            if (MIDX(log_probs, i, j) > MIDX(log_probs, i, pred_class)) {
                pred_class = j;
            }
        }

        if (pred_class == target) correct++;
    }

    return (double)correct/log_probs->batch;
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
// NOTE: we assume mean reduction from NLLLoss
void cross_entropy_backward(LogSoftLayer *const l, uint32_t *labels)
{
    TIMER_FUNC();

    uint32_t num_classes = l->num_classes;

    double scale = 1.0 / l->num_nodes;
#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++) {
        uint32_t target = labels[i];
        for (size_t j = 0; j < num_classes; j++) {
            double softmax_val = exp(MIDX(l->output, i, j));
            if (j == target)
                MIDX(l->grad_input, i, j) = (softmax_val - 1) * scale;
            else
                MIDX(l->grad_input, i, j) = softmax_val * scale;
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
        // Sum gradients across batch dimension
        double t0 = omp_get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < l->num_nodes; i++) {
            for (size_t j = 0; j < l->out_dim; j++) {
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

    uint32_t dim = l->dim;

#pragma omp parallel
    {
        Matrix* grad_local = matrix_create(dim, dim);
#pragma omp for
        for (size_t i = 0; i < l->num_nodes; i++) {
            for (size_t j = 0; j < dim; j++) {
                for (size_t k = 0; k < dim; k++) {
                    MIDX(grad_local, j, k) = - MIDX(l->output, i, j) * MIDX(l->output, i, k); // line: 358

                    if (j == k) {     // Kronecker delta
                        MIDX(grad_local, j, k) = 1 + MIDX(grad_local, j, k);
                    }

                    MIDX(grad_local, j, k) *= MIDX(l->recip_mag, i, 0);
                }
            }

            for (size_t j = 0; j < dim; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < dim; k++) {
                    sum += MIDX(l->grad_output, i, k) * MIDX(grad_local, k, j);
                }
                MIDX(l->grad_input, i, j) = sum;
            }
        }

        matrix_free(grad_local);
    }

    nob_log(NOB_INFO, "normalize_backward: ok");
}

void relu_backward(ReluLayer *const l)
{
    TIMER_FUNC();

    uint32_t dim = l->dim;

    // TODO: grad_input = grad_output * fmaxf(0.0f, copysignf(1.0f, output));
#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++) {
        for (size_t j = 0; j < dim; j++) {
            MIDX(l->grad_input, i, j) = MIDX(l->grad_output, i, j);
            if (MIDX(l->output, i, j) <= 0.0) {
                MIDX(l->grad_input, i, j) = 0;
            }
        }
    }

    nob_log(NOB_INFO, "relu_backward: ok");
}

void sageconv_backward(SageLayer *const l)
{
    TIMER_FUNC();

    size_t ldd_output = l->grad_output->stride;
    double *d_output = l->grad_output->data;

    double *input = l->input->data;
    size_t ldinput = l->input->stride;
    double *d_Wroot = l->grad_Wroot->data;
    size_t ldd_Wroot = l->grad_Wroot->stride;
    TIMER_BLOCK(l->timer_dWroot, {
            dgemm(l->in_dim, l->out_dim, l->num_nodes,
                  LinalgTrans, LinalgNoTrans,
                  1.0,
                  input, ldinput,
                  d_output, ldd_output,
                  0.0,
                  d_Wroot, ldd_Wroot);
        });

    double *agg = l->agg->data;
    size_t ldagg = l->agg->stride;
    double *d_Wagg = l->grad_Wagg->data;
    size_t ldd_Wagg = l->grad_Wagg->stride;
    TIMER_BLOCK(l->timer_dWagg, {
            dgemm(l->in_dim, l->out_dim, l->num_nodes,
                  LinalgTrans, LinalgNoTrans,
                  1.0,
                  agg, ldagg,
                  d_output, ldd_output,
                  0.0,
                  d_Wagg, ldd_Wagg);
                });

    size_t ldd_input = l->grad_input->stride;
    size_t ldWroot = l->Wroot->stride;
    double *d_input = l->grad_input->data;
    double *Wroot = l->Wroot->data;
    TIMER_BLOCK(l->timer_dinput, {
            dgemm(l->num_nodes, l->in_dim, l->out_dim,
                  LinalgNoTrans, LinalgTrans,
                  1.0,
                  d_output, ldd_output,
                  Wroot, ldWroot,
                  0.0,
                  d_input, ldd_input);
        });

    double t0 = omp_get_wtime();
#pragma omp parallel for
    for (size_t e = 0; e < l->num_edges; e++) {
        // With source_to_target flow, src gets gradient from dst's computation
        // (outgoing gradient) grad_input[src] += grad_output[dst] @ Wagg^T
        for (size_t i = 0; i < l->in_dim; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < l->out_dim; j++)
            {
                sum += MIDX(l->grad_output, l->edges.dst[e], j) * MIDX(l->Wagg, i, j);
            }
#pragma omp atomic
            MIDX(l->grad_input, l->edges.src[e], i) += sum * l->mean_scale[l->edges.dst[e]];
        }
    }
    timer_record(l->timer_dneigh, omp_get_wtime() - t0, NULL);

    nob_log(NOB_INFO, "sageconv_backward: ok");
}

// Update weights
// void sage_layer_update_weights(SageLayer* const l, float lr, bool do_adam)
// {
//     if (do_adam)
//     {
//         adam_step()
//     }
//     else
//     {
//         daxpy(l->Wroot->M*l->Wroot->N, -lr,
//               l->grad_Wroot->data, 1,
//               l->Wroot->data, 1);

//         daxpy(l->Wagg->M*l->Wagg->N, -lr,
//               l->grad_Wagg->data, 1,
//               l->Wagg->data, 1);
//     }

//     nob_log(NOB_INFO, "update_sageconv_weights: ok");
// }

// void linear_layer_update_weights(LinearLayer* const l, float lr)
// {
//     daxpy(l->W->M*l->W->N, -lr,
//           l->grad_W->data, 1,
//           l->W->data, 1);

//     if (l->grad_bias != NULL) {
//         daxpy(l->bias->N, -lr,
//               l->grad_bias->data, 1,
//               l->bias->data, 1);
//     }

//     nob_log(NOB_INFO, "update_linear_weights: ok");
// }

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

void normalize_layer_zero_gradients(L2NormLayer* l)
{
    matrix_zero(l->grad_output);
    matrix_zero(l->grad_input);
}

void linear_layer_zero_gradients(LinearLayer* l)
{
    matrix_zero(l->grad_output);
    matrix_zero(l->grad_input);
    matrix_zero(l->grad_W);
    matrix_zero(l->grad_bias);
}

void logsoft_layer_zero_gradients(LogSoftLayer* l)
{
    matrix_zero(l->grad_input);
}
