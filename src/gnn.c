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
void aggregate(SageLayer *const l)
{
    TIMER_FUNC();

    uint32_t num_edges = l->data->num_edges;
    uint32_t num_inputs = l->data->num_inputs;
    uint32_t in_dim = l->in_dim;

    EdgeIndex edges = l->data->edges;

    double* X = l->input->data;
    size_t ldX = l->input->stride;

    int nthreads = omp_get_max_threads();
    double* t_gather = calloc(sizeof(double), nthreads);
    if (!t_gather) ERROR("Could not calloc t_gather");
    double* t_pool = calloc(sizeof(double), nthreads);
    if (!t_pool) ERROR("Could not calloc t_pool");

    // printf("aggregate(): in_dim==l->agg->stride: %s\n", (in_dim==l->agg->stride) ? "True" : "False");
    memset(l->agg->data, 0, num_inputs*in_dim*sizeof(*l->agg->data));
    memset(l->mean_scale, 0, num_inputs * sizeof(*l->mean_scale));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t* adj = malloc(num_edges * sizeof(size_t));

#pragma omp for
        for (size_t i = 0; i < num_inputs; i++) {
            size_t neigh_count = 0;
            double *Y = l->agg->data + i * l->agg->stride;

            // NOTE: Collects neighbors from only incoming direction.

            double t0 = omp_get_wtime();
            // Find neighbors of count sample size
            for (size_t edge = 0; edge < num_edges; edge++) {
                uint32_t src = EDGE_SRC(&edges, edge);
                uint32_t dst = EDGE_DST(&edges, edge);

                // The source_to_target flow is the default of torch_geometric.nn.conv.message_passing,
                // and as far as I can tell non of the entries of OGB leaderboard that uses SageCONV
                // changes this to target_to_source. But, many seems to transform ogb-arxiv (directed graph)

                // to have symmetric edges, making it a undirected graph. GraphSAGE paper also uses undirected
                // citation graph dataset for their experiments.

                if (i == dst) { // paper src cites i (source_to_target)
                    adj[neigh_count++] = src;
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
    size_t num_inputs = l->data->num_inputs;
    size_t dim = l->dim;

#pragma omp parallel for
    for (size_t i = 0; i < num_inputs; i++) {
        for (size_t j = 0; j < dim; j++) {
            MIDX(l->output, i, j) = fmax(0.0, MIDX(l->input, i, j));
        }
    }
}

void normalize(L2NormLayer *const l)
{
    TIMER_FUNC();

    const double eps = 1e-12;
    size_t num_inputs = l->data->num_inputs;
    size_t dim = l->dim;

#pragma omp parallel for
    for (size_t i = 0; i < num_inputs; i++) {
        double sum_sq = 0.0;
        for (size_t j = 0; j < dim; j++) {
            double val = MIDX(l->input, i, j);
            sum_sq += val * val;
        }

        double norm = sqrt(sum_sq);
        double recip_mag = 1.0 / fmax(norm, eps);

        for (size_t j = 0; j < dim; j++) {
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
        uint32_t num_inputs = l->data->num_inputs;
        uint32_t out_dim = l->in_dim;

#pragma omp parallel for
        for (size_t i = 0; i < num_inputs; i++) {
            for (size_t j = 0; j < out_dim; j++) {
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

    size_t num_inputs = l->data->num_inputs;
    size_t num_classes = l->data->num_classes;


#pragma omp parallel for
    for (size_t i = 0; i < num_inputs; i++) {
        double max = MIDX(l->input, i, 0);

        for (size_t j = 1; j < num_classes; j++) {
            max = fmax(max, MIDX(l->input, i, j));
        }

        double logsumexp = 0.0;
        for (size_t j = 0; j < num_classes; j++) {
            logsumexp += exp(MIDX(l->input, i, j) - max);
        }

        logsumexp = log(logsumexp);

        for (size_t j = 0; j < num_classes; j++) {
            MIDX(l->output, i, j) = MIDX(l->input, i, j) - max - logsumexp;
        }
    }

    nob_log(NOB_INFO, "log_softmax: ok");
}

double nll_loss(Matrix *const log_probs, uint32_t *labels, Slice slice)
{
    TIMER_FUNC();

    size_t offset = slice.node.offset;
    size_t count = slice.node.count;

    double loss = 0.0;
#pragma omp parallel for reduction(+:loss)
	for (size_t i = offset; i < count+offset; i++) {
        uint32_t target = labels[i];
        double log_prob = MIDX(log_probs, i, target);
        loss -= log_prob;
    }

    return loss/count;
}

double accuracy(Matrix *const log_probs, uint32_t *labels, uint32_t num_classes, Slice slice)
{
    TIMER_FUNC();

    size_t offset = slice.node.offset;
    size_t count = slice.node.count;

    uint64_t correct = 0.0;
#pragma omp parallel for reduction(+:correct)
    for (size_t i = offset; i < count+offset; i++) {
        uint32_t target = labels[i];
        uint32_t pred_class = 0;
        for (size_t j = 1; j < num_classes; j++) {
            if (MIDX(log_probs, i, j) > MIDX(log_probs, i, pred_class)) {
                pred_class = j;
            }
        }

        if (pred_class == target) correct++;
    }

    return (double)correct/count;
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
// NOTE: we assume mean reduction from NLLLoss
void cross_entropy_backward(LogSoftLayer *const l, Slice train)
{
    TIMER_FUNC();

    uint32_t offset = train.node.offset;
    uint32_t count = train.node.count;
    uint32_t num_classes = l->data->num_classes;
    uint32_t *labels = l->data->labels;

    double scale = 1.0 / count;
#pragma omp parallel for
    for (size_t i = offset; i < count+offset; i++) {
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

void linear_backward(LinearLayer *const l, Slice train)
{
    TIMER_FUNC();

    uint32_t offset = train.node.offset;
    uint32_t count = train.node.count;
    uint32_t num_features = l->data->num_features;

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
        for (size_t i = offset; i < count+offset; i++) {
            for (size_t j = 0; j < num_features; j++) {
                // Accumulate the bias used by all batch samples
#pragma omp atomic
                MIDX(l->grad_bias, 0, j) += MIDX(l->grad_output, i, j);
            }
        }

        timer_record("grad_bias", omp_get_wtime() - t0, NULL);
    }

    nob_log(NOB_INFO, "linear_backward: ok");
}

void normalize_backward(L2NormLayer *const l, Slice train)
{
    TIMER_FUNC();

    uint32_t offset = train.node.offset;
    uint32_t count = train.node.count;
    uint32_t num_features = l->data->num_features;

#pragma omp parallel
    {
        Matrix* grad_local = matrix_create(num_features, num_features);
#pragma omp for
        for (size_t i = offset; i < count+offset; i++) {
            for (size_t j = 0; j < num_features; j++) {
                for (size_t k = 0; k < num_features; k++) {
                    MIDX(grad_local, j, k) = - MIDX(l->output, i, j) * MIDX(l->output, i, k);

                    if (j == k) {     // Kronecker delta
                        MIDX(grad_local, j, k) = 1 + MIDX(grad_local, j, k);
                    }

                    MIDX(grad_local, j, k) *= MIDX(l->recip_mag, i, 0);
                }
            }

            for (size_t j = 0; j < num_features; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < num_features; k++) {
                    sum += MIDX(l->grad_output, i, k) * MIDX(grad_local, k, j);
                }
                MIDX(l->grad_input, i, j) = sum;
            }
        }

        matrix_destroy(grad_local);
    }

    nob_log(NOB_INFO, "normalize_backward: ok");
}

void relu_backward(ReluLayer *const l, Slice train)
{
    TIMER_FUNC();

    uint32_t offset = train.node.offset;
    uint32_t count = train.node.count;
    uint32_t num_features = l->data->num_features;

    // TODO: grad_input = grad_output * fmaxf(0.0f, copysignf(1.0f, output));
#pragma omp parallel for
    for (size_t i = offset; i < count+offset; i++) {
        for (size_t j = 0; j < num_features; j++) {
            MIDX(l->grad_input, i, j) = MIDX(l->grad_output, i, j);
            if (MIDX(l->output, i, j) <= 0.0) {
                MIDX(l->grad_input, i, j) = 0;
            }
        }
    }

    nob_log(NOB_INFO, "relu_backward: ok");
}

void sageconv_backward(SageLayer *const l, Slice train)
{
    TIMER_FUNC();

    uint32_t offset = train.node.offset;
    uint32_t count = train.node.count;
    uint32_t edge_count = train.edge.count;
    uint32_t edge_offset = train.edge.offset;
    EdgeIndex edges = l->data->edges;

    TIMER_BLOCK(l->timer_dWroot, {
            double *input = l->input->data + offset * l->input->stride;
            double *grad_output = l->grad_output->data + offset * l->grad_output->stride;
            double *grad_Wroot = l->grad_Wroot->data;
            dgemm(l->in_dim, l->out_dim, count,
                  LinalgTrans, LinalgNoTrans,
                  1.0,
                  input, l->input->stride,
                  grad_output, l->grad_output->stride,
                  0.0,
                  grad_Wroot, l->grad_Wroot->stride);
        });

    TIMER_BLOCK(l->timer_dWagg, {
            double *agg = l->agg->data + offset * l->agg->stride;
            double *grad_output = l->grad_output->data + offset * l->grad_output->stride;
            double *grad_Wagg = l->grad_Wagg->data;
            dgemm(l->in_dim, l->out_dim, count,
                  LinalgTrans, LinalgNoTrans,
                  1.0,
                  agg, l->agg->stride,
                  grad_output, l->grad_output->stride,
                  0.0,
                  grad_Wagg, l->grad_Wagg->stride);
                });

    TIMER_BLOCK(l->timer_dinput, {
            double *grad_output = l->grad_output->data + offset * l->grad_output->stride;
            double *Wroot = l->Wroot->data;
            double *grad_input = l->grad_input->data + offset * l->grad_input->stride;
            dgemm(count, l->in_dim, l->out_dim,
                  LinalgNoTrans, LinalgTrans,
                  1.0,
                  grad_output, l->grad_output->stride,
                  Wroot, l->Wroot->stride,
                  0.0,
                  grad_input, l->grad_input->stride);
        });

    double t0 = omp_get_wtime();
#pragma omp parallel for
    for (size_t edge = edge_offset; edge < edge_count+edge_offset; edge++) {
        uint32_t src = EDGE_SRC(&edges, edge);
        uint32_t dst = EDGE_DST(&edges, edge);

        // With source_to_target flow, src gets gradient from dst's computation
        // (outgoing gradient) grad_input[src] += grad_output[dst] @ Wagg^T
        for (size_t i = 0; i < l->in_dim; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < l->out_dim; j++) {
                sum += MIDX(l->grad_output, dst, j) * MIDX(l->Wagg, i, j);
            }
#pragma omp atomic
            MIDX(l->grad_input, src, i) += sum * l->mean_scale[dst];
        }
    }
    timer_record(l->timer_dneigh, omp_get_wtime() - t0, NULL);

    nob_log(NOB_INFO, "sageconv_backward: ok");
}

// Update weights
void sage_layer_update_weights(SageLayer* const l, float lr, Slice train)
{
    (void)train;
    // uint32_t offset = train.node.offset;
    // uint32_t count = train.node.count;
    // uint32_t num_features = l->data->num_features;

    double scale = (double)-lr;

    daxpy(l->Wroot->M*l->Wroot->N, scale,
          l->grad_Wroot->data, 1,
          l->Wroot->data, 1);

    daxpy(l->Wagg->M*l->Wagg->N, scale,
          l->grad_Wagg->data, 1,
          l->Wagg->data, 1);

    nob_log(NOB_INFO, "update_sageconv_weights: ok");
}

void linear_layer_update_weights(LinearLayer* const l, float lr, Slice train)
{
    (void)train;
    // uint32_t count = train.node.count

    double scale = (double)-lr;

    daxpy(l->W->M*l->W->N, scale,
          l->grad_W->data, 1,
          l->W->data, 1);

    if (l->grad_bias != NULL) {
        daxpy(l->bias->N, scale,
              l->grad_bias->data, 1,
              l->bias->data, 1);
    }

    nob_log(NOB_INFO, "update_linear_weights: ok");
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
#ifdef USE_PREDICTION_HEAD
    linear_layer_zero_gradients(net->linear);
#endif
    logsoft_layer_zero_gradients(net->logsoft);
}
