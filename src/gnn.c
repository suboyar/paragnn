#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>
#include <cblas.h>

#include "core.h"
#include "gnn.h"
#include "dataset.h"
#include "timer.h"
#include "linalg/linalg.h"

#include "../nob.h"

// Forward propagation
void aggregate(SageLayer *const l)
{
    TIMER_FUNC();

    uint32_t in_dim = l->in_dim;

    double* X = l->input;
    size_t ldX = l->in_dim;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t *tid_adj = l->tls_adj + tid * l->num_nodes;

#pragma omp for
        for (size_t i = 0; i < l->num_nodes; i++) {
            size_t neigh_count = 0;
            double *Y = l->agg + i * l->in_dim;

            // NOTE: Collects neighbors from only incoming direction.

            for (size_t e = 0; e < l->num_edges; e++) {

                // The source_to_target flow is the default of torch_geometric.nn.conv.message_passing,
                // and as far as I can tell non of the entries of OGB leaderboard that uses SageCONV
                // changes this to target_to_source. But, many seems to transform ogb-arxiv (directed graph)
                // to have symmetric edges, making it an undirected graph. GraphSAGE paper also uses undirected
                // citation graph dataset for their experiments.

                // paper src cites i (source_to_target)
                if (i == l->edges.dst[e])
                {
                    tid_adj[neigh_count++] = l->edges.src[e];
                }

                // else if (i == l->edges.src[e]) { // paper i cites dst (target_to_source)
                //     adj[neigh_count++] = l->edges.dst[e];
                // }
            }

            if (neigh_count == 0) continue;

            double scale = 1.0 / neigh_count;

            for (size_t j = 0; j < in_dim; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < neigh_count; k++) {
                    sum += X[tid_adj[k] * ldX + j];
                }
                Y[j] = sum * scale;
            }
            l->mean_scale[i] = scale;
        }
    }
}

void sageconv(SageLayer *const l)
{
    TIMER_FUNC();

    size_t M = l->num_nodes;
    size_t N = l->out_dim;
    size_t K = l->in_dim;

    // output = input @ Wroot
    // input:  num_nodes x in_dim
    // Wroot:  in_dim x out_dim
    // output: num_nodes x out_dim
    TIMER_BLOCK("Wroot", {
            dgemm(M, N, K,
                  LinalgNoTrans,
                  LinalgNoTrans,
                  1.0,
                  l->input, l->in_dim,
                  l->Wroot, l->out_dim,
                  0.0,
                  l->output, l->out_dim);
        });

// #if defined(BASELINE) || defined(AGGREGATE_BASELINE)
//     aggregate_coo(l, nodes, edges);
// #else
//     aggregate_ccs(l, nodes, edges);
// #endif
    aggregate(l);

    // output += agg @ Wagg
    // agg: num_nodes x in_dim
    // Wagg: in_dim x out_dim
    // output: num_nodes x out_dim
    TIMER_BLOCK("Wagg", {
            dgemm(M, N, K,
                  LinalgNoTrans,
                  LinalgNoTrans,
                  1.0,
                  l->agg, l->in_dim,
                  l->Wagg, l->out_dim,
                  1.0,
                  l->output, l->out_dim);
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
            l->output[i*l->dim+j] = fmax(0.0, l->input[i*l->dim+j]);
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
            double val = l->input[i*l->dim+j];
            sum_sq += val * val;
        }

        double norm = sqrt(sum_sq);
        double recip_mag = 1.0 / fmax(norm, eps);

        for (size_t j = 0; j < l->dim; j++) {
            l->output[i*l->dim+j] = l->input[i*l->dim+j] * recip_mag;
        }

        l->recip_mag[i] = recip_mag;
    }

    nob_log(NOB_INFO, "normalize: ok");
}

void linear(LinearLayer *const l)
{
    TIMER_FUNC();

    size_t M = l->num_nodes;
    size_t N = l->out_dim;
    size_t K = l->in_dim;

    // output = input @ Wroot
    // input:  num_nodes x in_dim
    // W:  in_dim x out_dim
    // output: num_nodes x out_dim
    dgemm(M, N, K,
          LinalgNoTrans,
          LinalgNoTrans,
          1.0,
          l->input, l->in_dim,
          l->W, l->out_dim,
          0.0,
          l->output, l->out_dim);

#pragma omp parallel for
        for (size_t i = 0; i < l->num_nodes; i++) {
            for (size_t j = 0; j < l->out_dim; j++) {
                l->output[i*l->out_dim+j] += l->bias[j];
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
        double max = l->input[i*l->dim];

        for (size_t j = 1; j < l->num_classes; j++)
        {
            max = fmax(max, l->input[i*l->dim+j]);
        }

        double logsumexp = 0.0;
        for (size_t j = 0; j < l->num_classes; j++)
        {
            logsumexp += exp(l->input[i*l->dim+j] - max);
        }

        logsumexp = log(logsumexp);

        for (size_t j = 0; j < l->num_classes; j++)
        {
            l->output[i*l->dim+j] = l->input[i*l->dim+j] - max - logsumexp;
        }
    }

    nob_log(NOB_INFO, "log_softmax: ok");
}

float nll_loss(LogSoftLayer *l, const uint32_t *labels)
{
    TIMER_FUNC();

    float loss = 0.0;
#pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        loss -= (float)l->output[i * l->num_classes + labels[i]];
    }

    return loss / l->num_nodes;
}

float accuracy(const LogSoftLayer *l, const uint32_t *labels)
{
    TIMER_FUNC();

    uint64_t correct = 0;
#pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        const Real *row = &l->output[i * l->num_classes];
        uint32_t pred_class = 0;
        for (size_t j = 1; j < l->num_classes; j++)
        {
            if (row[j] > row[pred_class])
                pred_class = j;
        }

        if (pred_class == labels[i])
            correct++;
    }

    return (float)correct / l->num_nodes;
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
            double softmax_val = exp(l->output[i*l->dim+j]);
            if (j == target)
                l->grad_input[i*l->dim+j] = (softmax_val - 1) * scale;
            else
                l->grad_input[i*l->dim+j] = softmax_val * scale;
        }
    }

    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}

void linear_backward(LinearLayer *const l)
{
    TIMER_FUNC();

    // grad_input = grad_output @ W^T
    // grad_output: num_nodes * out_dim
    // W:           in_dim * out_dim
    // grad_input:  num_nodes * in_dim
    TIMER_BLOCK("grad_input", {
            dgemm(l->num_nodes, l->in_dim, l->out_dim,
                  LinalgNoTrans, LinalgTrans,
                  1.0,
                  l->grad_output, l->out_dim,
                  l->W,           l->out_dim,
                  0.0,
                  l->grad_input,  l->in_dim);
        });

    // grad_W = input^T @ grad_output
    // input:       num_nodes * in_dim
    // grad_output: num_nodes * out_dim
    // grad_W:      in_dim * out_dim
    TIMER_BLOCK("grad_W", {
            dgemm(l->in_dim, l->out_dim, l->num_nodes,
                  LinalgTrans, LinalgNoTrans,
                  1.0,
                  l->input,       l->in_dim,
                  l->grad_output, l->out_dim,
                  0.0,
                  l->grad_W,      l->out_dim);
        });

    double t0 = omp_get_wtime();

    // Zero first, then accumulate
    for (size_t j = 0; j < l->out_dim; j++)
        l->grad_bias[j] = 0.0;

#pragma omp parallel for
    for (size_t j = 0; j < l->out_dim; j++) {
        Real sum = 0.0;
        for (size_t i = 0; i < l->num_nodes; i++) {
            sum += l->grad_output[i * l->out_dim + j];
        }
        l->grad_bias[j] = sum;
    }

    timer_record("grad_bias", omp_get_wtime() - t0, NULL);

    nob_log(NOB_INFO, "linear_backward: ok");
}

void normalize_backward(L2NormLayer *const l)
{
    TIMER_FUNC();

    size_t dim = l->dim;

#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        Real *out_i      = &l->output[i * dim];
        Real *grad_out_i = &l->grad_output[i * dim];
        Real *grad_in_i  = &l->grad_input[i * dim];
        Real rm           = l->recip_mag[i];

        // dot(y, grad_output) for this node
        Real dot = 0.0;
        for (size_t j = 0; j < dim; j++)
            dot += out_i[j] * grad_out_i[j];

        // grad_input = recip_mag * (grad_output - output * dot)
        for (size_t j = 0; j < dim; j++)
            grad_in_i[j] = rm * (grad_out_i[j] - out_i[j] * dot);
    }

    nob_log(NOB_INFO, "normalize_backward: ok");
}

void relu_backward(ReluLayer *const l)
{
    TIMER_FUNC();

    size_t n = l->num_nodes * l->dim;

    // TODO: grad_input = grad_output * fmaxf(0.0f, copysignf(1.0f, output));
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        l->grad_input[i] = (l->output[i] > 0.0) ? l->grad_output[i] : 0.0;
    }

    nob_log(NOB_INFO, "relu_backward: ok");
}

void sageconv_backward(SageLayer *const l)
{
    TIMER_FUNC();

    // grad_Wroot = input^T @ grad_output
    // input:       num_nodes x in_dim
    // grad_output: num_nodes x out_dim
    // grad_Wroot:  in_dim x out_dim
    TIMER_BLOCK(l->timer_dWroot, {
        dgemm(l->in_dim, l->out_dim, l->num_nodes,
              LinalgTrans, LinalgNoTrans,
              1.0,
              l->input,      l->in_dim,
              l->grad_output, l->out_dim,
              0.0,
              l->grad_Wroot, l->out_dim);
    });

    // grad_Wagg = agg^T @ grad_output
    // agg:         num_nodes x in_dim
    // grad_output: num_nodes x out_dim
    // grad_Wagg:   in_dim x out_dim
    TIMER_BLOCK(l->timer_dWagg, {
        dgemm(l->in_dim, l->out_dim, l->num_nodes,
              LinalgTrans, LinalgNoTrans,
              1.0,
              l->agg,        l->in_dim,
              l->grad_output, l->out_dim,
              0.0,
              l->grad_Wagg,  l->out_dim);
    });

    // grad_input = grad_output @ Wroot^T
    // grad_output: num_nodes x out_dim
    // Wroot:       in_dim x out_dim
    // grad_input:  num_nodes x in_dim
    TIMER_BLOCK(l->timer_dinput, {
        dgemm(l->num_nodes, l->in_dim, l->out_dim,
              LinalgNoTrans, LinalgTrans,
              1.0,
              l->grad_output, l->out_dim,
              l->Wroot,       l->out_dim,
              0.0,
              l->grad_input,  l->in_dim);
    });

    // Neighbor gradient scatter:
    // grad_input[src] += mean_scale[dst] * grad_output[dst] @ Wagg^T
    double t0 = omp_get_wtime();
#pragma omp parallel for
    for (size_t e = 0; e < l->num_edges; e++)
    {
        uint32_t src = l->edges.src[e];
        uint32_t dst = l->edges.dst[e];
        Real scale = l->mean_scale[dst];
        Real *grad_out_dst = &l->grad_output[dst * l->out_dim];
        Real *grad_in_src  = &l->grad_input[src * l->in_dim];

        for (size_t i = 0; i < l->in_dim; i++)
        {
            Real sum = 0.0;
            for (size_t j = 0; j < l->out_dim; j++)
            {
                sum += grad_out_dst[j] * l->Wagg[i * l->out_dim + j];
            }
#pragma omp atomic
            grad_in_src[i] += sum * scale;
        }
    }
    timer_record(l->timer_dneigh, omp_get_wtime() - t0, NULL);

    nob_log(NOB_INFO, "sageconv_backward: ok");
}

// Reset gradient
#define GRAD_PARALLEL_THRESHOLD 32768  // 256KB of doubles i.e L2 cache
static inline void zero_out(Real *a, size_t n)
{
    if (n < GRAD_PARALLEL_THRESHOLD)
    {
        memset(a, 0, n * sizeof(Real));
    }
    else
    {
#pragma omp parallel for simd
        for (size_t i = 0; i < n; i++)
        {
            a[i] = 0.0;
        }
    }
}

void sage_layer_zero_gradients(SageLayer* l)
{
    zero_out(l->grad_input, l->num_nodes * l->in_dim);
    zero_out(l->grad_Wagg, l->in_dim * l->out_dim);
    zero_out(l->grad_Wroot, l->in_dim * l->out_dim);
}

void relu_layer_zero_gradients(ReluLayer* l)
{
    zero_out(l->grad_input, l->num_nodes * l->dim);
}

void normalize_layer_zero_gradients(L2NormLayer* l)
{
    zero_out(l->grad_input, l->num_nodes * l->dim);
}

void linear_layer_zero_gradients(LinearLayer* l)
{
    zero_out(l->grad_input, l->num_nodes * l->in_dim);
    zero_out(l->grad_W, l->in_dim * l->out_dim);
    zero_out(l->grad_bias, l->out_dim);
}

void logsoft_layer_zero_gradients(LogSoftLayer* l)
{
    zero_out(l->grad_input, l->num_nodes * l->dim);
}
