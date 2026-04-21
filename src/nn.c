#include <stdint.h>

#include "core.h"
#include "layers.h"
#include "matmul_naive.h"
#include "timer.h"

void relu(ReluLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t dim       = l->dim;

    const Real *restrict input  = l->input;
    Real       *restrict output = l->output;

    int64_t n = num_nodes * dim;
#pragma omp parallel for simd
    for (int64_t i = 0; i < n; i++)
    {
        output[i] = (input[i] > REAL(0.0)) ? input[i] : REAL(0.0);
    }
}

void grad_relu(ReluLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t dim       = l->dim;

    const Real *restrict output      = l->output;
    const Real *restrict grad_output = l->grad_output;
    Real       *restrict grad_input  = l->grad_input;

    int64_t n = num_nodes * dim;
#pragma omp parallel for simd
    for (int64_t i = 0; i < n; i++)
    {
        grad_input[i] = (output[i] > REAL(0.0)) ? grad_output[i] : REAL(0.0);
    }
}


void l2norm(L2NormLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t dim       = l->dim;

    const Real *restrict input     = l->input;
    Real       *restrict output    = l->output;
    Real       *restrict recip_mag = l->recip_mag;

    const Real eps = REAL(1e-12);
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        Real sum_sq = 0.0;
#pragma omp simd reduction(+:sum_sq)
        for (int64_t j = 0; j < dim; j++)
        {
            Real val = input[i*dim+j];
            sum_sq += val * val;
        }

        Real safe_sum_sq = real_fmax(sum_sq, eps * eps);
        Real recip = REAL(1.0) / real_sqrt(safe_sum_sq);

        const Real *restrict in_row  = &input[i*dim];
        Real       *restrict out_row = &output[i*dim];
#pragma omp simd
        for (int64_t j = 0; j < dim; j++)
        {
            out_row[j] = in_row[j] * recip;
        }

        recip_mag[i] = recip;
    }
}

void grad_l2norm(L2NormLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t dim       = l->dim;

    const Real *restrict output      = l->output;
    const Real *restrict grad_output = l->grad_output;
    Real       *restrict grad_input  = l->grad_input;
    const Real *restrict recip_mag   = l->recip_mag;

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        const Real *out_row = &output[i * dim];
        const Real *go_row  = &grad_output[i * dim];
        Real *gi_row  = &grad_input[i * dim];
        const Real recip    = recip_mag[i];

        // dot(y, grad_output) for this node
        Real dot = 0.0;
        for (int64_t j = 0; j < dim; j++)
        {
            dot += out_row[j] * go_row[j];
        }

        // grad_input = recip_mag * (grad_output - output * dot)
        for (int64_t j = 0; j < dim; j++)
        {
            gi_row[j] = recip * (go_row[j] - out_row[j] * dot);
        }
    }
}

void linear(LinearLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;
    int64_t out_dim   = l->out_dim;

    // output = input @ Wroot
    matmul(MatmulNoTrans, MatmulNoTrans,
           num_nodes, in_dim, out_dim,
           1.0,
           l->input,  in_dim,
           l->W,      out_dim,
           0.0,
           l->output, out_dim);

    const Real *restrict bias   = l->bias;
    Real       *restrict output = l->output;
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        Real *restrict out_row = &output[i*out_dim];
        for (int64_t j = 0; j < out_dim; j++)
        {
            out_row[j] += bias[j];
        }
    }
}

void grad_linear(LinearLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;
    int64_t out_dim   = l->out_dim;

    // grad_input = grad_output @ W^T
    TIMER_BLOCK("dinput", {
            matmul(MatmulNoTrans, MatmulTrans,
                   num_nodes, in_dim, out_dim,
                   1.0,
                   l->grad_output, out_dim,
                   l->W,           out_dim,
                   0.0,
                   l->grad_input,  in_dim);
        });

    // grad_W = input^T @ grad_output
    TIMER_BLOCK("dW", {
            matmul(MatmulTrans, MatmulNoTrans,
                   in_dim, out_dim, num_nodes,
                   1.0,
                   l->input,       in_dim,
                   l->grad_output, out_dim,
                   0.0,
                   l->grad_W,      out_dim);
        });

    double t = omp_get_wtime();

    const Real *restrict grad_output = l->grad_output;
    Real       *restrict grad_bias   = l->grad_bias;
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        const Real *restrict go_row = &grad_output[i*out_dim];
#pragma omp simd
        for (int64_t j = 0; j < out_dim; j++)
        {
            grad_bias[j] += go_row[j];
        }
    }

    timer_record("grad_bias", omp_get_wtime() - t, NULL);
}

/*
 * Log Sum Exp: https://stackoverflow.com/a/61570752
 */
void logsoftmax(LogSoftmaxLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes   = l->num_nodes;
    int64_t dim         = l->dim; // number of classes

    const Real *restrict input  = l->input;
    Real       *restrict output = l->output;

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        const Real *restrict in_row = &input[i*dim];
        Real max_logit = *in_row;
#pragma omp simd reduction(max:max_logit)
        for (int64_t j = 1; j < dim; j++)
        {
            max_logit = in_row[j] > max_logit ? in_row[j] : max_logit;
        }

        Real logsumexp = 0.0;
#pragma omp simd reduction(+:logsumexp)
        for (int64_t j = 0; j < dim; j++)
        {
            logsumexp += real_exp(in_row[j] - max_logit);
        }
        logsumexp = real_log(logsumexp);

        Real *restrict out_row = &output[i*dim];
#pragma omp simd
        for (int64_t j = 0; j < dim; j++)
        {
            out_row[j] = in_row[j] - max_logit - logsumexp;
        }
    }
}

Real nll_loss(LogSoftmaxLayer *l, const int64_t *labels)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t dim       = l->dim; // number of classes

    const Real *restrict output = l->output;

    Real loss = 0.0;
#pragma omp parallel for reduction(+:loss)
    for (int64_t i = 0; i < num_nodes; i++)
    {
        loss -= output[i*dim+labels[i]];
    }

    return loss / num_nodes;
}

Real accuracy(const LogSoftmaxLayer *l, const int64_t *labels)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t dim       = l->dim; // number of classses

    const Real *restrict output = l->output;

    uint64_t correct = 0;
#pragma omp parallel for reduction(+:correct)
    for (int64_t i = 0; i < l->num_nodes; i++)
    {
        const Real *out_row = &output[i*dim];
        int64_t pred_class = 0;
        for (int64_t j = 1; j < l->dim; j++)
        {
            if (out_row[j] > out_row[pred_class])
            {
                pred_class = j;
            }
        }

        if (pred_class == labels[i])
        {
            correct++;
        }
    }

    return (Real)correct / num_nodes;
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
// NOTE: we assume mean reduction from NLLLoss
void grad_cross_entropy(LogSoftmaxLayer *const l, int64_t *labels)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    uint32_t dim      = l->dim; // number of classes

    const Real *restrict output     = l->output;
    Real       *restrict grad_input = l->grad_input;

    Real scale = REAL(1.0) / num_nodes;
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        const Real *restrict out_row = &output[i*dim];
        Real       *restrict gi_row  = &grad_input[i*dim];

        int64_t target = labels[i];
        for (int64_t j = 0; j < dim; j++)
        {
            Real softmax_val = real_exp(out_row[j]);
            if (j == target)
            {
                gi_row[j] = (softmax_val - 1) * scale;
            }
            else
            {
                gi_row[j] = softmax_val * scale;
            }
        }
    }
}

// Reset gradient
void sage_layer_zero_gradients(SageLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->in_dim);
    real_zero_out(l->grad_Wagg, l->in_dim * l->out_dim);
    real_zero_out(l->grad_Wroot, l->in_dim * l->out_dim);
}

void relu_layer_zero_gradients(ReluLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->dim);
}

void normalize_layer_zero_gradients(L2NormLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->dim);
}

void linear_layer_zero_gradients(LinearLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->in_dim);
    real_zero_out(l->grad_W, l->in_dim * l->out_dim);
    real_zero_out(l->grad_bias, l->out_dim);
}

void logsoft_layer_zero_gradients(LogSoftmaxLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->dim);
}
