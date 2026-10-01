#include <stdint.h>

#include "core.h"
#include "layers.h"
#include "matmul_naive.h"
#include "timer.h"

__attribute__((optimize("unroll-loops")))
void relu(ReluLayer *const l)
{
    TIMER_FUNC();

    Real *restrict x  = l->x;

    int64_t n = l->node_count * l->dim;
#pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; i++)
        x[i] = (x[i] > REAL(0.0)) ? x[i] : REAL(0.0);
}

__attribute__((optimize("unroll-loops")))
void grad_relu(ReluLayer *const l)
{
    TIMER_FUNC();

    const Real *restrict x = l->x;
    Real *restrict dx      = l->dx;

    int64_t n = l->node_count * l->dim;
#pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; i++)
        dx[i] = (x[i] > REAL(0.0)) ? dx[i] : REAL(0.0);
}

void l2norm(L2NormLayer *const l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t dim       = l->dim;

    Real *restrict x        = l->x;
    Real *restrict inv_norm = l->inv_norm;

    const Real eps = REAL(1e-12);
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real sum_sq = 0.0;
#pragma omp simd reduction(+:sum_sq)
        for (int64_t d = 0; d < dim; d++)
        {
            Real val = x[n*dim+d];
            sum_sq += val * val;
        }

        Real safe_sum_sq = real_fmax(sum_sq, eps * eps);
        Real scale = REAL(1.0) / real_sqrt(safe_sum_sq);

        Real *restrict x_ptr = &x[n*dim];
#pragma omp simd
        for (int64_t d = 0; d < dim; d++)
            x_ptr[d] = x_ptr[d] * scale;

        inv_norm[n] = scale;
    }
}

void grad_l2norm(L2NormLayer *const l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t dim       = l->dim;

    const Real *restrict x         = l->x;
    Real       *restrict dx        = l->dx;
    const Real *restrict inv_norm  = l->inv_norm;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        const Real *x_ptr  = &x[n*dim];
        Real       *dx_ptr = &dx[n*dim];
        const Real  scale  = inv_norm[n];

        // dot(y, grad_output) for this node
        Real dot = 0.0;
        for (int64_t d = 0; d < dim; d++)
            dot += x_ptr[d] * dx_ptr[d];

        // grad_input = scale * (grad_output - output * dot)
        for (int64_t d = 0; d < dim; d++)
            dx_ptr[d] = scale * (dx_ptr[d] - x_ptr[d] * dot);
    }
}

/*
 * Log Sum Exp: https://stackoverflow.com/a/61570752
 */
void logsoftmax(LogSoftmaxLayer *const l)
{
    TIMER_FUNC();

    int64_t node_count   = l->node_count;
    int64_t dim         = l->dim; // number of classes

    Real *restrict x = l->x;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real *restrict x_ptr = &x[n*dim];
        Real max_logit = *x_ptr;
#pragma omp simd reduction(max:max_logit)
        for (int64_t j = 1; j < dim; j++)
            max_logit = x_ptr[j] > max_logit ? x_ptr[j] : max_logit;

        Real logsumexp = 0.0;
#pragma omp simd reduction(+:logsumexp)
        for (int64_t j = 0; j < dim; j++)
            logsumexp += real_exp(x_ptr[j] - max_logit);

        logsumexp = real_log(logsumexp);

#pragma omp simd
        for (int64_t j = 0; j < dim; j++)
            x_ptr[j] -= max_logit + logsumexp;
    }
}

Real nll_loss(LogSoftmaxLayer *l, const int64_t *y)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t dim        = l->dim; // number of classes

    Real *restrict x = l->x;

    Real loss = 0.0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (int64_t n = 0; n < node_count; n++)
        loss -= x[n*dim+y[n]];

    return loss / node_count;
}

Real accuracy(const LogSoftmaxLayer *l, const int64_t *labels)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t dim        = l->dim; // number of classses

    const Real *restrict x = l->x;

    uint64_t correct = 0;
#pragma omp parallel for schedule(static) reduction(+:correct)
    for (int64_t n = 0; n < l->node_count; n++)
    {
        const Real *x_ptr = &x[n*dim];
        int64_t pred_class = 0;
        for (int64_t d = 1; d < dim; d++)
        {
            if (x_ptr[d] > x_ptr[pred_class])
                pred_class = d;
        }

        if (pred_class == labels[n])
            correct++;
    }

    return (Real)correct / node_count;
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
// NOTE: we assume mean reduction from NLLLoss
void grad_logsoftmax_nll(LogSoftmaxLayer *const l, int64_t *labels)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    uint32_t dim       = l->dim; // number of classes

    const Real *restrict x  = l->x;
    Real       *restrict dx = l->dx;

    Real scale = REAL(1.0) / node_count;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        const Real *restrict x_ptr  = &x[n*dim];
        Real       *restrict dx_ptr = &dx[n*dim];

        int64_t target = labels[n];
        for (int64_t d = 0; d < dim; d++)
        {
            Real softmax_val = real_exp(x_ptr[d]);
            dx_ptr[d] = softmax_val * scale;
        }
        dx_ptr[target] -= scale;
    }
}
