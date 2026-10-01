#include <stdint.h>

#include "core.h"
#include "grad_mean_aggregate.h"
#include "layers.h"

void naive_touch(int64_t M, int64_t N, int64_t K,
                 Real *restrict A, int64_t lda,
                 Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; i++)
    {
        Real *C_i = &C[i*ldc];
        memset(C_i, 0, N * sizeof(*C_i));
        for (int64_t j = 0; j < N; j++)
        {
            for (int64_t k = 0; k < K; k++)
            {
                A[k*lda + i] = 0.0;
                B[k*ldb + j] = 0.0;
            } // end for j
        } // end for k
    } // end for i
}


void naive(int64_t M, int64_t N, int64_t K,
           const Real *restrict A, int64_t lda,
           const Real *restrict B, int64_t ldb,
           Real *restrict C, int64_t ldc)
{
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; i++)
    {
        Real *C_i = &C[i*ldc];
        for (int64_t j = 0; j < N; j++)
        {
            Real acc = 0;
            for (int64_t k = 0; k < K; k++)
            {
                acc += A[k*lda + i] * B[k*ldb + j];
            } // end for j
            C_i[j] = acc;
        } // end for k
    } // end for i
}

void grad_sageconv_naive(SageLayer *l)
{
    // grad_Wroot = input^T @ grad_output
    naive(l->in_dim, l->out_dim, l->num_nodes,
          l->input,       l->in_dim,
          l->grad_output, l->out_dim,
          l->grad_Wroot,  l->ldW);

    // grad_Wagg = agg^T @ grad_output
    naive(l->in_dim, l->out_dim, l->num_nodes,
          l->agg,         l->in_dim,
          l->grad_output, l->out_dim,
          l->grad_Wagg,   l->ldW);

    // grad_input = grad_output @ Wroot^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output, l->out_dim,
                l->Wroot,       l->out_dim,
                0.0,
                l->grad_input,  l->in_dim);

    // grad_scatter = grad_output @ Wagg^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output,  l->out_dim,
                l->Wagg,         l->out_dim,
                0.0,
                l->grad_scatter, l->in_dim);

    grad_mean_aggregate(l);
}
