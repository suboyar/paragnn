#include <stdint.h>
#include <cblas.h>

#include "core.h"
#include "grad_mean_aggregate.h"
#include "layers.h"

void cblas_gemm_touch(int64_t M, int64_t N, int64_t K,
                      Real *restrict A, int64_t lda,
                      Real *restrict B, int64_t ldb,
                      Real *restrict C, int64_t ldc)
{
    // BLAS internal thread mapping is opaque. A standard row-wise
    // OpenMP touch is used to fault the pages into memory.
#pragma omp parallel
    {
#pragma omp for
        for (int64_t k = 0; k < K; k++)
        {
            memset(&A[k * lda], 0, M * sizeof(Real));
            memset(&B[k * ldb], 0, N * sizeof(Real));
        }

#pragma omp for
        for (int64_t i = 0; i < M; i++)
        {
            memset(&C[i * ldc], 0, N * sizeof(Real));
        }
    }
}

void cblas_gemm(int64_t M, int64_t N, int64_t K,
          const Real *restrict A, int64_t lda,
          const Real *restrict B, int64_t ldb,
          Real *restrict C, int64_t ldc)
{
    cblas_rgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                M, N, K,
                1.0,
                A, lda,
                B, ldb,
                0.0,
                C, ldc);
}

void grad_sageconv_cblas_gemm(SageLayer *l)
{
    // grad_Wroot = input^T @ grad_output
    cblas_gemm(l->in_dim, l->out_dim, l->num_nodes,
               l->input,       l->in_dim,
               l->grad_output, l->out_dim,
               l->grad_Wroot,  l->ldW);

    // grad_Wagg = agg^T @ grad_output
    cblas_gemm(l->in_dim, l->out_dim, l->num_nodes,
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
