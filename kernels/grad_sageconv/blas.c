#include <stdint.h>
#include <cblas.h>

#include "core.h"

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
