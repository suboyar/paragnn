#include <stdint.h>

#include "core.h"

void naive_touch(int64_t M, int64_t N, int64_t K,
                 Real *restrict A, int64_t lda,
                 Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
#pragma omp parallel for
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
#pragma omp parallel for
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
