#include <stdint.h>

#include "core.h"

void outer_tn_v1(int64_t M, int64_t N, int64_t K,
                 const Real *restrict A, int64_t lda,
                 const Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
#pragma omp parallel for
    for (int64_t i = 0; i < M; i++)
    {
        Real *c_row = &C[i*ldc];
        {
            for (int64_t j = 0; j < N; j++)
            {
                for (int64_t k = 0; k < K; k++)
                {
                c_row[j] += A[k*lda + i] * B[k*ldb + j];
                } // end for j
            } // end for k
        } // end for i
    }
}
