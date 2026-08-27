/*
 * This version introduces OpenMP multithreading over K-split
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "core.h"

void outer_tn_v1_touch(int64_t M, int64_t N, int64_t K,
                       Real *restrict A, int64_t lda,
                       Real *restrict B, int64_t ldb,
                       Real *restrict C, int64_t ldc)
{
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int64_t kk = 0; kk < K; kk++)
        {
            Real *A_kk = &A[kk*lda];
            memset(A_kk, 0, M * sizeof(*A_kk));
            Real *B_kk = &B[kk*ldb];
            memset(B_kk, 0, N * sizeof(*B_kk));
        }

#pragma omp for schedule(static)
        for (int64_t i = 0; i < M; i++)
        {
            Real *C_i = &C[i * ldc];
            memset(C_i, 0, N);
        }
    }
}

void outer_tn_v1(int64_t M, int64_t N, int64_t K,
                 const Real *restrict A, int64_t lda,
                 const Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
    int nthreads = omp_get_max_threads();

    // One workate C per thread, cache-line aligned to avoid false sharing
    int64_t ldcl = N;  // each thread's C is M×N, stride N
    Real **all_Cl = malloc(nthreads * sizeof(Real*));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

        Real *Cl = cache_aligned_alloc((size_t)M * ldcl * sizeof(Real));
        memset(Cl, 0, (size_t)M * ldcl * sizeof(Real));
        all_Cl[tid] = Cl;

#pragma omp for schedule(static)
        for (int64_t k = 0; k < K; k++)
        {
            const Real *A_k = &A[k*lda];
            const Real *B_k = &B[k*ldb];
            for (int64_t i = 0; i < M; i++)
            {
                Real a = A_k[i];
                Real *Cl_i = &Cl[i*ldcl];
#pragma omp simd
                for (int64_t j = 0; j < N; j++)
                {
                    Cl_i[j] += a * B_k[j];
                } // end for jj
            } // end for ii
        } // end for kk

        // Reduction
#pragma omp for schedule(static)
        for (int64_t i = 0; i < M; i++)
        {
            Real *C_i = &C[i * ldc];
            for (int t = 0; t < nthreads; t++)
            {
                const Real *Cl_i = &all_Cl[t][i * ldcl];
#pragma omp simd
                for (int64_t j = 0; j < N; j++)
                    C_i[j] += Cl_i[j];
            }
        }
        free(Cl);
    }
    free(all_Cl);
}
