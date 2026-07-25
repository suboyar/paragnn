/*
 * This version introduces OpenMP multithreading over K-split
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "core.h"

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

#pragma omp for
        for (int64_t k = 0; k < K; k++)
        {
            const Real *a_row = &A[k*lda];
            const Real *b_row = &B[k*ldb];
            for (int64_t i = 0; i < M; i++)
            {
                Real a = a_row[i];
                Real *c_row = &Cl[i*ldcl];
#pragma omp simd
                for (int64_t j = 0; j < N; j++)
                {
                    c_row[j] += a * b_row[j];
                } // end for j
            } // end for i
        } // end for k

        // Reduction
#pragma omp for
        for (int64_t i = 0; i < M; i++)
        {
            Real *c_row = &C[i * ldc];
            for (int t = 0; t < nthreads; t++)
            {
                const Real *cl_row = &all_Cl[t][i * ldcl];
#pragma omp simd
                for (int64_t j = 0; j < N; j++)
                    c_row[j] += cl_row[j];
            }
        }

        free(Cl);
    }
    free(all_Cl);

}
