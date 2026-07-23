/*
 * This version introduces a register-blocked microkernel and cache blocking
 * along the K dimension
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "core.h"
#include "vreg.h"

#if defined(TARGET_CPU_XEONMAX9480)        /* xeonmaxq */
    #define KC 448
    #define K_UNROLL 8

#elif defined(TARGET_CPU_XEON8360Y)        /* habanaq  */
    #define KC 448
    #define K_UNROLL 8

#elif TARGET_CPU_XEON6960P                 /* h200q    */
    #define KC 448
    #define K_UNROLL 8

#elif defined(TARGET_CPU_EPYC7601)         /* defq     */
    #define KC 320
    #define K_UNROLL 1

#elif defined(TARGET_CPU_EPYC7302P)        /* rome16q  */
    #define KC 320
    #define K_UNROLL 1

#elif defined(TARGET_CPU_EPYC7413)         /* fpgaq    */
    #define KC 384
    #define K_UNROLL 2

#elif defined(TARGET_CPU_EPYC7763)         /* milanq   */
    #define KC 384
    #define K_UNROLL 2

#elif defined(TARGET_CPU_EPYC9684X)        /* genoaxq  */
    #define KC 256
    #define K_UNROLL 8

#elif defined(TARGET_CPU_THUNDERX2)       /* armq     */
    #define KC 512
    #define K_UNROLL 4

#elif defined(TARGET_CPU_KUNPENG920)       /* huaq     */
    #define KC 512
    #define K_UNROLL 4

#elif defined(TARGET_CPU_NEOVERSEV2)       /* gh200q   */
    #define KC 512
    #define K_UNROLL 4

#else                                      /* fallback */
    #define KC 256
    #define K_UNROLL 1
#endif

static void microkernel_MRxNR(int64_t k,
                              const Real *restrict A, int64_t lda,
                              const Real *restrict B, int64_t ldb,
                              Real *restrict C, int64_t ldc)
{
    Real c[MR][NR];
    PRAGMA_UNROLL(MR)
    for (int mr = 0; mr < MR; mr++)
    {
        PRAGMA_UNROLL(NR)
        for (int nr = 0; nr < NR; nr++)
        {
            c[mr][nr] = *(const Real*)(C + mr*ldc + nr);
        }
    }

    PRAGMA_UNROLL(K_UNROLL)
    for (int64_t i = 0; i < k; i++)
    {
        const Real *restrict a_ptr = &A[i*lda];
        const Real *restrict b_ptr = &B[i*ldb];

        Real b[NR];
        PRAGMA_UNROLL(NR)
        for (int nr = 0; nr < NR; nr++)
        {
            b[nr] = *(const Real*)(b_ptr + nr);
        }

        Real a;
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            a = a_ptr[mr];
            PRAGMA_UNROLL(NR)
            for (int nr = 0; nr < NR; nr++)
            {
                c[mr][nr] += a * b[nr];
            }
        }
    }

    PRAGMA_UNROLL(MR)
    for (int mr = 0; mr < MR; mr++)
    {
        PRAGMA_UNROLL(NR)
        for (int nr = 0; nr < NR; nr++)
        {
            C[mr*ldc + nr] = c[mr][nr];
        }
    }
}

void outer_tn_v4(int64_t M, int64_t N, int64_t K,
                 const Real *restrict A, int64_t lda,
                 const Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
    int nthreads = omp_get_max_threads();

    const int64_t M_tiled = (M / MR) * MR;
    const int64_t N_tiled = (N / NR) * NR;
    const int64_t ldcl = ((N + NR - 1) / NR) * NR;
    Real *C_work = aligned_alloc(64, (int64_t)nthreads * M * ldcl * sizeof(Real));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *C_tile = &C_work[(int64_t)tid * M * ldcl];
        memset(C_tile, 0, M * ldcl * sizeof(Real));

#pragma omp for nowait
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t k_end = MIN(kk + KC, K);

            for (int64_t ii = 0; ii < M_tiled; ii += MR)
            {
                for (int64_t jj = 0; jj < N_tiled; jj += NR)
                {
                    microkernel_MRxNR(k_end - kk,
                                      &A[kk*lda + ii], lda,
                                      &B[kk*ldb + jj], ldb,
                                      &C_tile[ii*ldcl + jj], ldcl);
                } // end for jj

                for (int64_t j = N_tiled; j < N; j++)
                {
                    Real c[MR];
                    PRAGMA_UNROLL(MR)
                    for (int mr = 0; mr < MR; mr++)
                    {
                        c[mr] = *(const Real*)(C_tile + (ii+mr)*ldcl + j);
                    }

                    for (int64_t k = kk; k < k_end; k++)
                    {
                        const Real *a_row = &A[k*lda];
                        Real b = B[k*ldb + j];
                        PRAGMA_UNROLL(MR)
                        for (int mr = 0; mr < MR; mr++)
                        {
                            c[mr] += a_row[ii+mr] * b;
                        }
                    } // end for kk

                    PRAGMA_UNROLL(MR)
                    for (int mr = 0; mr < MR; mr++)
                    {
                        C_tile[(ii+mr)*ldcl + j] = c[mr];
                    }
                } // end for j
            } // end for ii

            for (int64_t i = M_tiled; i < M; i++)
            {
                for (int64_t k = kk; k < k_end; k++)
                {
                    Real a = A[k*lda + i];
                    const Real *b_row = &B[k*ldb];
                    Real *c_row = &C_tile[i*ldcl];
#pragma omp simd
                    for (int64_t j = 0; j < N; j++)
                    {
                        c_row[j] += a * b_row[j];
                    } // end for j
                } // end for k
            } // end for i
        } // end for kk

        // Reduction
#pragma omp for
        for (int64_t i = 0; i < M; i++)
        {
            for (int t = 0; t < nthreads; t++)
            {
                const Real *tile_row = &C_work[(int64_t)t * M * ldcl + i * ldcl];
                Real *c_dest = &C[i*ldc];
#pragma omp simd
                for (int64_t j = 0; j < N; j++)
                {
                    c_dest[j] += tile_row[j];
                } // end for j
            } // end for t
        } // end for i
    }
    free(C_work);
}
