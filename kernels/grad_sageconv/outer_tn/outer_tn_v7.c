/*
 * This version introduces blocking for the NR loop which we refer to as NC
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "core.h"
#include "vreg.h"

#if defined(TARGET_CPU_XEONMAX9480)        /* xeonmaxq */
    #define KC 448
    #define NC 1024
    #define K_UNROLL 8

#elif defined(TARGET_CPU_XEON8360Y)        /* habanaq  */
    #define KC 448
    #define NC 656
    #define K_UNROLL 8

#elif TARGET_CPU_XEON6960P                 /* h200q    */
    #define KC 448
    #define NC 1024
    #define K_UNROLL 8

#elif defined(TARGET_CPU_EPYC7601)         /* defq     */
    #define KC 320
    #define NC 304
    #define K_UNROLL 1

#elif defined(TARGET_CPU_EPYC7302P)        /* rome16q  */
    #define KC 320
    #define NC 304
    #define K_UNROLL 1

#elif defined(TARGET_CPU_EPYC7413)         /* fpgaq    */
    #define KC 384
    #define NC 256
    #define K_UNROLL 2

#elif defined(TARGET_CPU_EPYC7763)         /* milanq   */
    #define KC 384
    #define NC 256
    #define K_UNROLL 2

#elif defined(TARGET_CPU_EPYC9684X)        /* genoaxq  */
    #define KC 256
    #define NC 768
    #define K_UNROLL 8

#elif defined(TARGET_CPU_THUNDERX2)       /* armq     */
    #define KC 512
    #define NC 96
    #define K_UNROLL 4

#elif defined(TARGET_CPU_KUNPENG920)       /* huaq     */
    #define KC 512
    #define NC 192
    #define K_UNROLL 4

#elif defined(TARGET_CPU_NEOVERSEV2)       /* gh200q   */
    #define KC 512
    #define NC 384
    #define K_UNROLL 4

#else                                      /* fallback */
    #define KC 256
    #define NC 48
    #define K_UNROLL 1
#endif

static inline __attribute__((always_inline))
void pack_panel(const Real *restrict X, int64_t ldx,
                   Real *restrict Xp,
                   int64_t n, int64_t n_pad, int64_t kb,
                   int64_t panel)
{
    const VReal vzero = (VReal){0};

    for (int64_t pp = 0; pp < n_pad; pp += panel)
    {
        int64_t rem = n - pp;
        if (rem < 0)     rem = 0;
        if (rem > panel) rem = panel;

        Real *restrict dst = Xp + pp * kb;

        for (int64_t k = 0; k < kb; k++)
        {
            const Real *src = X + k * ldx + pp;
            Real *out       = dst + k * panel;
            int64_t i = 0;

            /* copy valid elements */
            for (; i + N_VEC <= rem; i += N_VEC)
                vrstore_u(out + i, vrload_u(src + i));
            for (; i < rem; i++)
                out[i] = src[i];

            /* zero padding */
            for (; i + N_VEC <= panel; i += N_VEC)
                vrstore_u(out + i, vzero);
            for (; i < panel; i++)
                out[i] = 0.0f;
        }
    }
}

_Static_assert(NV * (MR + 1) + 1 <= NUM_REGS, "MR/NR combination exceeds available registers");
static void microkernel_MRxNR(int64_t k,
                              const Real *restrict A,
                              const Real *restrict B,
                              Real *restrict C, int64_t ldc,
                              int first_time)
{
    // Registers = NV * MR
    VReal c[MR][NV];
    if (first_time)
    {
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            PRAGMA_UNROLL(NV)
            for (int nv = 0; nv < NV; nv++)
            {
                c[mr][nv] = vrbcast((Real) 0.0);
            }
        }
    }
    else
    {
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            PRAGMA_UNROLL(NV)
            for (int nv = 0; nv < NV; nv++)
            {
                c[mr][nv] = *(const VReal*)(C + mr*ldc + nv*N_VEC);
            }
        }
    }

    PRAGMA_UNROLL(K_UNROLL)
    for (int64_t i = 0; i < k; i++)
    {
        // Registers = NV
        VReal b[NV];
        PRAGMA_UNROLL(NV)
        for (int nv = 0; nv < NV; nv++)
        {
            b[nv] = *(const VReal*)(B + nv*N_VEC);
        }

        // Registers = 1
        VReal a;
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            a = vrbcast(A[mr]);
            PRAGMA_UNROLL(NV)
            for (int nv = 0; nv < NV; nv++)
            {
                c[mr][nv] += a * b[nv];
            }
        }
        A += MR;
        B += NR;
    }

    PRAGMA_UNROLL(MR)
    for (int mr = 0; mr < MR; mr++)
    {
        PRAGMA_UNROLL(NV)
        for (int nv = 0; nv < NV; nv++)
        {
            *(VReal*)(C + mr*ldc + nv*N_VEC) = c[mr][nv];
        }
    }
}

void outer_tn_v7(int64_t M, int64_t N, int64_t K,
                 const Real *restrict A, int64_t lda,
                 const Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
    int nthreads = omp_get_max_threads();

    const int64_t M_pad = ((M + MR - 1) / MR) * MR;
    const int64_t N_pad = ((N + NR - 1) / NR) * NR;
    const int64_t ldcl  = N_pad;

#pragma omp parallel
    {
        Real *Cl = cache_aligned_alloc((size_t)M_pad * ldcl * sizeof(Real));
        Real *Ap = cache_aligned_alloc((size_t)KC * M_pad * sizeof(Real));
        Real *Bp = cache_aligned_alloc((size_t)KC * N_pad * sizeof(Real));

#pragma omp for
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int first_time = (kk == 0);
            int64_t kb = MIN(KC, K - kk);

            // Pack A
            const Real *A_kk = &A[kk * lda];

            pack_panel(A_kk, lda, Ap, M, M_pad, kb, MR);

            // Pack B
            const Real *B_kk = &B[kk * ldb];
            pack_panel(B_kk, ldb, Bp, N, N_pad, kb, NR);

            for (int64_t jj_outer = 0; jj_outer < N_pad; jj_outer += NC)
            {
                int64_t j_end = MIN(jj_outer + NC, N_pad);

                for (int64_t ii = 0; ii < M_pad; ii += MR)
                {
                    for (int64_t jj = jj_outer; jj < j_end; jj += NR)
                    {
                        microkernel_MRxNR(kb,
                                          &Ap[ii * kb],
                                          &Bp[jj * kb],
                                          &Cl[ii*ldcl + jj], ldcl,
                                          first_time);
                    } // end for jj
                } // end for ii
            } // end for jj_outer
        } // end for kk

        free(Ap);
        free(Bp);

#pragma omp barrier

        // Reduction
#pragma omp for
        for (int64_t i = 0; i < M; i++)
        {

            const Real *Cl_row = Cl + M_pad*ldcl + i*ldcl;
            Real *c_row = &C[i*ldc];
#pragma omp simd
                for (int64_t j = 0; j < N; j++)
                {
                    c_row[j] += Cl_row[j];
                } // end for j
        } // end for i

        free(Cl);
    }

}
