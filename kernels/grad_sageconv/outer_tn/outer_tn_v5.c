/*
 * This version introduces explicit data packing for panels of matrices A and B
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

// Pack a matrix X (K * n, stored as X[k*ldx + i]) into panels of width `panel`.
// The packed layout is: for each panel starting at offset `pp`, store
// panel-width elements contiguously for each of the `kb` k-steps.
// Zero-pads if pp+i >= n.
static void pack_panel(const Real *restrict X, int64_t ldx,
                       Real *restrict Xp,
                       int64_t n, int64_t n_pad, int64_t kb,
                       int64_t panel)
{
    for (int64_t pp = 0; pp < n_pad; pp += panel)
    {
        Real *dst = &Xp[pp * kb];
        for (int64_t k = 0; k < kb; k++)
        {
            for (int64_t i = 0; i < panel; i++)
            {
                int64_t gi = pp + i;
                dst[k * panel + i] = (gi < n) ? X[k * ldx + gi] : REAL(0.0);
            }
        }
    }
}

static void microkernel_MRxNR(int64_t k,
                              const Real *restrict A,
                              const Real *restrict B,
                              Real *restrict C, int64_t ldc)
{
    // Registers = MR * NR
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
        // Registers = NR
        Real b[NR];
        PRAGMA_UNROLL(NR)
        for (int nr = 0; nr < NR; nr++)
        {
            b[nr] = *(const Real*)(B + nr);
        }

        // Registers = 1
        Real a;
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            a = *(const Real*)(A + mr);
            PRAGMA_UNROLL(NR)
            for (int nr = 0; nr < NR; nr++)
            {
                c[mr][nr] += a * b[nr];
            }
        }
        A += MR;
        B += NR;
    }

    PRAGMA_UNROLL(MR)
    for (int mr = 0; mr < MR; mr++)
    {
        PRAGMA_UNROLL(NR)
        for (int nr = 0; nr < NR; nr++)
        {
            *(Real*)(C + mr*ldc + nr) = c[mr][nr];
        }
    }
}

void outer_tn_v5(int64_t M, int64_t N, int64_t K,
              const Real *restrict A, int64_t lda,
              const Real *restrict B, int64_t ldb,
              Real *restrict C, int64_t ldc)
{
    int nthreads = omp_get_max_threads();

    const int64_t M_pad = ((M + MR - 1) / MR) * MR;
    const int64_t N_pad = ((N + NR - 1) / NR) * NR;
    const int64_t ldcl  = N_pad;

    Real *Cwork = aligned_alloc(64, (int64_t)nthreads * M_pad * ldcl * sizeof(Real));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *Cl = &Cwork[(int64_t)tid * M_pad * ldcl];
        memset(Cl, 0, M_pad * ldcl * sizeof(Real));

        Real *Ap = cache_aligned_alloc((size_t)KC * M_pad * sizeof(Real));
        Real *Bp = cache_aligned_alloc((size_t)KC * N_pad * sizeof(Real));

#pragma omp for
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t kb = MIN(KC, K - kk);

            // Pack A
            const Real *A_kk = &A[kk * lda];
            pack_panel(A_kk, lda, Ap, M, M_pad, kb, MR);

            // Pack B
            const Real *B_kk = &B[kk * ldb];
            pack_panel(B_kk, ldb, Bp, N, N_pad, kb, NR);

            for (int64_t ii = 0; ii < M_pad; ii += MR)
            {
                for (int64_t jj = 0; jj < N_pad; jj += NR)
                {
                    microkernel_MRxNR(kb,
                                      &Ap[ii * kb],
                                      &Bp[jj * kb],
                                      &Cl[ii*ldcl + jj], ldcl);
                } // end for jj
            } // end for ii
        } // end for kk

        free(Ap);
        free(Bp);

#pragma omp for
        for (int64_t i = 0; i < M; i++)
        {
            for (int t = 0; t < nthreads; t++)
            {
                const Real *Cl_row = Cwork + t*M_pad*ldcl + i*ldcl;
                Real *c_row = &C[i*ldc];
#pragma omp simd
                for (int64_t j = 0; j < N; j++)
                {
                    c_row[j] += Cl_row[j];
                } // end for j
            } // end for t
        } // end for i
    }

    free(Cwork);
}
