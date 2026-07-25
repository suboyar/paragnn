/*
 * This version introduces panel packing and blocking over KC and NC
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "outer_tn_params.h"
#include "core.h"
#include "vreg.h"

// Packs a block of 'panel' columns from each row sequentially into Xp.
//
// Xp memory layout for the first panel:
// [ X[0][0:panel] | X[1][0:panel] | ... | X[rows-1][0:panel] ]
static void pack_panel(const Real *restrict X, int64_t ldx,
                       Real *restrict Xp,
                       int64_t rows, int64_t cols, int64_t cols_aligned,
                       int64_t panel)
{
    Real *restrict aligned_Xp = (Real *)__builtin_assume_aligned(Xp, VLEN_BYTES);

    for (int64_t pp = 0; pp < cols_aligned; pp += panel)
    {
        int64_t rem = cols - pp;
        if (rem > panel) rem = panel;

        Real *restrict dst = aligned_Xp + pp * rows;

        for (int64_t rr = 0; rr < rows; rr++)
        {
            const Real *src = X + rr * ldx + pp;
            Real *out       = dst + rr * panel;
            int64_t i = 0;

            for (; i < rem; i++)
                out[i] = src[i];

            for (; i < panel; i++)
                out[i] = 0.0;
        }
    }
}

static void microkernel_MRxNR(int64_t k,
                              const Real *restrict A,
                              const Real *restrict B,
                              Real *restrict C, int64_t ldc,
                              int first_time)
{
    // Registers = MR * (NR/N_VEC)
    Real c[MR][NR];
    if (first_time)
    {
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            PRAGMA_UNROLL(NR)
            for (int nr = 0; nr < NR; nr++)
            {
                c[mr][nr] = (Real)0.0;
            }
        }
    }
    else
    {
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            PRAGMA_UNROLL(NR)
            for (int nr = 0; nr < NR; nr++)
            {
                c[mr][nr] = *(const Real*)(C + mr*ldc + nr);
            }
        }
    }

    PRAGMA_UNROLL(K_UNROLL)
    for (int64_t i = 0; i < k; i++)
    {
        // Registers = (NR/N_VEC)
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

void outer_tn_v2(int64_t M, int64_t N, int64_t K,
              const Real *restrict A, int64_t lda,
              const Real *restrict B, int64_t ldb,
              Real *restrict C, int64_t ldc)
{
    int nthreads = omp_get_max_threads();

    const int64_t M_pad = ((M + MR - 1) / MR) * MR;
    const int64_t N_pad = ((N + NR - 1) / NR) * NR;
    const int64_t ldcl  = N_pad;

    Real *all_Cl[nthreads];

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

        static thread_local Real* Ap = NULL;
        static thread_local Real* Bp = NULL;
        static thread_local Real* Cl = NULL;

        static thread_local int64_t local_M_pad = 0;
        static thread_local int64_t local_N_pad = 0;
        bool needs_cl_realloc = 0;

        if (M_pad != local_M_pad)
        {
            free(Ap);
            Ap = cache_aligned_alloc((size_t)KC * M_pad * sizeof(Real));
            needs_cl_realloc = 1;
        }

        if (N_pad != local_N_pad)
        {
            free(Bp);
            Bp = cache_aligned_alloc((size_t)KC * N_pad * sizeof(Real));
            needs_cl_realloc = 1;
        }

        if (needs_cl_realloc)
        {
            free(Cl);
            Cl = cache_aligned_alloc((size_t)M_pad * ldcl * sizeof(Real));
        }

        local_M_pad = M_pad;
        local_N_pad = N_pad;

        // memset(Cl, 0, (size_t)M_pad * ldcl * sizeof(Real));
        all_Cl[tid] = Cl;

        // NUMA First-Touch initialization
        int first_time = 1;

#pragma omp for
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t kb = MIN(KC, K - kk);

            // Pack A
            const Real *A_kk = &A[kk * lda];
            pack_panel(A_kk, lda, Ap, kb, M, M_pad, MR);

            // Pack B
            const Real *B_kk = &B[kk * ldb];
            pack_panel(B_kk, ldb, Bp, kb, N, N_pad, NR);

            for (int64_t ii = 0; ii < M_pad; ii += MR)
            {
                for (int64_t jj = 0; jj < N_pad; jj += NR)
                {
                    microkernel_MRxNR(kb,
                                      &Ap[ii * kb],
                                      &Bp[jj * kb],
                                      &Cl[ii*ldcl + jj], ldcl,
                                      first_time);
                } // end for jj
            } // end for ii
            first_time = 0;
        } // end for kk

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

    }
    // free(all_Cl);
}
