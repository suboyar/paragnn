/*
 * This version introduces gcc vector extension
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include <omp.h>

#include "outer_tn_params.h"
#include "core.h"
#include "vreg.h"

static void pack_A(const Real *restrict A, int64_t lda, Real *restrict Ap,
                   int64_t cols, int64_t rows)
{
    Real *restrict aligned_Ap = (Real *)__builtin_assume_aligned(Ap, VLEN_BYTES);
    int64_t full_rows = (rows / MR) * MR;

    int64_t pp = 0;
    for ( ; pp < full_rows; pp += MR)
    {
        PRAGMA_UNROLL(K_UNROLL)
        for (int64_t k = 0; k < cols; k++)
        {
            const Real *src = A + pp + k * lda;
            PRAGMA_UNROLL(MR)
            for (int i = 0; i < MR; i++)
                *aligned_Ap++ = src[i];
        }
    }

    if (pp < rows)
    {
        int64_t rem = rows - pp;
        PRAGMA_UNROLL(K_UNROLL)
        for (int64_t k = 0; k < cols; k++)
        {
            const Real *src = A + pp + k * lda;
            // Absolute fringe of the matrix
            int i = 0;
            for (; i < rem; i++)
                *aligned_Ap++ = src[i];
            for (; i < MR; i++)
                *aligned_Ap++ = 0.0;
        }
    }
}

static void pack_B(const Real *restrict B, int64_t ldb, Real *restrict Bp,
                   int64_t rows, int64_t cols)
{
    Real *restrict aligned_Bp = (Real *)__builtin_assume_aligned(Bp, VLEN_BYTES);

    int64_t cols_aligned = ((cols + NR - 1) / NR) * NR;
    const VReal vzero = (VReal){0};

    for (int64_t pp = 0; pp < cols_aligned; pp += NR)
    {
        int64_t rem = cols - pp;
        if (rem > NR) rem = NR;

        Real *restrict dst = aligned_Bp + pp * rows;

#pragma GCC unroll 4
        for (int64_t k = 0; k < rows; k++)
        {
            const Real *src = B + k * ldb + pp;
            Real *out       = dst + k * NR;
            int64_t i = 0;

            // Copy valid elements in full N_VEC chunks
            for (; i + N_VEC <= rem; i += N_VEC)
                vrstore(out + i, vrload_u(src + i));

            if (i < rem)
            {
                Real tmp[N_VEC] __attribute__((aligned(VLEN_BYTES))) = {0};

                for (int64_t j = 0; i + j < rem; j++)
                    tmp[j] = src[i + j];

                vrstore(out + i, vrload(tmp));
                i += N_VEC;
            }

            for (; i < NR; i += N_VEC)
                vrstore(out + i, vzero);
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

static void reduction(int64_t M, int64_t N,
                      int64_t M_pad, int64_t N_pad,
                      int nthreads,
                      Real *restrict C, int64_t ldc,
                      Real *restrict Cl, int64_t ldcl,
                      Real *all_Cl[])
{
#pragma omp for
    for (int64_t i = 0; i < M; i++)
    {
        Real *out_row = &C[i * ldc];

        int64_t j = 0;
#define UNROLL_FACTOR (NUM_REGS/2)
        for (; j + UNROLL_FACTOR * N_VEC <= N; j += UNROLL_FACTOR * N_VEC)
        {
            VReal sum[UNROLL_FACTOR];

            PRAGMA_UNROLL(UNROLL_FACTOR)
            for (int iv = 0; iv < UNROLL_FACTOR; iv++) {
                sum[iv] = vrbcast((Real)0.0);
            }

            PRAGMA_UNROLL(4)
            for (int t = 0; t < nthreads; t++)
            {
                const Real *in_row = &all_Cl[t][i * ldcl];

                PRAGMA_UNROLL(UNROLL_FACTOR)
                for (int iv = 0; iv < UNROLL_FACTOR; iv++) {
                    sum[iv] += vrload(in_row + j + iv * N_VEC);
                }
            }

            PRAGMA_UNROLL(UNROLL_FACTOR)
            for (int iv = 0; iv < UNROLL_FACTOR; iv++) {
                stream_vrstore(out_row + j + iv * N_VEC, sum[iv]);
            }
        }
#undef UNROLL_FACTOR

        for (; j < N; j++)
        {
            Real sum_scalar = (Real)0.0;
            for (int t = 0; t < nthreads; t++)
            {
                sum_scalar += all_Cl[t][i * ldcl + j];
            }
            out_row[j] = sum_scalar;
        }
    }
}

void outer_tn_v3(int64_t M, int64_t N, int64_t K,
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

        all_Cl[tid] = Cl;

        // NUMA First-Touch initialization
        int first_time = 1;

#pragma omp for
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t kb = MIN(KC, K - kk);

            const Real *A_kk = &A[kk * lda];
            pack_A(A_kk, lda, Ap, kb, M);
            const Real *B_kk = &B[kk * ldb];
            pack_B(B_kk, ldb, Bp, kb, N);

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
            first_time = 0;
        } // end for kk

        // If this thread did no work, its buffer contains garbage. Zero it out
        // before the reduction reads from it
        if (first_time)
        {
            memset(Cl, 0, (size_t)M_pad * ldcl * sizeof(Real));
        }

        #pragma omp barrier

        // Reduction
        reduction(M, N, M_pad, N_pad, nthreads, C, ldc, Cl, ldcl, all_Cl);
    }
}
