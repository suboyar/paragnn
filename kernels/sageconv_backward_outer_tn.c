#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "layers.h"
#include "sageconv_backward_common.h"
#include "vreg.h"

#ifndef KC
#ifndef MR
#ifndef NR

#if defined(TARGET_CPU_THUNDERX2)          /* armq     */
    #define KC  64
    #define MR  6
    #define NR  16

#elif defined(TARGET_CPU_EPYC7601)         /* defq     */
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(TARGET_CPU_EPYC7413)         /* fpgaq    */
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(TARGET_CPU_EPYC9684X)        /* genoaxq  */
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(TARGET_CPU_NEOVERSEV2)       /* gh200q   */
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(TARGET_CPU_XEON8360Y)        /* habanaq  */
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(TARGET_CPU_KUNPENG920)       /* huaq     */
    #define KC 128
    #define MR 9
    #define NR 12

#elif defined(TARGET_CPU_EPYC7763)         /* milanq   */
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(TARGET_CPU_EPYC7302P)        /* rome16q  */
    #define KC  64
    #define MR  6
    #define NR  16

#elif defined(TARGET_CPU_XEONMAX9480)      /* xeonmaxq */
    // #define KC  64
    // #define MR  6
    // #define NR  16

/*  ISA-level fallbacks */

#elif defined(__AVX512F__)
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(__AVX2__)
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(__ARM_FEATURE_SVE2)
    // #define KC  64
    // #define MR  6
    // #define NR  16

#elif defined(__ARM_NEON) || defined(__ARM_FEATURE_SIMD32)
    // #define KC  64
    // #define MR  6
    // #define NR  16

#else
    #define KC  32
    #define MR  6
    #define NR  8
#endif

#endif /* NR */
#endif /* MR */
#endif /* KC */


#ifdef USE_DOUBLE
    #define NR_FULL NR
    #undef  NR
    #define NR (NR_FULL / 2)
    #undef NR_FULL
#endif

#define NV (NR / VLEN)

_Static_assert(NR % VLEN == 0, "NR must be a multiple of VLEN");
_Static_assert(NV * (MR + 1) + 1 <= NUM_REGS, "MR/NR combination exceeds available registers");

// Pack a matrix X (K * n, stored as X[k*ldx + i]) into panels of width `panel`.
// The packed layout is: for each panel starting at offset `pp`, store
// panel-width elements contiguously for each of the `kb` k-steps.
// Zero-pads if pp+i >= n.
static void pack_panel_v1(const Real *restrict X, int64_t ldx,
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


static inline __attribute__((always_inline))
void pack_panel_v2(const Real *restrict X, int64_t ldx,
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
            for (; i + VLEN <= rem; i += VLEN)
                vstore_u(out + i, vload_u(src + i));
            for (; i < rem; i++)
                out[i] = src[i];

            /* zero padding */
            for (; i + VLEN <= panel; i += VLEN)
                vstore_u(out + i, vzero);
            for (; i < panel; i++)
                out[i] = 0.0f;
        }
    }
}

#define DEFINE_PACK(PANEL)                                          \
    __attribute__((noinline))                                       \
    void pack_panel_##PANEL(const Real *restrict X, int64_t ldx,   \
                            Real *restrict Xp,                     \
                            int64_t n, int64_t n_pad, int64_t kb)   \
    {                                                               \
        pack_panel_v2(X, ldx, Xp, n, n_pad, kb, PANEL);             \
    }

// pack_panel_NR
DEFINE_PACK(NR)
// pack_panel_MR
DEFINE_PACK(MR)

// Not packed
static void outer_tn_microkernel_MRxNR_v1(int64_t k,
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

// Assumes A and B are packed
static void outer_tn_microkernel_MRxNR_v2(int64_t k,
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

// GCC vector extension
static void outer_tn_microkernel_MRxNR_v3(int64_t k,
                                          const Real *restrict A,
                                          const Real *restrict B,
                                          Real *restrict C, int64_t ldc)
{
    // Registers = NV * MR
    VReal c[MR][NV];
    PRAGMA_UNROLL(MR)
    for (int mr = 0; mr < MR; mr++)
    {
        PRAGMA_UNROLL(NV)
        for (int nv = 0; nv < NV; nv++)
        {
            c[mr][nv] = *(const VReal*)(C + mr*ldc + nv*VLEN);;
        }
    }

    for (int64_t i = 0; i < k; i++)
    {
        // Registers = NV
        VReal b[NV];
        PRAGMA_UNROLL(NV)
        for (int nv = 0; nv < NV; nv++)
        {
            b[nv] = *(const VReal*)(B + nv*VLEN);
        }

        // Registers = 1
        VReal a;
        PRAGMA_UNROLL(MR)
        for (int mr = 0; mr < MR; mr++)
        {
            a = bcast(A[mr]);
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
            *(VReal*)(C + mr*ldc + nv*VLEN) = c[mr][nv];
        }
    }
}

void outer_tn_v1(int64_t M, int64_t N, int64_t K,
                 const Real *restrict A, int64_t lda,
                 const Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
#pragma omp parallel for
    for (int64_t i = 0; i < M; i++)
    {
        Real *c_row = &C[i*ldc];
        for (int64_t k = 0; k < K; k++)
        {
            Real a = A[k*lda + i];
            const Real *b_row = &B[k*ldb];
            for (int64_t j = 0; j < N; j++)
            {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

void outer_tn_v2(int64_t M, int64_t N, int64_t K,
                 const Real *restrict A, int64_t lda,
                 const Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
    int nthreads = omp_get_max_threads();

    // One workate C per thread, cache-line aligned to avoid false sharing
    int64_t ldcl = N;  // each thread's C is M×N, stride N
    Real *Cwork = cache_aligned_alloc((int64_t)nthreads * M * ldcl * sizeof(Real));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *Cl = &Cwork[(int64_t)tid * M * ldcl];
        memset(Cl, 0, M * ldcl * sizeof(Real));

#pragma omp for nowait
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
                }
            }
        }
    }

#pragma omp parallel for
    for (int64_t i = 0; i < M; i++)
    {
        for (int t = 0; t < nthreads; t++)
        {
            const Real *Cl_row = &Cwork[(int64_t)t * M * ldcl + i * ldcl];
            Real *c_row = &C[i*ldc];
#pragma omp simd
            for (int64_t j = 0; j < N; j++)
            {
                c_row[j] += Cl_row[j];
            }
        }
    }

    free(Cwork);
}

void outer_tn_v3(int64_t M, int64_t N, int64_t K,
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
                    outer_tn_microkernel_MRxNR_v1(k_end - kk,
                                                  &A[kk*lda + ii], lda,
                                                  &B[kk*ldb + jj], ldb,
                                                  &C_tile[ii*ldcl + jj], ldcl);
                }

                // N tail: jj >= N_tiled, columns N_tiled..N-1
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
                    }
                    PRAGMA_UNROLL(MR)
                    for (int mr = 0; mr < MR; mr++)
                    {
                        C_tile[(ii+mr)*ldcl + j] = c[mr];
                    }
                }
            }

            // M tail: ii >= M_tiled, rows M_tiled..M-1
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
                    }
                }
            }
        }
    }

    // Reduction
#pragma omp parallel for
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
            }
        }
    }

    free(C_work);
}

void outer_tn_v4(int64_t M, int64_t N, int64_t K,
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

        Real *Ap = aligned_alloc(64, KC * M_pad * sizeof(Real));
        Real *Bp = aligned_alloc(64, KC * N_pad * sizeof(Real));

#pragma omp for
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t kb = MIN(KC, K - kk);

            // Pack A
            const Real *A_kk = &A[kk * lda];
            pack_panel_v1(A_kk, lda, Ap, M, M_pad, kb, MR);

            // Pack B
            const Real *B_kk = &B[kk * ldb];
            pack_panel_v1(B_kk, ldb, Bp, N, N_pad, kb, NR);

            for (int64_t ii = 0; ii < M_pad; ii += MR)
            {
                for (int64_t jj = 0; jj < N_pad; jj += NR)
                {
                    outer_tn_microkernel_MRxNR_v2(kb,
                                                  &Ap[ii * kb],
                                                  &Bp[jj * kb],
                                                  &Cl[ii*ldcl + jj], ldcl);
                }
            }
        }

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
                }
            }
        }
    }

    free(Cwork);
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

        Real *Ap = aligned_alloc(64, KC * M_pad * sizeof(Real));
        Real *Bp = aligned_alloc(64, KC * N_pad * sizeof(Real));

#pragma omp for
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t kb = MIN(KC, K - kk);

            // Pack A
            const Real *A_kk = &A[kk * lda];
            pack_panel_MR(A_kk, lda, Ap, M, M_pad, kb);

            // Pack B
            const Real *B_kk = &B[kk * ldb];
            pack_panel_NR(B_kk, ldb, Bp, N, N_pad, kb);

            for (int64_t ii = 0; ii < M_pad; ii += MR) {
                for (int64_t jj = 0; jj < N_pad; jj += NR) {
                    outer_tn_microkernel_MRxNR_v3(kb,
                                                  &Ap[ii * kb],
                                                  &Bp[jj * kb],
                                                  &Cl[ii*ldcl + jj], ldcl);
                }
            }
        }

        free(Ap);
        free(Bp);

        // Reduction
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
                }
            }
        }

    }

    free(Cwork);
}

typedef void (*outer_fn)(int64_t M, int64_t N, int64_t K,
                         const Real *restrict A, int64_t lda,
                         const Real *restrict B, int64_t ldb,
                         Real *restrict C, int64_t ldc);

static void sageconv_backward_impl(SageLayer *l, outer_fn kernel)
{
    // grad_Wroot = input^T @ grad_output
    kernel(l->in_dim, l->out_dim, l->num_nodes,
           l->input,       l->in_dim,
           l->grad_output, l->out_dim,
           l->grad_Wroot,  l->out_dim);

    // grad_Wagg = agg^T @ grad_output
    kernel(l->in_dim, l->out_dim, l->num_nodes,
           l->agg,         l->in_dim,
           l->grad_output, l->out_dim,
           l->grad_Wagg,   l->out_dim);

    // grad_input = grad_output @ Wroot^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output, l->out_dim,
                l->Wroot,       l->out_dim,
                0.0,
                l->grad_input,  l->in_dim);

    // grad_scatter = grad_output @ Wagg^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output,  l->out_dim,
                l->Wagg,         l->out_dim,
                0.0,
                l->grad_scatter, l->in_dim);

    grad_mean_aggregate(l);
}

void sageconv_backward_outer_v1(SageLayer *l)   { sageconv_backward_impl(l, outer_tn_v1); }
void sageconv_backward_outer_v2(SageLayer *l)   { sageconv_backward_impl(l, outer_tn_v2); }
void sageconv_backward_outer_v3(SageLayer *l)   { sageconv_backward_impl(l, outer_tn_v3); }
void sageconv_backward_outer_v4(SageLayer *l)   { sageconv_backward_impl(l, outer_tn_v4); }
void sageconv_backward_outer_v5(SageLayer *l)   { sageconv_backward_impl(l, outer_tn_v5); }
