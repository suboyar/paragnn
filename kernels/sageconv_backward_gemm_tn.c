#include <omp.h>
#include <cblas.h>
#include <stdint.h>
#include <string.h>

#include "core.h"
#include "layers.h"
#include "layers.h"
#include "sageconv_backward_common.h"

#ifndef KC
#define KC 256
#endif // KC

#ifndef MR
#define MR 4
#endif // MR

#ifndef NR
#define NR 4
#endif // NR


//
// Using signed integers for loop indicies allows the compiler to optimize the
// loop much more aggresive, since signed integers are not allowed to overflow in C.
// Reference: https://kristerw.blogspot.com/2016/02/how-undefined-signed-overflow-enables.html
//


//
// When doing doing matmul only B needs to be traversed column-wise,
// but since matrix A's shape is of MxN and B's shape is MxK, this forces us
// to traverse A in column-wise order too. Which is not ideal for cache utilization,
// especially when A and B are of a tall-and-skinny matrix. This seems to be called
// Tall & Skinny Matrix Transposed times Tall & Skinny Matrix (TSMTTSM) in litterateur.
//

/* Loop order i->k->j, so A[k,i] gets reused N times */
void gemm_tn_v1(int64_t M, int64_t N, int64_t K,
                const Real *restrict A, int64_t lda,
                const Real *restrict B, int64_t ldb,
                Real *restrict C, int64_t ldc)
{
#pragma omp parallel for
    for (int64_t i = 0; i < M; i++)
    {
        Real *c_row = &C[i*ldc];
        memset(c_row, 0, N * sizeof(Real));

        for (int64_t k = 0; k < K; k++)
        {
            register Real a = A[k*lda+i];
            const Real *b_row = &B[k*ldb];
#pragma omp simd
            for (int64_t j = 0; j < N; j++)
            {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

/* K cache block (ikj) */
void gemm_tn_v2(int64_t M, int64_t N, int64_t K,
                const Real *restrict A, int64_t lda,
                const Real *restrict B, int64_t ldb,
                Real *restrict C, int64_t ldc)
{

#pragma omp parallel
    {
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t k_end = MIN(kk+KC, K);

#pragma omp for
            for (int64_t i = 0; i < M; i++)
            {
                for (int64_t k = kk; k < k_end; k++)
                {
                    register const Real a = A[k*lda+i];
                    const Real *b_row = &B[k*ldb];
                    Real *c_row = &C[i*ldc];
#pragma omp simd
                    for (int64_t j = 0; j < N; j++)
                    {
                        c_row[j] += a * b_row[j];
                    }
                }
            }
        }
    }
}

/* K block + unroll-jam M (ikj) */
void gemm_tn_v3(int64_t M, int64_t N, int64_t K,
                const Real *restrict A, int64_t lda,
                const Real *restrict B, int64_t ldb,
                Real *restrict C, int64_t ldc)
{
    const int64_t M_tiled = (M / MR) * MR;

#pragma omp parallel
    {
        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t k_end = MIN(kk+KC, K);

#pragma omp for
            for (int64_t ii = 0; ii < M_tiled; ii+=MR)
            {
                for (int64_t k = kk; k < k_end; k++)
                {
                    Real a[MR];
                    const Real *b_row = &B[k*ldb];
                    Real *c_row[MR];

                    PRAGMA_UNROLL(MR)
                    for (int mr = 0; mr < MR; mr++)
                    {
                        a[mr] = A[k*lda+ii+mr];
                    }

                    PRAGMA_UNROLL(MR)
                    for (int mr = 0; mr < MR; mr++)
                    {
                        c_row[mr] = &C[(ii+mr)*ldc];
                    }
#pragma omp simd
                    for (int64_t j = 0; j < N; j++)
                    {
                        register const Real b = b_row[j];
                        for (int mr = 0; mr < MR; mr++)
                        {
                            c_row[mr][j] += a[mr] * b;
                        }
                    }
                }
            }

#pragma omp for
            for (int64_t i = M_tiled; i < M; i++)
            {
                for (int64_t k = kk; k < k_end; k++)
                {
                    register const Real a = A[k*lda+i];
                    const Real *b_row = &B[k*ldb];
                    Real *c_row = &C[i*ldc];
#pragma omp simd
                    for (int64_t j = 0; j < N; j++)
                    {
                        c_row[j] += a * b_row[j];
                    }
                }
            }
        }
    }
}

/* K block + unroll-jam M + unroll N (ikj) */
void gemm_tn_v4(int64_t M, int64_t N, int64_t K,
                const Real *restrict A, int64_t lda,
                const Real *restrict B, int64_t ldb,
                Real *restrict C, int64_t ldc)
{
    const int64_t M_tiled = (M / MR) * MR;
    const int64_t N_tiled = (N / NR) * NR;

#pragma omp parallel
    {

        for (int64_t kk = 0; kk < K; kk += KC)
        {
            int64_t k_end = MIN(KC+kk, K);

#pragma omp for
            for (int64_t ii = 0; ii < M_tiled; ii+=MR)
            {
                for (int64_t k = kk; k < k_end; k++)
                {
                    Real a[MR];
                    PRAGMA_UNROLL(MR)
                    for (int mr = 0; mr < MR; mr++)
                    {
                        a[mr] = A[k*lda+ii+mr];
                    }
#pragma omp simd
                    for (int64_t jj = 0; jj < N_tiled; jj+=NR)
                    {
                        Real b[NR];
                        PRAGMA_UNROLL(NR)
                        for (int nr = 0; nr < NR; nr++)
                        {
                            b[nr] = B[k*ldb+jj+nr];
                        }

                        PRAGMA_UNROLL(MR)
                        for (int mr = 0; mr < MR; mr++)
                        {
                            PRAGMA_UNROLL(NR)
                            for (int nr = 0; nr < NR; nr++)
                            {
                                C[(ii+mr)*ldc+jj+nr] += a[mr] * b[nr];
                            }
                        }
                    }

#pragma omp simd
                    for (int64_t j = N_tiled; j < N; j++)
                    {
                        register Real b = B[k*ldb+j];
                        PRAGMA_UNROLL(MR)
                        for (int mr = 0; mr < MR; mr++)
                        {
                            C[(ii+mr)*ldc+j] += a[mr] * b;
                        }
                    }
                }
            }

#pragma omp for
            for (int64_t i = M_tiled; i < M; i++)
            {
                for (int64_t k = kk; k < k_end; k++) {
                    register Real a = A[k*lda+i];
#pragma omp simd
                    for (int64_t j = 0; j < N; j++)
                    {
                        C[i*ldc+j] += a * B[k*ldb+j];
                    }
                }
            }
        }
    }
}

void gemm_tn_blas(int64_t M, int64_t N, int64_t K,
                  const Real *A, int64_t lda,
                  const Real *B, int64_t ldb,
                  Real *C, int64_t ldc)
{
    cblas_rgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                M, N, K,
                1.0,
                A, lda,
                B, ldb,
                0.0,
                C, ldc);
}

typedef void (*gemm_tn_fn)(int64_t M, int64_t N, int64_t K,
                           const Real *restrict A, int64_t lda,
                           const Real *restrict B, int64_t ldb,
                           Real *restrict C, int64_t ldc);

static void sageconv_backward_impl(SageLayer *l, gemm_tn_fn kernel)
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

    // grad_input  = grad_output @ Wroot^T
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

void sageconv_backward_gemm_tn_v1(SageLayer *l)   { sageconv_backward_impl(l, gemm_tn_v1); }
void sageconv_backward_gemm_tn_v2(SageLayer *l)   { sageconv_backward_impl(l, gemm_tn_v2); }
void sageconv_backward_gemm_tn_v3(SageLayer *l)   { sageconv_backward_impl(l, gemm_tn_v3); }
void sageconv_backward_gemm_tn_v4(SageLayer *l)   { sageconv_backward_impl(l, gemm_tn_v4); }
void sageconv_backward_gemm_tn_blas(SageLayer *l) { sageconv_backward_impl(l, gemm_tn_blas); }
