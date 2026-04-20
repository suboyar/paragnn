#include <stdint.h>

#include "matmul_naive.h"

static void beta_kernel(int64_t M, int64_t N, Real beta, Real *restrict C, int64_t ldc)
{
#pragma omp parallel for simd
    for (int64_t i = 0; i < M; i++)
    {
        for (int64_t j = 0; j < N; j++)
        {
            C[i*ldc+j] *= beta;
        }
    }
}

static void matmul_nn(int64_t M, int64_t N, int64_t K,
                      const Real alpha,
                      const Real *restrict A, int64_t lda,
                      const Real *restrict B, int64_t ldb,
                      const Real beta,
                      Real *restrict C, int64_t ldc)
{
    if (beta != 0.0 && beta != 1.0)
    {
        beta_kernel(M, N, beta, C, ldc);
    }

    if (beta == 0.0)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            const Real *restrict a_row = &A[i*lda];
            Real *restrict c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += a_row[k] * B[k*ldb+j];
                }
                c_row[j] = alpha * sum;
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            const Real *restrict a_row = &A[i*lda];
            Real *restrict c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += a_row[k] * B[k*ldb+j];
                }
                c_row[j] += alpha * sum;
            }
        }
    }
}

static void matmul_nt(int64_t M, int64_t N, int64_t K,
                      const Real alpha,
                      const Real *restrict A, int64_t lda,
                      const Real *restrict B, int64_t ldb,
                      const Real beta,
                      Real *restrict C, int64_t ldc)
{
    if (beta != 0.0 && beta != 1.0)
    {
        beta_kernel(M, N, beta, C, ldc);
    }

    if (beta == 0.0)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            const Real *restrict a_row = &A[i*lda];
            Real *restrict c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                const Real *restrict b_row = &B[j*ldb];
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += a_row[k] * b_row[k];
                }
                c_row[j] = alpha * sum;
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            const Real *restrict a_row = &A[i*lda];
            Real *restrict c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                const Real *restrict b_row = &B[j*ldb];
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += a_row[k] * b_row[k];
                }
                c_row[j] += alpha * sum;
            }
        }
    }
}


static void matmul_tn(int64_t M, int64_t N, int64_t K,
                      const Real alpha,
                      const Real *restrict A, int64_t lda,
                      const Real *restrict B, int64_t ldb,
                      const Real beta,
                      Real *restrict C, int64_t ldc)
{
    if (beta != 0.0 && beta != 1.0)
    {
        beta_kernel(M, N, beta, C, ldc);
    }

    if (beta == 0.0)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            Real *restrict c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += A[k*lda+i] * B[k*ldb+j];
                }
                c_row[j] = alpha * sum;
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            Real *c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += A[k*lda+i] * B[k*ldb+j];
                }
                c_row[j] += alpha * sum;
            }
        }
    }
}

static void matmul_tt(int64_t M, int64_t N, int64_t K,
                      const Real alpha,
                      const Real *restrict A, int64_t lda,
                      const Real *restrict B, int64_t ldb,
                      const Real beta,
                      Real *restrict C, int64_t ldc)
{
    if (beta != 0.0 && beta != 1.0)
    {
        beta_kernel(M, N, beta, C, ldc);
    }

    if (beta == 0.0)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            Real *restrict c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                const Real *restrict b_row = &B[j*ldb];
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += A[k*lda+i] * b_row[k];
                }
                c_row[j] = alpha * sum;
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int64_t i = 0; i < M; i++)
        {
            Real *restrict c_row = &C[i*ldc];
            for (int64_t j = 0; j < N; j++)
            {
                const Real *restrict b_row = &B[j*ldb];
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (int64_t k = 0; k < K; k++)
                {
                    sum += A[k*lda+i] * b_row[k];
                }
                c_row[j] += alpha * sum;
            }
        }
    }
}

typedef void (*matmulfunc)(int64_t M, int64_t N, int64_t K,
                           const Real alpha,
                           const Real *restrict A, int64_t lda,
                           const Real *restrict B, int64_t ldb,
                           const Real beta,
                           Real *restrict C, int64_t ldc);

void matmul(enum MATMUL_TRANSPOSE TransA,
            enum MATMUL_TRANSPOSE TransB,
            int64_t M, int64_t N, int64_t K,
            Real alpha,
            Real *restrict A, int64_t lda,
            Real *restrict B, int64_t ldb,
            Real beta,
            Real *restrict C, int64_t ldc)
{
    matmulfunc funcs[] = {matmul_nn, matmul_nt, matmul_tn, matmul_tt};
    int idx = TransA << 1 | TransB;
    funcs[idx](M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
