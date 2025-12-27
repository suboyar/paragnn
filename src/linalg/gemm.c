#include <cblas.h>

#include "linalg.h"

#if !defined(USE_CBLAS) && !defined(USE_CBLAS_DGEMM)
typedef void (*gemmfunc)(size_t M, size_t N, size_t K,
                         double alpha,
                         double *restrict A, size_t lda,
                         double *restrict B, size_t ldb,
                         double beta,
                         double *restrict C, size_t ldc);

static inline void dgemm_beta(size_t M, size_t N, double beta, size_t ldc, double *C)
{
    if (beta != 1.0) {
        if (beta == 0.0) {
#pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    C[i*ldc+j] = 0.0;
                }
            }
        } else {
#pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    C[i*ldc+j] *= beta;
                }
            }
        }
    }
}

static void dgemm_nn(size_t M, size_t N, size_t K,
              double alpha,
              double *restrict A, size_t lda,
              double *restrict B, size_t ldb,
              double beta,
              double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < K; k++) {
                sum += A[i*lda+k] * B[k*ldb+j];
            }
            C[i*ldc+j] += alpha * sum;
        }
    }
}

static void dgemm_nt(size_t M, size_t N, size_t K,
                double alpha,
                double *restrict A, size_t lda,
                double *restrict B, size_t ldb,
                double beta,
                double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < K; k++) {
                sum += A[i*lda+k] * B[j*ldb+k];
            }
            C[i*ldc+j] += alpha * sum;
        }
    }
}

static void dgemm_tn(size_t M, size_t N, size_t K,
                double alpha,
                double *restrict A, size_t lda,
                double *restrict B, size_t ldb,
                double beta,
                double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < K; k++) {
                sum += A[k*lda+i] * B[k*ldb+j];
            }
            C[i*ldc+j] += alpha * sum;
        }
    }
}

static void dgemm_tt(size_t M, size_t N, size_t K,
                double alpha,
                double *restrict A, size_t lda,
                double *restrict B, size_t ldb,
                double beta,
                double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < K; k++) {
                sum += A[k*lda+i] * B[j*ldb+k];
            }
            C[i*ldc+j] += alpha * sum;
        }
    }
}

#endif // !defined(USE_CBLAS) && !defined(USE_CBLAS_DGEMM)

void dgemm(size_t M, size_t N, size_t K,
                  enum LINALG_TRANSPOSE TransA,
                  enum LINALG_TRANSPOSE TransB,
                  double alpha,
                  double *restrict A, size_t lda,
                  double *restrict B, size_t ldb,
                  double beta,
                  double *restrict C, size_t ldc)
{
#if defined(USE_CBLAS) || defined(USE_CBLAS_DGEMM)
    cblas_dgemm(CblasRowMajor,
                (TransA ? CblasTrans : CblasNoTrans),
                (TransB ? CblasTrans : CblasNoTrans),
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
#else
    gemmfunc funcs[] = {dgemm_nn, dgemm_nt, dgemm_tn, dgemm_tt};
    int idx = TransA << 1 | TransB;
    funcs[idx](M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif // #if defined(USE_CBLAS) || defined(USE_CBLAS_DGEMM)
}
