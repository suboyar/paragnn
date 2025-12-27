#ifndef GEMM_H
#define GEMM_H

#include "linalg.h"

void dgemm(size_t M, size_t N, size_t K,
           enum LINALG_TRANSPOSE TransA,
           enum LINALG_TRANSPOSE TransB,
           double alpha,
           double *restrict A, size_t lda,
           double *restrict B, size_t ldb,
           double beta,
           double *restrict C, size_t ldc);

#endif // GEMM_H
