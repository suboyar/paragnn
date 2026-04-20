#ifndef MATMUL_NAIVE_H
#define MATMUL_NAIVE_H

#include <stdint.h>
#include "core.h"

enum MATMUL_TRANSPOSE {
    MatmulNoTrans = 0,
    MatmulTrans = 1,
};

void matmul(enum MATMUL_TRANSPOSE TransA,
            enum MATMUL_TRANSPOSE TransB,
            int64_t M, int64_t N, int64_t K,
            Real alpha,
            Real *restrict A, int64_t lda,
            Real *restrict B, int64_t ldb,
            Real beta,
            Real *restrict C, int64_t ldc);

#endif // MATMUL_NAIVE_H
