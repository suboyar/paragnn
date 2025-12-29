#include <stdlib.h>
#include <cblas.h>

#include "../../nob.h"

void daxpy(const size_t N, const double alpha, const double *restrict X,
           const size_t incX, double *restrict Y, const size_t incY)
{
#if defined(USE_CBLAS) || defined(USE_CBLAS_DAXPY)
    cblas_daxpy(N, alpha, X, incX, Y, incY);
#else
    if (incX == 1 && incY == 1) {
#pragma omp parallel for simd
        for (size_t i = 0; i < N; i++) {
            Y[i] += alpha * X[i];
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            Y[i * incY] += alpha * X[i * incX];
        }
    }
#endif // defined(USE_CBLAS) || defined(USE_CBLAS_DAXPY)
}
