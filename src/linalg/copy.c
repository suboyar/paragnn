#include <string.h>
#include <cblas.h>

#include "linalg.h"

void dcopy(size_t N, double *restrict X, size_t incX, double *restrict Y, size_t incY) {
#if defined(USE_CBLAS) || defined(USE_CBLAS_DCOPY)
    cblas_dcopy(N, X, incX, Y, incY);
#else
    if (incX == 1 && incY == 1) {
        memcpy(Y, X, N * sizeof(double));
// #pragma omp parallel for simd
        // for (size_t i = 0; i < N; i++) {
        //     Y[i] = X[i];
        // }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            Y[i * incY] = X[i * incX];
        }
    }
#endif // defined(USE_CBLAS) || defined(USE_CBLAS_DCOPY)
}
