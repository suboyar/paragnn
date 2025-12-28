#include <cblas.h>

#include "linalg.h"

void dscal(size_t N, const double alpha, double *restrict X, size_t incX)
{
#if defined(USE_CBLAS) || defined(USE_CBLAS_DSCAL)
    cblas_dscal(N, alpha, X, incX);
#else
    if (incX == 1) {
#pragma omp parallel for simd
        for (size_t i = 0; i < N; i++) {
            X[i] *= alpha;
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            X[i * incX] *= alpha;
        }
    }
#endif // defined(USE_CBLAS) || defined(USE_CBLAS_DSCAL)
}
