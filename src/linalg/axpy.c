#include <stdlib.h>
#include <cblas.h>

#include "../../nob.h"

void daxpy(const size_t N, const double alpha, const double *restrict X,
           const size_t incX, double *restrict Y, const size_t incY)
{
#ifdef USE_CBLAS
    cblas_daxpy(N, alpha, X, incX, Y, incY);
#else
    if (incX == 1 && incY == 1) {
        for (size_t i = 0; i < N; i++) {
            Y[i] += alpha * X[i];
        }
    } else {
        NOB_TODO("Implement daxpy for case of increment not equal to 1");
    }
#endif // USE_CBLAS
}
