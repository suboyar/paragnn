#ifndef LINALG_AXPY_H
#define LINALG_AXPY_H

void daxpy(const size_t N, const double alpha, const double *restrict X,
           const size_t incX, double *restrict Y, const size_t incY);

#endif // LINALG_AXPY_H
