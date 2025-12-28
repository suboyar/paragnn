#ifndef LINALG_COPY_H
#define LINALG_COPY_H

void dcopy(size_t N, double *restrict X, size_t incX, double *restrict Y, size_t incY);

#endif // LINALG_COPY_H
