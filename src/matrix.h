#ifndef MATRIX_H
#define MATRIX_H

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "core.h"
#include "perf.h"
#include "linalg/linalg.h"

#include "../nob.h"

#define MIDX(m, i, j) (m)->data[(i)*(m)->stride+(j)]

typedef struct {
    double *data;
    union {
        size_t rows;
        size_t M;
        size_t batch;
    };
    union {
        size_t cols;
        size_t N;
        size_t features;
    };
    size_t stride;
} Matrix;

Matrix* matrix_create(size_t M, size_t N);
void matrix_destroy(Matrix *m);
void matrix_zero(Matrix *m);
void matrix_fill_random(Matrix *m, double low, double high);

// linalg interfaces
void matrix_dgemm(enum LINALG_TRANSPOSE TransA,
                  enum LINALG_TRANSPOSE TransB,
                  double alpha,
                  const Matrix *A,
                  const Matrix *B,
                  double beta,
                  Matrix *C);

// Interspection functions
#define MPRINT(m) matrix_print(m, #m, 0)
void matrix_spec(Matrix *m, const char* name);
#define MSPEC(m) matrix_spec(m, #m)
const char* matrix_shape(Matrix *m);

#endif // MATRIX_H
