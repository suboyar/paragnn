// matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define IDX(i, j, width) ((i) * (width) + (j))

typedef struct {
    double *data;
    size_t width;
    size_t height;
    size_t capacity;
} matrix_t;


matrix_t* matrix_create(size_t height, size_t width);
void matrix_destroy(matrix_t *mat);
inline double mat_get(const matrix_t *mat, size_t i, size_t j);
inline void mat_set(matrix_t *mat, size_t i, size_t j, double value);
double* matrix_row(matrix_t *mat, size_t i);
void matrix_fill(matrix_t *matrix, double value);
void dot(matrix_t *A, matrix_t *B, matrix_t *C);
void dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt);
void fmatrix_print(FILE* out, const matrix_t *matrix, const char *name);

#endif // MATRIX_H
