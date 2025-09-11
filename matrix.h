#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define IDX(i, j, width) ((i) * (width) + (j))
#define MAT_AT(m, i, j) (m)->data[(i)*(m)->width + (j)]
#define MAT_ROW(m, row) (m)->data[(row)*(m)->width]

#define MAT_ASSERT(M1, M2) do {assert((M1)->height == (M2)->height); assert((M1)->width == (M2)->width);} while(0)

typedef struct {
    double *data;
    size_t width;
    size_t height;
    size_t capacity;
} matrix_t;

matrix_t* mat_create(size_t height, size_t width);
void mat_destroy(matrix_t *mat);
double mat_get(const matrix_t *mat, size_t i, size_t j);
void mat_set(matrix_t *mat, size_t i, size_t j, double value);
double* mat_row(matrix_t *mat, size_t i);
void mat_cpy(matrix_t* dst, matrix_t* src);
void mat_copy_row(matrix_t* dst, size_t dst_row, matrix_t* src, size_t src_row);
void mat_sum(matrix_t* dst, matrix_t* A);
void mat_fill(matrix_t *matrix, double value);
void mat_rand(matrix_t* m, float low, float high);
void dot(matrix_t *A, matrix_t *B, matrix_t *C); // C = A . B
void dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt);
bool mat_equal(matrix_t *A, matrix_t *B, size_t *row, size_t *col);
void mat_print(matrix_t* mat, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)
void mat_spec(matrix_t* mat, const char* name);
#define MAT_SPEC(m) mat_spec(m, #m)

#endif // MATRIX_H
