#ifndef MATRIX_H
#define MATRIX_H

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "nob.h"

#define IDX(i, j, width) ((i) * (width) + (j))
#define MAT_AT(m, i, j) (m)->data[(i)*(m)->width + (j)]
#define MAT_ROW(m, row) MAT_AT((m), row, 0)
#define MAT_COL(m, col) MAT_AT((m), 0, col)
#ifndef NDEBUG
    #define MAT_BOUNDS_CHECK(m, i, j) do {                                            \
            if ((i) >= (m)->height) {                                                 \
                fprintf(stderr, "%s:%d: error: %s: Row index %zu out of bounds\n",    \
                        __FILE__, __LINE__, __func__, (size_t)(i));                   \
                abort();                                                              \
            }                                                                         \
            if ((j) >= (m)->width) {                                                  \
                fprintf(stderr, "%s:%d: error: %s: Column index %zu out of bounds\n", \
                __FILE__, __LINE__, __func__, (size_t)(j));                           \
                abort();                                                              \
            }                                                                         \
        } while(0)
#else
    #define MAT_BOUNDS_CHECK(m, i, j) (void)(0)
#endif

#define MAT_ASSERT(M1, M2) do {assert((M1)->height == (M2)->height); assert((M1)->width == (M2)->width);} while(0)
#define MAT_ASSERT_H(M1, M2) do {assert((M1)->height == (M2)->height);} while(0)
#define MAT_ASSERT_W(M1, M2) do {assert((M1)->width == (M2)->width);} while(0)
#define MAT_ASSERT_DOT(M1, M2) do {assert((M1)->width == (M2)->height);} while(0)

// Matrix creation macro which handles format switching internally
#ifdef ROW_MAJOR
    #define MAT_CREATE(dim1, dim2) mat_create(dim1, dim2)
#else
    #define MAT_CREATE(dim1, dim2) mat_create(dim2, dim1)  // Swap dimensions
#endif

typedef struct {
    double *data;
    size_t width;
    size_t height;
    size_t capacity;
} matrix_t;

matrix_t* mat_create(size_t height, size_t width);
void mat_destroy(matrix_t *mat);
// inline double *mat_at(matrix_t *m, size_t i, size_t j); // Safer variant to MAT_AT()
double mat_get(const matrix_t *mat, size_t i, size_t j);
void mat_set(matrix_t *mat, size_t i, size_t j, double value);
double* mat_row(matrix_t *mat, size_t i);
void mat_cpy(matrix_t* dst, matrix_t* src);
void mat_copy_row(matrix_t* dst, size_t dst_row, matrix_t* src, size_t src_row);
void mat_sum(matrix_t* dst, matrix_t* A);
void mat_fill(matrix_t *matrix, double value);
void mat_rand(matrix_t* m, float low, float high);
void mat_transpose(matrix_t *m);
void dot(matrix_t *A, matrix_t *B, matrix_t *C); // C = A @ B
void dot_agg(matrix_t *A, matrix_t *B, matrix_t *C); // C += A @ B
void dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt);
void dot_agg_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt);
bool mat_equal(matrix_t *A, matrix_t *B, size_t *row, size_t *col);
void mat_print(matrix_t* mat, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)
void mat_spec(matrix_t* mat, const char* name);
#define MAT_SPEC(m) mat_spec(m, #m)
const char* mat_shape(matrix_t* mat);

#endif // MATRIX_H
