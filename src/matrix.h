#ifndef MATRIX_H
#define MATRIX_H

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "core.h"
#include "perf.h"

#include "nob.h"

#define BATCH_DIM(m) ((m)->height)  // batch_size = height
#define NODE_DIM(m)  ((m)->width)   // features = width
#define MAT_CREATE(batch, features) mat_create(batch, features)
#define MAT_AT(m, i, j) (m)->data[(i)*(m)->width + (j)]

#define MAT_ROW(m, row) NOB_TODO("MAT_ROW is depricated")
#define MAT_COL(m, col) NOB_TODO("MAT_COL is depricated")

#ifndef NDEBUG
    #define MAT_BOUNDS_CHECK(m, i, j) do {                                                           \
            if ((i) >= BATCH_DIM(m)) {                                                               \
                ERROR("Batch index %zu out of bounds (max %zu)", (size_t)(i), (size_t)BATCH_DIM(m)); \
                abort();                                                                             \
            }                                                                                        \
            if ((j) >= NODE_DIM(m)) {                                                                \
                ERROR("Node index %zu out of bounds (max %zu)", (size_t)(j), (size_t)NODE_DIM(m));   \
                abort();                                                                             \
            }                                                                                        \
    } while(0)
#else
    #define MAT_BOUNDS_CHECK(m, i, j) (void)(0)
#endif

#define MAT_ASSERT(A, B) do {assert((A)->height == (B)->height); assert((A)->width == (B)->width);} while(0)
#define MAT_ASSERT_BATCH(A, B) do {assert(BATCH_DIM((A)) == BATCH_DIM((B)));} while(0)
#define MAT_ASSERT_NODE(A, B) do {assert(NODE_DIM((A)) == NODE_DIM((B)));} while(0)
#define MAT_ASSERT_DOT(A, B) do {assert((A)->width == (B)->height);} while(0)
#define MAT_ASSERT_DOT_EX(A, B, AT, BT) assert(((AT) ? (A)->height : (A)->width) == ((BT) ? (B)->width : (B)->height))
#define MAT_ASSERT_H(A, B) NOB_TODO("MAT_ASSERT_H is depricated, maybe use MAT_ASSERT_BATCH")
#define MAT_ASSERT_W(A, B) NOB_TODO("MAT_ASSERT_W is depricated, maybe use MAT_ASSERT_NODE")


typedef struct {
    double *data;
    size_t width;
    size_t height;
    size_t capacity;
} matrix_t;

matrix_t* mat_create(size_t height, size_t width);
void mat_destroy(matrix_t *mat);
// inline double *mat_at(matrix_t *m, size_t i, size_t j); // Safer variant to MAT_AT()
void mat_zero(matrix_t *matrix);
double mat_get(const matrix_t *mat, size_t i, size_t j);
void mat_set(matrix_t *mat, size_t i, size_t j, double value);
double* mat_row(matrix_t *mat, size_t i);
void mat_cpy(matrix_t* dst, matrix_t* src);
void mat_copy_row(matrix_t* dst, size_t dst_row, matrix_t* src, size_t src_row);
void mat_sum(matrix_t* dst, matrix_t* A);
void mat_fill(matrix_t *matrix, double value);
void mat_rand(matrix_t* m, float low, float high);
void mat_transpose(matrix_t *m);
void mat_transpose_to(matrix_t *A, matrix_t *B);
OpMetrics dot(matrix_t *A, matrix_t *B, matrix_t *C); // C = A @ B
OpMetrics dot_agg(matrix_t *A, matrix_t *B, matrix_t *C); // C += A @ B
OpMetrics dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt);
OpMetrics dot_agg_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt);
bool mat_equal(matrix_t *A, matrix_t *B, size_t *row, size_t *col);
void mat_print(matrix_t* mat, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)
void mat_spec(matrix_t* mat, const char* name);
#define MAT_SPEC(m) mat_spec(m, #m)
const char* mat_shape(matrix_t* mat);

#endif // MATRIX_H
