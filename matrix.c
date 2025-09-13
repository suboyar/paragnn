#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

matrix_t* mat_create(size_t height, size_t width)
{
    matrix_t *mat = malloc(sizeof(matrix_t));
    if (!mat) return NULL;

    mat->data = calloc(height * width, sizeof(*mat->data));
    if (!mat->data) {
        assert(false && "Could not create matrix");
        free(mat);
        return NULL;
    }

    mat->height = height;
    mat->width = width;
    mat->capacity = height * width;
    return mat;
}

void mat_destroy(matrix_t *mat)
{
    free(mat->data);
    free(mat);
}

double mat_get(const matrix_t *mat, size_t i, size_t j)
{
    assert(mat != NULL && mat->data != NULL);
    assert(i < mat->height && j < mat->width);
    return mat->data[IDX(i, j, mat->width)];
}

void mat_set(matrix_t *mat, size_t i, size_t j, double value)
{
    assert(mat != NULL && mat->data != NULL);
    assert(i < mat->height && j < mat->width);
    mat->data[IDX(i, j, mat->width)] = value;
}

double* mat_row(matrix_t *mat, size_t i)
{
    assert(i < mat->height);
    return mat->data + (i * mat->width);
}

void mat_cpy(matrix_t* dst, matrix_t* src)
{
    MAT_ASSERT(dst, src);
    for (size_t i = 0; i < dst->height; ++i) {
        for (size_t j = 0; j < dst->width; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_copy_row(matrix_t* dst, size_t dst_row, matrix_t* src, size_t src_row)
{
    assert(dst->width == src->width);
    assert(dst_row < dst->height);
    assert(src_row < src->height);

    memcpy(&MAT_AT(dst, dst_row, 0),
           &MAT_AT(src, src_row, 0),
           src->width * sizeof(double));
}

void mat_sum(matrix_t* dst, matrix_t* A)
{
    MAT_ASSERT(dst, A);
    for (size_t i = 0; i < dst->height; ++i) {
        for (size_t j = 0; j < dst->width; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(A, i, j);
        }
    }
}

void mat_fill(matrix_t *matrix, double value)
{
    if (!matrix || !matrix->data) return;

    for (size_t i = 0; i < matrix->capacity; i++) {
        matrix->data[i] = value;
    }
}

void mat_rand(matrix_t* m, float low, float high)
{
    for (size_t i = 0; i < m->height; ++i) {
        for (size_t j = 0; j < m->width; ++j) {
            MAT_AT(m, i, j) = ((float)rand() / RAND_MAX) * (high - low) + low;
        }
    }
}

void dot(matrix_t *A, matrix_t *B, matrix_t *C)
{
    // Verify inner dimensions match
    assert(A->width == B->height);
    assert(C->height == A->height);
    assert(C->width == B->width);

    size_t M = A->height;
    size_t N = A->width;
    size_t P = B->width;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += A->data[IDX(i, k, A->width)] * B->data[IDX(k, j, B->width)];
            }
            C->data[IDX(i, j, C->width)] = sum;
        }
    }
}

void dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt)
{
    // Verify inner dimensions match
    assert((at ? A->height : A->width) == (bt ? B->width : B->height));
    assert(C->height == (at ? A->width : A->height));
    assert(C->width == (bt ? B->height : B->width));

    size_t M = at ? A->width : A->height;
    size_t N = at ? A->height : A->width;
    size_t P = bt ? B->height : B->width;

    // Precompute strides for each matrix
    size_t a_row_stride = at ? 1 : A->width;
    size_t a_col_stride = at ? A->width : 1;
    size_t b_row_stride = bt ? 1 : B->width;
    size_t b_col_stride = bt ? B->width : 1;

    // TODO unroll and jam
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += A->data[i*a_row_stride + k*a_col_stride] *
                       B->data[k*b_row_stride + j*b_col_stride];
            }
            MAT_AT(C, i, j) = sum;
        }
    }
}

void dot_agg_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt)
{
    // Verify inner dimensions match
    assert((at ? A->height : A->width) == (bt ? B->width : B->height));
    assert(C->height == (at ? A->width : A->height));
    assert(C->width == (bt ? B->height : B->width));

    size_t M = at ? A->width : A->height;
    size_t N = at ? A->height : A->width;
    size_t P = bt ? B->height : B->width;

    // Precompute strides for each matrix
    size_t a_row_stride = at ? 1 : A->width;
    size_t a_col_stride = at ? A->width : 1;
    size_t b_row_stride = bt ? 1 : B->width;
    size_t b_col_stride = bt ? B->width : 1;

    // TODO unroll and jam
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += A->data[i*a_row_stride + k*a_col_stride] *
                       B->data[k*b_row_stride + j*b_col_stride];
            }
            MAT_AT(C, i, j) += sum;
        }
    }
}


// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
bool mat_equal(matrix_t *A, matrix_t *B, size_t *row, size_t *col)
{
    // Calculate the difference.
    MAT_ASSERT(A, B);
    for (size_t i = 0; i < A->height; i++) {
        for (size_t j = 0; j < A->width; j++) {
            double a = MAT_AT(A, i, j);
            double b = MAT_AT(B, i, j);
            float diff = fabs(a - b);
            a = fabs(a);
            b = fabs(b);
            // Find the largest
            float largest = (b > a) ? b : a;

            if (diff > largest * DBL_EPSILON) {
                if (row != NULL && col != NULL) {
                    *row = i;
                    *col = j;
                }
                return false;
            }
        }
    }
    return true;
}

void mat_spec(matrix_t* mat, const char* name)
{
    if (mat != NULL) {
        printf("%s (%zux%zu)\n", name, mat->height, mat->width);
    } else {
        printf("%s (nil)\n", name);
    }
}

void mat_print(matrix_t* mat, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < mat->height; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < mat->width; ++j) {
            printf("%f ", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}
