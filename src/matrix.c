#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "matrix.h"
#include "perf.h"

#include "nob.h"

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

void mat_zero(matrix_t *matrix)
{
    if (!matrix || !matrix->data) return;
    memset(matrix->data, 0, matrix->capacity * sizeof(*matrix->data));
}

double mat_get(const matrix_t *m, size_t i, size_t j)
{
    assert(m != NULL && m->data != NULL);
    assert(i < m->height && j < m->width);
    return MAT_AT(m, i, j);
}

void mat_set(matrix_t *m, size_t i, size_t j, double value)
{
    assert(m != NULL && m->data != NULL);
    assert(i < BATCH_DIM(m));
    assert(j < NODE_DIM(m));
    MAT_AT(m, i, j) = value;
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
            MAT_AT(m, i, j) = ((double)rand() / RAND_MAX) * (high - low) + low;
        }
    }
}

void mat_transpose(matrix_t *m)
{
    size_t height = m->height;
    size_t width = m->width;

    double *temp_data = malloc(height * width * sizeof(double));
    assert(temp_data != NULL);

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            temp_data[j * height + i] = MAT_AT(m, i, j);
        }
    }

    memcpy(m->data, temp_data, height * width * sizeof(double));
    free(temp_data);

    m->height = width;
    m->width = height;
}

void mat_transpose_to(matrix_t *A, matrix_t *B)
{
    assert(A->height == B->width);
    assert(A->width == B->height);

    size_t height = A->height;
    size_t width = B->width;
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            MAT_AT(B, j, i) = MAT_AT(A, i, j);
        }
    }
}

OpMetrics dot(matrix_t *A, matrix_t *B, matrix_t *C)
{
    // Verify inner dimensions match
    assert(A->width == B->height && "Inner dimensions must match for matrix multiplication");

    assert(C->height == A->height && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

    size_t M = A->height;
    size_t N = A->width;
    size_t P = B->width;

    uint64_t flops = 2ULL * M * P * N;
    uint64_t bytes = (2ULL * M * N * P) + (M * P) * sizeof(double);

#pragma omp parallel for
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += MAT_AT(A, i, k) * MAT_AT(B, k, j);
            }
            MAT_AT(C, i, j) = sum;
        }
    }

    return (OpMetrics){.flops=flops, .bytes=bytes};
}

OpMetrics dot_agg(matrix_t *A, matrix_t *B, matrix_t *C)
{
    // Verify inner dimensions match

    assert(A->width == B->height);
    assert(C->height == A->height);
    assert(C->width == B->width);

    size_t M = A->height;
    size_t N = A->width;
    size_t P = B->width;

    uint64_t flops = (2ULL * M * P * N) + (M * P);
    uint64_t bytes = (2ULL * M * N * P) + (2ULL * M * P) * sizeof(double);

#pragma omp parallel for
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += MAT_AT(A, i, k) * MAT_AT(B, k, j);
            }
            MAT_AT(C, i, j) += sum;
        }
    }

    return (OpMetrics){.flops=flops, .bytes=bytes};
}

OpMetrics dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt)
{
    // Calculate effective dimensions after potential transposition
    size_t eff_A_rows = at ? A->width : A->height;
    size_t eff_A_cols = at ? A->height : A->width;
    size_t eff_B_rows = bt ? B->width : B->height;
    size_t eff_B_cols = bt ? B->height : B->width;

    // Verify dimensions for valid matrix multiplication: eff_A * eff_B = C
    // Inner dimensions must match: columns of eff_A = rows of eff_B
    assert(eff_A_cols == eff_B_rows && "Inner dimensions must match for matrix multiplication");

    // Output dimensions must match: C = eff_A_rows @ eff_B_cols
    assert(C->height == eff_A_rows && "Output height must match effective rows of operand A");
    assert(C->width == eff_B_cols && "Output width must match effective columns of operand B");

    size_t M = eff_A_rows;
    size_t N = eff_A_cols;
    size_t P = eff_B_cols;

    uint64_t flops = 2ULL * M * N * P;
    uint64_t bytes = ((2ULL * N * P * M) + (P * M)) * sizeof(double);

    // Precompute strides for each matrix
    size_t a_row_stride = at ? 1 : A->width;
    size_t a_col_stride = at ? A->width : 1;
    size_t b_row_stride = bt ? 1 : B->width;
    size_t b_col_stride = bt ? B->width : 1;

    // TODO unroll and jam
#pragma omp parallel for
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

    return (OpMetrics){.flops=flops, .bytes=bytes};
}

OpMetrics dot_agg_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt)
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

    uint64_t flops = (2ULL * M * P * N) + (M * P);
    uint64_t bytes = (2ULL * M * N * P) + (2ULL * M * P) * sizeof(double);

    // TODO unroll and jam
#pragma omp parallel for
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
    return (OpMetrics){.flops=flops, .bytes=bytes};
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
            double diff = fabs(a - b);
            a = fabs(a);
            b = fabs(b);
            // Find the largest
            double largest = (b > a) ? b : a;

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

const char* mat_shape(matrix_t* m)
{
    if (m == NULL ) {
        return "(nil)";
    }
    return nob_temp_sprintf("(%zux%zu)", m->height, m->width);

}
