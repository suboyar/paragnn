#include <assert.h>
#include <stdlib.h>

#include "matrix.h"

matrix_t* matrix_create(size_t height, size_t width)
{
    matrix_t *mat = malloc(sizeof(matrix_t));
    if (!mat) return NULL;

    mat->data = calloc(height * width, sizeof(*mat->data));
    if (!mat->data) {
        free(mat);
        return NULL;
    }

    mat->height = height;
    mat->width = width;
    mat->capacity = height * width;
    return mat;
}

void matrix_destroy(matrix_t *mat)
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

double* matrix_row(matrix_t *mat, size_t i)
{
    assert(i < mat->height);
    return mat->data + (i * mat->width);
}

void matrix_fill(matrix_t *matrix, double value)
{
    if (!matrix || !matrix->data) return;

    for (size_t i = 0; i < matrix->capacity; i++) {
        matrix->data[i] = value;
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
            C->data[IDX(i, j, C->width)] = sum;
        }
    }
}

void fmatrix_print(FILE* out, const matrix_t *matrix, const char *name)
{
    if (!matrix || !matrix->data) {
        fprintf(out, "%s = None\n", name ? name : "matrix");
        return;
    }

    fprintf(out, "%s = array([", name ? name : "matrix");

    for (size_t i = 0; i < matrix->height; i++) {
        if (i > 0) fprintf(out, ", ");
        fprintf(out, "[");
        for (size_t j = 0; j < matrix->width; j++) {
            if (j > 0) fprintf(out, ", ");
            fprintf(out, "%.2f", matrix->data[IDX(i, j, matrix->width)]);
        }
        fprintf(out, "]");
    }

    fprintf(out, "])\n");
}
