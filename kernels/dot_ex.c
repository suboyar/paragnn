#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/param.h>

#include <omp.h>
#include <openblas/cblas.h>     // NOTE: This might not work on eX3, but is needed for OpenSUSE

#define NOB_IMPLEMENTATION      // Needed for matrix.h etc...
#include "matrix.h"
#include "perf.h"
#include "cache_counter.h"

#define NUM_TRAIN_NODES 90941
#define NUM_VALID_NODES 29799
#define NUM_TEST_NODES 48603

#ifndef NTIMES
#    define NTIMES 10
#endif // NTIMES

#ifndef ENABLE_INNER_SIMD
#  define OUTER_SIMD simd
#  define INNER_SIMD
#else
#  define OUTER_SIMD
#  define INNER_SIMD simd
#endif

#define ARRAY_LEN(array) (sizeof(array)/sizeof(array[0]))

FileHandler csv_out = {0};
cache_counter_t* thread_counters = NULL;

typedef enum {
    RESTORE_NONE = 0,
    RESTORE_A    = 1 << 0,
    RESTORE_B    = 1 << 1,
} RestoreFlags;

typedef struct {
    const char* name;
    void (*fn) (matrix_t*, matrix_t*, matrix_t*);
    RestoreFlags restore;
} MatmulKernel;

#define NEW_KERNEL(fn, restore) (MatmulKernel){.name=#fn, (fn), (restore)}

//
// When doing doing matmul only B needs to be traversed column-wise,
// but since matrix A's shape is of MxN and B's shape is MxK, this forces us
// to traverse A in column-wise order too. Which is not ideal for cache utilization,
// especially when A and B are of a tall-and-skinny matrix.
//

void matmul_naive(matrix_t *A, matrix_t *B, matrix_t *C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

#pragma omp parallel for OUTER_SIMD
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;

#pragma omp INNER_SIMD
            for (size_t k = 0; k < K; k++) {
                sum += MAT_AT(A, k, i) * MAT_AT(B, k, j);
            }
            MAT_AT(C, i, j) = sum;
        }
    }
}

void matmul_naive_restrict(matrix_t *A, matrix_t *B, matrix_t *C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    double* restrict a_data = A->data;
    double* restrict b_data = B->data;
    double* restrict c_data = C->data;

    size_t a_width = A->width;
    size_t b_width = B->width;
    size_t c_width = C->width;

#pragma omp parallel for OUTER_SIMD
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;

#pragma omp INNER_SIMD
            for (size_t k = 0; k < K; k++) {
                sum += a_data[k*a_width+i] * b_data[k*b_width+j];
            }
            c_data[i*c_width+j] = sum;
        }
    }
}

void matmul_unroll(matrix_t *A, matrix_t *B, matrix_t *C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

#pragma omp parallel for OUTER_SIMD
    for (size_t i = 0; i < M; i++) {
        size_t j;
        for (j = 0; j + 3 < N; j+=4) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

#pragma omp INNER_SIMD
            for (size_t k = 0; k < K; k++) {
                double a_ik = MAT_AT(A, k, i);
                sum0 += a_ik * MAT_AT(B, k, (j+0));
                sum1 += a_ik * MAT_AT(B, k, (j+1));
                sum2 += a_ik * MAT_AT(B, k, (j+2));
                sum3 += a_ik * MAT_AT(B, k, (j+3));
            }

            MAT_AT(C, i, j+0) = sum0;
            MAT_AT(C, i, j+1) = sum1;
            MAT_AT(C, i, j+2) = sum2;
            MAT_AT(C, i, j+3) = sum3;
        }

        for (; j < N; j++) {
            double sum = 0.0;

#pragma omp INNER_SIMD
            for (size_t k = 0; k < N; k++) {
                sum += MAT_AT(A, k, i) * MAT_AT(B, k, j);
            }
            MAT_AT(C, i, j) = sum;
        }
    }
}

void matmul_tiled(matrix_t *A, matrix_t *B, matrix_t *C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for OUTER_SIMD
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);
            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);
                for (size_t i = istart; i < istop; i++) {
                    for (size_t j = jstart; j < jstop; j++) {
                        double sum = 0.0;

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += MAT_AT(A, k, i) * MAT_AT(B, k, j);
                        }

                        MAT_AT(C, i, j) += sum;
                    }
                }
            }
        }
    }
}

void matmul_tiled_1x4(matrix_t* A, matrix_t* B, matrix_t* C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for OUTER_SIMD
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);

            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                for (size_t i = istart; i < istop; i++) {

                    size_t j;
                    for (j = jstart; j + 3 < jstop; j += 4) {
                        double sum0 = MAT_AT(C, i, j+0);
                        double sum1 = MAT_AT(C, i, j+1);
                        double sum2 = MAT_AT(C, i, j+2);
                        double sum3 = MAT_AT(C, i, j+3);

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = MAT_AT(A, k, i);
                            sum0 += a_ik * MAT_AT(B, k, j+0);
                            sum1 += a_ik * MAT_AT(B, k, j+1);
                            sum2 += a_ik * MAT_AT(B, k, j+2);
                            sum3 += a_ik * MAT_AT(B, k, j+3);
                        }

                        MAT_AT(C, i, j+0) = sum0;
                        MAT_AT(C, i, j+1) = sum1;
                        MAT_AT(C, i, j+2) = sum2;
                        MAT_AT(C, i, j+3) = sum3;
                    }

                    for (; j < jstop; j++) {
                        double sum = MAT_AT(C, i, j+0);

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += MAT_AT(A, k, i) * MAT_AT(B, k, j);
                        }

                        MAT_AT(C, i, j) = sum;
                    }
                }

            }
        }
    }
}

void matmul_tiled_1x4_restrict(matrix_t* A, matrix_t* B, matrix_t* C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    double* restrict a_data = A->data;
    double* restrict b_data = B->data;
    double* restrict c_data = C->data;

    size_t a_width = A->width;
    size_t b_width = B->width;
    size_t c_width = C->width;

    size_t b = 64;

#pragma omp parallel for OUTER_SIMD
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);

            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                for (size_t i = istart; i < istop; i++) {

                    size_t j;
                    for (j = jstart; j + 3 < jstop; j += 4) {
                        double sum0 = c_data[i*c_width+(j+0)];
                        double sum1 = c_data[i*c_width+(j+1)];
                        double sum2 = c_data[i*c_width+(j+2)];
                        double sum3 = c_data[i*c_width+(j+3)];

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = a_data[k*a_width+i];
                            sum0 += a_ik * b_data[k*b_width+(j+0)];
                            sum1 += a_ik * b_data[k*b_width+(j+1)];
                            sum2 += a_ik * b_data[k*b_width+(j+2)];
                            sum3 += a_ik * b_data[k*b_width+(j+3)];
                        }

                        c_data[i*c_width+(j+0)] = sum0;
                        c_data[i*c_width+(j+1)] = sum1;
                        c_data[i*c_width+(j+2)] = sum2;
                        c_data[i*c_width+(j+3)] = sum3;
                    }

                    for (; j < jstop; j++) {
                        double sum = c_data[i*c_width+j];

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += a_data[k*a_width+i] * b_data[k*b_width+i];
                        }

                        c_data[i*c_width+j] = sum;
                    }
                }

            }
        }
    }
}

void matmul_tiled_2x2(matrix_t*  A, matrix_t* B, matrix_t* C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for OUTER_SIMD
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);
            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                size_t j;
                for (j = jstart; j + 1 < jstop; j += 2) {
                    size_t i;
                    for (i = istart; i + 1 < istop; i += 2) {
                        double sum00 = MAT_AT(C, i+0, j+0);
                        double sum01 = MAT_AT(C, i+0, j+1);
                        double sum10 = MAT_AT(C, i+1, j+0);
                        double sum11 = MAT_AT(C, i+1, j+1);

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_i0k = MAT_AT(A, k, i+0);
                            double a_i1k = MAT_AT(A, k, i+1);
                            double b_kj0 = MAT_AT(B, k, j+0);
                            double b_kj1 = MAT_AT(B, k, j+1);

                            sum00 += a_i0k * b_kj0;
                            sum01 += a_i0k * b_kj1;
                            sum10 += a_i1k * b_kj0;
                            sum11 += a_i1k * b_kj1;
                        }

                        MAT_AT(C, i+0, j+0) = sum00;
                        MAT_AT(C, i+0, j+1) = sum01;
                        MAT_AT(C, i+1, j+0) = sum10;
                        MAT_AT(C, i+1, j+1) = sum11;
                    }

                    for (; i < istop; i++) {
                        double sum0 = 0.0;
                        double sum1 = 0.0;

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = MAT_AT(A, k, i);
                            sum0 += a_ik * MAT_AT(B, k, j+0);
                            sum1 += a_ik * MAT_AT(B, k, j+1);
                        }

                        MAT_AT(C, i, j+0) += sum0;
                        MAT_AT(C, i, j+1) += sum1;
                    }
                }

                for (; j < jstop; j++) {
                    size_t i;

                    for (i = istop; i + 1 < istop; i += 2) {
                        double sum0 = 0.0;
                        double sum1 = 0.0;

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double b_kj = MAT_AT(B, k, j);
                            sum0 += MAT_AT(A, k, i+0) * b_kj;
                            sum1 += MAT_AT(A, k, i+1) * b_kj;
                        }
                        MAT_AT(C, i+0, j) += sum0;
                        MAT_AT(C, i+1, j) += sum1;
                    }

                    for (; i < istop; i++) {
                        double sum = 0.0;
#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += MAT_AT(A, k, i) * MAT_AT(B, k, j);
                        }
                        MAT_AT(C, i, j) += sum;
                    }
                }
            }
        }
    }
}

void transpose_inplace(matrix_t *src)
{
    size_t height = src->height;
    size_t width = src->width;
    size_t b = 64;

    matrix_t *dst = mat_create(width, height);

    size_t s_width = src->width;
    size_t d_width = dst->width;

    double* restrict s = src->data;
    double* restrict d = dst->data;

#pragma omp parallel for OUTER_SIMD
    for (size_t jj = 0; jj < width; jj += b) {
        size_t jstop = (jj + b < width) ? jj + b : width;

        for (size_t ii = 0; ii < height; ii += b) {
            size_t istop = (ii + b < height) ? ii + b : height;

            size_t i;
            for (i = ii; i + 3 < istop; i+=4) {
                double* restrict s_row0 = &s[(i+0) * s_width];
                double* restrict s_row1 = &s[(i+1) * s_width];
                double* restrict s_row2 = &s[(i+2) * s_width];
                double* restrict s_row3 = &s[(i+3) * s_width];

#pragma omp INNER_SIMD
                for (size_t j = jj; j < jstop; j++) {
                    d[d_width * j + (i+0)] = s_row0[j];
                    d[d_width * j + (i+1)] = s_row1[j];
                    d[d_width * j + (i+2)] = s_row2[j];
                    d[d_width * j + (i+3)] = s_row3[j];
                }
            }

            for (; i < istop; i++) {
                double* restrict s_row = &s[i * s_width];

#pragma omp INNER_SIMD
                for (size_t j = jj; j < jstop; j++) {
                    d[j * d_width + i] = s_row[j];
                }
            }
        }
    }

    double *temp = src->data;
    src->data = dst->data;
    src->height = width;
    src->width = height;
    free(temp);
    free(dst);
}

double* transpose(const double *restrict s, size_t height, size_t width)
{
    size_t alignment = 64;      // hardcoded cacheline size
    size_t size = height * width * sizeof(double);
    size_t padded_size = (size + alignment - 1) & ~(alignment - 1);
    double *restrict d = aligned_alloc(alignment, padded_size);

    size_t s_width = width;
    size_t d_width = height;

    size_t b = 64;
#pragma omp parallel for OUTER_SIMD
    for (size_t jj = 0; jj < width; jj += b) {
        size_t jstop = (jj + b < width) ? jj + b : width;

        for (size_t ii = 0; ii < height; ii += b) {
            size_t istop = (ii + b < height) ? ii + b : height;

            size_t i;
            for (i = ii; i + 3 < istop; i+=4) {
                const double* restrict s_row0 = &s[(i+0) * s_width];
                const double* restrict s_row1 = &s[(i+1) * s_width];
                const double* restrict s_row2 = &s[(i+2) * s_width];
                const double* restrict s_row3 = &s[(i+3) * s_width];

#pragma omp INNER_SIMD
                for (size_t j = jj; j < jstop; j++) {
                    d[d_width * j + (i+0)] = s_row0[j];
                    d[d_width * j + (i+1)] = s_row1[j];
                    d[d_width * j + (i+2)] = s_row2[j];
                    d[d_width * j + (i+3)] = s_row3[j];
                }
            }

            for (; i < istop; i++) {
                const double* restrict s_row = &s[i * s_width];

#pragma omp INNER_SIMD
                for (size_t j = jj; j < jstop; j++) {
                    d[j * d_width + i] = s_row[j];
                }
            }
        }
    }

    return d;
}


//
// While previously both A and B were being traversed in column-major
// (read comment above matmul()), we can remedy this issue by
// transposing both of these matrices at the beginning, which allows
// us to traverse both matrices in row-major order. The idea is by
// doing some pre-work will lead to faster computation of matmul.
//

void matmul_tiled_transposed(matrix_t *A, matrix_t *B, matrix_t *C)
{
    matrix_t At = {
        .height = A->width,
        .width = A->height,
        .data = transpose(A->data, A->height, A->width)
    };

    matrix_t Bt = {
        .height = B->width,
        .width = B->height,
        .data = transpose(B->data, B->height, B->width)
    };

    size_t M = At.height;
    size_t K = At.width;       // <=> Bt->width
    size_t N = Bt.height;

    size_t b = 64;

#pragma omp parallel for OUTER_SIMD
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);
            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                for (size_t i = istart; i < istop; i++) {
                    for (size_t j = jstart; j < jstop; j++) {
                        double sum = MAT_AT(C, i, j);

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += MAT_AT(&At, i, k) * MAT_AT(&Bt, j, k);
                        }

                        MAT_AT(C, i, j) += sum;
                    }
                }
            }
        }
    }

    free(At.data);
    free(Bt.data);
}

void matmul_tiled_2x2_transposed(matrix_t* A, matrix_t* B, matrix_t* C)
{
    //
    // We can do in place transposing since in GNN the neurons in
    // activation layers (A & B) doesn't really mater after we have
    // gone through this layer. Only the weight matrix (C) is of
    // importance.
    //

    matrix_t At = {
        .height = A->width,
        .width = A->height,
        .data = transpose(A->data, A->height, A->width)
    };

    matrix_t Bt = {
        .height = B->width,
        .width = B->height,
        .data = transpose(B->data, B->height, B->width)
    };

    size_t M = At.height;
    size_t K = At.width;       // <=> Bt->width
    size_t N = Bt.height;

    size_t b = 64;

#pragma omp parallel for OUTER_SIMD
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);
            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                size_t j;
                for (j = jstart; j + 1 < jstop; j += 2) {
                    size_t i;
                    for (i = istart; i + 1 < istop; i += 2) {
                        double sum00 = MAT_AT(C, i+0, j+0);
                        double sum01 = MAT_AT(C, i+0, j+1);
                        double sum10 = MAT_AT(C, i+1, j+0);
                        double sum11 = MAT_AT(C, i+1, j+1);

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_i0k = MAT_AT(&At, i+0, k);
                            double a_i1k = MAT_AT(&At, i+1, k);
                            double b_kj0 = MAT_AT(&Bt, j+0, k);
                            double b_kj1 = MAT_AT(&Bt, j+1, k);

                            sum00 += a_i0k * b_kj0;
                            sum01 += a_i0k * b_kj1;
                            sum10 += a_i1k * b_kj0;
                            sum11 += a_i1k * b_kj1;
                        }

                        MAT_AT(C, i+0, j+0) = sum00;
                        MAT_AT(C, i+0, j+1) = sum01;
                        MAT_AT(C, i+1, j+0) = sum10;
                        MAT_AT(C, i+1, j+1) = sum11;
                    }

                    for (; i < istop; i++) {
                        double sum0 = 0.0;
                        double sum1 = 0.0;

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = MAT_AT(&At, k, i);
                            sum0 += a_ik * MAT_AT(&Bt, j+0, k);
                            sum1 += a_ik * MAT_AT(&Bt, j+1, k);
                        }

                        MAT_AT(C, i, j+0) += sum0;
                        MAT_AT(C, i, j+1) += sum1;
                    }
                }

                for (; j < jstop; j++) {
                    size_t i;

                    for (i = istart; i + 1 < istop; i += 2) {
                        double sum0 = 0.0;
                        double sum1 = 0.0;

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double b_kj = MAT_AT(&Bt, j, k);
                            sum0 += MAT_AT(&At, i+0, k) * b_kj;
                            sum1 += MAT_AT(&At, i+1, k) * b_kj;
                        }
                        MAT_AT(C, i+0, j) += sum0;
                        MAT_AT(C, i+1, j) += sum1;
                    }

                    for (; i < istop; i++) {
                        double sum = 0.0;

#pragma omp INNER_SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += MAT_AT(&At, i, k) * MAT_AT(&Bt, j, k);
                        }
                        MAT_AT(C, i, j) += sum;
                    }
                }
            }
        }
    }

    free(At.data);
    free(Bt.data);
}

void matmul_tiled_1x4_transposed(matrix_t* A, matrix_t* B, matrix_t* C)
{
    matrix_t At = {
        .height = A->width,
        .width = A->height,
        .data = transpose(A->data, A->height, A->width)
    };

    matrix_t Bt = {
        .height = B->width,
        .width = B->height,
        .data = transpose(B->data, B->height, B->width)
    };

    size_t M = At.height;
    size_t K = At.width;       // <=> Bt->width
    size_t N = Bt.height;

    size_t b = 64;

#pragma omp parallel for
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);

            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                for (size_t i = istart; i < istop; i++) {

                    size_t j;
                    for (j = jstart; j + 3 < jstop; j += 4) {
                        double sum0 = MAT_AT(C, i, j+0);
                        double sum1 = MAT_AT(C, i, j+1);
                        double sum2 = MAT_AT(C, i, j+2);
                        double sum3 = MAT_AT(C, i, j+3);

#pragma omp simd
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = MAT_AT(&At, i, k);
                            sum0 += a_ik * MAT_AT(&Bt, j+0, k);
                            sum1 += a_ik * MAT_AT(&Bt, j+1, k);
                            sum2 += a_ik * MAT_AT(&Bt, j+2, k);
                            sum3 += a_ik * MAT_AT(&Bt, j+3, k);
                        }

                        MAT_AT(C, i, j+0) = sum0;
                        MAT_AT(C, i, j+1) = sum1;
                        MAT_AT(C, i, j+2) = sum2;
                        MAT_AT(C, i, j+3) = sum3;
                    }

                    for (; j < jstop; j++) {
                        double sum = MAT_AT(C, i, j);

#pragma omp simd
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += MAT_AT(&At, i, k) * MAT_AT(&Bt, j, k);
                        }

                        MAT_AT(C, i, j) = sum;
                    }
                }

            }
        }
    }

    free(At.data);
    free(Bt.data);
}

void matmul_tiled_1x4_transposed_restrict(matrix_t* A, matrix_t* B, matrix_t* C)
{
    double* restrict a_data = transpose(A->data, A->height, A->width);
    double* restrict b_data = transpose(B->data, B->height, B->width);
    double* restrict c_data = C->data;

    // A and B is transposed
    size_t a_width = A->height;
    size_t b_width = B->height;
    size_t c_width = C->width;

    // Use the old shape pre-transpose
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);

            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                for (size_t i = istart; i < istop; i++) {
                    double* restrict c_row = &c_data[i * c_width];
                    double* restrict a_row = &a_data[i * a_width];

                    size_t j;
                    for (j = jstart; j + 3 < jstop; j += 4) {
                        double sum0 = c_row[j+0];
                        double sum1 = c_row[j+1];
                        double sum2 = c_row[j+2];
                        double sum3 = c_row[j+3];

#pragma omp simd
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = a_row[k];
                            sum0 += a_ik * b_data[(j+0) * b_width + k];
                            sum1 += a_ik * b_data[(j+1) * b_width + k];
                            sum2 += a_ik * b_data[(j+2) * b_width + k];
                            sum3 += a_ik * b_data[(j+3) * b_width + k];
                        }

                        c_row[j+0] = sum0;
                        c_row[j+1] = sum1;
                        c_row[j+2] = sum2;
                        c_row[j+3] = sum3;
                    }

                    for (; j < jstop; j++) {
                        double sum = c_row[j];
#pragma omp simd
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = a_row[k];
                            sum += a_ik * b_data[j * b_width + k];
                        }
                        c_row[j] = sum;
                    }
                }
            }
        }
    }

    free(a_data);
    free(b_data);
}

void matmul_tiled_1x4_transposed_A_restrict(matrix_t* A, matrix_t* B, matrix_t* C)
{
    double* restrict a_data = transpose(A->data, A->height, A->width);
    double* restrict b_data = B->data;
    double* restrict c_data = C->data;

    // A is transposed
    size_t a_width = A->height;
    size_t b_width = B->width;
    size_t c_width = C->width;

    // Use the old shape pre-transpose
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t jj = 0; jj < N; jj += b) {
            size_t jstart = jj, jstop = MIN(jj+b, N);

            for (size_t kk = 0; kk < K; kk += b) {
                size_t kstart = kk, kstop = MIN(kk+b, K);

                for (size_t i = istart; i < istop; i++) {
                    double* restrict c_row = &c_data[i * c_width];
                    double* restrict a_row = &a_data[i * a_width];

                    size_t j;
                    for (j = jstart; j + 3 < jstop; j += 4) {
                        double sum0 = c_row[j+0];
                        double sum1 = c_row[j+1];
                        double sum2 = c_row[j+2];
                        double sum3 = c_row[j+3];

#pragma omp simd
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = a_row[k];
                            sum0 += a_ik * b_data[b_width * k + (j+0)];
                            sum1 += a_ik * b_data[b_width * k + (j+1)];
                            sum2 += a_ik * b_data[b_width * k + (j+2)];
                            sum3 += a_ik * b_data[b_width * k + (j+3)];
                        }

                        c_row[j+0] = sum0;
                        c_row[j+1] = sum1;
                        c_row[j+2] = sum2;
                        c_row[j+3] = sum3;
                    }

                    for (; j < jstop; j++) {
                        double sum = c_row[j];
#pragma omp simd
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = a_row[k];
                            sum += a_ik * b_data[k + b_width + j];
                        }
                        c_row[j] = sum;
                    }
                }
            }
        }
    }

    free(a_data);
}

void matmul_tiled_1x4_transposed_restrict_ikj(matrix_t* A, matrix_t* B, matrix_t* C)
{
    double* restrict a_data = transpose(A->data, A->height, A->width);
    double* restrict b_data = transpose(B->data, B->height, B->width);
    double* restrict c_data = C->data;

    // A is transposed
    size_t a_width = A->height;
    size_t b_width = B->height;
    size_t c_width = C->width;

    // Use the old shape pre-transpose
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t kk = 0; kk < K; kk += b) {
            size_t kstart = kk, kstop = MIN(kk+b, K);

            for (size_t jj = 0; jj < N; jj += b) {
                size_t jstart = jj, jstop = MIN(jj+b, N);

                for (size_t i = istart; i < istop; i++) {
                    double* restrict c_row = &c_data[i * c_width];
                    double* restrict a_row = &a_data[i * a_width];
#pragma omp simd
                    for (size_t k = kstart; k < kstop; k++) {
                        double a_ik = a_row[k];
                        size_t j;
                        for (j = jstart; j + 3 < jstop; j+=4) {
                            c_row[j+0] += a_ik * b_data[(j+0) * b_width + k];
                            c_row[j+1] += a_ik * b_data[(j+1) * b_width + k];
                            c_row[j+2] += a_ik * b_data[(j+2) * b_width + k];
                            c_row[j+3] += a_ik * b_data[(j+3) * b_width + k];
                        }

                        for (; j < jstop; j++) {
                            c_row[j] += a_ik * b_data[(j) * b_width + k];
                        }
                    }
                }
            }
        }
    }

    free(a_data);
    free(b_data);
}

void matmul_tiled_1x4_transposed_A_restrict_ikj(matrix_t* A, matrix_t* B, matrix_t* C)
{
    double* restrict a_data = transpose(A->data, A->height, A->width);
    double* restrict b_data = B->data;
    double* restrict c_data = C->data;

    // A and B is transposed
    size_t a_width = A->height;
    size_t b_width = B->width;
    size_t c_width = C->width;

    // Use the old shape pre-transpose
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t kk = 0; kk < K; kk += b) {
            size_t kstart = kk, kstop = MIN(kk+b, K);

            for (size_t jj = 0; jj < N; jj += b) {
                size_t jstart = jj, jstop = MIN(jj+b, N);

                for (size_t i = istart; i < istop; i++) {
                    double* restrict c_row = &c_data[i * c_width];
                    double* restrict a_row = &a_data[i * a_width];

#pragma omp simd
                    for (size_t k = kstart; k < kstop; k++) {
                        double a_ik = a_row[k];
                        size_t j;
                        for (j = jstart; j + 3 < jstop; j+=4) {
                            c_row[j+0] += a_ik * b_data[b_width * k + (j+0)];
                            c_row[j+1] += a_ik * b_data[b_width * k + (j+1)];
                            c_row[j+2] += a_ik * b_data[b_width * k + (j+2)];
                            c_row[j+3] += a_ik * b_data[b_width * k + (j+3)];
                        }

                        for (; j < jstop; j++) {
                            c_row[j] += a_ik * b_data[b_width * k + (j)];
                        }
                    }
                }
            }
        }
    }

    free(a_data);
}

void matmul_tiled_transposed_A_restrict_ikj(matrix_t* A, matrix_t* B, matrix_t* C)
{
    double* restrict a_data = transpose(A->data, A->height, A->width);
    double* restrict b_data = B->data;
    double* restrict c_data = C->data;

    // A is transposed
    size_t a_width = A->height;
    size_t b_width = B->width;
    size_t c_width = C->width;

    // Use the old shape pre-transpose for A
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    size_t b = 64;

#pragma omp parallel for
    for (size_t ii = 0; ii < M; ii += b) {
        size_t istart = ii, istop = MIN(ii+b, M);

        for (size_t kk = 0; kk < K; kk += b) {
            size_t kstart = kk, kstop = MIN(kk+b, K);

            for (size_t jj = 0; jj < N; jj += b) {
                size_t jstart = jj, jstop = MIN(jj+b, N);

                for (size_t i = istart; i < istop; i++) {
                    double* restrict c_row = &c_data[i * c_width];
                    double* restrict a_row = &a_data[i * a_width];

                    for (size_t k = kstart; k < kstop; k++) {
                        double a_ik = a_row[k];
#pragma omp simd
                        for (size_t j = jstart; j < jstop; j++) {
                            c_row[j] += a_ik * b_data[b_width * k + j];
                        }
                    }
                }
            }
        }
    }

    free(a_data);
}

void blas(matrix_t* A, matrix_t* B, matrix_t* C)
{
    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    CBLAS_TRANSPOSE TransA = CblasTrans;
    CBLAS_TRANSPOSE TransB = CblasNoTrans;

    cblas_dgemm(CblasRowMajor,
                TransA,
                TransB,
                M,
                N,
                K,
                1.0,
                A->data,
                A->width,
                B->data,
                B->width,
                0.0,
                C->data,
                C->width);

}

bool is_valid(matrix_t* src, matrix_t* ref)
{
    assert(src->height == ref->height && "The source and reference height doesn't match");
    assert(src->width == ref->width && "The source and reference width doesn't match");

    const double abs_tol = 1e-9;
    const double rel_tol = 1e-6;

    for (size_t i = 0; i < src->height; i++) {
        for (size_t j = 0; j < src->width; j++) {
            double src_val = MAT_AT(src, i, j);
            double ref_val = MAT_AT(ref, i, j);

            double abs_diff = fabs(src_val - ref_val);
            double abs_max = fmax(fabs(src_val), fabs(ref_val));

            if (abs_diff > abs_tol && abs_diff > rel_tol * abs_max) {
                return false;
            }
        }
    }

    return true;
}

void restore_if_needed(matrix_t* A, matrix_t* B, RestoreFlags flags)
{
    if (flags & RESTORE_A) transpose_inplace(A);
    if (flags & RESTORE_B) transpose_inplace(B);
}

void run_benchmark(matrix_t* A, matrix_t* B, matrix_t* C, matrix_t* ref, MatmulKernel kernel)
{
    assert(A->height == B->height && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->width && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

    // Validate once
    printf("%s: verify", kernel.name);
    fflush(stdout);
    kernel.fn(A, B, C);
    if (!is_valid(C, ref)) {
        printf("\n");
        ERROR("Result from '%s' implementation doesn't match reference", kernel.name);
    }
    mat_zero(C);
    restore_if_needed(A, B, kernel.restore);

    // Warm up
    printf("\r\033[K%s: warmup", kernel.name);
    fflush(stdout);
    for (size_t i = 0; i < 10; i++) {
        kernel.fn(A, B, C);
        mat_zero(C);
        restore_if_needed(A, B, kernel.restore);
        printf(".");
        fflush(stdout);
    }

    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;
    uint64_t flops = 2ULL * M * N * K;

    double min_time = DBL_MAX;
    uint64_t bytes = 0;
    uint64_t l3_local = 0;
    uint64_t l3_remote = 0;

    // Timed runs
    printf("\r\033[K%s: run", kernel.name);
    fflush(stdout);
    for (size_t i = 0; i < NTIMES; i++) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            cache_counter_start(&thread_counters[tid]);
        }

        double start = omp_get_wtime();
        kernel.fn(A, B, C);
        double elapsed = omp_get_wtime() - start;

        if (elapsed < min_time) {
            min_time = elapsed;
            bytes = 0;
            l3_local = 0;
            l3_remote = 0;
#pragma omp parallel reduction(+:bytes,l3_local,l3_remote)
            {
                int tid = omp_get_thread_num();
                cache_counter_stop(&thread_counters[tid]);

                bytes += cache_counter_get_bytes_loaded(&thread_counters[tid]);
                long long local = 0;
                long long remote = 0;
                cache_counter_get_cache_misses(&thread_counters[tid], &local, &remote);
                l3_local += (uint64_t)local;
                l3_remote += (uint64_t)remote;
            }
        }
        mat_zero(C);
        restore_if_needed(A, B, kernel.restore);
        putchar('.');
        fflush(stdout);
    }

    double bandwidth = bytes / min_time;
    double flops_per_s = (double) flops / min_time;
    double intensity = flops_per_s / bandwidth;
    printf("\r\033[K%s: %.3fs, %.2f MBytes/s, %.2f MFlops/s, %.2f flop/byte, "
           "L3-local(MB): %.2f, L3-remote(MB): %.2f\n",
           kernel.name, min_time, bandwidth/1e6, flops_per_s/1e6, intensity,
           (double)l3_local/1e6, (double)l3_remote/1e6);
}

int main(void)
{
    int num_threads = omp_get_max_threads();
    printf("Using %d threads\n", num_threads);
    // openblas_set_num_threads(num_threads);
    thread_counters = malloc(num_threads * sizeof(cache_counter_t));
    if (!thread_counters) {
        fprintf(stderr, "ERROR: Could not allocate thread_counters\n");
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_counters[tid] = cache_counter_init();
    }

#ifdef NDEBUG
    size_t K_values[] = {90941};
    size_t MN = 256;
#else
    size_t K_values[] = {64};
    size_t MN = 32;
#endif // NDEBUG

    for (size_t b = 0; b < ARRAY_LEN(K_values); b++) {
        size_t M = MN;
        size_t K = K_values[b];
        size_t N = MN;

        matrix_t* A = mat_create(K, M);
        matrix_t* B = mat_create(K, N);
        matrix_t* C = mat_create(M, N);

        matrix_t *ref = mat_create(M, N);

        for (size_t i = 0; i < A->height; ++i) {
            for (size_t j = 0; j < A->width; ++j) {
                MAT_AT(A, i, j) = i * A->width + j;
            }
        }

        for (size_t i = 0; i < B->height; ++i) {
            for (size_t j = 0; j < B->width; ++j) {
                MAT_AT(B, i, j) = (i * B->width + j) + 1000000;
            }
        }

        // Precompute the reference
        printf("Computing reference\n");
        blas(A, B, ref);

        MatmulKernel kernels[] = {
            NEW_KERNEL(matmul_naive, RESTORE_NONE),
            NEW_KERNEL(matmul_naive_restrict, RESTORE_NONE),

            NEW_KERNEL(matmul_unroll, RESTORE_NONE),
            NEW_KERNEL(matmul_tiled, RESTORE_NONE),

            NEW_KERNEL(matmul_tiled_1x4, RESTORE_NONE),
            NEW_KERNEL(matmul_tiled_1x4_restrict, RESTORE_NONE),

            NEW_KERNEL(matmul_tiled_2x2, RESTORE_NONE),
            NEW_KERNEL(matmul_tiled_transposed, RESTORE_NONE),
            NEW_KERNEL(matmul_tiled_2x2_transposed, RESTORE_NONE),

            NEW_KERNEL(matmul_tiled_1x4_transposed, RESTORE_NONE),
            NEW_KERNEL(matmul_tiled_1x4_transposed_restrict, RESTORE_NONE),

            NEW_KERNEL(matmul_tiled_1x4_transposed_A_restrict,  RESTORE_NONE),
            NEW_KERNEL(matmul_tiled_1x4_transposed_restrict_ikj, RESTORE_NONE),
            NEW_KERNEL(matmul_tiled_1x4_transposed_A_restrict_ikj, RESTORE_NONE),
            NEW_KERNEL(matmul_tiled_transposed_A_restrict_ikj, RESTORE_NONE),
            NEW_KERNEL(blas, RESTORE_NONE),
        };

        for (size_t i = 0; i < ARRAY_LEN(kernels); i++) {
            run_benchmark(A, B, C, ref, kernels[i]);
        }

        mat_destroy(A);
        mat_destroy(B);
        mat_destroy(C);
        mat_destroy(ref);
    }

    return 0;
}
