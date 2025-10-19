#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "matrix.h"
#include "perf.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

#define L3_CACHE_MB 128
// #define L3_CACHE_MB 2.3 * 1024


#ifdef NDEBUG
#define ARRAY_SIZE ((size_t)L3_CACHE_MB * 4 * 1024 * 1024 / 8)
const size_t iter = 10;
#else
#define ARRAY_SIZE ((size_t)L3_CACHE_MB * 1 * 1024 * 1024 / 8)
const size_t iter = 1;
#endif // NDEBUG

typedef struct {
    size_t M, N, P;  // A is (M x N), B is (N x P), C is (M x P)
} MatrixSizes;

typedef enum {
    BASELINE,
    BASELINE_SIMD,
    BASELINE_INNER_SIMD,
    COLLAPSE,
    COLLAPSE_SIMD,
    COLLAPSE_INNER_SIMD,
    UNROLL,
    UNROLL_SIMD,
    UNROLL_INNER_SIMD,
    BLOCKED_UNROLL,
    BLOCKED_UNROLL_SIMD,
} Strategy;

typedef struct {
    const char* name;
    bool at;      // transpose A
    bool bt;      // transpose B
    bool ct;      // transpose C for validation
    Strategy strategy;
} BenchConfig;

MatrixSizes get_sizes_for_target_memory(size_t base_M, size_t base_N, size_t base_P)
{
    // Calculate memory for base shape
    size_t base_elements = base_M * base_N + base_N * base_P + base_M * base_P;
    size_t base_bytes = base_elements;

    // Scale uniformly to hit target
    double scale_factor = sqrt((double)ARRAY_SIZE / base_bytes);

    size_t M = (size_t)(base_M * scale_factor);
    size_t N = (size_t)(base_N * scale_factor);
    size_t P = (size_t)(base_P * scale_factor);

    size_t actual_bytes = (M*N + N*P + M*P) * sizeof(double);
    printf("Scaled (%zu,%zu,%zu) -> (%zu,%zu,%zu) = %.1f MB\n",
           base_M, base_N, base_P, M, N, P, actual_bytes / 1024.0 / 1024.0);

    return (MatrixSizes){.M = M, .N = N, .P = P};
}

OpMetrics bench_dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt, Strategy strat)
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

    switch (strat) {

    case BASELINE: {
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
    }break;

    case BASELINE_SIMD: {
#pragma omp parallel for simd
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
    }break;

    case BASELINE_INNER_SIMD: {
#pragma omp parallel for
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < P; j++) {
                double sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < N; k++) {
                    sum += A->data[i*a_row_stride + k*a_col_stride] *
                           B->data[k*b_row_stride + j*b_col_stride];
                }
                MAT_AT(C, i, j) = sum;
            }
        }
    }break;

    case COLLAPSE: {
#pragma omp parallel for collapse(2)
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
    }break;

    case COLLAPSE_SIMD: {
#pragma omp parallel for simd collapse(2)
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
    }break;

    case COLLAPSE_INNER_SIMD: {
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < P; j++) {
                double sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < N; k++) {
                    sum += A->data[i*a_row_stride + k*a_col_stride] *
                           B->data[k*b_row_stride + j*b_col_stride];
                }
                MAT_AT(C, i, j) = sum;
            }
        }
    }break;

    case UNROLL: {
        bytes = ((5ULL * N * (P/4) * M) + (P * M)) * sizeof(double);
#pragma omp parallel for
        for (size_t i = 0; i < M; i++) {
            size_t j;
            for (j = 0; j + 3 < P; j+=4) {
                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (size_t k = 0; k < N; k++) {
                    double a_ik = A->data[i*a_row_stride + k*a_col_stride];
                    sum0 += a_ik * B->data[k*b_row_stride + ((j+0)*b_col_stride)];
                    sum1 += a_ik * B->data[k*b_row_stride + ((j+1)*b_col_stride)];
                    sum2 += a_ik * B->data[k*b_row_stride + ((j+2)*b_col_stride)];
                    sum3 += a_ik * B->data[k*b_row_stride + ((j+3)*b_col_stride)];
                }

                MAT_AT(C, i, j+0) = sum0;
                MAT_AT(C, i, j+1) = sum1;
                MAT_AT(C, i, j+2) = sum2;
                MAT_AT(C, i, j+3) = sum3;
            }

            for (; j < P; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < N; k++) {
                    sum += A->data[i*a_row_stride + k*a_col_stride] *
                           B->data[k*b_row_stride + j*b_col_stride];
                }
                MAT_AT(C, i, j) = sum;
            }
        }
    }break;

    case UNROLL_SIMD: {
        bytes = ((5ULL * N * (P/4) * M) + (P * M)) * sizeof(double);
#pragma omp parallel for simd
        for (size_t i = 0; i < M; i++) {
            size_t j;
            for (j = 0; j + 3 < P; j+=4) {
                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (size_t k = 0; k < N; k++) {
                    double a_ik = A->data[i*a_row_stride + k*a_col_stride];
                    sum0 += a_ik * B->data[k*b_row_stride + ((j+0)*b_col_stride)];
                    sum1 += a_ik * B->data[k*b_row_stride + ((j+1)*b_col_stride)];
                    sum2 += a_ik * B->data[k*b_row_stride + ((j+2)*b_col_stride)];
                    sum3 += a_ik * B->data[k*b_row_stride + ((j+3)*b_col_stride)];
                }

                MAT_AT(C, i, j+0) = sum0;
                MAT_AT(C, i, j+1) = sum1;
                MAT_AT(C, i, j+2) = sum2;
                MAT_AT(C, i, j+3) = sum3;
            }

            for (; j < P; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < N; k++) {
                    sum += A->data[i*a_row_stride + k*a_col_stride] *
                           B->data[k*b_row_stride + j*b_col_stride];
                }
                MAT_AT(C, i, j) = sum;
            }
        }
    }break;

    case UNROLL_INNER_SIMD: {
        bytes = ((5ULL * N * (P/4) * M) + (P * M)) * sizeof(double);
#pragma omp parallel for
        for (size_t i = 0; i < M; i++) {
            size_t j;
            for (j = 0; j + 3 < P; j+=4) {
                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
#pragma omp simd reduction(+:sum0, sum1, sum2, sum3)
                for (size_t k = 0; k < N; k++) {
                    double a_ik = A->data[i*a_row_stride + k*a_col_stride];
                    sum0 += a_ik * B->data[k*b_row_stride + ((j+0)*b_col_stride)];
                    sum1 += a_ik * B->data[k*b_row_stride + ((j+1)*b_col_stride)];
                    sum2 += a_ik * B->data[k*b_row_stride + ((j+2)*b_col_stride)];
                    sum3 += a_ik * B->data[k*b_row_stride + ((j+3)*b_col_stride)];
                }

                MAT_AT(C, i, j+0) = sum0;
                MAT_AT(C, i, j+1) = sum1;
                MAT_AT(C, i, j+2) = sum2;
                MAT_AT(C, i, j+3) = sum3;
            }

            for (; j < P; j++) {
                double sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < N; k++) {
                    sum += A->data[i*a_row_stride + k*a_col_stride] *
                           B->data[k*b_row_stride + j*b_col_stride];
                }
                MAT_AT(C, i, j) = sum;
            }
        }
    }break;

    case BLOCKED_UNROLL: {
        size_t ib = 64;
        size_t jb = 64;
        size_t kb = 256;
#pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < M; ii += ib) {
            for (size_t jj = 0; jj < P; jj += jb) {
                for (size_t kk = 0; kk < N; kk += kb) {
                    size_t i_end = (ii + ib < M) ? ii + ib : M;
                    size_t j_end = (jj + jb < P) ? jj + jb : P;
                    size_t k_end = (kk + kb < N) ? kk + kb : N;

                    size_t j;
                    for (j = jj; j + 1 < j_end; j += 2) {
                        size_t i;
                        for (i = ii; i + 1 < i_end; i += 2) {
                            double sum00 = MAT_AT(C, i+0, j+0);
                            double sum01 = MAT_AT(C, i+0, j+1);
                            double sum10 = MAT_AT(C, i+1, j+0);
                            double sum11 = MAT_AT(C, i+1, j+1);

                            for (size_t k = kk; k < k_end; k++) {
                                double a_i0k = MAT_STRIDED(A, i+0, k, a_row_stride, a_col_stride);
                                double a_i1k = MAT_STRIDED(A, i+1, k, a_row_stride, a_col_stride);
                                double b_kj0 = MAT_STRIDED(B, k, j+0, b_row_stride, b_col_stride);
                                double b_kj1 = MAT_STRIDED(B, k, j+1, b_row_stride, b_col_stride);

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

                        for (; i < i_end; i++) {
                            double sum0 = MAT_AT(C, i, j+0);
                            double sum1 = MAT_AT(C, i, j+1);

                            for (size_t k = kk; k < k_end; k++) {
                                double a_ik = MAT_STRIDED(A, i, k, a_row_stride, a_col_stride);
                                sum0 += a_ik * MAT_STRIDED(B, k, j+0, b_row_stride, b_col_stride);
                                sum1 += a_ik * MAT_STRIDED(B, k, j+1, b_row_stride, b_col_stride);
                            }

                            MAT_AT(C, i, j+0) = sum0;
                            MAT_AT(C, i, j+1) = sum1;
                        }
                    }

                    for (; j < j_end; j++) {
                        size_t i;

                        for (i = ii; i + 1 < i_end; i += 2) {
                            double sum0 = MAT_AT(C, i+0, j);
                            double sum1 = MAT_AT(C, i+1, j);

                            for (size_t k = kk; k < k_end; k++) {
                                double b_kj = MAT_STRIDED(B, k, j, b_row_stride, b_col_stride);
                                sum0 += MAT_STRIDED(A, i+0, k, a_row_stride, a_col_stride) * b_kj;
                                sum1 += MAT_STRIDED(A, i+1, k, a_row_stride, a_col_stride) * b_kj;
                            }
                            MAT_AT(C, i+0, j) = sum0;
                            MAT_AT(C, i+1, j) = sum1;
                        }

                        for (; i < i_end; i++) {
                            double sum = MAT_AT(C, i, j);
                            for (size_t k = kk; k < k_end; k++) {
                                sum += MAT_STRIDED(A, i, k, a_row_stride, a_col_stride)
                                    * MAT_STRIDED(B, k, j, b_row_stride, b_col_stride);
                            }
                            MAT_AT(C, i, j) = sum;
                        }
                    }
                }
            }
        }
    } break;

    case BLOCKED_UNROLL_SIMD: {
        size_t ib = 64;
        size_t jb = 64;
        size_t kb = 256;

#pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < M; ii += ib) {
            for (size_t jj = 0; jj < P; jj += jb) {
                for (size_t kk = 0; kk < N; kk += kb) {
                    size_t i_end = (ii + ib < M) ? ii + ib : M;
                    size_t j_end = (jj + jb < P) ? jj + jb : P;
                    size_t k_end = (kk + kb < N) ? kk + kb : N;

                    size_t j;
                    for (j = jj; j + 1 < j_end; j += 2) {
                        size_t i;
                        for (i = ii; i + 1 < i_end; i += 2) {
                            double sum00 = MAT_AT(C, i+0, j+0);
                            double sum01 = MAT_AT(C, i+0, j+1);
                            double sum10 = MAT_AT(C, i+1, j+0);
                            double sum11 = MAT_AT(C, i+1, j+1);

#pragma omp simd
                            for (size_t k = kk; k < k_end; k++) {
                                double a_i0k = MAT_STRIDED(A, i+0, k, a_row_stride, a_col_stride);
                                double a_i1k = MAT_STRIDED(A, i+1, k, a_row_stride, a_col_stride);
                                double b_kj0 = MAT_STRIDED(B, k, j+0, b_row_stride, b_col_stride);
                                double b_kj1 = MAT_STRIDED(B, k, j+1, b_row_stride, b_col_stride);

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

                        for (; i < i_end; i++) {
                            double sum0 = MAT_AT(C, i, j+0);
                            double sum1 = MAT_AT(C, i, j+1);

#pragma omp simd
                            for (size_t k = kk; k < k_end; k++) {
                                double a_ik = MAT_STRIDED(A, i, k, a_row_stride, a_col_stride);
                                sum0 += a_ik * MAT_STRIDED(B, k, j+0, b_row_stride, b_col_stride);
                                sum1 += a_ik * MAT_STRIDED(B, k, j+1, b_row_stride, b_col_stride);
                            }

                            MAT_AT(C, i, j+0) = sum0;
                            MAT_AT(C, i, j+1) = sum1;
                        }
                    }

                    for (; j < j_end; j++) {
                        size_t i;

                        for (i = ii; i + 1 < i_end; i += 2) {
                            double sum0 = MAT_AT(C, i+0, j);
                            double sum1 = MAT_AT(C, i+1, j);

#pragma omp simd
                            for (size_t k = kk; k < k_end; k++) {
                                double b_kj = MAT_STRIDED(B, k, j, b_row_stride, b_col_stride);
                                sum0 += MAT_STRIDED(A, i+0, k, a_row_stride, a_col_stride) * b_kj;
                                sum1 += MAT_STRIDED(A, i+1, k, a_row_stride, a_col_stride) * b_kj;
                            }
                            MAT_AT(C, i+0, j) = sum0;
                            MAT_AT(C, i+1, j) = sum1;
                        }

                        for (; i < i_end; i++) {
                            double sum = MAT_AT(C, i, j);
#pragma omp simd
                            for (size_t k = kk; k < k_end; k++) {  // FIXED: was "k = 0; k < N"
                                sum += MAT_STRIDED(A, i, k, a_row_stride, a_col_stride)
                                    * MAT_STRIDED(B, k, j, b_row_stride, b_col_stride);
                            }
                            MAT_AT(C, i, j) = sum;
                        }
                    }
                }
            }
        }
    } break;

    default:
        ERROR("Got a unknown strat type, exiting...");
    }


    return (OpMetrics){.flops=flops, .bytes=bytes};
}

bool is_valid(matrix_t* src, matrix_t* ref, bool srcT, bool refT)
{
    size_t eff_src_rows = srcT ? src->width : src->height;
    size_t eff_src_cols = srcT ? src->height : src->width;
    size_t eff_ref_rows = refT ? ref->width : ref->height;
    size_t eff_ref_cols = refT ? ref->height : ref->width;

    if (eff_src_rows != eff_ref_rows) {
        ERROR("Effective row count mismatch:\n"
              "  src: %zux%zu%s -> %zu rows\n"
              "  ref: %zux%zu%s -> %zu rows",
              src->height, src->width, srcT ? " (transposed)" : "",
              eff_src_rows,
              ref->height, ref->width, refT ? " (transposed)" : "",
              eff_ref_rows);
    }

    if (eff_src_cols != eff_ref_cols) {
        ERROR("Effective column count mismatch:\n"
              "  src: %zux%zu%s -> %zu cols\n"
              "  ref: %zux%zu%s -> %zu cols",
              src->height, src->width, srcT ? " (transposed)" : "",
              eff_src_cols,
              ref->height, ref->width, refT ? " (transposed)" : "",
              eff_ref_cols);
    }

    const double abs_tol = 1e-9;
    const double rel_tol = 1e-6;

    for (size_t i = 0; i < eff_src_rows; i++) {
        for (size_t j = 0; j < eff_src_cols; j++) {
            double *src_val;
            double *ref_val;
            if (srcT) src_val = &MAT_AT(src, j, i);
            else src_val = &MAT_AT(src, i, j);

            if (refT) ref_val = &MAT_AT(ref, j, i);
            else ref_val = &MAT_AT(ref, i, j);

            double abs_diff = fabs(*src_val - *ref_val);
            double abs_max = fmax(fabs(*src_val), fabs(*ref_val));

            if (abs_diff > abs_tol && abs_diff > rel_tol * abs_max) {
                return false;
            }
        }
    }

    return true;
}

void run_benchmark(matrix_t* A, matrix_t* B, matrix_t* C, matrix_t* ref,
                   const BenchConfig* config)
{
    // Validate once
    bench_dot_ex(A, B, C, config->at, config->bt, config->strategy);

    if (!is_valid(C, ref, config->ct, false)) {
        ERROR("Result for '%s' doesn't match reference", config->name);
    }

    // Warm up
    for (size_t i = 0; i < iter; i++) {
        bench_dot_ex(A, B, C, config->at, config->bt, config->strategy);
    }

    // Timed runs
    for (size_t i = 0; i < iter; i++) {
        PERF_CALL(config->name,
                  bench_dot_ex(A, B, C, config->at, config->bt, config->strategy));
    }

    mat_zero(C);
    printf("%s finished\n", config->name);
}

int main(void)
{
    // sage_bwd_grad_Wroot: A=(90941x256), B=(90941x40), C=(256x40)
    MatrixSizes sz = get_sizes_for_target_memory(90941, 256, 40);

    matrix_t *A = mat_create(sz.M, sz.N);
    matrix_t *B = mat_create(sz.M, sz.P);
    matrix_t *C = mat_create(sz.N, sz.P);

    matrix_t *ref = mat_create(sz.N, sz.P);

    // const size_t dim = (size_t)(sqrt(ARRAY_SIZE) + 0.5);
    // printf("old dim: %zu = %.1fMB\n", dim, ((((dim*dim + dim*dim + dim*dim) * sizeof(double)) / 1024.0) / 1024.0));

    // matrix_t *A = mat_create(dim, dim);
    // matrix_t *B = mat_create(dim, dim);
    // matrix_t *C = mat_create(dim, dim);

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
    dot_ex(A, B, ref, true, false);

    // Normal variants
    // run_all_variants(A, B, C, ref, "", true, false, false);
    BenchConfig configs[] = {
        // {"baseline",          .at=true,  .bt=false, .ct=false, .strategy=BASELINE},
        {"baseline-simd", .at=true,  .bt=false, .ct=false, .strategy=BASELINE_SIMD},
        // {"baseline-inner-simd", .at=true,  .bt=false, .ct=false, .strategy=BASELINE_INNER_SIMD},
        // {"baseline-collapse", .at=true,  .bt=false, .ct=false, .strategy=COLLAPSE},
        // {"baseline-collapse-simd", .at=true,  .bt=false, .ct=false, .strategy=COLLAPSE_SIMD},
        // {"baseline-collapse-inner-simd", .at=true,  .bt=false, .ct=false, .strategy=COLLAPSE_INNER_SIMD},
        {"baseline-unroll",   .at=true,  .bt=false, .ct=false, .strategy=UNROLL},
        {"baseline-unroll-simd",   .at=true,  .bt=false, .ct=false, .strategy=UNROLL_SIMD},
        // {"baseline-unroll-inner-simd",   .at=true,  .bt=false, .ct=false, .strategy=UNROLL_INNER_SIMD},
        {"baseline-blocked-unroll",   .at=true,  .bt=false, .ct=false, .strategy=BLOCKED_UNROLL},
        {"baseline-blcoked-unroll-simd",   .at=true,  .bt=false, .ct=false, .strategy=BLOCKED_UNROLL_SIMD},
    };

    for (size_t i = 0; i < sizeof(configs)/sizeof(configs[0]); i++) {
        run_benchmark(A, B, C, ref, &configs[i]);
    }

    // Affine variants
    matrix_t* At = mat_create(sz.N, sz.M);
    matrix_t* Bt = mat_create(sz.P, sz.M);
    matrix_t* Ct = mat_create(sz.P, sz.N);

    mat_transpose_to(A, At);
    mat_transpose_to(B, Bt);

    // run_all_variants(Bt, At, Ct, ref, "affine-", false, true, true);
    BenchConfig affine_configs[] = {
        {"affine-baseline",  .at=false, .bt=true, .ct=true, .strategy=BASELINE},
        {"affine-baseline-simd",  .at=false, .bt=true, .ct=true, .strategy=BASELINE_SIMD},
        // {"affine-baseline-inner-simd",  .at=false, .bt=true, .ct=true, .strategy=BASELINE_INNER_SIMD},
        // {"affine-collapse",  .at=false, .bt=true, .ct=true, .strategy=COLLAPSE},
        // {"affine-collapse-simd",  .at=false, .bt=true, .ct=true, .strategy=COLLAPSE_SIMD},
        // {"affine-collapse-inner-simd",  .at=false, .bt=true, .ct=true, .strategy=COLLAPSE_INNER_SIMD},
        {"affine-unroll",    .at=false, .bt=true, .ct=true, .strategy=UNROLL},
        {"affine-unroll-simd",    .at=false, .bt=true, .ct=true, .strategy=UNROLL_SIMD},
        // {"affine-unroll-inner-simd",    .at=false, .bt=true, .ct=true, .strategy=UNROLL_INNER_SIMD},
        {"affine-blocked-unroll",   .at=false,  .bt=true, .ct=true, .strategy=BLOCKED_UNROLL},
        {"affine-blocked-unroll-simd",   .at=false,  .bt=true, .ct=true, .strategy=BLOCKED_UNROLL_SIMD},
    };

    for (size_t i = 0; i < sizeof(affine_configs)/sizeof(affine_configs[0]); i++) {
        run_benchmark(Bt, At, Ct, ref, &affine_configs[i]);
    }

    PERF_PRINT();

    return 0;
}
