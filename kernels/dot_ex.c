#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "matrix.h"
#include "perf.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

// #define ARRAY_SIZE 524288000 // genoaxq
// #define ARRAY_SIZE 33554432 // rome16q
// #define ARRAY_SIZE 117964800 // xeonmaxq

#ifndef ARRAY_SIZE
#    define ARRAY_SIZE 10000000
#endif // ARRAY_SIZE

#ifndef NTIMES
#    define NTIMES 10
#endif // NTIMES

#define SIMD_ENABLED

#ifdef SIMD_ENABLED
#    define SIMD simd
#    define SIMD_SFX "_simd"
#else
#    define SIMD
#    define SIMD_SFX ""
#endif

typedef struct {
    size_t M, N, P;  // A is (M x N), B is (N x P), C is (M x P)
} MatrixSizes;

typedef enum {
    BASELINE,
    BASELINE_SIMD,
    COLLAPSE,
    COLLAPSE_SIMD,
    UNROLL,
    UNROLL_SIMD,
    BLOCKED,
    BLOCKED_UNROLL,
    BLOCKED_UNROLL_SIMD,
} Strategy;

typedef struct {
    const char* name;
    bool at;      // transpose A
    bool bt;      // transpose B
    bool ct;      // transpose C for validation
    bool pre_trans;   // Pre transposing
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

void transpose(matrix_t* src, matrix_t* dst)
{
    PERF_FUNC_START();

    size_t height = src->height;
    size_t width = src->width;
    size_t b = 64;


#pragma omp parallel for collapse(2)
    for (size_t jj = 0; jj < width; jj += b) {
        for (size_t ii = 0; ii < height; ii += b) {
            size_t jstop = (jj + b < width) ? jj + b : width;
            size_t istop = (ii + b < height) ? ii + b : height;

            size_t j;
            for (j = jj; j + 3 < jstop; j += 4) {
#pragma omp SIMD
                for (size_t i = ii; i < istop; i++) {
                    MAT_AT(dst, j+0, i) = MAT_AT(src, i, j+0);
                    MAT_AT(dst, j+1, i) = MAT_AT(src, i, j+1);
                    MAT_AT(dst, j+2, i) = MAT_AT(src, i, j+2);
                    MAT_AT(dst, j+3, i) = MAT_AT(src, i, j+3);
                }
            }

            for (; j < jstop; j++) {
#pragma omp SIMD
                for (size_t i = ii; i < istop; i++) {
                    MAT_AT(dst, j, i) = MAT_AT(src, i, j);
                }
            }
        }
    }

    PERF_FUNC_END();
}

OpMetrics bench_dot_ex(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt, bool pre_trans, Strategy strat)
{
    matrix_t *At, *Bt;//*Ct, *Corig;

    if (pre_trans) {
        At = mat_create(A->width, A->height);
        Bt = mat_create(B->width, B->height);

        transpose(A, At);
        transpose(B, Bt);

        A = At;
        B = Bt;
    }

    // Calculate effective dimensions after potential transposition
    size_t eff_A_rows = at ? A->width : A->height;
    size_t eff_A_cols = at ? A->height : A->width;
    size_t eff_B_rows = bt ? B->width : B->height;
    size_t eff_B_cols = bt ? B->height : B->width;

    assert(eff_A_cols == eff_B_rows && "Inner dimensions must match");
    assert(C->height == eff_A_rows && "Output height must match");
    assert(C->width == eff_B_cols && "Output width must match");

    size_t M = eff_A_rows;
    size_t N = eff_A_cols;
    size_t P = eff_B_cols;

    uint64_t flops = 2ULL * M * N * P;
    uint64_t bytes = ((2ULL * N * P * M) + (P * M)) * sizeof(double);

    // Precompute strides for each matrix
    size_t a_row_stride = at ? 1 : A->width;
    size_t a_col_stride = at ? A->width : 1;
    size_t b_row_stride = bt ? 1 : B->width;
    size_t b_col_stride = bt ? B->width : 1;xo

    // Used for unroll variants
    size_t m = 4;

    // Used for block variants
    size_t ib = 64;
    size_t jb = 64;
    size_t kb = 64;
    size_t m_block = 2;

    switch (strat) {

    case BASELINE: {
#pragma omp parallel for SIMD
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

    case COLLAPSE: {
#pragma omp parallel for SIMD collapse(2)
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

    case UNROLL: {
        // bytes = ((5ULL * N * (P/4) * M) + (P * M)) * sizeof(double);
#pragma omp parallel for SIMD
        for (size_t i = 0; i < M; i++) {
            size_t j;
            for (j = 0; j + 3 < P; j+=m) {
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
    } break;

    case BLOCKED: {
#pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < M; ii += ib) {
            for (size_t jj = 0; jj < P; jj += jb) {
                for (size_t kk = 0; kk < N; kk += kb) {
                    size_t i_end = (ii + ib < M) ? ii + ib : M;
                    size_t j_end = (jj + jb < P) ? jj + jb : P;
                    size_t k_end = (kk + kb < N) ? kk + kb : N;

                    for (size_t j = jj; j < j_end; j++) {
                        size_t i;
                        for (i = ii; i + 1 < i_end; i += m_block) {
                            double sum0 = MAT_AT(C, i+0, j);
                            double sum1 = MAT_AT(C, i+1, j);

#pragma omp SIMD
                            for (size_t k = kk; k < k_end; k++) {
                                double b_kj = MAT_STRIDED(B, k, j, b_row_stride, b_col_stride);
                                sum0 += MAT_STRIDED(A, i+0, k, a_row_stride, a_col_stride) * b_kj;
                                sum1 += MAT_STRIDED(A, i+1, k, a_row_stride, a_col_stride) * b_kj;
                            }
                            MAT_AT(C, i+0, j) += sum0;
                            MAT_AT(C, i+1, j) += sum1;
                        }

                        // Handle remaining row
                        for (; i < i_end; i++) {
                            double sum = MAT_AT(C, i, j);
#pragma omp SIMD
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

    case BLOCKED_UNROLL: {
#pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < M; ii += ib) {
            for (size_t jj = 0; jj < P; jj += jb) {
                for (size_t kk = 0; kk < N; kk += kb) {
                    size_t i_end = (ii + ib < M) ? ii + ib : M;
                    size_t j_end = (jj + jb < P) ? jj + jb : P;
                    size_t k_end = (kk + kb < N) ? kk + kb : N;

                    size_t j;
                    for (j = jj; j + 1 < j_end; j += m_block) {
                        size_t i;
                        for (i = ii; i + 1 < i_end; i += m_block) {
                            double sum00 = MAT_AT(C, i+0, j+0);
                            double sum01 = MAT_AT(C, i+0, j+1);
                             double sum10 = MAT_AT(C, i+1, j+0);
                            double sum11 = MAT_AT(C, i+1, j+1);

#pragma omp SIMD
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

#pragma omp SIMD
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

                        for (i = ii; i + 1 < i_end; i += m_block) {
                            double sum0 = MAT_AT(C, i+0, j);
                            double sum1 = MAT_AT(C, i+1, j);

#pragma omp SIMD
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
#pragma omp SIMD
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

    default:
        ERROR("Got a unknown strat type, exiting...");
    }

    if (pre_trans) {
        mat_destroy(At);
        mat_destroy(Bt);
        // transpose(C, Corig);
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

void run_benchmark(matrix_t* A, matrix_t* B, matrix_t* C, matrix_t* ref, const BenchConfig* config)
{
    // Validate once
    bench_dot_ex(A, B, C, config->at, config->bt, config->pre_trans, config->strategy);

    if (!is_valid(C, ref, config->ct, false)) {
        ERROR("Result for '%s' doesn't match reference", config->name);
    }

    // Warm up
    for (size_t i = 0; i < NTIMES; i++) {
        bench_dot_ex(A, B, C, config->at, config->bt, config->pre_trans, config->strategy);
    }

    // Timed runs
    for (size_t i = 0; i < NTIMES; i++) {
        PERF_CALL(config->name,
                  bench_dot_ex(A, B, C, config->at, config->bt, config->pre_trans, config->strategy));
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
    BenchConfig baseline_configs[] = {
        {"baseline"SIMD_SFX,                .at = true, .bt=false, .ct=false, .pre_trans = false, .strategy=BASELINE},
        {"baseline-collapse"SIMD_SFX,       .at = true, .bt=false, .ct=false, .pre_trans = false, .strategy=COLLAPSE},
        {"baseline-unroll"SIMD_SFX,         .at = true, .bt=false, .ct=false, .pre_trans = false, .strategy=UNROLL},
        {"baseline-blocked"SIMD_SFX,        .at = true, .bt=false, .ct=false, .pre_trans = false, .strategy=BLOCKED},
        {"baseline-blocked-unroll"SIMD_SFX, .at = true, .bt=false, .ct=false, .pre_trans = false, .strategy=BLOCKED_UNROLL},
    };

    for (size_t i = 0; i < sizeof(baseline_configs)/sizeof(baseline_configs[0]); i++) {
        run_benchmark(A, B, C, ref, &baseline_configs[i]);
    }

    // Affine variants
    matrix_t* At = mat_create(sz.N, sz.M);
    matrix_t* Bt = mat_create(sz.P, sz.M);
    matrix_t* Ct = mat_create(sz.P, sz.N);

    mat_transpose_to(A, At);
    mat_transpose_to(B, Bt);

    BenchConfig affine_configs[] = {
        {"affine-baseline"SIMD_SFX,       .at = false, .bt=true, .ct=true, .pre_trans = false, .strategy=BASELINE},
        {"affine-collapse"SIMD_SFX,       .at = false, .bt=true, .ct=true, .pre_trans = false, .strategy=COLLAPSE},
        {"affine-unroll"SIMD_SFX,         .at = false, .bt=true, .ct=true, .pre_trans = false, .strategy=UNROLL},
        {"affine-blocked"SIMD_SFX,        .at = false, .bt=true, .ct=true, .pre_trans = false, .strategy=BLOCKED},
        {"affine-blocked-unroll"SIMD_SFX, .at = false, .bt=true, .ct=true, .pre_trans = false, .strategy=BLOCKED_UNROLL},
    };

    for (size_t i = 0; i < sizeof(affine_configs)/sizeof(affine_configs[0]); i++) {
        run_benchmark(Bt, At, Ct, ref, &affine_configs[i]);
    }

    BenchConfig pre_trans_configs[] = {
        {"pre-transpose"SIMD_SFX,                .at = false, .bt=true, .ct=false, .pre_trans = true, .strategy=BASELINE},
        {"pre-transpose-collapse"SIMD_SFX,       .at = false, .bt=true, .ct=false, .pre_trans = true, .strategy=COLLAPSE},
        {"pre-transpose-unroll"SIMD_SFX,         .at = false, .bt=true, .ct=false, .pre_trans = true, .strategy=UNROLL},
        {"pre-transpose-blocked"SIMD_SFX,        .at = false, .bt=true, .ct=false, .pre_trans = true, .strategy=BLOCKED},
        {"pre-transpose-blocked-unroll"SIMD_SFX, .at = false, .bt=true, .ct=false, .pre_trans = true, .strategy=BLOCKED_UNROLL},
    };

    for (size_t i = 0; i < sizeof(pre_trans_configs)/sizeof(pre_trans_configs[0]); i++) {
        run_benchmark(A, B, C, ref, &pre_trans_configs[i]);
    }

    PERF_PRINT();

    mat_destroy(A);
    mat_destroy(B);
    mat_destroy(C);
    mat_destroy(At);
    mat_destroy(Bt);
    mat_destroy(Ct);

    return 0;
}
