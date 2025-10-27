#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "matrix.h"
#include "perf.h"
#include "cache_counter.h"

#define NOB_IMPLEMENTATION
#include "nob.h"


// #define ARRAY_SIZE 524288000 // genoaxq
// #define ARRAY_SIZE 33554432 // rome16q
// #define ARRAY_SIZE 117964800 // xeonmaxq

// #define ARRAY_SIZE 100000
#ifndef ARRAY_SIZE
#    define ARRAY_SIZE 10000000
#endif // ARRAY_SIZE

#ifndef NTIMES
#    define NTIMES 10
#endif // NTIMES

#ifdef SIMD_ENABLED
#    define SIMD simd
#    define SIMD_SFX "_simd"
#else
#    define SIMD
#    define SIMD_SFX ""
#endif

#define BENCH_NAME(name, height, width) nob_temp_sprintf("(%zux%zu)%s", (height), (width), (name))

#define BASELINE_CONFIG(name, strat) \
    {BENCH_NAME((name), height, width), .at = true, .bt = false, .ct = false, .pre_trans = false, .strategy = (strat)}

#define AFFINE_CONFIG(name, strat) \
    {BENCH_NAME((name), height, width), .at = false, .bt = true, .ct = false, .pre_trans = true, .strategy = (strat)}


FileHandler csv_out = {0};
cache_counter_t* thread_counters = NULL;

typedef struct {
    size_t M, N, P;  // A is (M x N), B is (N x P), C is (M x P)
} MatrixSizes;

typedef enum {
    BASELINE,
    COLLAPSE,
    UNROLL,
    BLOCKED,
    BLOCKED_UNROLL,
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

typedef struct {
    size_t M, N, P;
    size_t ar, ac;
    size_t br, bc;
} PrologueCtx;

void prologue(matrix_t* A, matrix_t* B, matrix_t* C, bool at, bool bt, PrologueCtx* ctx)
{
    // Calculate effective dimensions after potential transposition
    size_t eff_A_rows = at ? A->width : A->height;
    size_t eff_A_cols = at ? A->height : A->width;
    size_t eff_B_rows = bt ? B->width : B->height;
    size_t eff_B_cols = bt ? B->height : B->width;

    assert(eff_A_cols == eff_B_rows && "Inner dimensions must match");
    assert(C->height == eff_A_rows && "Output height must match");
    assert(C->width == eff_B_cols && "Output width must match");

    ctx->M = eff_A_rows;
    ctx->N = eff_A_cols;
    ctx->P = eff_B_cols;

    // Precompute strides for each matrix
    ctx->ar = at ? 1 : A->width;
    ctx->ac = at ? A->width : 1;
    ctx->br = bt ? 1 : B->width;
    ctx->bc = bt ? B->width : 1;
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

void baseline(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt, bool pre_trans)
{
    matrix_t *At, *Bt;

    if (pre_trans) {
        At = mat_create(A->width, A->height);
        Bt = mat_create(B->width, B->height);

        transpose(A, At);
        transpose(B, Bt);

        A = At;
        B = Bt;
    }

    PrologueCtx ctx = {0};
    prologue(A, B, C, at, bt, &ctx);

#pragma omp parallel for SIMD
    for (size_t i = 0; i < ctx.M; i++) {
        for (size_t j = 0; j < ctx.P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < ctx.N; k++) {
                sum += MAT_STRIDED(A, i, k, ctx.ar, ctx.ac) * MAT_STRIDED(B, k, j, ctx.br, ctx.bc);
            }
            MAT_AT(C, i, j) = sum;
        }
    }

    if (pre_trans) {
        mat_destroy(At);
        mat_destroy(Bt);
    }
}

void collapse(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt, bool pre_trans)
{
    matrix_t *At, *Bt;

    if (pre_trans) {
        At = mat_create(A->width, A->height);
        Bt = mat_create(B->width, B->height);

        transpose(A, At);
        transpose(B, Bt);

        A = At;
        B = Bt;
    }

    PrologueCtx ctx = {0};
    prologue(A, B, C, at, bt, &ctx);

#pragma omp parallel for SIMD collapse(2)
    for (size_t i = 0; i < ctx.M; i++) {
        for (size_t j = 0; j < ctx.P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < ctx.N; k++) {
                sum += MAT_STRIDED(A, i, k, ctx.ar, ctx.ac) * MAT_STRIDED(B, k, j, ctx.br, ctx.bc);
            }
            MAT_AT(C, i, j) = sum;
        }
    }

    if (pre_trans) {
        mat_destroy(At);
        mat_destroy(Bt);
    }
}

void unroll(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt, bool pre_trans)
{
    matrix_t *At, *Bt;

    if (pre_trans) {
        At = mat_create(A->width, A->height);
        Bt = mat_create(B->width, B->height);

        transpose(A, At);
        transpose(B, Bt);

        A = At;
        B = Bt;
    }

    PrologueCtx ctx = {0};
    prologue(A, B, C, at, bt, &ctx);


#pragma omp parallel for SIMD
    for (size_t i = 0; i < ctx.M; i++) {
        size_t j;
        for (j = 0; j + 3 < ctx.P; j+=4) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            for (size_t k = 0; k < ctx.N; k++) {
                double a_ik = MAT_STRIDED(A, i, k, ctx.ar, ctx.ac);
                sum0 += a_ik * MAT_STRIDED(B, k, (j+0), ctx.br, ctx.bc);
                sum1 += a_ik * MAT_STRIDED(B, k, (j+1), ctx.br, ctx.bc);
                sum2 += a_ik * MAT_STRIDED(B, k, (j+2), ctx.br, ctx.bc);
                sum3 += a_ik * MAT_STRIDED(B, k, (j+3), ctx.br, ctx.bc);
            }

            MAT_AT(C, i, j+0) = sum0;
            MAT_AT(C, i, j+1) = sum1;
            MAT_AT(C, i, j+2) = sum2;
            MAT_AT(C, i, j+3) = sum3;
        }

        for (; j < ctx.P; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < ctx.N; k++) {
                sum += MAT_STRIDED(A, i, k, ctx.ar, ctx.ac) * MAT_STRIDED(B, k, j, ctx.br, ctx.bc);
            }
            MAT_AT(C, i, j) = sum;
        }
    }

    if (pre_trans) {
        mat_destroy(At);
        mat_destroy(Bt);
    }
}

void blocked(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt, bool pre_trans)
{
    matrix_t *At, *Bt;

    if (pre_trans) {
        At = mat_create(A->width, A->height);
        Bt = mat_create(B->width, B->height);

        transpose(A, At);
        transpose(B, Bt);

        A = At;
        B = Bt;
    }

    PrologueCtx ctx = {0};
    prologue(A, B, C, at, bt, &ctx);

    size_t ib = 64;
    size_t jb = 64;
    size_t kb = 64;

#pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < ctx.M; ii += ib) {
        for (size_t jj = 0; jj < ctx.P; jj += jb) {
            for (size_t kk = 0; kk < ctx.N; kk += kb) {
                size_t i_end = (ii + ib < ctx.M) ? ii + ib : ctx.M;
                size_t j_end = (jj + jb < ctx.P) ? jj + jb : ctx.P;
                size_t k_end = (kk + kb < ctx.N) ? kk + kb : ctx.N;

                for (size_t j = jj; j < j_end; j++) {
                    size_t i;
                    for (i = ii; i + 1 < i_end; i += 2) {
                        double sum0 = 0.0;
                        double sum1 = 0.0;

#pragma omp SIMD
                        for (size_t k = kk; k < k_end; k++) {
                            double b_kj = MAT_STRIDED(B, k, j, ctx.br, ctx.bc);
                            sum0 += MAT_STRIDED(A, i+0, k, ctx.ar, ctx.ac) * b_kj;
                            sum1 += MAT_STRIDED(A, i+1, k, ctx.ar, ctx.ac) * b_kj;
                        }

                        MAT_AT(C, i+0, j) += sum0;
                        MAT_AT(C, i+1, j) += sum1;
                    }

                    for (; i < i_end; i++) {
                        double sum = 0.0;
#pragma omp SIMD
                        for (size_t k = kk; k < k_end; k++) {
                            sum += MAT_STRIDED(A, i, k, ctx.ar, ctx.ac) * MAT_STRIDED(B, k, j, ctx.br, ctx.bc);
                        }
                        MAT_AT(C, i, j) += sum;
                    }
                }
            }
        }
    }

    if (pre_trans) {
        mat_destroy(At);
        mat_destroy(Bt);
    }
}

void blocked_unroll(matrix_t *A, matrix_t *B, matrix_t *C, bool at, bool bt, bool pre_trans)
{
    matrix_t *At, *Bt;

    if (pre_trans) {
        At = mat_create(A->width, A->height);
        Bt = mat_create(B->width, B->height);

        transpose(A, At);
        transpose(B, Bt);

        A = At;
        B = Bt;
    }

    PrologueCtx ctx = {0};
    prologue(A, B, C, at, bt, &ctx);

    size_t ib = 64;
    size_t jb = 64;
    size_t kb = 64;

#pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < ctx.M; ii += ib) {
        for (size_t jj = 0; jj < ctx.P; jj += jb) {
            for (size_t kk = 0; kk < ctx.N; kk += kb) {
                size_t i_end = (ii + ib < ctx.M) ? ii + ib : ctx.M;
                size_t j_end = (jj + jb < ctx.P) ? jj + jb : ctx.P;
                size_t k_end = (kk + kb < ctx.N) ? kk + kb : ctx.N;

                size_t j;
                for (j = jj; j + 1 < j_end; j += 2) {
                    size_t i;
                    for (i = ii; i + 1 < i_end; i += 2) {
                        double sum00 = 0.0;
                        double sum01 = 0.0;
                        double sum10 = 0.0;
                        double sum11 = 0.0;

#pragma omp SIMD
                        for (size_t k = kk; k < k_end; k++) {
                            double a_i0k = MAT_STRIDED(A, i+0, k, ctx.ar, ctx.ac);
                            double a_i1k = MAT_STRIDED(A, i+1, k, ctx.ar, ctx.ac);
                            double b_kj0 = MAT_STRIDED(B, k, j+0, ctx.br, ctx.bc);
                            double b_kj1 = MAT_STRIDED(B, k, j+1, ctx.br, ctx.bc);

                            sum00 += a_i0k * b_kj0;
                            sum01 += a_i0k * b_kj1;
                            sum10 += a_i1k * b_kj0;
                            sum11 += a_i1k * b_kj1;
                        }

                        MAT_AT(C, i+0, j+0) += sum00;
                        MAT_AT(C, i+0, j+1) += sum01;
                        MAT_AT(C, i+1, j+0) += sum10;
                        MAT_AT(C, i+1, j+1) += sum11;
                    }

                    for (; i < i_end; i++) {
                        double sum0 = 0.0;
                        double sum1 = 0.0;

#pragma omp SIMD
                        for (size_t k = kk; k < k_end; k++) {
                            double a_ik = MAT_STRIDED(A, i, k, ctx.ar, ctx.ac);
                            sum0 += a_ik * MAT_STRIDED(B, k, j+0, ctx.br, ctx.bc);
                            sum1 += a_ik * MAT_STRIDED(B, k, j+1, ctx.br, ctx.bc);
                        }

                        MAT_AT(C, i, j+0) += sum0;
                        MAT_AT(C, i, j+1) += sum1;
                    }
                }

                for (; j < j_end; j++) {
                    size_t i;

                    for (i = ii; i + 1 < i_end; i += 2) {
                        double sum0 = 0.0;
                        double sum1 = 0.0;

#pragma omp SIMD
                        for (size_t k = kk; k < k_end; k++) {
                            double b_kj = MAT_STRIDED(B, k, j, ctx.br, ctx.bc);
                            sum0 += MAT_STRIDED(A, i+0, k, ctx.ar, ctx.ac) * b_kj;
                            sum1 += MAT_STRIDED(A, i+1, k, ctx.ar, ctx.ac) * b_kj;
                        }
                        MAT_AT(C, i+0, j) += sum0;
                        MAT_AT(C, i+1, j) += sum1;
                    }

                    for (; i < i_end; i++) {
                        double sum = 0.0;
#pragma omp SIMD
                        for (size_t k = kk; k < k_end; k++) {
                            sum += MAT_STRIDED(A, i, k, ctx.ar, ctx.ac) * MAT_STRIDED(B, k, j, ctx.br, ctx.bc);
                        }
                        MAT_AT(C, i, j) += sum;
                    }
                }
            }
        }
    }

    if (pre_trans) {
        mat_destroy(At);
        mat_destroy(Bt);
    }
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

void run_benchmark(matrix_t* A, matrix_t* B, matrix_t* C, matrix_t* ref, const BenchConfig* conf)
{
    // Validate once
    switch (conf->strategy) {
    case BASELINE:
        baseline(A, B, C, conf->at, conf->bt, conf->pre_trans);
        break;
    case COLLAPSE:
        collapse(A, B, C, conf->at, conf->bt, conf->pre_trans);
        break;
    case UNROLL:
        unroll(A, B, C, conf->at, conf->bt, conf->pre_trans);
        break;
    case BLOCKED:
        blocked(A, B, C, conf->at, conf->bt, conf->pre_trans);
        break;
    case BLOCKED_UNROLL:
        blocked_unroll(A, B, C, conf->at, conf->bt, conf->pre_trans);
        break;
    default:
        ERROR("Got a unknown strategy type, exiting...");
    }

    if (!is_valid(C, ref, conf->ct, false)) {
        ERROR("Result from '%s' implementation doesn't match reference", conf->name);
    }

    mat_zero(C);

    // Warm up
    for (size_t i = 0; i < NTIMES; i++) {
        switch (conf->strategy) {
        case BASELINE:
            baseline(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case COLLAPSE:
            collapse(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case UNROLL:
            unroll(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case BLOCKED:
            blocked(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case BLOCKED_UNROLL:
            blocked_unroll(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        default:
            ERROR("Got a unknown strategy type, exiting...");
        }

        mat_zero(C);
    }

    uint64_t bytes = 0;
    uint64_t cache_misses_local = 0;
    uint64_t cache_misses_remote = 0;

    // Timed runs
    printf("%s", conf->name);
    double total_time = 0.0;
    for (size_t i = 0; i < NTIMES; i++) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            cache_counter_start(&thread_counters[tid]);
        }
        double start = omp_get_wtime();
        switch (conf->strategy) {
        case BASELINE:
            baseline(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case COLLAPSE:
            collapse(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case UNROLL:
            unroll(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case BLOCKED:
            blocked(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        case BLOCKED_UNROLL:
            blocked_unroll(A, B, C, conf->at, conf->bt, conf->pre_trans);
            break;
        default:
            ERROR("Got a unknown strategy type, exiting...");
        }
        total_time += omp_get_wtime() - start;

#pragma omp parallel reduction(+:bytes,cache_misses_local,cache_misses_remote)
        {
            int tid = omp_get_thread_num();
            cache_counter_stop(&thread_counters[tid]);

            bytes += cache_counter_get_bytes_loaded(&thread_counters[tid]);
            long long local = 0;
            long long remote = 0;
            cache_counter_get_cache_misses(&thread_counters[tid], &local, &remote);
            cache_misses_local += (uint64_t)local;
            cache_misses_remote += (uint64_t)remote;
        }

        putchar('.');
        fflush(stdout);
        mat_zero(C);
    }

    // Compute FLOPs
    if (conf->pre_trans) {
        matrix_t* At = mat_create(A->width, A->height);
        matrix_t* Bt = mat_create(B->width, B->height);

        transpose(A, At);
        transpose(B, Bt);

        A = At;
        B = Bt;
    }

    PrologueCtx ctx = {0};
    prologue(A, B, C, conf->at, conf->bt, &ctx);

    double avg_time = total_time / NTIMES;
    double avg_bytes = (double) bytes / NTIMES;
    double avg_cache_miss_local = (double)cache_misses_local / NTIMES;
    double avg_cache_miss_remote = (double)cache_misses_remote / NTIMES;

    double gb_per_s = avg_bytes / avg_time / 1e9;
    uint64_t flops = 2ULL * ctx.M * ctx.N * ctx.P;
    double gflops_per_s = (double) flops / avg_time / 1e9;
    double intensity = gflops_per_s / gb_per_s;

    printf("\r%s: %.3fs, %.2f GB/s, %.2f GFLOP/s, %.2f flop/byte, "
           "L3-miss-local: %.0f, L3-miss-remote: %.0f\n",
           conf->name, avg_time, gb_per_s, gflops_per_s, intensity,
           avg_cache_miss_local, avg_cache_miss_remote);

    if (conf->pre_trans) {
        mat_destroy(A);
        mat_destroy(B);
    }
}

int main(void)
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            int num_threads = omp_get_num_threads();
            thread_counters = malloc(num_threads * sizeof(cache_counter_t));
        }

        int tid = omp_get_thread_num();
        thread_counters[tid] = cache_counter_init();
    }

    // sage_bwd_grad_Wroot: A=(90941x256), B=(90941x40), C=(256x40)

// #define NUM_TRAIN_NODES 90941
// #define NUM_VALID_NODES 29799
// #define NUM_TEST_NODES 48603

    // 32, 64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, VALID_SIZE, 32K, TEST_SIZE, 64K, TRAIN_SIZE, 128K, 256K, 512K

    size_t batch_sizes[15];
    batch_sizes[0] = 32;
    for (size_t b = 1; b < sizeof(batch_sizes)/sizeof(batch_sizes[0]); b++) {
        batch_sizes[b] = batch_sizes[b-1] * 2;
    }

    for (size_t b = 0; b < sizeof(batch_sizes)/sizeof(batch_sizes[0]); b++) {
        size_t height = batch_sizes[b];
        size_t width = 256;

        matrix_t* A = mat_create(height, width);
        matrix_t* B = mat_create(height, width);
        matrix_t* C = mat_create(width, width);

        matrix_t *ref = mat_create(256, 256);

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
        printf("Finished computing reference\n");

        BenchConfig configs[] = {
            BASELINE_CONFIG("baseline", BASELINE),
            BASELINE_CONFIG("blocked", BLOCKED),
            AFFINE_CONFIG("affine", BASELINE),
            AFFINE_CONFIG("affine-blocked", BLOCKED),
            AFFINE_CONFIG("affine-blocked-unroll", BLOCKED_UNROLL),
        };

        for (size_t i = 0; i < sizeof(configs)/sizeof(configs[0]); i++) {
            run_benchmark(A, B, C, ref, &configs[i]);
        }

        mat_destroy(A);
        mat_destroy(B);
        mat_destroy(C);
        mat_destroy(ref);
    }

    PERF_PRINT();


    return 0;
}
