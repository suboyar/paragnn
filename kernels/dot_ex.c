#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/param.h>

#include <omp.h>
#include <cblas.h>

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
#else
#    define SIMD
#endif

#define ARRAY_LEN(array) (sizeof(array)/sizeof(array[0]))

FileHandler csv_out = {0};
cache_counter_t* thread_counters = NULL;

typedef struct {
    const char* name;
    void (*fn) (matrix_t*, matrix_t*, matrix_t*);
    bool restore; // If set, A and B will be transposed after each time the kernel is run
} MatmulKernel;

//
// When doing doing matmul only B needs to be traversed column-wise,
// but since matrix A's shape is of MxN and B's shape is MxK, this forces us
// to traverse A in column-wise order too. Which is not ideal for cache utilization,
// especially when A and B are of a tall-and-skinny matrix.
//

void matmul_naive(matrix_t *A, matrix_t *B, matrix_t *C)
{
    assert(A->height == B->height && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->width && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

    size_t M = A->width;
    size_t K = A->height;       // <=> B->height (is a small number)
    size_t N = B->width;

#pragma omp parallel for SIMD
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += MAT_AT(A, k, i) * MAT_AT(B, k, j); // segfaults here
            }
            MAT_AT(C, i, j) = sum;
        }
    }
}

void matmul_unroll(matrix_t *A, matrix_t *B, matrix_t *C)
{
    assert(A->height == B->height && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->width && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

#pragma omp parallel for SIMD
    for (size_t i = 0; i < M; i++) {
        size_t j;
        for (j = 0; j + 3 < N; j+=4) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
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
            for (size_t k = 0; k < N; k++) {
                sum += MAT_AT(A, k, i) * MAT_AT(B, k, j);
            }
            MAT_AT(C, i, j) = sum;
        }
    }
}

void matmul_tiled(matrix_t *A, matrix_t *B, matrix_t *C)
{
    assert(A->height == B->height && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->width && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

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
                    for (size_t j = jstart; j < jstop; j++) {
                        double sum = 0.0;
#pragma omp SIMD
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
    assert(A->height == B->height && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->width && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

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

                    size_t j;
                    for (j = jstart; j + 1 < jstop; j += 4) {
                        double sum0 = MAT_AT(C, i, j+0);
                        double sum1 = MAT_AT(C, i, j+1);
                        double sum2 = MAT_AT(C, i, j+2);
                        double sum3 = MAT_AT(C, i, j+3);

#pragma omp SIMD
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

#pragma omp SIMD
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

void matmul_tiled_2x2(matrix_t*  A, matrix_t* B, matrix_t* C)
{
    assert(A->height == B->height && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->width && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

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

                size_t j;
                for (j = jstart; j + 1 < jstop; j += 2) {
                    size_t i;
                    for (i = istart; i + 1 < istop; i += 2) {
                        double sum00 = MAT_AT(C, i+0, j+0);
                        double sum01 = MAT_AT(C, i+0, j+1);
                        double sum10 = MAT_AT(C, i+1, j+0);
                        double sum11 = MAT_AT(C, i+1, j+1);

#pragma omp SIMD
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

#pragma omp SIMD
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

#pragma omp SIMD
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
#pragma omp SIMD
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

void transpose_inplace(matrix_t* src)
{
    PERF_FUNC_START();

    size_t height = src->height;
    size_t width = src->width;
    size_t b = 64;

    matrix_t *dst = mat_create(width, height);

#pragma omp parallel for
    for (size_t ii = 0; ii < height; ii += b) {
        for (size_t jj = 0; jj < width; jj += b) {
            size_t istop = (ii + b < height) ? ii + b : height;
            size_t jstop = (jj + b < width) ? jj + b : width;
            size_t i;
            for (i = ii; i+3 < istop; i+=4) {
                for (size_t j = jj; j < jstop; j++) {
#pragma omp SIMD
                    MAT_AT(dst, j, i+0) = MAT_AT(src, i+0, j);
                    MAT_AT(dst, j, i+1) = MAT_AT(src, i+1, j);
                    MAT_AT(dst, j, i+2) = MAT_AT(src, i+2, j);
                    MAT_AT(dst, j, i+3) = MAT_AT(src, i+3, j);
                }
            }

            for (; i < istop; i++) {
#pragma omp SIMD
                for (size_t j = jj; j < jstop; j++) {
                    MAT_AT(dst, j, i) = MAT_AT(src, i, j);
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

    PERF_FUNC_END();
}

//
// While previously both A and B were being traversed in column-major
// (read comment above matmul()), we can remedy this issue by
// transposing both of these matrices at the beginning, which allows
// us to traverse both matrices in row-major order. The idea is by
// doing some pre-work will lead to faster computation of matmul.
//

void matmul_tiled_2x2_transposed(matrix_t*  A, matrix_t* B, matrix_t* C)
{
    //
    // We can do in place transposing since in GNN the neurons in
    // activation layers (A & B) doesn't really mater after we have
    // gone through this layer. Only the weight matrix (C) is of
    // importance.
    //

    transpose_inplace(A);
    transpose_inplace(B);

    assert(A->width == B->width && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->height && "Output height must match A height");
    assert(C->width == B->height && "Output width must match B width");

    size_t b = 64;
    size_t M = A->height;
    size_t K = A->width;       // <=> B->width
    size_t N = B->height;

#pragma omp parallel for
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

#pragma omp SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_i0k = MAT_AT(A, i+0, k);
                            double a_i1k = MAT_AT(A, i+1, k);
                            double b_kj0 = MAT_AT(B, j+0, k);
                            double b_kj1 = MAT_AT(B, j+1, k);

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

#pragma omp SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double a_ik = MAT_AT(A, k, i);
                            sum0 += a_ik * MAT_AT(B, j+0, k);
                            sum1 += a_ik * MAT_AT(B, j+1, k);
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

#pragma omp SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            double b_kj = MAT_AT(B, j, k);
                            sum0 += MAT_AT(A, i+0, k) * b_kj;
                            sum1 += MAT_AT(A, i+1, k) * b_kj;
                        }
                        MAT_AT(C, i+0, j) += sum0;
                        MAT_AT(C, i+1, j) += sum1;
                    }

                    for (; i < istop; i++) {
                        double sum = 0.0;
#pragma omp SIMD
                        for (size_t k = kstart; k < kstop; k++) {
                            sum += MAT_AT(A, i, k) * MAT_AT(B, j, k);
                        }
                        MAT_AT(C, i, j) += sum;
                    }
                }
            }
        }
    }
}


void blas(matrix_t* A, matrix_t* B, matrix_t* C)
{
    assert(A->height == B->height && "Inner dimensions must match for matrix multiplication");
    assert(C->height == A->width && "Output height must match A height");
    assert(C->width == B->width && "Output width must match B width");

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

void run_benchmark(matrix_t* A, matrix_t* B, matrix_t* C, matrix_t* ref, MatmulKernel kernel)
{
    printf("Running %s\n", kernel.name);

    // Validate once
    kernel.fn(A, B, C);
    if (!is_valid(C, ref)) {
        ERROR("Result from '%s' implementation doesn't match reference", kernel.name);
    }
    mat_zero(C);
    if (kernel.restore) {
        transpose_inplace(A);
        transpose_inplace(B);
    }

    // Warm up
    for (size_t i = 0; i < NTIMES; i++) {
        kernel.fn(A, B, C);
        mat_zero(C);
        if (kernel.restore) {
            transpose_inplace(A);
            transpose_inplace(B);
        }
    }

    uint64_t bytes = 0;
    uint64_t cache_misses_local = 0;
    uint64_t cache_misses_remote = 0;

    // Timed runs
    double total_time = 0.0;
    for (size_t i = 0; i < NTIMES; i++) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            cache_counter_start(&thread_counters[tid]);
        }
        double start = omp_get_wtime();
        kernel.fn(A, B, C);
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
        if (kernel.restore) {
            transpose_inplace(A);
            transpose_inplace(B);
        }
    }

    size_t M = A->width;
    size_t K = A->height;       // <=> B->height
    size_t N = B->width;

    double avg_time = total_time / NTIMES;
    double avg_bytes = (double) bytes / NTIMES;
    double avg_cache_miss_local = (double)cache_misses_local / NTIMES;
    double avg_cache_miss_remote = (double)cache_misses_remote / NTIMES;

    double gb_per_s = avg_bytes / avg_time / 1e9;
    uint64_t flops = 2ULL * M * N * K;
    double gflops_per_s = (double) flops / avg_time / 1e9;
    double intensity = gflops_per_s / gb_per_s;

    printf("\r%s: %.3fs, %.2f GB/s, %.2f GFLOP/s, %.2f flop/byte, "
           "L3-miss-local: %.0f, L3-miss-remote: %.0f\n",
           kernel.name, avg_time, gb_per_s, gflops_per_s, intensity,
           avg_cache_miss_local, avg_cache_miss_remote);
}

int main(void)
{
    int num_threads = omp_get_max_threads();
    openblas_set_num_threads(num_threads);
    thread_counters = malloc(num_threads * sizeof(cache_counter_t));
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_counters[tid] = cache_counter_init();
    }
    printf("finished setting up thread_counters\n");
    // sage_bwd_grad_Wroot: A=(90941x256), B=(90941x265), C=(256x265)

// #define NUM_TRAIN_NODES 90941
// #define NUM_VALID_NODES 29799
// #define NUM_TEST_NODES 48603

    // 32, 64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, VALID_SIZE, 32K, TEST_SIZE, 64K, TRAIN_SIZE, 128K, 256K, 512K

#ifdef NDEBUG
    size_t batch_sizes[15];
    batch_sizes[0] = 32;
    for (size_t b = 1; b < ARRAY_LEN(batch_sizes); b++) {
        batch_sizes[b] = batch_sizes[b-1] * 2;
    }
    size_t width = 256;
#else
    size_t batch_sizes[] = {32};
    size_t width = 16;
#endif // NDEBUG

    for (size_t b = 0; b < ARRAY_LEN(batch_sizes); b++) {
        size_t height = batch_sizes[b];

        matrix_t* A = mat_create(height, width);
        matrix_t* B = mat_create(height, width);
        matrix_t* C = mat_create(width, width);

        matrix_t *ref = mat_create(width, width);

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
        matmul_naive(A, B, ref);
        printf("Finished computing reference\n");

        MatmulKernel kernels[] = {
            {"naive",                matmul_naive,                .restore=false},
            {"unroll",               matmul_unroll,               .restore=false},
            {"tiled",                matmul_tiled,                .restore=false},
            {"tiled_1x4",            matmul_tiled_1x4,            .restore=false},
            {"tiled_2x2",            matmul_tiled_2x2,            .restore=false},
            {"tiled_2x2_transposed", matmul_tiled_2x2_transposed, .restore=true},
            {"blas",                 blas,                        .restore=false},
        };

        for (size_t i = 0; i < ARRAY_LEN(kernels); i++) {
            run_benchmark(A, B, C, ref, kernels[i]);
        }

        mat_destroy(A);
        mat_destroy(B);
        mat_destroy(C);
        mat_destroy(ref);
    }

    PERF_PRINT();


    return 0;
}
