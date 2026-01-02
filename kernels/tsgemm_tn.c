#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/param.h>

#include <omp.h>
#include <cblas.h>

#define NOB_IMPLEMENTATION      // Needed for matrix.h etc...
#include "matrix.h"
#include "timer.h"
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

typedef void (*gemmfunc)(size_t M, size_t N, size_t K,
                         double alpha,
                         double *restrict A, size_t lda,
                         double *restrict B, size_t ldb,
                         double beta,
                         double *restrict C, size_t ldc);

typedef enum {
    RESTORE_NONE = 0,
    RESTORE_A    = 1 << 0,
    RESTORE_B    = 1 << 1,
} RestoreFlags;

typedef struct {
    const char* name;
    const char* desc;
    gemmfunc fn;
} MatmulKernel;

#define NEW_KERNEL(fn_, desc_) (MatmulKernel){.name=#fn_, .desc=(desc_), .fn=(fn_)}

inline static void dgemm_beta(size_t M, size_t N, double beta, size_t ldc, double *C)
{
    if (beta != 1.0) {
        if (beta == 0.0) {
#pragma omp parallel for simd
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    C[i*ldc+j] = 0.0;
                }
            }
        } else {
#pragma omp parallel for simd
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    C[i*ldc+j] *= beta;
                }
            }
        }
    }
}

//
// When doing doing matmul only B needs to be traversed column-wise,
// but since matrix A's shape is of MxN and B's shape is MxK, this forces us
// to traverse A in column-wise order too. Which is not ideal for cache utilization,
// especially when A and B are of a tall-and-skinny matrix.
//

void naive_v1(size_t M, size_t N, size_t K,
              double alpha,
              double *restrict A, size_t lda,
              double *restrict B, size_t ldb,
              double beta,
              double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

#pragma omp parallel for simd
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += A[k*lda+i] * B[k*ldb+j];
            }
            C[i*ldc+j] += alpha * sum;
        }
    }
}

void naive_v2(size_t M, size_t N, size_t K,
              double alpha,
              double *restrict A, size_t lda,
              double *restrict B, size_t ldb,
              double beta,
              double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

#pragma omp parallel for simd
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            register double a = A[k*lda+i] * alpha;
            for (size_t j = 0; j < N; j++) {
                C[i*ldc+j] += a * B[k*ldb+j];
            }
        }
    }
}

void unroll_v1(size_t M, size_t N, size_t K,
               double alpha,
               double *restrict A, size_t lda,
               double *restrict B, size_t ldb,
               double beta,
               double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

    const size_t M_unroll = (M / 4) * 4;

#pragma omp parallel for simd
    for (size_t i = 0; i < M_unroll; i+=4) {
        for (size_t k = 0; k < K; k++) {
            register double a0 = A[k*lda+i+0] * alpha;
            register double a1 = A[k*lda+i+1] * alpha;
            register double a2 = A[k*lda+i+2] * alpha;
            register double a3 = A[k*lda+i+3] * alpha;

            for (size_t j = 0; j < N; j++) {
                C[(i+0)*ldc+j] += a0 * B[k*ldb+j];
                C[(i+1)*ldc+j] += a1 * B[k*ldb+j];
                C[(i+2)*ldc+j] += a2 * B[k*ldb+j];
                C[(i+3)*ldc+j] += a3 * B[k*ldb+j];
            }
        }
    }

#pragma omp parallel for simd
    for (size_t i = M_unroll; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            register double a = A[k*lda+i] * alpha;
            for (size_t j = 0; j < N; j++) {
                C[i*ldc+j] += a * B[k*ldb+j];
            }
        }
    }
}

void unroll_v2(size_t M, size_t N, size_t K,
               double alpha,
               double *restrict A, size_t lda,
               double *restrict B, size_t ldb,
               double beta,
               double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

    const size_t M_unroll = (M / 4) * 4;
    const size_t K_unroll = (K / 4) * 4;


#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < M_unroll; i+=4) {
            for (size_t k = 0; k < K_unroll; k += 4) {
                register double a00 = alpha * A[(k+0)*lda + (i+0)];
                register double a01 = alpha * A[(k+1)*lda + (i+0)];
                register double a02 = alpha * A[(k+2)*lda + (i+0)];
                register double a03 = alpha * A[(k+3)*lda + (i+0)];

                register double a10 = alpha * A[(k+0)*lda + (i+1)];
                register double a11 = alpha * A[(k+1)*lda + (i+1)];
                register double a12 = alpha * A[(k+2)*lda + (i+1)];
                register double a13 = alpha * A[(k+3)*lda + (i+1)];

                register double a20 = alpha * A[(k+0)*lda + (i+2)];
                register double a21 = alpha * A[(k+1)*lda + (i+2)];
                register double a22 = alpha * A[(k+2)*lda + (i+2)];
                register double a23 = alpha * A[(k+3)*lda + (i+2)];

                register double a30 = alpha * A[(k+0)*lda + (i+3)];
                register double a31 = alpha * A[(k+1)*lda + (i+3)];
                register double a32 = alpha * A[(k+2)*lda + (i+3)];
                register double a33 = alpha * A[(k+3)*lda + (i+3)];

#pragma omp simd
                for (size_t j = 0; j < N; j++) {
                    double b0 = B[(k+0)*ldb + j];
                    double b1 = B[(k+1)*ldb + j];
                    double b2 = B[(k+2)*ldb + j];
                    double b3 = B[(k+3)*ldb + j];

                    C[(i+0)*ldc + j] += a00*b0 + a01*b1 + a02*b2 + a03*b3;
                    C[(i+1)*ldc + j] += a10*b0 + a11*b1 + a12*b2 + a13*b3;
                    C[(i+2)*ldc + j] += a20*b0 + a21*b1 + a22*b2 + a23*b3;
                    C[(i+3)*ldc + j] += a30*b0 + a31*b1 + a32*b2 + a33*b3;
                }
            }

            for (size_t k = K_unroll; k < K; k++) {
                register double a0 = alpha * A[k*lda + (i+0)];
                register double a1 = alpha * A[k*lda + (i+1)];
                register double a2 = alpha * A[k*lda + (i+2)];
                register double a3 = alpha * A[k*lda + (i+3)];

#pragma omp simd
                for (size_t j = 0; j < N; j++) {
                    double b = B[k*ldb + j];
                    C[(i+0)*ldc + j] += a0 * b;
                    C[(i+1)*ldc + j] += a1 * b;
                    C[(i+2)*ldc + j] += a2 * b;
                    C[(i+3)*ldc + j] += a3 * b;
                }
            }
        }

#pragma omp for simd
        for (size_t i = M_unroll; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                register double a = A[k*lda+i] * alpha;
#pragma omp for
                for (size_t j = 0; j < N; j++) {
                    C[i*ldc+j] += a * B[k*ldb+j];
                }
            }
        }
    }
}

void unroll_v3(size_t M, size_t N, size_t K,
               double alpha,
               double *restrict A, size_t lda,
               double *restrict B, size_t ldb,
               double beta,
               double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

    const size_t M_unroll = (M / 4) * 4;
    const size_t N_unroll = (N / 4) * 4;


#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < M_unroll; i+=4) {
            for (size_t k = 0; k < K; k++) {
                register double a0 = A[k*lda+i+0] * alpha;
                register double a1 = A[k*lda+i+1] * alpha;
                register double a2 = A[k*lda+i+2] * alpha;
                register double a3 = A[k*lda+i+3] * alpha;

#pragma omp simd
                for (size_t j = 0; j < N_unroll; j+=4) {
                    register double b0 = B[k*ldb+j+0];
                    register double b1 = B[k*ldb+j+1];
                    register double b2 = B[k*ldb+j+2];
                    register double b3 = B[k*ldb+j+3];

                    C[(i+0)*ldc+j+0] += a0 * b0;
                    C[(i+1)*ldc+j+0] += a1 * b0;
                    C[(i+2)*ldc+j+0] += a2 * b0;
                    C[(i+3)*ldc+j+0] += a3 * b0;

                    C[(i+0)*ldc+j+1] += a0 * b1;
                    C[(i+1)*ldc+j+1] += a1 * b1;
                    C[(i+2)*ldc+j+1] += a2 * b1;
                    C[(i+3)*ldc+j+1] += a3 * b1;

                    C[(i+0)*ldc+j+2] += a0 * b2;
                    C[(i+1)*ldc+j+2] += a1 * b2;
                    C[(i+2)*ldc+j+2] += a2 * b2;
                    C[(i+3)*ldc+j+2] += a3 * b2;

                    C[(i+0)*ldc+j+3] += a0 * b3;
                    C[(i+1)*ldc+j+3] += a1 * b3;
                    C[(i+2)*ldc+j+3] += a2 * b3;
                    C[(i+3)*ldc+j+3] += a3 * b3;
                }
            }
        }

#pragma omp for
        for (size_t i = M_unroll; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                register double a = A[k*lda+i] * alpha;
#pragma omp simd
                for (size_t j = 0; j < N; j++) {
                    C[i*ldc+j] += a * B[k*ldb+j];
                }
            }
        }
    }
}

void cache_block_v1(size_t M, size_t N, size_t K,
                    double alpha,
                    double *restrict A, size_t lda,
                    double *restrict B, size_t ldb,
                    double beta,
                    double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

    const size_t K_block = 256;
    const size_t M_unroll = (M / 4) * 4;

#pragma omp parallel
    {
        for (size_t kb = 0; kb < K; kb += K_block) {
            size_t k_end = MIN(kb+K_block, K);

#pragma omp for simd
            for (size_t i = 0; i < M_unroll; i+=4) {
                for (size_t k = kb; k < k_end; k++) {
                    register double a0 = A[k*lda+i+0] * alpha;
                    register double a1 = A[k*lda+i+1] * alpha;
                    register double a2 = A[k*lda+i+2] * alpha;
                    register double a3 = A[k*lda+i+3] * alpha;

                    for (size_t j = 0; j < N; j++) {
                        C[(i+0)*ldc+j] += a0 * B[k*ldb+j];
                        C[(i+1)*ldc+j] += a1 * B[k*ldb+j];
                        C[(i+2)*ldc+j] += a2 * B[k*ldb+j];
                        C[(i+3)*ldc+j] += a3 * B[k*ldb+j];
                    }
                }
            }

#pragma omp for simd
            for (size_t i = M_unroll; i < M; i++) {
                for (size_t k = 0; k < K; k++) {
                    register double a = A[k*lda+i] * alpha;
                    for (size_t j = 0; j < N; j++) {
                        C[i*ldc+j] += a * B[k*ldb+j];
                    }
                }
            }
        }
    }
}

void cache_block_v2(size_t M, size_t N, size_t K,
                    double alpha,
                    double *restrict A, size_t lda,
                    double *restrict B, size_t ldb,
                    double beta,
                    double *restrict C, size_t ldc)
{
    dgemm_beta(M, N, beta, ldc, C);

    const size_t K_block = 256;
    const size_t M_unroll = (M / 4) * 4;
    const size_t N_unroll = (N / 4) * 4;

#pragma omp parallel
    {
        for (size_t kb = 0; kb < K; kb += K_block) {
            size_t k_end = MIN(kb+K_block, K);

#pragma omp for
            for (size_t i = 0; i < M_unroll; i+=4) {
                for (size_t k = kb; k < k_end; k++) {
                    register double a0 = A[k*lda+i+0] * alpha;
                    register double a1 = A[k*lda+i+1] * alpha;
                    register double a2 = A[k*lda+i+2] * alpha;
                    register double a3 = A[k*lda+i+3] * alpha;

#pragma omp simd
                    for (size_t j = 0; j < N_unroll; j+=4) {
                        register double b0 = B[k*ldb+j+0];
                        register double b1 = B[k*ldb+j+1];
                        register double b2 = B[k*ldb+j+2];
                        register double b3 = B[k*ldb+j+3];

                        C[(i+0)*ldc+j+0] += a0 * b0;
                        C[(i+1)*ldc+j+0] += a1 * b0;
                        C[(i+2)*ldc+j+0] += a2 * b0;
                        C[(i+3)*ldc+j+0] += a3 * b0;

                        C[(i+0)*ldc+j+1] += a0 * b1;
                        C[(i+1)*ldc+j+1] += a1 * b1;
                        C[(i+2)*ldc+j+1] += a2 * b1;
                        C[(i+3)*ldc+j+1] += a3 * b1;

                        C[(i+0)*ldc+j+2] += a0 * b2;
                        C[(i+1)*ldc+j+2] += a1 * b2;
                        C[(i+2)*ldc+j+2] += a2 * b2;
                        C[(i+3)*ldc+j+2] += a3 * b2;

                        C[(i+0)*ldc+j+3] += a0 * b3;
                        C[(i+1)*ldc+j+3] += a1 * b3;
                        C[(i+2)*ldc+j+3] += a2 * b3;
                        C[(i+3)*ldc+j+3] += a3 * b3;
                    }
                }
            }
        }
#pragma omp for
        for (size_t i = M_unroll; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                register double a = A[k*lda+i] * alpha;
#pragma omp simd
                for (size_t j = 0; j < N; j++) {
                    C[i*ldc+j] += a * B[k*ldb+j];
                }
            }
        }
    }
}

void blas(size_t M, size_t N, size_t K,
          double alpha,
          double *restrict A, size_t lda,
          double *restrict B, size_t ldb,
          double beta,
          double *restrict C, size_t ldc)
{
    cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
}

bool is_valid(Matrix* src, Matrix* ref)
{
    const double abs_tol = 1e-9;
    const double rel_tol = 1e-6;

    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    size_t M = src->M;
    size_t N = src->N;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double src_val = MIDX(src, i, j);
            double ref_val = MIDX(ref, i, j);

            double abs_diff = fabs(src_val - ref_val);
            double abs_max = fmax(fabs(src_val), fabs(ref_val));

            if (abs_diff > abs_tol && abs_diff > rel_tol * abs_max) {
                return false;
            }
        }
    }

    return true;
}

void run_benchmark(Matrix* A, Matrix* B, Matrix* C, Matrix* ref, MatmulKernel kernel)
{
    size_t M = A->N; // TransA==LinalgTrans
    size_t K = A->M;  // TransA==LinalgTrans
    size_t N = B->N;

    double alpha = 1.0;
    double beta = 0.0;

    size_t lda = A->stride;
    size_t ldb = B->stride;
    size_t ldc = C->stride;

    // Validate once
    printf("%s: verify", kernel.name);
    fflush(stdout);
    kernel.fn(M, N, K,
              alpha,
              A->data, lda,
              B->data, ldb,
              beta,
              C->data, ldc);
    if (!is_valid(C, ref)) {
        printf("\n");
        ERROR("Result from '%s' implementation doesn't match reference", kernel.name);
    }
    matrix_zero(C);

    // Warm up
    printf("\r\033[K%s: warmup", kernel.name);
    fflush(stdout);
    for (size_t i = 0; i < 10; i++) {
        kernel.fn(M, N, K,
                  alpha,
                  A->data, lda,
                  B->data, ldb,
                  beta,
                  C->data, ldc);
        matrix_zero(C);
        printf(".");
        fflush(stdout);
    }

    assert(A->M == B->M && "K dimension mismatch (rows of A and B must match for TN gemm)");


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
        kernel.fn(M, N, K,
                  alpha,
                  A->data, lda,
                  B->data, ldb,
                  beta,
                  C->data, ldc);
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
        matrix_zero(C);
        putchar('.');
        fflush(stdout);
    }

    double bandwidth = bytes / min_time;
    double flops_per_s = (double) flops / min_time;
    double intensity = flops_per_s / bandwidth;
    printf("\r\033[K%s: %s\n", kernel.name, kernel.desc);
    printf("    %.3fs, %.2f MFlops/s, %.2f MBytes/s, %.2f flop/byte, "
           "%.2f MB(L3-local), %.2f MB(L3-remote)\n",
           min_time, flops_per_s/1e6, bandwidth/1e6, intensity,
           (double)l3_local/1e6, (double)l3_remote/1e6);
}

int main(void)
{
    printf("OpenBLAS config: %s\n", openblas_get_config());
    printf("OpenBLAS coretype: %s\n", openblas_get_corename());

    openblas_set_num_threads(omp_get_max_threads());
    int omp_num_threads = omp_get_max_threads();
    int openblas_num_threads = openblas_get_num_threads();
    printf("Using %d threads(omp) and %d threads(openblas)\n", omp_num_threads, openblas_num_threads);
    thread_counters = malloc(omp_num_threads * sizeof(cache_counter_t));
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

        Matrix* A = matrix_create(K, M);
        Matrix* B = matrix_create(K, N);
        Matrix* C = matrix_create(M, N);

        Matrix *ref = matrix_create(M, N);

        for (size_t i = 0; i < A->M; ++i) {
            for (size_t j = 0; j < A->N; ++j) {
                MIDX(A, i, j) = i * A->N + j;
            }
        }

        for (size_t i = 0; i < B->M; ++i) {
            for (size_t j = 0; j < B->N; ++j) {
                MIDX(B, i, j) = (i * B->N + j) + 1000000;
            }
        }

        // Precompute the reference
        printf("Computing reference\n");
        blas(A->N, B->N, A->M,
             1.0,
             A->data, A->stride,
             B->data, B->stride,
             0.0,
             ref->data, ref->stride);

        MatmulKernel kernels[] = {
            // NEW_KERNEL(naive_v1, "baseline (ijk)"),
            // NEW_KERNEL(naive_v2, "baseline (ikj)"),
            // NEW_KERNEL(unroll_v1, "unroll-jam M (ikj)"),
            // NEW_KERNEL(unroll_v2, "unroll-jam M + unroll K (ikj)"),
            // NEW_KERNEL(unroll_v3, "unroll-jam M + unroll N (ikj)"),
            // NEW_KERNEL(cache_block_v1, "K block + unroll-jam M (ikj)"),
            NEW_KERNEL(cache_block_v2, "K block + unroll-jam M + unroll N(ikj)"),
            NEW_KERNEL(pack_v1, ""),
            NEW_KERNEL(blas, "OpenBLAS (crème de la crème)"),
        };

        for (size_t i = 0; i < ARRAY_LEN(kernels); i++) {
            run_benchmark(A, B, C, ref, kernels[i]);
        }

        matrix_destroy(A);
        matrix_destroy(B);
        matrix_destroy(C);
        matrix_destroy(ref);
    }

    return 0;
}
