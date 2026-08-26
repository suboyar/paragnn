#ifndef KERNELS_H
#define KERNELS_H

#include "core.h"
#include "layers.h"

#define KERNEL_TOUCH_ARGS                       \
    int64_t M, int64_t N, int64_t K,            \
    Real *restrict A, int64_t lda,              \
    Real *restrict B, int64_t ldb,              \
    Real *restrict C, int64_t ldc

#define KERNEL_ARGS                             \
    int64_t M, int64_t N, int64_t K,            \
    const Real *restrict A, int64_t lda,        \
    const Real *restrict B, int64_t ldb,        \
    Real *restrict C, int64_t ldc

typedef void (*KernelFunc)(KERNEL_ARGS);
typedef void (*TouchFunc)(KERNEL_TOUCH_ARGS);

// Kernel declerations
void naive_touch(KERNEL_TOUCH_ARGS);
void naive(KERNEL_ARGS);

void cblas_gemm_touch(KERNEL_TOUCH_ARGS);
void cblas_gemm(KERNEL_ARGS);

void outer_tn_v1_touch(KERNEL_TOUCH_ARGS);
void outer_tn_v1(KERNEL_ARGS);

void outer_tn_v2_touch(KERNEL_TOUCH_ARGS);
void outer_tn_v2(KERNEL_ARGS);

void outer_tn_v3_touch(KERNEL_TOUCH_ARGS);
void outer_tn_v3(KERNEL_ARGS);

void grad_sageconv(SageLayer *l, KernelFunc kernel);

#endif // KERNELS_H
