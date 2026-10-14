/*
 * ParaGNN
 * Copyright (C) 2026  Suleyman Boyar
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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

#endif // KERNELS_H
