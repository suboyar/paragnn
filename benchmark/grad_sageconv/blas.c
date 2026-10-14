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

#include <stdint.h>
#include <cblas.h>

#include "core.h"
#include "layers.h"

void cblas_gemm_touch(int64_t M, int64_t N, int64_t K,
                      Real *restrict A, int64_t lda,
                      Real *restrict B, int64_t ldb,
                      Real *restrict C, int64_t ldc)
{
    // BLAS internal thread mapping is opaque. A standard row-wise
    // OpenMP touch is used to fault the pages into memory.
#pragma omp parallel
    {
#pragma omp for
        for (int64_t k = 0; k < K; k++)
        {
            memset(&A[k * lda], 0, M * sizeof(Real));
            memset(&B[k * ldb], 0, N * sizeof(Real));
        }

#pragma omp for
        for (int64_t i = 0; i < M; i++)
        {
            memset(&C[i * ldc], 0, N * sizeof(Real));
        }
    }
}

void cblas_gemm(int64_t M, int64_t N, int64_t K,
          const Real *restrict A, int64_t lda,
          const Real *restrict B, int64_t ldb,
          Real *restrict C, int64_t ldc)
{
    cblas_rgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                M, N, K,
                1.0,
                A, lda,
                B, ldb,
                0.0,
                C, ldc);
}
