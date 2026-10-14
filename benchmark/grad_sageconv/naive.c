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

#include "core.h"
#include "layers.h"

void naive_touch(int64_t M, int64_t N, int64_t K,
                 Real *restrict A, int64_t lda,
                 Real *restrict B, int64_t ldb,
                 Real *restrict C, int64_t ldc)
{
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; i++)
    {
        Real *C_i = &C[i*ldc];
        memset(C_i, 0, N * sizeof(*C_i));
        for (int64_t j = 0; j < N; j++)
        {
            for (int64_t k = 0; k < K; k++)
            {
                A[k*lda + i] = 0.0;
                B[k*ldb + j] = 0.0;
            } // end for j
        } // end for k
    } // end for i
}


void naive(int64_t M, int64_t N, int64_t K,
           const Real *restrict A, int64_t lda,
           const Real *restrict B, int64_t ldb,
           Real *restrict C, int64_t ldc)
{
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; i++)
    {
        Real *C_i = &C[i*ldc];
        for (int64_t j = 0; j < N; j++)
        {
            Real acc = 0;
            for (int64_t k = 0; k < K; k++)
            {
                acc += A[k*lda + i] * B[k*ldb + j];
            } // end for j
            C_i[j] = acc;
        } // end for k
    } // end for i
}
