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

#ifndef MATMUL_NAIVE_H
#define MATMUL_NAIVE_H

#include <stdint.h>
#include "core.h"

enum MATMUL_TRANSPOSE {
    MatmulNoTrans = 0,
    MatmulTrans = 1,
};

void matmul(enum MATMUL_TRANSPOSE TransA,
            enum MATMUL_TRANSPOSE TransB,
            int64_t M, int64_t N, int64_t K,
            Real alpha,
            Real *restrict A, int64_t lda,
            Real *restrict B, int64_t ldb,
            Real beta,
            Real *restrict C, int64_t ldc);

#endif // MATMUL_NAIVE_H
