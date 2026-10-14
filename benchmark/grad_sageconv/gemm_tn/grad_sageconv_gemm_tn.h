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

#ifndef GRAD_SAGECONV_GEMM_H
#define GRAD_SAGECONV_GEMM_H

void grad_sageconv_gemm_tn_v1(SageLayer *l);
void grad_sageconv_gemm_tn_v2(SageLayer *l);
void grad_sageconv_gemm_tn_v3(SageLayer *l);
void grad_sageconv_gemm_tn_v4(SageLayer *l);
void grad_sageconv_gemm_tn_blas(SageLayer *l);

#endif // GRAD_SAGECONV_GEMM_H
