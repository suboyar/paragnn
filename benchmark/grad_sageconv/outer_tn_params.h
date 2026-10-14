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

#ifndef OUTER_TN_PARAMS_H
#define OUTER_TN_PARAMS_H
#include "vreg.h"

#if defined(TARGET_CPU_XEON6960P) /*h200q*/
    #define DEF_KC 64
    #define DEF_MR 8
    #define DEF_NR 16
    #define K_UNROLL 8
#elif defined(TARGET_CPU_XEONMAX9480) /*xeonmaxq*/
    #define DEF_KC 32
    #define DEF_MR 8
    #define DEF_NR 16
    #define K_UNROLL 8
#elif defined(TARGET_CPU_XEON8360Y) /*habanaq*/
    #define DEF_KC 32
    #define DEF_MR 8
    #define DEF_NR 16
    #define K_UNROLL 1
#elif defined(TARGET_CPU_EPYC9684X) /*genoaxq*/
    #define DEF_KC 128
    #define DEF_MR 6
    #define DEF_NR 64
    #define K_UNROLL 1
#elif defined(TARGET_CPU_EPYC7763) /*milanq*/
    #define DEF_KC 64
    #define DEF_MR 4
    #define DEF_NR 24
    #define K_UNROLL 1
#elif defined(TARGET_CPU_EPYC7413) /*fpgaq*/
    #define DEF_KC 128
    #define DEF_MR 8
    #define DEF_NR 8
    #define K_UNROLL 4
#elif defined(TARGET_CPU_EPYC7302P) /*rome16q*/
    #define DEF_KC 128
    #define DEF_MR 5
    #define DEF_NR 16
    #define K_UNROLL 2
#elif defined(TARGET_CPU_EPYC7601) /*defq*/
    #define DEF_KC 128
    #define DEF_MR 5
    #define DEF_NR 16
    #define K_UNROLL 2
#elif defined(TARGET_CPU_NEOVERSEV2) /*gh200q*/
    #define DEF_KC 64
    #define DEF_MR 8
    #define DEF_NR 8
    #define K_UNROLL 4
#elif defined(TARGET_CPU_KUNPENG920) /*huaq*/
    #define DEF_KC 32
    #define DEF_MR 6
    #define DEF_NR 16
    #define K_UNROLL 1
#elif defined(TARGET_CPU_THUNDERX2) /*armq*/
    #define DEF_KC 64
    #define DEF_MR 6
    #define DEF_NR 8
    #define K_UNROLL 4
#else
    #error "Target CPU not supported or defined."
#endif

#ifndef MR
    #define MR DEF_MR
#endif

#ifndef NR
    #define NR DEF_NR
#endif

#ifndef KC
    #ifdef DEF_KC
        #define KC DEF_KC
    #endif
#endif

#ifndef K_UNROLL
    #define K_UNROLL DEF_K_UNROLL
#endif

// _Static_assert(NR % N_VEC == 0, "NR must be a multiple of N_VEC");

#define NV (NR / N_VEC)

#ifdef USE_DOUBLE
    #error "Double precision parameters missing. Regenerate using param.py for DP."
#endif

#endif // OUTER_TN_PARAMS_H
