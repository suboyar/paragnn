#ifndef OUTER_TN_PARAMS_H
#define OUTER_TN_PARAMS_H

#include "vreg.h"

#if defined(TARGET_CPU_XEONMAX9480) /* xeonmaxq */
    #define MR 6
    #define NR 64
    #define KC 160
    #define K_UNROLL 1
#elif defined(TARGET_CPU_XEON8360Y) /* habanaq */
    #define MR 6
    #define NR 64
    #define KC 160
    #define K_UNROLL 1
#elif defined(TARGET_CPU_XEON6960P) /* h200q */
    #define MR 6
    #define NR 64
    #define KC 160
    #define K_UNROLL 1
#elif defined(TARGET_CPU_EPYC7601) /* defq */
    #define MR 5
    #define NR 16
    #define KC 320
    #define K_UNROLL 1
#elif defined(TARGET_CPU_EPYC7302P) /* rome16q */
    #define MR 5
    #define NR 16
    #define KC 320
    #define K_UNROLL 1
#elif defined(TARGET_CPU_EPYC7413) /* fpgaq */
    #define MR 8
    #define NR 8
    #define KC 384
    #define K_UNROLL 2
#elif defined(TARGET_CPU_EPYC7763) /* milanq */
    #define MR 8
    #define NR 8
    #define KC 384
    #define K_UNROLL 2
#elif defined(TARGET_CPU_EPYC9684X) /* genoaxq */
    #define MR 6
    #define NR 64
    #define KC 96
    #define K_UNROLL 1
#elif defined(TARGET_CPU_THUNDERX2) /* armq */
    #define MR 6
    #define NR 16
    #define KC 320
    #define K_UNROLL 1
#elif defined(TARGET_CPU_KUNPENG920) /* huaq */
    #define MR 6
    #define NR 16
    #define KC 512
    #define K_UNROLL 1
#elif defined(TARGET_CPU_NEOVERSEV2) /* gh200q */
    #define MR 6
    #define NR 16
    #define KC 512
    #define K_UNROLL 1
#else
    #error "Target CPU not supported or defined."
#endif

_Static_assert(NR % N_VEC == 0, "NR must be a multiple of N_VEC");

#define NV (NR / N_VEC)

#ifdef USE_DOUBLE
    #error "Double precision parameters missing. Regenerate using param.py for DP."
#endif


#endif // OUTER_TN_PARAMS_H
