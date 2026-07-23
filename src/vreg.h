#ifndef VREG_H
#define VREG_H

#include "core.h"

typedef float  v4f  __attribute__((vector_size(16)));
typedef float  v8f  __attribute__((vector_size(32)));
typedef float  v16f __attribute__((vector_size(64)));

typedef double v2d  __attribute__((vector_size(16)));
typedef double v4d  __attribute__((vector_size(32)));
typedef double v8d  __attribute__((vector_size(64)));

#if defined(__AVX512F__)
    #define NUM_REGS 32
    #define N_VEC (64 / (int)sizeof(Real))
#elif defined(__AVX2__)
    #define NUM_REGS 16
    #define N_VEC (32 / (int)sizeof(Real))
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define NUM_REGS 32
    #define N_VEC (16 / (int)sizeof(Real))
#elif defined(__SSE2__)
    #define NUM_REGS 16
    #define N_VEC (16 / (int)sizeof(Real))
#else
    #error "Unsupported platform: need AVX2, AVX-512, or ARM NEON"
#endif

// ARM Specific Microarchitectures
#if defined(TARGET_CPU_THUNDERX2)      // armq
    #define MR 6
    #define NR 8

#elif defined(TARGET_CPU_KUNPENG920)   // huaq
    #define MR 4
    #define NR 8

#elif defined(TARGET_CPU_NEOVERSEV2)   // gh200q
    #define MR 8
    #define NR 8

/* Since __AVX2__ includes Zen 1, Zen 2, and Zen 3, where Zen 1 and Zen 2 have
 * higher FMA latency, we define optimal micro kernel parameters for them here
 * on a per-TARGET basis.
 */
#elif defined(TARGET_CPU_EPYC7601) || defined(TARGET_CPU_EPYC7302P) // defq, rome16q
    #define MR 5
    #define NR 16

/* ISA fallbacks define the default micro kernel parameters for the baseline
 * architectures in each group. CPUs that share the ISA but require distinct
 * tuning (e.g., due to different FMA latencies) are explicitly handled above.
 */

// AVX-512 Group (xeonmaxq, habanaq, h200q, genoaxq)
#elif defined(__AVX512F__)
    #define MR 8
    #define NR 16

// AVX2 Group (fpgaq, milanq)
#elif defined(__AVX2__)
    #define MR 8
    #define NR 8

// Global Fallback
#else
    #define MR 4
    #define NR 4
#endif

/* NR is chosen to be a multiple of N_VEC, where N_VEC is
 * computed from (VLEN / 32).
 */
#ifdef USE_DOUBLE
    #define NR_FULL NR
    #undef  NR
    #define NR (NR_FULL / 2)
    #undef NR_FULL
#endif

_Static_assert(NR % N_VEC == 0, "NR must be a multiple of N_VEC");

#define NV (NR / N_VEC)

typedef Real VReal __attribute__((vector_size(N_VEC * sizeof(Real))));
// Unaligned, alias-safe
typedef VReal VReal_u __attribute__((may_alias, aligned(1)));

static inline __attribute__((unused))
VReal vrload(const Real *p) { return *(const VReal *)p; }
static inline __attribute__((unused))
VReal vrload_u(const Real *p) { return *(const VReal_u *)p; }

static inline __attribute__((unused))
void vrstore(Real *p, VReal v) { *(VReal *)p = v; }
static inline __attribute__((unused))
void vrstore_u(Real *p, VReal v) { *(VReal_u *)p = v; }

static inline __attribute__((unused))
VReal vrbcast(Real x) { return (VReal){} + x; }

#endif // VREG_H
