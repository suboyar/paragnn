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
    #define VLEN (64 / (int)sizeof(Real))
#elif defined(__AVX2__)
    #define NUM_REGS 16
    #define VLEN (32 / (int)sizeof(Real))
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define NUM_REGS 32
    #define VLEN (16 / (int)sizeof(Real))
#elif defined(__SSE2__)
    #define NUM_REGS 16
    #define VLEN (16 / (int)sizeof(Real))
#else
    #error "Unsupported platform: need AVX2, AVX-512, or ARM NEON"
#endif

typedef Real VReal __attribute__((vector_size(VLEN * sizeof(Real))));
// Unaligned, alias-safe
typedef VReal VReal_u __attribute__((may_alias, aligned(1)));

static inline VReal vload(const Real *p)      { return *(const VReal *)p; }
static inline VReal vload_u(const Real *p)    { return *(const VReal_u *)p; }
static inline void vstore(Real *p, VReal v)   { *(VReal *)p = v; }
static inline void vstore_u(Real *p, VReal v) { *(VReal_u *)p = v; }

static inline __attribute__((unused)) VReal bcast(Real x)
{
    return (VReal){} + x;
}

#endif // VREG_H
