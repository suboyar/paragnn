#ifndef VREG_H
#define VREG_H

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
// Include SVE headers if targeting Neoverse V2 (Grace Hopper)
#if defined(TARGET_CPU_NEOVERSEV2) || defined(__ARM_FEATURE_SVE)
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>
#endif
#endif

#include "core.h"

typedef float  v4f  __attribute__((vector_size(16)));
typedef float  v8f  __attribute__((vector_size(32)));
typedef float  v16f __attribute__((vector_size(64)));

typedef double v2d  __attribute__((vector_size(16)));
typedef double v4d  __attribute__((vector_size(32)));
typedef double v8d  __attribute__((vector_size(64)));

#if defined(__AVX512F__)
    #define NUM_REGS 32
    #define VLEN 512
    #define VLEN_BYTES 64
#elif defined(__AVX2__)
    #define NUM_REGS 16
    #define VLEN 256
    #define VLEN_BYTES 32
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define NUM_REGS 32
    #define VLEN 128
    #define VLEN_BYTES 16
#elif defined(__SSE2__)
    #define NUM_REGS 16
    #define VLEN 128
    #define VLEN_BYTES 16
#else
    #error "Unsupported platform: need AVX2, AVX-512, or ARM NEON"
#endif
#define N_VEC (VLEN_BYTES / (int)sizeof(Real))

typedef Real VReal __attribute__((vector_size(N_VEC * sizeof(Real))));
// Unaligned, alias-safe
typedef VReal VReal_u __attribute__((may_alias, aligned(1)));

static inline __attribute__((always_inline, unused))
VReal vrload(const Real *p) { return *(const VReal *)p; }
static inline __attribute__((always_inline, unused))
VReal vrload_u(const Real *p) { return *(const VReal_u *)p; }

static inline __attribute__((always_inline, unused))
void vrstore(Real *p, VReal v) { *(VReal *)p = v; }
static inline __attribute__((always_inline, unused))
void vrstore_u(Real *p, VReal v) { *(VReal_u *)p = v; }

static inline __attribute__((always_inline, unused))
VReal vrbcast(Real x) { return (VReal){} + x; }

static inline __attribute__((always_inline, unused))
void stream_vrstore(Real* __restrict dst, VReal v)
{
#if defined(__AVX512F__)
    _mm512_stream_ps(dst, (__m512)v);
#elif defined(__AVX2__)
    _mm256_stream_ps(dst, (__m256)v);
#elif defined(TARGET_CPU_NEOVERSEV2) || defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_pat_b32(SV_VL4);
    svfloat32_t sv_v = svset_neonq_f32(svundef_f32(), (float32x4_t)v);
    svstnt1_f32(pg, dst, sv_v);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t vx = (float32x4_t)v;
    float64x1_t lo = vget_low_f64(vreinterpretq_f64_f32(vx));
    float64x1_t hi = vget_high_f64(vreinterpretq_f64_f32(vx));
    __asm__ volatile (
        "stnp %d[lo], %d[hi], [%[ptr]]"
        : /* No outputs */
        : [lo]  "w" (lo),
          [hi]  "w" (hi),
          [ptr] "r" (dst)
        : "memory"
    );
#else
    // Fallback
    vrstore(dst, v);
#endif
}

#endif // VREG_H
