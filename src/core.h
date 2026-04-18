#ifndef CORE_H
#define CORE_H

#include <stdio.h>
#include <math.h>
#include <cblas.h>

#define STRINGIFY(x) #x
#define PRAGMA_UNROLL(n) _Pragma(STRINGIFY(GCC unroll n))

#ifdef USE_DOUBLE
#define Real      double
#define real_sqrt sqrt
#define real_pow  pow
#define real_fabs fabs
#define real_fmax fmax
#define cblas_rgemm cblas_dgemm
#else
#define Real      float
#define real_sqrt sqrtf
#define real_pow  powf
#define real_fabs fabsf
#define real_fmax fmaxf
#define cblas_rgemm cblas_sgemm
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif // MIN
#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif // MAX

#ifndef ERROR
#define ERROR(fmt, ...) do {                                            \
        fflush(stdout);                                                 \
        fprintf(stderr, "%s:%d: error: %s: " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
        abort();                                                        \
    } while(0)
#endif

#ifndef OMP_ERROR
#define OMP_ERROR(fmt, ...)                                             \
        _Pragma("omp critical")                                         \
        {                                                               \
            ERROR((fmt), ##__VA_ARGS__);                                \
        }                                                               \
    } while(0)
#endif

#ifndef TODO
#define TODO(fmt, ...) do {                                                 \
        fflush(stdout);                                                 \
        _Pragma("omp critical")                                         \
        {                                                               \
            fprintf(stderr, "%s:%d: TODO: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            abort();                                                    \
        }                                                               \
    } while(0)
#endif

#ifndef NDEBUG
    #if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
        #define BREAKPOINT() __asm__ __volatile__("int $3")
    #elif defined(__arm__) || defined(_M_ARM)
        #define BREAKPOINT() __asm__ __volatile__("bkpt #0")
    #elif defined(__aarch64__) || defined(_M_ARM64)
        #define BREAKPOINT() __asm__ __volatile__("brk #0")
    #elif defined(__riscv)
        #define BREAKPOINT() __asm__ __volatile__("ebreak")
    #elif defined(_WIN32)
        // Fallback for Windows - requires including <intrin.h>
        #define BREAKPOINT() __debugbreak()
    #elif defined(__GNUC__)
        // GCC builtin fallback
        #define BREAKPOINT() __builtin_trap()
    #else
        #include <signal.h>
        #define BREAKPOINT() raise(SIGTRAP)
    #endif
#else
    // No-op in release builds
    #define BREAKPOINT() ((void)0)
#endif // NDEBUG

typedef struct {
    FILE* fp;
    char* filename;
} FileHandler;

void *cache_aligned_alloc(size_t size);
void real_zero_out(Real *a, size_t n);

#endif // CORE_H
