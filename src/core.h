#ifndef CORE_H
#define CORE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

#define STRINGIFY(x) #x
#define PRAGMA_UNROLL(n) _Pragma(STRINGIFY(GCC unroll n))

#if defined(USE_DOUBLE)
#define Real      double
#define REAL(x)   ((double)(x))
#define REAL_MAX  DBL_MAX
#define real_sqrt sqrt
#define real_pow  pow
#define real_fabs fabs
#define real_fmax fmax
#define real_exp  exp
#define real_log  log
#define cblas_rgemm cblas_dgemm
#else
#define Real      float
#define REAL(x)   ((float)(x))
#define REAL_MAX  FLT_MAX
#define real_sqrt sqrtf
#define real_pow  powf
#define real_fabs fabsf
#define real_fmax fmaxf
#define real_exp  expf
#define real_log  logf
#define cblas_rgemm cblas_sgemm
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif // MIN
#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif // MAX

#ifndef LOG_ERROR
#define LOG_ERROR(fmt, ...) do {                                        \
        fflush(stdout);                                                 \
        fprintf(stderr, "%s:%d: error: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    } while(0)
#endif

#ifndef ERROR
#define ERROR(fmt, ...) do {                                            \
        LOG_ERROR(fmt, ##__VA_ARGS__);                                  \
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

#define UNREACHABLE(fmt, ...) \
    do { fprintf(stderr, "UNREACHABLE %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); abort(); } while (0)

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

/*
 * Target CPU name for console output. Auto-detected from
 * TARGET_CPU_* defines.
 */
#if defined(TARGET_CPU_GENERIC)
    #define TARGET_NAME "Generic"
#elif defined(TARGET_CPU_THUNDERX2)        /* armq     */
    #define TARGET_NAME "ThunderX2 99xx"
#elif defined(TARGET_CPU_EPYC7601)         /* defq     */
    #define TARGET_NAME "EPYC 7601"
#elif defined(TARGET_CPU_EPYC7413)         /* fpgaq    */
    #define TARGET_NAME "EPYC 7413"
#elif defined(TARGET_CPU_EPYC9684X)        /* genoaxq  */
    #define TARGET_NAME "EPYC 9684X"
#elif defined(TARGET_CPU_NEOVERSEV2)       /* gh200q   */
    #define TARGET_NAME "Neoverse-V2"
#elif defined(TARGET_CPU_XEON8360Y)        /* habanaq  */
    #define TARGET_NAME "Xeon Platinum 8360Y"
#elif defined(TARGET_CPU_KUNPENG920)       /* huaq     */
    #define TARGET_NAME "Kunpeng-920"
#elif defined(TARGET_CPU_EPYC7763)         /* milanq   */
    #define TARGET_NAME "EPYC 7763"
#elif defined(TARGET_CPU_EPYC7302P)        /* rome16q  */
    #define TARGET_NAME "EPYC 7302P"
#elif defined(TARGET_CPU_XEONMAX9480)      /* xeonmaxq */
    #define TARGET_NAME "Xeon Max 9480"
#else
    #define TARGET_NAME "Unknown"
#endif

typedef struct {
    FILE* fp;
    char* filename;
} FileHandler;

void *cache_aligned_alloc(size_t size);
void real_zero_out(Real *a, size_t n);
char *expand_path(const char *path);
void mkdir_recursive(const char *path);
const char *path_name(const char *path);
bool file_exists(const char *file_path);

#endif // CORE_H
