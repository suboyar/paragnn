#ifndef CORE_H
#define CORE_H

#define ERROR(fmt, ...) do { \
    fprintf(stderr, "%s:%d: error: %s: " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    abort(); \
} while(0)

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

#endif // CORE_H
