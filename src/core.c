#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core.h"

static long get_cache_line_size(void)
{
    long size;

    size = sysconf(_SC_LEVEL4_CACHE_LINESIZE);
    if (size > 0) return size;

    size = sysconf(_SC_LEVEL3_CACHE_LINESIZE);
    if (size > 0) return size;

    size = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
    if (size > 0) return size;

    size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (size > 0) return size;

    // Cache line size unavailable via sysconf(), using default of 64 bytes
    return 64;
}

void *cache_aligned_alloc(size_t size)
{
    size_t alignment = get_cache_line_size();
    size_t padded_size = (size + alignment - 1) & ~(alignment - 1);

    return aligned_alloc(alignment, padded_size);
}

#ifndef PARALLEL_ZERO_THRESHOLD
#ifdef USE_DOUBLE
#define PARALLEL_ZERO_THRESHOLD 32768  // 256KB of doubles i.e L2 cache
#else
#define PARALLEL_ZERO_THRESHOLD 65536 // 256KB of floats i.e L2 cache
#endif
#endif

void real_zero_out(Real *a, size_t n)
{
    if (n < PARALLEL_ZERO_THRESHOLD)
    {
        memset(a, 0, n * sizeof(Real));
    }
    else
    {
#pragma omp parallel for simd
        for (size_t i = 0; i < n; i++)
        {
            a[i] = 0.0;
        }
    }
}
