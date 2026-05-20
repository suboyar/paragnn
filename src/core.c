#include <errno.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <wordexp.h>

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
    if (n < PARALLEL_ZERO_THRESHOLD || omp_in_parallel())
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

char *expand_path(const char *path)
{
    if (path == NULL || path[0] == '\0') return NULL;

    wordexp_t result;
    if (wordexp(path, &result, WRDE_NOCMD | WRDE_UNDEF) != 0)
        return NULL;

    char *expanded = (result.we_wordc > 0) ? strdup(result.we_wordv[0]) : NULL;
    wordfree(&result);
    return expanded;
}

void mkdir_recursive(const char *path)
{
    if (path == NULL || path[0] == '\0')
    {
        ERROR("cannot create directory from empty path");
    }

    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode))
    {
        goto cleanup;
    }

    char *tmp = strdup(path);

    for (char *p = tmp + 1; *p; p++)
    {
        if (*p == '/')
        {
            *p = '\0';
            if (mkdir(tmp, 0755) < 0 && errno != EEXIST)
            {
                ERROR("could not create directory '%s': %s", tmp, strerror(errno));
            }
            *p = '/';
        }
    }

    // create the final component
    if (mkdir(tmp, 0755) < 0 && errno != EEXIST)
    {
        ERROR("could not create directory '%s': %s", tmp, strerror(errno));
    }

cleanup:
    free(tmp);
}

const char *path_name(const char *path)
{
    const char *p = strrchr(path, '/');
    return p ? p + 1 : path;
}

bool file_exists(const char *file_path)
{
    struct stat statbuf;
    if (stat(file_path, &statbuf) < 0)
    {
        if (errno == ENOENT) return false;
        ERROR("Could not check if file %s exists: %s", file_path, strerror(errno));
    }
    return true;
}
