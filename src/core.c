#include "core.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <wordexp.h>

#include <omp.h>
#include <numa.h>
#include <numaif.h>

int get_active_numa_nodes(void)
{
    static int _numa_nodes = 0;
    int numa_nodes;
#pragma omp atomic read
    numa_nodes = _numa_nodes;

    if (__builtin_expect(numa_nodes == 0, 0))
    {
        struct bitmask *total_nodes = numa_allocate_nodemask();

#pragma omp parallel
        {
            struct bitmask *thread_nodes = numa_get_run_node_mask();
#pragma omp critical
            {
                for (unsigned int i = 0; i < total_nodes->size; i++) {
                    if (numa_bitmask_isbitset(thread_nodes, i)) {
                        numa_bitmask_setbit(total_nodes, i);
                    }
                }
            }
            numa_bitmask_free(thread_nodes);
        }

        numa_nodes = numa_bitmask_weight(total_nodes);
        numa_bitmask_free(total_nodes);

#pragma omp atomic write
        _numa_nodes = numa_nodes;
    }

    return numa_nodes;
}

size_t get_cache_linesize(void)
{
    static size_t cache_linesize = 0;
    size_t val;

#pragma omp atomic read
    val = cache_linesize;

    if (__builtin_expect(val == 0, 0))
    {
#pragma omp critical
        {
#pragma omp atomic read
            val = cache_linesize;

            if (val == 0)
            {
                long tmp;
                if ((tmp = sysconf(_SC_LEVEL4_CACHE_LINESIZE)) > 0) val = tmp;
                else if ((tmp = sysconf(_SC_LEVEL3_CACHE_LINESIZE)) > 0) val = tmp;
                else if ((tmp = sysconf(_SC_LEVEL2_CACHE_LINESIZE)) > 0) val = tmp;
                else if ((tmp = sysconf(_SC_LEVEL1_DCACHE_LINESIZE)) > 0) val = tmp;
                else val = 64;

#pragma omp atomic write
                cache_linesize = val;
            }
        }
    }
    return val;
}

void *alloc_local(size_t size)
{
    size_t alignment = get_cache_linesize();
    return aligned_alloc(alignment, size);
}

/* A page is typical 4KB, such that its also cachline aligned */
void *alloc_interleaved(size_t size)
{
    int page_size = getpagesize();
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);
    void *ptr = aligned_alloc(page_size, aligned_size);
    if (!ptr) return NULL;

    struct bitmask *nodes = numa_get_run_node_mask();
    if (!nodes) {
        free(ptr);
        return NULL;
    }

    // Apply NUMA interleave policy
    if (mbind(ptr, aligned_size, MPOL_INTERLEAVE, nodes->maskp, nodes->size, 0) < 0) {
        free(ptr);
        numa_bitmask_free(nodes);
        return NULL;
    }

    numa_bitmask_free(nodes);
    return ptr;
}

void *alloc_shared(size_t size)
{
    if (get_active_numa_nodes() > 1) return alloc_interleaved(size);
    else return alloc_local(size);
}

#ifndef PARALLEL_ZERO_THRESHOLD
#ifdef USE_DOUBLE
#define PARALLEL_ZERO_THRESHOLD 32768  // 256KB of doubles i.e ~L2 cache
#else
#define PARALLEL_ZERO_THRESHOLD 65536 // 256KB of floats i.e ~L2 cache
#endif
#endif

void real_zero_out(Real *a, size_t n)
{
    if (n < PARALLEL_ZERO_THRESHOLD || omp_in_parallel())
        memset(a, 0, n * sizeof(Real));
    else
    {
#pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < n; i++)
        {
            a[i] = 0.0;
        }
    }
}

// TODO: Check out getrusage from <sys/resource.h>
size_t get_memory_usage(void)
{
    const char *path = "/proc/self/status";
    FILE* file = fopen(path, "r");
    if (!file) ERROR("Could not open %s: %s", path, strerror(errno));
    size_t kb = 0;
    char line[128];
    while (fgets(line, 128, file) != NULL)
    {
        if (strncmp(line, "VmRSS:", 6) == 0)
        {
            sscanf(line + 6, "%zu", &kb);
            if (errno != 0) ERROR("sscanf failed: %s", strerror(errno));
            break;
        }
    }
    fclose(file);
    return kb;
}

void print_memory_usage(void)
{
    size_t kb = get_memory_usage();
    if (kb >= 1024 * 1024)
    {
        printf("Memory usage: %.2f GB\n", (double)kb / (1024.0 * 1024.0));
    }
    else if (kb >= 1024)
    {
        printf("Memory usage: %.2f MB\n", (double)kb / 1024.0);
    }
    else
    {
        printf("Memory usage: %zu kB\n", kb);
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
        ERROR("cannot create directory from empty path");

    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode))
        return;

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
        ERROR("could not create directory '%s': %s", tmp, strerror(errno));

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

char* fd_to_path(FILE *fp)
{
    if (!fp) return strdup("");
    if (fp == stdout) return strdup("stdout");
    if (fp == stderr) return strdup("stderr");

    char proc_path[64];
    snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fileno(fp));

    char tmp[PATH_MAX];
    ssize_t len = readlink(proc_path, tmp, sizeof(tmp));
    if (len < 0) return strdup("unknown");

    return strndup(tmp, (size_t)len);
}

MmapInfo map_file(const char *file, int prot, int flags)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    if (fstat(fd, &sb) < 0) ERROR("fstat failed: %s", strerror(errno));
    void *data = mmap(NULL, sb.st_size, prot, flags, fd, 0);
    if (data == MAP_FAILED) ERROR("mmap failed: %s", strerror(errno));
    return (MmapInfo){data, (size_t)sb.st_size, fd};
}

void unmap_file(MmapInfo *info)
{
    if (munmap(info->data, info->bytes) < 0) ERROR("munmap failed: %s", strerror(errno));
    close(info->fd);
}
