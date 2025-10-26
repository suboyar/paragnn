#define _GNU_SOURCE
#include <dirent.h>
#include <linux/perf_event.h>    /* Definition of PERF_* constants */
#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <sys/syscall.h>         /* Definition of SYS_* constants */
#include <sys/ioctl.h>
#include <unistd.h>

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cache_counter.h"
#include "core.h"

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

typedef enum {
    CPU_AMD,
    CPU_INTEL,
    CPU_ARM,
    CPU_UNKNOWN
} CPUVendor;

static CPUVendor detect_cpu_vendor(void)
{
#if defined(__x86_64__) || defined(__i386__)
    uint32_t ebx, ecx, edx;
    char vendor[13];

    // CPUID with eax=0 returns vendor string
    __asm__ __volatile__(
        "cpuid"
        : "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0)
    );

    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "AuthenticAMD") == 0) return CPU_AMD;
    if (strcmp(vendor, "GenuineIntel") == 0) return CPU_INTEL;

    return CPU_UNKNOWN;
#elif defined(__aarch64__) || defined(__arm__)
    return CPU_ARM;
#endif
    return CPU_UNKNOWN;
}

int get_amdzen1_l3_pmu_type(void)
{
    FILE *f = fopen("/sys/devices/amd_l3/type", "r");
    if (!f) {
        return -1;  // L3 PMU not available
    }

    int type;
    fscanf(f, "%d", &type);
    fclose(f);
    return type;
}

cache_counter_t cache_counter_init(void)
{
    cache_counter_t counter = {0};
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    CPUVendor vendor = detect_cpu_vendor();

    // ref: https://github.com/torvalds/linux/tree/566771afc7a81e343da9939f0bd848d3622e2501/tools/perf/pmu-events/arch/
    if (vendor == CPU_AMD) {
        pe.type = PERF_TYPE_RAW;

        pe.config = (0x08 << 8) | 0x43; // Doesn't seem to have be supported on amdzen1
        counter.l3_miss_local.fd = perf_event_open(&pe, 0, -1, -1, 0);
        counter.l3_miss_local.available = counter.l3_miss_local.fd != -1;
        if (!counter.l3_miss_local.available) {
            ERROR("Could not start LS_DMND_FILLS_FROM_SYS.MEM_IO_LOCAL (amd) raw event: %s", strerror(errno));
        }

        pe.config = (0x40 << 8) | 0x43; // Doesn't seem to have be supported on amdzen1
        counter.l3_miss_remote.fd = perf_event_open(&pe, 0, -1, -1, 0);
        counter.l3_miss_remote.available = counter.l3_miss_remote.fd != -1;
        if (!counter.l3_miss_remote.available) {
            ERROR("Could not start LS_DMND_FILLS_FROM_SYS.MEM_IO_REMOTE (amd) raw event: %s", strerror(errno));
        }
    }

    else if (vendor == CPU_INTEL) {
        pe.type = PERF_TYPE_RAW;

        pe.config = (0x01 << 8) | 0xd3;
        counter.l3_miss_local.fd = perf_event_open(&pe, 0, -1, -1, 0);
        counter.l3_miss_local.available = counter.l3_miss_local.fd != -1;
        if (!counter.l3_miss_local.available) {
            ERROR("Could not start MEM_LOAD_L3_MISS_RETIRED.LOCAL_DRAM (intel) raw event: %s", strerror(errno));
        }

        pe.config = (0x02 << 8) | 0xd3;
        counter.l3_miss_remote.fd = perf_event_open(&pe, 0, -1, -1, 0);
        counter.l3_miss_remote.available = counter.l3_miss_remote.fd != -1;
        if (!counter.l3_miss_remote.available) {
            ERROR("Could not start MEM_LOAD_L3_MISS_RETIRED.REMOTE_DRAM (intel) raw event: %s", strerror(errno));
        }
    }

    else if (vendor == CPU_ARM) {
        pe.type = PERF_TYPE_RAW;

        pe.config = 0x400B;
        counter.l3_miss_local.fd = perf_event_open(&pe, 0, -1, -1, 0);
        counter.l3_miss_local.available = counter.l3_miss_local.fd != -1;
        if (!counter.l3_miss_local.available) {
            ERROR("Could not start L3D_CACHE_LMISS_RD (neoversev2) raw event: %s", strerror(errno));
        }

        counter.l3_miss_remote.available = false; // gh200q has only one socket
    }

    else {
        int llc_errno = 0, generic_errno = 0;

        pe.type = PERF_TYPE_HW_CACHE;
        pe.config = (PERF_COUNT_HW_CACHE_LL) |
            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        counter.l3_miss_generic.fd = perf_event_open(&pe, 0, -1, -1, 0);

        if (counter.l3_miss_generic.fd == -1) {
            // Fall back to generic cache misses
            llc_errno = errno;
            fprintf(stderr, "warning: LLC event not available (%s), falling back to generic cache misses\n",
                    strerror(llc_errno));

            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_CACHE_MISSES;
            counter.l3_miss_generic.fd = perf_event_open(&pe, 0, -1, -1, 0);

            if (counter.l3_miss_generic.fd == -1) {
                generic_errno = errno;
            }
        }

        counter.l3_miss_generic.available = (counter.l3_miss_generic.fd != -1);

        if (!counter.l3_miss_generic.available) {
            ERROR("Could not start any cache events. LLC: %s, Generic: %s",
                  strerror(llc_errno), strerror(generic_errno));
        }
    }

    return counter;
}

void cache_counter_start(cache_counter_t* counter)
{
    if (counter->l3_miss_local.available) {
        ioctl(counter->l3_miss_local.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(counter->l3_miss_local.fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    if (counter->l3_miss_remote.available) {
        ioctl(counter->l3_miss_remote.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(counter->l3_miss_remote.fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    if (counter->l3_miss_generic.available) {
        ioctl(counter->l3_miss_generic.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(counter->l3_miss_generic.fd, PERF_EVENT_IOC_ENABLE, 0);
    }
}

void cache_counter_stop(cache_counter_t* counter)
{
    int ret;

    if (counter->l3_miss_local.available) {
        ioctl(counter->l3_miss_local.fd, PERF_EVENT_IOC_DISABLE, 0);
        ret = read(counter->l3_miss_local.fd, &counter->l3_miss_local.count, sizeof(counter->l3_miss_local.count));
        if (ret != sizeof(counter->l3_miss_local.count)) {
            ERROR("Failed to read L3 local miss counter: expected %zu bytes, got %d: %s",
                  sizeof(counter->l3_miss_local.count), ret, ret < 0 ? strerror(errno) : "partial read");
        }
    }

    if (counter->l3_miss_remote.available) {
        ioctl(counter->l3_miss_remote.fd, PERF_EVENT_IOC_DISABLE, 0);
        ret = read(counter->l3_miss_remote.fd, &counter->l3_miss_remote.count, sizeof(counter->l3_miss_remote.count));
        if (ret != sizeof(counter->l3_miss_remote.count)) {
            ERROR("Failed to read L3 remote miss counter: expected %zu bytes, got %d: %s",
                  sizeof(counter->l3_miss_remote.count), ret, ret < 0 ? strerror(errno) : "partial read");
        }
    }

    if (counter->l3_miss_generic.available) {
        ioctl(counter->l3_miss_generic.fd, PERF_EVENT_IOC_DISABLE, 0);
        ret = read(counter->l3_miss_generic.fd, &counter->l3_miss_generic.count, sizeof(counter->l3_miss_generic.count));
        if (ret != sizeof(counter->l3_miss_generic.count)) {
            ERROR("Failed to read L3 generic miss counter: expected %zu bytes, got %d: %s",
                  sizeof(counter->l3_miss_generic.count), ret, ret < 0 ? strerror(errno) : "partial read");
        }
    }
}

void cache_counter_close(cache_counter_t* counter)
{
    if (counter->l3_miss_local.available) {
        close(counter->l3_miss_local.fd);
    }

    if (counter->l3_miss_remote.available) {
        close(counter->l3_miss_remote.fd);
    }

    if (counter->l3_miss_generic.available) {
        close(counter->l3_miss_generic.fd);
    }
}

void cache_counter_print(cache_counter_t counter)
{
    CPUVendor vendor = detect_cpu_vendor();

    if (vendor == CPU_AMD || vendor == CPU_INTEL || vendor == CPU_ARM) {
        if (counter.l3_miss_local.available) {
            printf("L3 misses (local):  %12lld\n", counter.l3_miss_local.count);
        }
        if (counter.l3_miss_remote.available) {
            printf("L3 misses (remote): %12lld\n", counter.l3_miss_remote.count);
        }
    } else {
        if (counter.l3_miss_generic.available) {
            printf("L3 misses:          %12lld\n", counter.l3_miss_generic.count);
        }
    }
}

static long get_cache_line_size()
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

#pragma omp single
    fprintf(stderr, "warning: Cache line size unavailable via sysconf(), using default of 64 bytes\n");

    return 64;
}

void cache_counter_get_cache_misses(cache_counter_t* counter, long long* cache_misses_local, long long* cache_misses_remote)
{
    *cache_misses_local = 0;
    *cache_misses_remote = 0;

    if (counter->l3_miss_local.available) {
        *cache_misses_local = counter->l3_miss_local.count;
    }

    if (counter->l3_miss_remote.available) {
        *cache_misses_remote = counter->l3_miss_remote.count;
    }

    if (counter->l3_miss_generic.available) {
        *cache_misses_local = counter->l3_miss_generic.count;
    }
}

uint64_t cache_counter_get_bytes_loaded(cache_counter_t* counter)
{
    size_t cacheline = get_cache_line_size();
    uint64_t total_misses = 0;

    if (counter->l3_miss_local.available) {
        total_misses += counter->l3_miss_local.count;
    }

    if (counter->l3_miss_remote.available) {
        total_misses += counter->l3_miss_remote.count;
    }

    if (counter->l3_miss_generic.available) {
        return counter->l3_miss_generic.count * cacheline;
    }

    return total_misses * cacheline;
}
