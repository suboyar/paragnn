#define _GNU_SOURCE
#include <dirent.h>
#include <linux/perf_event.h>    /* Definition of PERF_* constants */
#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <sys/syscall.h>         /* Definition of SYS_* constants */
#include <sys/ioctl.h>
#include <unistd.h>

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

cache_counter_t cache_counter_init(void)
{
    cache_counter_t counter = {0};
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    if (detect_cpu_vendor() == CPU_AMD) {
        pe.type = PERF_TYPE_RAW;

        /*
         * IDX	    : 943718413
         * PMU name : amd64_fam19h_zen3 (AMD64 Fam19h Zen3)
         * Name     : DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM
         * Equiv	: None
         * Flags    : None
         * Desc     : Demand Data Cache fills by data source
         * Code     : 0x43
         * Umask-00 : 0x01 : PMU : [LCL_L2] : None : Fill from local L2 to the core
         * Umask-01 : 0x02 : PMU : [INT_CACHE] : None : Fill from L3 or different L2 in same CCX
         * Umask-02 : 0x04 : PMU : [EXT_CACHE_LCL] : None : Fill from cache of different CCX in same node
         * Umask-03 : 0x08 : PMU : [MEM_IO_LCL] : None : Fill from DRAM or IO connected in same node
         * Umask-04 : 0x10 : PMU : [EXT_CACHE_RMT] : None : Fill from CCX cache in different node
         * Umask-05 : 0x40 : PMU : [MEM_IO_RMT] : None : Fill from DRAM or IO connected in different node
         * Modif-00 : 0x00 : PMU : [k] : monitor at priv level 0 (boolean)
         * Modif-01 : 0x01 : PMU : [u] : monitor at priv level 1, 2, 3 (boolean)
         * Modif-02 : 0x02 : PMU : [e] : edge level (boolean)
         * Modif-03 : 0x03 : PMU : [i] : invert (boolean)
         * Modif-04 : 0x04 : PMU : [c] : counter-mask in range [0-255] (integer)
         * Modif-05 : 0x05 : PMU : [h] : monitor in hypervisor (boolean)
         * Modif-06 : 0x06 : PMU : [g] : measure in guest (boolean)
         */

        pe.config = (0x08 << 8) | 0x43;
        counter.demand_local_mem.fd = perf_event_open(&pe, 0, -1, -1, 0);
        counter.demand_local_mem.available = counter.demand_local_mem.fd != -1;
        if (!counter.demand_local_mem.available) {
            ERROR("Could not start DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM raw event");
        }
        // printf("Reading from DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM event\n");


    } else {
        // pe.type = PERF_TYPE_HARDWARE;
        // pe.config = PERF_COUNT_HW_CACHE_MISSES;
        // counter.cache_misses.fd = perf_event_open(&pe, 0, -1, -1, 0);
        // counter.cache_misses.available = counter.cache_misses.fd != -1;

        pe.type = PERF_TYPE_HW_CACHE;
        pe.config = (PERF_COUNT_HW_CACHE_LL) |
            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        counter.llc_misses.fd = perf_event_open(&pe, 0, -1, -1, 0);
        counter.llc_misses.available = counter.llc_misses.fd != -1;
        if (!counter.llc_misses.available) {
            ERROR("Could not start PERF_COUNT_HW_CACHE_LL event");
        }

        // printf("Reading from PERF_COUNT_HW_CACHE_LL event\n");

        // pe.type = PERF_TYPE_HARDWARE;
        // pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
        // counter.cache_refs.fd = perf_event_open(&pe, 0, -1, -1, 0);
        // counter.cache_refs.available = counter.cache_refs.fd != -1;
    }
    return counter;
}

void cache_counter_start(cache_counter_t* counter)
{
    if (counter->demand_local_mem.available) {
        ioctl(counter->demand_local_mem.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(counter->demand_local_mem.fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    if (counter->cache_misses.available) {
        ioctl(counter->cache_misses.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(counter->cache_misses.fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    if (counter->llc_misses.available) {
        ioctl(counter->llc_misses.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(counter->llc_misses.fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    if (counter->cache_refs.available) {
        ioctl(counter->cache_refs.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(counter->cache_refs.fd, PERF_EVENT_IOC_ENABLE, 0);
    }
}

void cache_counter_stop(cache_counter_t* counter)
{
    if (counter->demand_local_mem.available) {
        ioctl(counter->demand_local_mem.fd, PERF_EVENT_IOC_DISABLE, 0);
        read(counter->demand_local_mem.fd, &counter->demand_local_mem.count, sizeof(counter->demand_local_mem.count));
    }
    if (counter->cache_misses.available) {
        ioctl(counter->cache_misses.fd, PERF_EVENT_IOC_DISABLE, 0);
        read(counter->cache_misses.fd, &counter->cache_misses.count, sizeof(counter->cache_misses.count));
    }

    if (counter->llc_misses.available) {
        ioctl(counter->llc_misses.fd, PERF_EVENT_IOC_DISABLE, 0);
        read(counter->llc_misses.fd, &counter->llc_misses.count, sizeof(counter->llc_misses.count));
    }

    if (counter->cache_refs.available) {
        ioctl(counter->cache_refs.fd, PERF_EVENT_IOC_DISABLE, 0);
        read(counter->cache_refs.fd, &counter->cache_refs.count, sizeof(counter->cache_refs.count));
    }
}

void cache_counter_close(cache_counter_t* counter)
{
    if (counter->demand_local_mem.available) {
        close(counter->demand_local_mem.fd);
    }

    if (counter->cache_misses.available) {
        close(counter->cache_misses.fd);
    }

    if (counter->llc_misses.available) {
        close(counter->llc_misses.fd);
    }

    if (counter->cache_refs.available) {
        close(counter->cache_refs.fd);
    }
}

void cache_counter_print(cache_counter_t counter)
{
    if (detect_cpu_vendor() == CPU_AMD) {
        if (counter.demand_local_mem.available)
            printf("LLC misses:       %12lld\n", counter.demand_local_mem.count);
    } else {
        if (counter.llc_misses.available)
            printf("LLC misses:       %12lld\n", counter.llc_misses.count);

        // if (counter.cache_misses.available)
        //     printf("Cache misses:     %12lld\n", counter.cache_misses.count);

        // if (counter.cache_refs.available)
        //     printf("Cache references: %12lld\n", counter.cache_refs.count);

        // if (counter.cache_misses.available &&
        //     counter.cache_refs.available &&
        //     counter.cache_refs.count > 0)
        //     printf("Miss rate:        %11.2f%%\n",
        //            100.0 * counter.cache_misses.count / cache_counter.cache_refs.count);
    }
}


static size_t get_cache_line_size()
{
    long size = sysconf(_SC_LEVEL3_CACHE_LINESIZE);
    if (size > 0) return (size_t)size;

    size = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
    if (size > 0) return (size_t)size;

    size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (size > 0) return (size_t)size;

    fprintf(stderr, "INFO: Cache line size unavailable via sysconf(), using default of 64 bytes\n");
    return 64;
}

uint64_t cache_get_bytes_loaded(cache_counter_t* counter)
{
    size_t cacheline = get_cache_line_size();
    // printf("cacheline = %zu\n", cacheline);

    if (detect_cpu_vendor() == CPU_AMD) {
        if (counter->demand_local_mem.available) {
            // printf("counter->demand_local_mem.count = %llu\n", counter->demand_local_mem.count);
            return counter->demand_local_mem.count * get_cache_line_size();
        }
    } else {
        if (counter->llc_misses.available) {
            // printf("counter->llc_misses.count = %llu\n", counter->llc_misses.count);
            return counter->llc_misses.count * get_cache_line_size();
        }
    }
    fprintf(stderr, "INFO: No counters was started, can't return number of bytes\n");
    return 0;
}
