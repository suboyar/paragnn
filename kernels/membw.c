#include "membw.h"

#include <dirent.h>
#include <linux/perf_event.h>    /* Definition of PERF_* constants */
#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <sys/syscall.h>         /* Definition of SYS_* constants */
#include <sys/ioctl.h>
#include <unistd.h>
#include <omp.h>
#include <stddef.h>

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"

// Extracted form cputype.h: https://github.com/torvalds/linux/blob/54e82e93ca93e49cb4c33988adec5c8cb9d0df31/arch/arm64/include/asm/cputype.h
#define ARM_CPU_IMP_ARM    0x41
#define ARM_CPU_IMP_CAVIUM 0x43
#define ARM_CPU_IMP_HISI   0x48

#define ARM_CPU_PART_NEOVERSE_V2  0xD4F
#define CAVIUM_CPU_PART_THUNDERX2 0x0AF
#define HISI_CPU_PART_TSV110      0xD01

#define COUNTER_OP(counter_ptr, op, ...) do { \
    if ((counter_ptr)->available) { \
        op((counter_ptr)->fd, ##__VA_ARGS__); \
    } \
} while(0)

typedef enum {
    CPU_UNKNOWN,
    CPU_AMD,
    CPU_INTEL,
    CPU_ARM_NEOVERSE_V2,
    CPU_ARM_THUNDERX2,
    CPU_ARM_KUNPENG_920,
} CPUVendor;

typedef struct {
    int fd;
    long long count;
    bool available;
} PerfCounter;

typedef struct {
    PerfCounter llc_load_miss;
    PerfCounter llc_store_miss;
    PerfCounter l3_miss_local;
    PerfCounter l3_miss_remote;
} MemBWMetrics;

typedef void (*membw_action_t)(MemBWMetrics*);

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);
static CPUVendor detect_cpu_vendor(void);
static void alloc_state(int num_instances);
static MemBWMetrics init_events(void);
static inline void apply_action_1(membw_action_t action);
static inline void apply_action_all(membw_action_t action);
static void start_events(MemBWMetrics* metric);
static void stop_events(MemBWMetrics* metric);
static void close_events(MemBWMetrics* metric);
static void read_counter(PerfCounter *c, const char* name);
static int64_t get_metric_all(size_t offset);
static int64_t get_metric_1(size_t offset);


static MemBWMetrics *global_metrics = NULL;

void membw_init_1(void)   { alloc_state(1); }
void membw_init_all(void) { alloc_state(omp_get_max_threads()); }

void membw_start_1(void)   { apply_action_1(start_events);   }
void membw_start_all(void) { apply_action_all(start_events); }

void membw_stop_1(void)    { apply_action_1(stop_events);    }
void membw_stop_all(void)  { apply_action_all(stop_events);  }

void membw_close_1(void)   { apply_action_1(close_events);   }
void membw_close_all(void) { apply_action_all(close_events); }

int64_t membw_get_llc_load_miss_1()          { return get_metric_1(offsetof(MemBWMetrics, llc_load_miss));    }
int64_t membw_get_llc_load_miss_all()        { return get_metric_all(offsetof(MemBWMetrics, llc_load_miss));  }

int64_t membw_get_llc_store_miss_1()         { return get_metric_1(offsetof(MemBWMetrics, llc_store_miss));   }
int64_t membw_get_llc_store_miss_all()       { return get_metric_all(offsetof(MemBWMetrics, llc_store_miss)); }

int64_t membw_get_l3_local_cache_miss_1()    { return get_metric_1(offsetof(MemBWMetrics, l3_miss_local));    }
int64_t membw_get_l3_local_cache_miss_all()  { return get_metric_all(offsetof(MemBWMetrics, l3_miss_local));  }

int64_t membw_get_l3_remote_cache_miss_1()   { return get_metric_1(offsetof(MemBWMetrics, l3_miss_remote));   }
int64_t membw_get_l3_remote_cache_miss_all() { return get_metric_all(offsetof(MemBWMetrics, l3_miss_remote)); }

uint64_t membw_get_bytes_loaded_1()
{
    int64_t bytes_load = membw_get_llc_load_miss_1();
    int64_t bytes_store = membw_get_llc_store_miss_1();
    if (bytes_load != -1 && bytes_store != -1) return bytes_load + bytes_store;

    // Fallback to raw events if standard LLC events failed
    int64_t bytes_local = membw_get_l3_local_cache_miss_1();
    int64_t bytes_remote = membw_get_l3_remote_cache_miss_1();
    if (bytes_local != -1 || bytes_remote != -1) {
        uint64_t total = 0;
        if (bytes_local != -1) total += bytes_local;
        if (bytes_remote != -1) total += bytes_remote;
        return total;
    }

    return -1;
}

uint64_t membw_get_bytes_loaded_all()
{
    int64_t bytes_load = membw_get_llc_load_miss_all();
    int64_t bytes_store = membw_get_llc_store_miss_all();
    if (bytes_load != -1 && bytes_store != -1) return bytes_load + bytes_store;

    // Fallback to raw events if standard LLC events failed
    int64_t bytes_local = membw_get_l3_local_cache_miss_all();
    int64_t bytes_remote = membw_get_l3_remote_cache_miss_all();
    if (bytes_local != -1 || bytes_remote != -1) {
        uint64_t total = 0;
        if (bytes_local != -1) total += bytes_local;
        if (bytes_remote != -1) total += bytes_remote;
        return total;
    }
    return -1;
}

double membw_get_bw_1(double time)
{
    uint64_t bytes = membw_get_bytes_loaded_1();
    return (bytes != -1) ? (double)bytes / time : -1.0;
}

double membw_get_bw_all(double time)
{
    uint64_t bytes = membw_get_bytes_loaded_all();
    return (bytes != -1) ? (double)bytes / time : -1.0;
}

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static CPUVendor detect_cpu_vendor(void)
{
#if defined(__x86_64__) || defined(__i386__)
    uint32_t ebx, ecx, edx;
    char vendor[13];

    // CPUID with eax=0 returns vendor string
    __asm__ __volatile__("cpuid"
                         : "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(0));

    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "AuthenticAMD") == 0) return CPU_AMD;
    if (strcmp(vendor, "GenuineIntel") == 0) return CPU_INTEL;

#elif defined(__aarch64__) || defined(__arm__)
    // ref: https://developer.arm.com/documentation/107771/0102/AArch64-registers/AArch64-Identification-registers-summary/MIDR-EL1--Main-ID-Register
    uint64_t midr;
    __asm__ __volatile__("mrs %0, midr_el1" : "=r"(midr));

    uint8_t implementer = (midr >> 24) & 0xFF;
    uint16_t part_num = (midr >> 4) & 0xFFF;

    // eX3 has only these ARM CPUs, and I don't think they will add any new ones
    // before I finish my thesis. So this should be enough.

    if (implementer == ARM_CPU_IMP_ARM    && part_num == ARM_CPU_PART_NEOVERSE_V2)  return CPU_ARM_NEOVERSE_V2;
    if (implementer == ARM_CPU_IMP_CAVIUM && part_num == CAVIUM_CPU_PART_THUNDERX2) return CPU_ARM_THUNDERX2;
    if (implementer == ARM_CPU_IMP_HISI   && part_num == HISI_CPU_PART_TSV110)      return CPU_ARM_KUNPENG_920;
#endif
    return CPU_UNKNOWN;
}

static void alloc_state(int num_instances)
{
    if (global_metrics) ERROR("Tried to allocate global_metrics twice");
    global_metrics = ALLOC_OR_DIE(malloc(num_instances * sizeof(MemBWMetrics)));
    if (num_instances == 1) global_metrics[0] = init_events();
    else
    {
#pragma omp parallel
        {
            global_metrics[omp_get_thread_num()] = init_events();
        }
    }
}

static MemBWMetrics init_events(void)
{
    MemBWMetrics metric = {0};
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    pe.type = PERF_TYPE_HW_CACHE;
    pe.config = (PERF_COUNT_HW_CACHE_LL) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    metric.llc_load_miss.fd = perf_event_open(&pe, 0, -1, -1, 0);
    metric.llc_load_miss.available = (metric.llc_load_miss.fd != -1);
    pe.config = (PERF_COUNT_HW_CACHE_LL) |
                (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    metric.llc_store_miss.fd = perf_event_open(&pe, 0, -1, -1, 0);
    metric.llc_store_miss.available = (metric.llc_store_miss.fd != -1);

    // ref: https://github.com/torvalds/linux/tree/566771afc7a81e343da9939f0bd848d3622e2501/tools/perf/pmu-events/arch/
    CPUVendor vendor = detect_cpu_vendor();
    if (vendor == CPU_AMD)
    {
        pe.type = PERF_TYPE_RAW;
        pe.config = (0x08 << 8) | 0x43; // Doesn't seem to be supported on amdzen1
        metric.l3_miss_local.fd = perf_event_open(&pe, 0, -1, -1, 0);
        metric.l3_miss_local.available = metric.l3_miss_local.fd != -1;
        if (!metric.l3_miss_local.available) ERROR("Could not start LS_DMND_FILLS_FROM_SYS.MEM_IO_LOCAL (amd) raw event: %s", strerror(errno));
        pe.config = (0x40 << 8) | 0x43; // Doesn't seem to be supported on amdzen1
        metric.l3_miss_remote.fd = perf_event_open(&pe, 0, -1, -1, 0);
        metric.l3_miss_remote.available = metric.l3_miss_remote.fd != -1;
        if (!metric.l3_miss_remote.available) ERROR("Could not start LS_DMND_FILLS_FROM_SYS.MEM_IO_REMOTE (amd) raw event: %s", strerror(errno));
    }

    else if (vendor == CPU_INTEL)
    {
        pe.type = PERF_TYPE_RAW;
        pe.config = (0x01 << 8) | 0xd3;
        metric.l3_miss_local.fd = perf_event_open(&pe, 0, -1, -1, 0);
        metric.l3_miss_local.available = metric.l3_miss_local.fd != -1;
        if (!metric.l3_miss_local.available) ERROR("Could not start MEM_LOAD_L3_MISS_RETIRED.LOCAL_DRAM (intel) raw event: %s", strerror(errno));
        pe.config = (0x02 << 8) | 0xd3;
        metric.l3_miss_remote.fd = perf_event_open(&pe, 0, -1, -1, 0);
        metric.l3_miss_remote.available = metric.l3_miss_remote.fd != -1;
        if (!metric.l3_miss_remote.available) ERROR("Could not start MEM_LOAD_L3_MISS_RETIRED.REMOTE_DRAM (intel) raw event: %s", strerror(errno));
    }

    else if (vendor == CPU_ARM_NEOVERSE_V2)
    {
        pe.type = PERF_TYPE_RAW;
        pe.config = 0x400B;
        metric.l3_miss_local.fd = perf_event_open(&pe, 0, -1, -1, 0);
        metric.l3_miss_local.available = metric.l3_miss_local.fd != -1;
        if (!metric.l3_miss_local.available) ERROR("Could not start L3D_CACHE_LMISS_RD (neoversev2) raw event: %s", strerror(errno));
        metric.l3_miss_remote.available = false; // gh200q only has one socket
    }

    return metric;
}

static inline void apply_action_1(membw_action_t action)
{
    action(&global_metrics[0]);
}

static inline void apply_action_all(membw_action_t action)
{
#pragma omp parallel
    {
        action(&global_metrics[omp_get_thread_num()]);
    }
}

static void start_events(MemBWMetrics* metric)
{
    COUNTER_OP(&metric->llc_load_miss,  ioctl, PERF_EVENT_IOC_RESET,  0);
    COUNTER_OP(&metric->llc_load_miss,  ioctl, PERF_EVENT_IOC_ENABLE, 0);

    COUNTER_OP(&metric->llc_store_miss, ioctl, PERF_EVENT_IOC_RESET,  0);
    COUNTER_OP(&metric->llc_store_miss, ioctl, PERF_EVENT_IOC_ENABLE, 0);

    COUNTER_OP(&metric->l3_miss_local,  ioctl, PERF_EVENT_IOC_RESET,  0);
    COUNTER_OP(&metric->l3_miss_local,  ioctl, PERF_EVENT_IOC_ENABLE, 0);

    COUNTER_OP(&metric->l3_miss_remote, ioctl, PERF_EVENT_IOC_RESET,  0);
    COUNTER_OP(&metric->l3_miss_remote, ioctl, PERF_EVENT_IOC_ENABLE, 0);
}

static void read_counter(PerfCounter *c, const char* name)
{
    if (c->available)
    {
        ioctl(c->fd, PERF_EVENT_IOC_DISABLE, 0);
        int ret = read(c->fd, &c->count, sizeof(c->count));
        if (ret != sizeof(c->count)) ERROR("Failed to read %s: expected %zu bytes, got %d", name, sizeof(c->count), ret);
    }
}

static void stop_events(MemBWMetrics* metric)
{
    read_counter(&metric->llc_load_miss,  "LLC load miss");
    read_counter(&metric->llc_store_miss, "LLC store miss");
    read_counter(&metric->l3_miss_local,  "L3 local miss");
    read_counter(&metric->l3_miss_remote, "L3 remote miss");
}

static void close_events(MemBWMetrics* metric)
{
    COUNTER_OP(&metric->llc_load_miss, close);
    COUNTER_OP(&metric->llc_store_miss, close);
    COUNTER_OP(&metric->l3_miss_local, close);
    COUNTER_OP(&metric->l3_miss_remote, close);
}

static int64_t get_metric_1(size_t offset)
{
    PerfCounter *pc = (PerfCounter *)((char *)&global_metrics[0] + offset);
    if (!pc->available) return -1;

    return pc->count * get_cache_linesize();
}

static int64_t get_metric_all(size_t offset)
{
    PerfCounter *pc0 = (PerfCounter *)((char *)&global_metrics[0] + offset);
    if (!pc0->available) return -1;

    int64_t total_miss = 0;
    int num_threads = omp_get_max_threads();

#pragma omp parallel for reduction(+:total_miss)
    for (int t = 0; t < num_threads; t++) {
        PerfCounter *pc = (PerfCounter *)((char *)&global_metrics[t] + offset);
        total_miss += pc->count;
    }

    return total_miss * get_cache_linesize();
}
