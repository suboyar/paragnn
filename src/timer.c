#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "core.h"
#include "timer.h"

#ifndef TIMER_INDENT_SPACE
#define TIMER_INDENT_SPACE 2
#endif
#ifndef TIMER_MAX_LINE_WIDTH
#define TIMER_MAX_LINE_WIDTH 200
#endif
#ifndef TIMER_MAX_NAME_LEN
#define TIMER_MAX_NAME_LEN 128
#endif

#ifndef TIMER_MAX_STACK_DEPTH
#define TIMER_MAX_STACK_DEPTH 64
#endif

// 1024 contexts should be enough (famous last words)
#ifndef TIMER_HASHTABLE_SIZE
#define TIMER_HASHTABLE_SIZE 1024
#endif

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

static size_t timer_sample_size = 1000;

typedef struct TimerEntry TimerEntry;

struct TimerEntry {
    const char *name;
    TimerEntry *parent;
    double     *samples;
    size_t      count;
    double      current_start;
    bool        is_active;
    // After sampling
    bool        metrics_computed;
    double      min;
    double      max;
    double      total;
    double      avg;
    double      std;
    double      p99;
    double      p95;
};

typedef struct {
    TimerEntry* entries;
    size_t      count;
    size_t      capacity;
} TimerRegistry;

typedef struct {
    TimerEntry* entries[TIMER_MAX_STACK_DEPTH];
    int         depth;
} TimerStack;

static _Thread_local TimerStack timer_stack = { .depth = 0 };
static bool timer_enabled = false;

static TimerEntry entries[TIMER_HASHTABLE_SIZE];
static TimerRegistry reg = {
    .entries = entries,
    .count = 0,
    .capacity = TIMER_HASHTABLE_SIZE,
};

static inline TimerEntry* stack_top(void)
{
    return timer_stack.depth > 0
        ? timer_stack.entries[timer_stack.depth - 1]
        : NULL;
}

static inline void stack_push(TimerEntry* entry)
{
    if (timer_stack.depth >= TIMER_MAX_STACK_DEPTH) {
        ERROR("Timer stack overflow");
    }
    timer_stack.entries[timer_stack.depth++] = entry;
}

static inline void stack_pop(void)
{
    if (timer_stack.depth > 0) {
        timer_stack.depth--;
    }
}

// This uses FNV-1a hashing algorithm
static inline uint64_t hash_key(const TimerEntry* parent, const char* key)
{
    uint64_t hash = FNV_OFFSET;

    if (parent) {
        uint64_t ptr = (uint64_t)(uintptr_t)parent;
        for (int i = 0; i < 8; i++) {
            hash ^= (ptr >> (i * 8)) & 0xFF;
            hash *= FNV_PRIME;
        }
    }

    for (const char* p = key; *p; p++) {
        hash ^= (uint64_t)(uint8_t)(*p);
        hash *= FNV_PRIME;
    }
    return hash;
}

static inline size_t get_idx(const TimerEntry* parent, const char* key) {
    uint64_t hash = hash_key(parent, key);
    return (size_t)(hash & (reg.capacity-1));
}

void timer_set_timer_sample_size(size_t size) { timer_sample_size = size; }

TimerEntry* find_entry(const char* name)
{
    TimerEntry* parent = stack_top();
    size_t idx = get_idx(parent, name);
    TimerEntry* p = reg.entries + idx;
    TimerEntry* end = reg.entries + reg.capacity;

    while (p < end && p->name != NULL) {
        if (strcmp(p->name, name) == 0) {
            return p;
        }
        p++;
    }

    return NULL;
}

static TimerEntry* find_or_create_entry(const char* name) {
    if (reg.count >= reg.capacity) {
        return NULL;
    }

    TimerEntry* parent = stack_top();
    size_t idx = get_idx(parent, name);
    TimerEntry* p = reg.entries + idx;
    TimerEntry* end = reg.entries + reg.capacity;

    while (p < end && p->name != NULL) {
        if (strcmp(p->name, name) == 0) {
            return p;
        }
        p++;
    }

    p->name = name;
    p->parent = stack_top();
    p->samples = malloc(timer_sample_size * sizeof(*p->samples));
    p->count = 0;
    p->min = -1;
    p->max = -1;
    p->total = -1;
    p->avg = -1;
    p->std = -1;
    p->p99 = -1;
    p->p95 = -1;
    reg.count++;

    return p;
}

#ifndef LOG_ERROR
#define LOG_ERROR(fmt, ...) do {                                        \
        fflush(stdout);                                                 \
        fprintf(stderr, "%s:%d: error: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    } while(0)
#endif

#ifndef NDEBUG
#define ERROR(fmt, ...) do {                                            \
        LOG_ERROR(fmt, ##__VA_ARGS__);                                  \
        __builtin_trap();                                               \
    } while(0)
#else
#define ERROR(fmt, ...) do {                                            \
        LOG_ERROR(fmt, ##__VA_ARGS__);                                  \
        _exit(1);                                                       \
    } while(0)
#endif

void timer_record(const char* name, double elapsed, TimerEntry* entry)
{
    if (!timer_enabled) return;
    if (!entry && (entry = find_or_create_entry(name)) == NULL)
        ERROR("registry full for timer '%s'", name);
    if (entry->count >= timer_sample_size)
        ERROR("sample limit (%zu) exceeded for timer '%s' (increase with timer_set_timer_sample_size)", timer_sample_size, name);
    entry->samples[entry->count++] = elapsed;
}

void timer_record_parallel(const char* name, double* elapsed, int nthreads)
{
    if (!timer_enabled) return;

    double wall_time = 0.0;
    for (int t = 0; t < nthreads; t++) {
        wall_time = fmax(wall_time, elapsed[t]);
    }
    timer_record(name, wall_time, NULL);
}

void timer_enable(void)
{
    timer_enabled = true;
}

void timer_disable(void)
{
    timer_enabled = false;
}

void timer_reset(void)
{
    memset(entries, 0, sizeof(entries));
    reg.count = 0;
    timer_stack.depth = 0;
}

TimerEntry* __timer_scope_push(const char* name)
{
    if (!timer_enabled) return NULL;
    TimerEntry* entry = find_or_create_entry(name);
    if (entry == NULL) ERROR("registry full for timer '%s'", name);
    stack_push(entry);
    return entry;
}

void __timer_scope_end(TimerScope* scope)
{
    if (!timer_enabled) return;

    double elapsed = omp_get_wtime() - scope->start_time;

    stack_pop();
    timer_record(scope->name, elapsed, scope->entry);
}

static int cmp_double(const void* a, const void* b)
{
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da > db) return 1;
    if (da < db) return -1;
    return 0;
}

static void compute_metrics(TimerEntry *entry)
{
    if (entry->metrics_computed || entry->count == 0) return;
    entry->metrics_computed = true;
    qsort(entry->samples, entry->count, sizeof(double), cmp_double);

    double local_total = 0.0;
#pragma omp parallel for reduction(+:local_total) if (entry->count > 1000)
    for (size_t i = 0; i < entry->count; i++)
        local_total += entry->samples[i];
    entry->total = local_total;
    entry->min = entry->samples[0];
    entry->max = entry->samples[entry->count - 1];
    entry->avg = local_total / entry->count;
    if (entry->count > 1)
    {
        double sum_sq_diff = 0.0;
#pragma omp parallel for reduction(+:sum_sq_diff) if(entry->count > 1000)
        for (size_t i = 0; i < entry->count; i++)
        {
            double diff = entry->samples[i] - entry->avg;
            sum_sq_diff += diff * diff;
        }
        entry->std = sqrt(sum_sq_diff / entry->count);
    }
    // Nearest Rank
    entry->p95 = entry->samples[(size_t)((entry->count - 1) * 0.95)];
    entry->p99 = entry->samples[(size_t)((entry->count - 1) * 0.99)];

}

double timer_get_time(const char* name, enum TimerMetric metric)
{
    TimerEntry* entry = find_entry(name);
    if (entry == NULL) ERROR("timer entry '%s' not found", name);

    compute_metrics(entry);
    switch (metric)
    {
        case TIMER_MIN_TIME: return entry->min;
        case TIMER_MAX_TIME: return entry->max;
        case TIMER_TOTAL_TIME: return entry->total;
        case TIMER_AVG_TIME: return entry->avg;
        case TIMER_STD_TIME: return entry->std;
        case TIMER_P99_TIME: return entry->p99;
        case TIMER_P95_TIME: return entry->p95;
        default: ERROR("invalid metric %d for timer '%s'", metric, name);
    }
}
static int cmp_entry_min_time_desc(const void* a, const void* b)
{
    const TimerEntry* ea = *(const TimerEntry**)a;
    const TimerEntry* eb = *(const TimerEntry**)b;

    if (eb->min > ea->min) return 1;
    if (eb->min < ea->min) return -1;
    return 0;
}

static size_t get_valid_entry_ptrs(TimerEntry** out)
{
    size_t count = 0;
    for (size_t i = 0; i < reg.capacity; i++) {
        if (reg.entries[i].name != NULL && reg.entries[i].count > 0) {
            out[count++] = &reg.entries[i];
        }
    }
    return count;
}

static void print_tree(TimerEntry** all_entries, size_t total_count,
                       const TimerEntry* parent, int depth, int name_col_width)
{
    TimerEntry** children = malloc(total_count * sizeof(TimerEntry*));
    size_t num_children = 0;

    for (size_t i = 0; i < total_count; i++) {
        if (all_entries[i]->parent == parent) {
            children[num_children++] = all_entries[i];
        }
    }

    qsort(children, num_children, sizeof(TimerEntry*), cmp_entry_min_time_desc);

    for (size_t i = 0; i < num_children; i++) {
        TimerEntry* e = children[i];

        char indented_name[TIMER_MAX_LINE_WIDTH+1];
        int indent = depth * TIMER_INDENT_SPACE;

        snprintf(indented_name, sizeof(indented_name), "%*s%s", indent, "", e->name);

        if ((int)strlen(indented_name) > name_col_width) {
            indented_name[name_col_width - 3] = '.';
            indented_name[name_col_width - 2] = '.';
            indented_name[name_col_width - 1] = '.';
            indented_name[name_col_width] = '\0';
        }
        printf("%-*s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-8zu\n",
               name_col_width, indented_name, e->min, e->max, e->avg, e->total, e->std, e->p95, e->p99, e->count);
        print_tree(all_entries, total_count, e, depth + 1, name_col_width);
    }

    free(children);
}

void timer_print(void)
{
    TimerEntry** all_entries = malloc(reg.capacity * sizeof(TimerEntry*));
    size_t count = get_valid_entry_ptrs(all_entries);
    for (size_t i = 0; i < count; i++)
        compute_metrics(all_entries[i]);

    char fixed_cols[TIMER_MAX_LINE_WIDTH+1];
    int fixed_cols_width;
    fixed_cols_width = snprintf(fixed_cols, sizeof(fixed_cols),
                                "%-12s %-12s %-12s %-12s %-12s %-12s %-12s %-8s",
                                "min(s)", "max(s)", "avg(s)", "total(s)", "std", "95th", "99th", "calls");

    int name_col_width = 30;
    for (size_t i = 0; i < reg.capacity; i++) {
        if (reg.entries[i].name != NULL && reg.entries[i].count > 0) {
            TimerEntry* e = &reg.entries[i];
            int spaces = (int)strlen(e->name);;
            while (e->parent != NULL) {
                spaces += TIMER_INDENT_SPACE;
                e = e->parent;
            }
            name_col_width = MAX(name_col_width, spaces);
        }
    }

    int max_line_width = 120;
    int max_name_width = max_line_width - fixed_cols_width - 1;
    if (name_col_width > max_name_width) {
        name_col_width = max_name_width;
    }

    // HACK: The compiler thinks that fixed_cols is length of TIMERf_MAX_LINE_WIDTH,
    // which produces a warning. So we just multiply the size with 2.
    char heading[2*TIMER_MAX_LINE_WIDTH+1];
    snprintf(heading, sizeof(heading), "%-*s %s", name_col_width, "name", fixed_cols);
    printf("%s\n", heading);

    for (size_t i = 0; i < strlen(heading); i++) printf("-");
    printf("\n");

    print_tree(all_entries, count, NULL, 0, name_col_width);

    free(all_entries);
}

static void build_path(TimerEntry* e, char* buf, size_t buf_size)
{
    if (e->parent) {
        build_path(e->parent, buf, buf_size);
        strncat(buf, "/", buf_size - strlen(buf) - 1);
    }
    strncat(buf, e->name, buf_size - strlen(buf) - 1);
}

void timer_export_csv(FILE *fd)
{
    if (!fd) return;

    // +1 for / (slashes)
    char path[(TIMER_MAX_NAME_LEN+1)*TIMER_MAX_STACK_DEPTH];

    if (fd == stdout) fprintf(fd, "\n--- CSV_OUTPUT_BEGIN ---\n");
    fprintf(fd, "name,parent,min(s),max(s),avg(s),total(s),std,95th,99th,calls\n");
    for (size_t i = 0; i < reg.capacity; i++) {
        if (reg.entries[i].name != NULL && reg.entries[i].count > 0) {
            TimerEntry* e = &reg.entries[i];
            compute_metrics(e);
            path[0] = '\0';
            if (e->parent) build_path(e->parent, path, sizeof(path)/sizeof(path[0]));

            fprintf(fd, "%s,%s,%f,%f,%f,%f,%f,%f,%f,%zu\n",
                    e->name, path, e->min, e->max, e->avg, e->total, e->std, e->p95, e->p99, e->count);
        }
    }
    if (fd == stdout) fprintf(fd, "--- CSV_OUTPUT_END ---\n");
}
