#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "core.h"
#include "timer.h"
#include "../nob.h"

#ifndef TIMER_INDENT_SPACE
#define TIMER_INDENT_SPACE 2
#endif
#define TIMER_MAX_LINE_WIDTH 100

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

typedef struct TimerEntry TimerEntry;

struct TimerEntry {
    const char* name;
    TimerEntry* parent;
    double      total_time;
    double      min_time;
    double      max_time;
    double      current_start;
    int         thread_count;
    size_t      count;
    bool        is_active;
};

typedef struct {
    TimerEntry* entries;
    size_t      count;
    size_t      capacity;
} TimerRegistry;

#define MAX_STACK_DEPTH 64
typedef struct {
    TimerEntry* entries[MAX_STACK_DEPTH];
    int         depth;
} TimerStack;

static _Thread_local TimerStack timer_stack = { .depth = 0 };

// 1024 contexts should be enough (famous last words)
#define HASHTABLE_SIZE 1024
static TimerEntry entries[HASHTABLE_SIZE];
static TimerRegistry reg = {
    .entries = entries,
    .count = 0,
    .capacity = HASHTABLE_SIZE,
};

static inline TimerEntry* stack_top(void)
{
    return timer_stack.depth > 0
        ? timer_stack.entries[timer_stack.depth - 1]
        : NULL;
}

static inline void stack_push(TimerEntry* entry)
{
    if (timer_stack.depth >= MAX_STACK_DEPTH) {
        nob_log(NOB_ERROR, "Timer stack overflow");
        abort();
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
    p->total_time = 0.0;
    p->min_time = DBL_MAX;
    p->max_time = 0.0;
    p->thread_count = omp_in_parallel() ? omp_get_thread_num() : 1;
    p->count = 0;
    reg.count++;

    return p;
}

void timer_record(const char* name, double elapsed, TimerEntry* entry)
{
    if (!entry && (entry = find_or_create_entry(name)) == NULL) {
        nob_log(NOB_ERROR, "Timer '%s': registry full", name);
        abort();
    }

    entry->total_time += elapsed;
    entry->min_time = fmin(elapsed, entry->min_time);
    entry->max_time = fmax(elapsed, entry->max_time);
    entry->count++;
}

void timer_record_parallel(const char* name, double* elapsed, int nthreads)
{
    double wall_time = 0.0;
    for (int t = 0; t < nthreads; t++) {
        wall_time = fmax(wall_time, elapsed[t]);
    }
    timer_record(name, wall_time, NULL);
}

TimerEntry* __timer_scope_push(const char* name)
{
    TimerEntry* entry = find_or_create_entry(name);

    if (entry == NULL) {
        nob_log(NOB_ERROR, "Timer '%s': registry full", name);
        abort();
    }

    stack_push(entry);
    return entry;
}

void __timer_scope_end(TimerScope* scope)
{
    double elapsed = omp_get_wtime() - scope->start_time;

    stack_pop();
    timer_record(scope->name, elapsed, scope->entry);
}

double timer_get_time(const char* name, enum TimerMetric metric)
{
    TimerEntry* entry = find_entry(name);
    if (entry == NULL) {
        ERROR("Timer entry '%s' not found", name);
    }

    switch (metric) {
    case TIMER_TOTAL_TIME:
        return entry->total_time;
    case TIMER_MIN_TIME:
        return entry->min_time;
    case TIMER_MAX_TIME:
        return entry->max_time;
    }
    abort();
}

static int cmp_entry_ptr_by_total_time(const void* a, const void* b)
{
    const TimerEntry* ea = *(const TimerEntry**)a;
    const TimerEntry* eb = *(const TimerEntry**)b;

    if (eb->total_time > ea->total_time) return 1;
    if (eb->total_time < ea->total_time) return -1;
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

    qsort(children, num_children, sizeof(TimerEntry*), cmp_entry_ptr_by_total_time);

    for (size_t i = 0; i < num_children; i++) {
        TimerEntry* e = children[i];
        double avg = e->total_time / e->count;

        char indented_name[TIMER_MAX_LINE_WIDTH+1];
        int indent = depth * TIMER_INDENT_SPACE;

        snprintf(indented_name, sizeof(indented_name), "%*s%s", indent, "", e->name);

        if ((int)strlen(indented_name) > name_col_width) {
            indented_name[name_col_width - 3] = '.';
            indented_name[name_col_width - 2] = '.';
            indented_name[name_col_width - 1] = '.';
            indented_name[name_col_width] = '\0';
        }

        printf("%-*s %-12.6f %-12.6f %-12.6f %-12.6f %-8zu\n",
               name_col_width, indented_name, avg, e->total_time, e->min_time, e->max_time, e->count);

        print_tree(all_entries, total_count, e, depth + 1, name_col_width);
    }

    free(children);
}

void timer_print(void)
{
    char fixed_cols[TIMER_MAX_LINE_WIDTH+1];
    int fixed_cols_width = snprintf(fixed_cols, sizeof(fixed_cols),
                                    "%-12s %-12s %-12s %-12s %-8s",
                                    "avg(s)", "total(s)", "min(s)", "max(s)", "calls");

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

    const int max_line_width = 120;
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

    TimerEntry** all_entries = malloc(reg.capacity * sizeof(TimerEntry*));
    size_t count = get_valid_entry_ptrs(all_entries);

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

void timer_export_csv(const char *fname)
{
    if (!fname) return;
    FILE *f = (strcmp(fname, "stdout") == 0) ? stdout : fopen(fname, "w+");
    if (!f) ERROR("Could not open file %s for csv export: %s", fname, strerror(errno));

    // +1 for / (slashes)
    char path[(128+1)*MAX_STACK_DEPTH]; // TODO do #define MAX_LEN_NAME 128

    if (f == stdout) fprintf(f, "\n--- CSV_OUTPUT_BEGIN ---\n");
    fprintf(f, "name,parent,avg(s),total(s),min(s),max(s),calls\n");

    for (size_t i = 0; i < reg.capacity; i++) {
        if (reg.entries[i].name != NULL && reg.entries[i].count > 0) {
            TimerEntry* e = &reg.entries[i];
            path[0] = '\0';
            if (e->parent) build_path(e->parent, path, NOB_ARRAY_LEN(path));

            double avg = e->total_time / e->count;
            fprintf(f, "%s,%s,%f,%f,%f,%f,%zu\n",
                    e->name, path, avg, e->total_time, e->min_time, e->max_time, e->count);
        }
    }

    if (f == stdout) fprintf(f, "--- CSV_OUTPUT_END ---\n");
}
