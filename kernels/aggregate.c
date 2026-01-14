#define _GNU_SOURCE
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <error.h>
#include <omp.h>

#define NOB_IMPLEMENTATION
#include "cache_counter.h"
#include "core.h"
#include "dataset.h"

/**
 * Forces the compiler to assume `p` and the memory it points to are
 * externally observable, preventing dead code elimination.
 *
 * Source: Chandler Carruth's CppCon 2015 talk
 * "Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!"
 * https://youtu.be/nXaxk27zwlk?t=2473
 */
static void escape(void *p) {__asm__ volatile("" : : "g"(p) : "memory");}
static void clobber() {__asm__ volatile("" : : : "memory");}


#define ARRAY_LEN(array) (sizeof(array)/sizeof(array[0]))
#define EDGE_SRC(e, k)        ((e)->data[2 * (k) + 0])
#define EDGE_DST(e, k)        ((e)->data[2 * (k) + 1])

static size_t ntimes = 100;

static cache_counter_t* thread_counters = NULL;

typedef struct {
    uint32_t *data;
} COO;

typedef struct {
    uint32_t *row_ptr;          // src
    uint32_t *col_idx;          // dst
} CRS;

typedef struct {
    uint32_t *col_ptr;          // dst
    uint32_t *row_idx;          // src
    double *val;                // This should be filled on demand
} CCS;

// static CRS crs_edges = {0};
// static CCS ccs_edges = {0};

typedef void (*aggfunc)(size_t, size_t, size_t, void *, double *, size_t, double *, size_t);

typedef struct {
    const char* name;
    const char* desc;
    void *edge_index;
    aggfunc fn;
    uint64_t flops;
} AggregateKernel;

#define NEW_KERNEL(fn_, edge_index_, desc_, flops_) (AggregateKernel){.name=#fn_, .edge_index=(edge_index_), .desc=(desc_), .fn=(fn_), .flops=(flops_)}

static inline size_t range_len(Range r) { return r.end - r.start; }

void coo_v1(size_t node_count, size_t edge_count, size_t in_dim, void *edge_index,
            double *restrict X, size_t ldx, double *restrict Y, size_t ldy)
{
    COO *edges = (COO*)edge_index;

#pragma omp parallel for
    for (size_t i = 0; i < node_count; i++) {
        size_t degree = 0;
        double *y = Y + i * ldy;

        for (size_t j = 0; j < edge_count; j++) {
            uint32_t src = EDGE_SRC(edges, j);
            uint32_t dst = EDGE_DST(edges, j);

            if (i == dst) {
                degree++;
                for (size_t j = 0; j < in_dim; j++) {
                    y[j] += X[src * ldx + j];
                }
            }
        }

        if (degree == 0) continue;
        double scale = 1.0 / degree;

        for (size_t j = 0; j < in_dim; j++) {
            y[j] *= scale;
        }
    }
}

void crs_v1(size_t node_count, size_t edge_count, size_t in_dim, void *edge_index,
            double *restrict X, size_t ldx, double *restrict Y, size_t ldy)
{
    (void)edge_count;
    CRS *edges = (CRS*)edge_index;

#pragma omp parallel for
    for (size_t i = 0; i < node_count; i++) {
        size_t degree = 0;
        double *y = Y + i * ldy;

        for (size_t src = 0; src < node_count; src++) {
            for (size_t j = edges->row_ptr[src]; j < edges->row_ptr[src+1]; j++) {
                uint32_t dst = edges->col_idx[j];
                if (dst == i) {
                    degree++;
                    double *x = X + src * ldx;
                    for (size_t k = 0; k < in_dim; k++) {
                        y[k] += x[k];
                    }
                }
            }
        }

        if (degree == 0) continue;
        double scale = 1.0 / degree;

        for (size_t k = 0; k < in_dim; k++) {
            y[k] *= scale;
        }
    }
}

void ccs_v1(size_t node_count, size_t edge_count, size_t in_dim, void *edge_index,
            double *restrict X, size_t ldx, double *restrict Y, size_t ldy)
{
    (void)edge_count;
    CCS *edges = (CCS*)edge_index;

#pragma omp parallel for
    for (size_t i = 0; i < node_count; i++) {
        uint32_t degree = edges->col_ptr[i+1] - edges->col_ptr[i];

        if (degree == 0) continue;

        double *y = Y + i * ldy;
        double scale = 1.0 / degree;
        for (size_t j = edges->col_ptr[i]; j < edges->col_ptr[i+1]; j++) {
            double *x = X + edges->row_idx[j] * ldx;
            for (size_t k = 0; k < in_dim; k++) {
                y[k] += x[k] * scale;
            }
        }
    }
}

void ccs_v2(size_t node_count, size_t edge_count, size_t in_dim, void *edge_index,
            double *restrict X, size_t ldx, double *restrict Y, size_t ldy)
{
    (void)edge_count;
    CCS *edges = (CCS*)edge_index;
    const size_t K_block = 64;

#pragma omp parallel for
    for (size_t i = 0; i < node_count; i++) {
        uint32_t start = edges->col_ptr[i];
        uint32_t end = edges->col_ptr[i+1];
        uint32_t degree = end - start;
        if (degree == 0) continue;

        double *y = Y + i * ldy;

        for (size_t kb = 0; kb < in_dim; kb += K_block) {
            size_t k_end = MIN(kb+K_block, in_dim);

            for (size_t j = start; j < end; j++) {
                double *x = X + edges->row_idx[j] * ldx;
#pragma omp simd
                for (size_t k = kb; k < k_end; k++) {
                    y[k] += x[k];
                }
            }
        }

        double scale = 1.0 / degree;
#pragma omp simd
        for (size_t k = 0; k < in_dim; k++) {
            y[k] *= scale;
        }
    }
}

void ccs_v3(size_t node_count, size_t edge_count, size_t in_dim, void *edge_index,
            double *restrict X, size_t ldx, double *restrict Y, size_t ldy)
{
    (void)edge_count;
    CCS *edges = (CCS*)edge_index;

#pragma omp parallel for
    for (size_t i = 0; i < node_count; i++) {
        uint32_t start = edges->col_ptr[i];
        uint32_t end = edges->col_ptr[i+1];
        uint32_t degree = end - start;
        if (degree == 0) continue;

        double *y = Y + i * ldy;
        double scale = 1.0 / degree;

        // Process 4 edges at a time
        size_t j = start;
        for (; j + 3 < end; j += 4) {
            double *x0 = X + edges->row_idx[j+0] * ldx;
            double *x1 = X + edges->row_idx[j+1] * ldx;
            double *x2 = X + edges->row_idx[j+2] * ldx;
            double *x3 = X + edges->row_idx[j+3] * ldx;

#pragma omp simd
            for (size_t k = 0; k < in_dim; k++) {
                y[k] += (x0[k] + x1[k] + x2[k] + x3[k]) * scale;
            }
        }

        // Remainder
        for (; j < end; j++) {
            double *x = X + edges->row_idx[j] * ldx;
#pragma omp simd
            for (size_t k = 0; k < in_dim; k++) {
                y[k] += x[k] * scale;
            }
        }
    }
}

void ccs_v4(size_t node_count, size_t edge_count, size_t in_dim, void *edge_index,
            double *restrict X, size_t ldx, double *restrict Y, size_t ldy)
{
    (void)edge_count;
    CCS *edges = (CCS*)edge_index;

#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < node_count; i++) {
        uint32_t start = edges->col_ptr[i];
        uint32_t end = edges->col_ptr[i+1];
        uint32_t degree = end - start;
        if (degree == 0) continue;

        double *y = Y + i * ldy;
        double scale = 1.0 / degree;

        size_t j = start;
        for (; j + 3 < end; j += 4) {
            double *x0 = X + edges->row_idx[j+0] * ldx;
            double *x1 = X + edges->row_idx[j+1] * ldx;
            double *x2 = X + edges->row_idx[j+2] * ldx;
            double *x3 = X + edges->row_idx[j+3] * ldx;

#pragma omp simd
            for (size_t k = 0; k < in_dim; k++) {
                y[k] += (x0[k] + x1[k] + x2[k] + x3[k]) * scale;
            }
        }

        // Remainder
        for (; j < end; j++) {
            double *x = X + edges->row_idx[j] * ldx;
#pragma omp simd
            for (size_t k = 0; k < in_dim; k++) {
                y[k] += x[k] * scale;
            }
        }
    }
}

void coo2crs(size_t node_count, size_t edge_count, COO *coo_edges, CRS *crs_edges)
{
    memset(crs_edges->row_ptr, 0, (node_count+1)*sizeof(*crs_edges->row_ptr));
    for (size_t edge = 0; edge < edge_count; edge++) {
        uint32_t src = EDGE_SRC(coo_edges, edge);
        crs_edges->row_ptr[src+1]++;
    }

    for(size_t i = 1; i < (node_count+1); i++) {
        crs_edges->row_ptr[i] += crs_edges->row_ptr[i-1];
    }

    size_t *row_pos = malloc(node_count * sizeof(*row_pos));
    for (size_t i = 0; i < node_count; i++) {
        row_pos[i] = crs_edges->row_ptr[i];
    }

    for (size_t edge = 0; edge < edge_count; edge++) {
        uint32_t src = EDGE_SRC(coo_edges, edge);
        uint32_t dst = EDGE_DST(coo_edges, edge);
        crs_edges->col_idx[row_pos[src]++] = dst;
    }
    free(row_pos);
}

void coo2ccs(size_t node_count, size_t edge_count, COO *coo_edges, CCS *ccs_edges)
{
    memset(ccs_edges->col_ptr, 0, (node_count+1)*sizeof(*ccs_edges->col_ptr));
    for (size_t edge = 0; edge < edge_count; edge++) {
        uint32_t dst = EDGE_DST(coo_edges, edge);
        ccs_edges->col_ptr[dst+1]++;
    }

    for(size_t i = 1; i < (node_count+1); i++) {
        ccs_edges->col_ptr[i] += ccs_edges->col_ptr[i-1];
    }

    size_t *col_pos = malloc(node_count * sizeof(*col_pos));
    for (size_t i = 0; i < node_count; i++) {
        col_pos[i] = ccs_edges->col_ptr[i];
    }

    for (size_t edge = 0; edge < edge_count; edge++) {
        uint32_t src = EDGE_SRC(coo_edges, edge);
        uint32_t dst = EDGE_DST(coo_edges, edge);
        ccs_edges->row_idx[col_pos[dst]++] = src;
    }
    free(col_pos);
}

int cmp_uint32(const void *a, const void *b) {
    return (*(uint32_t *)a - *(uint32_t *)b);
}

void coo2ccs_sorted(size_t node_count, size_t edge_count, COO *coo_edges, CCS *ccs_edges)
{
    coo2ccs(node_count, edge_count, coo_edges, ccs_edges);

    for (size_t col = 0; col < node_count; col++) {
        uint32_t start = ccs_edges->col_ptr[col];
        uint32_t end = ccs_edges->col_ptr[col+1];
        qsort(&ccs_edges->row_idx[start], end - start, sizeof(uint32_t), cmp_uint32);
    }
}

bool is_valid(size_t M, double *restrict A, double *restrict B)
{
    const double abs_tol = 1e-9;
    const double rel_tol = 1e-6;

    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    for (size_t i = 0; i < M; i++) {
        double a = A[i];
        double b = B[i];

        double abs_diff = fabs(a - b);
        double abs_max = fmax(fabs(a), fabs(b));

        if (abs_diff > abs_tol && abs_diff > rel_tol * abs_max) {
            return false;
        }
    }
    return true;
}

void dzero(size_t M, double *restrict X)
{
    const size_t M_unroll = (M / 4) * 4;
#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < M_unroll; i+=4) {
            X[i+0] = 0;
            X[i+1] = 0;
            X[i+2] = 0;
            X[i+3] = 0;
        }

#pragma omp for
        for (size_t i = M_unroll; i < M; i++) {
            X[i] = 0;
        }
    }
}

void run_benchmark(size_t node_count, size_t edge_count, size_t in_dim,
                   double *restrict X, size_t ldx,
                   double *restrict Y, size_t ldy,
                   AggregateKernel kernel)
{
    bool interactive = isatty(STDOUT_FILENO);

    // Warm up
    if (interactive) {printf("\r\033[K%s: warmup", kernel.name); fflush(stdout);}
    for (size_t i = 0; i < 10; i++) {
        kernel.fn(node_count, edge_count, in_dim, kernel.edge_index, X, ldx, Y, ldy);
        dzero(node_count*in_dim, Y);
        if (interactive) {printf("."); fflush(stdout);}
    }

    double min_time = DBL_MAX;
    uint64_t bytes = 0;
    uint64_t l3_local = 0;
    uint64_t l3_remote = 0;

    if (interactive) {printf("\r\033[K%s: run", kernel.name); fflush(stdout);}
    for (size_t i = 0; i < ntimes; i++) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            cache_counter_start(&thread_counters[tid]);
        }
        double start = omp_get_wtime();
        kernel.fn(node_count, edge_count, in_dim, kernel.edge_index, X, ldx, Y, ldy);
        double elapsed = omp_get_wtime() - start;
        if (elapsed < min_time) {
            min_time = elapsed;
            bytes = 0;
            l3_local = 0;
            l3_remote = 0;
#pragma omp parallel reduction(+:bytes,l3_local,l3_remote)
            {
                int tid = omp_get_thread_num();
                cache_counter_stop(&thread_counters[tid]);

                bytes += cache_counter_get_bytes_loaded(&thread_counters[tid]);
                long long local = 0;
                long long remote = 0;
                cache_counter_get_cache_misses(&thread_counters[tid], &local, &remote);
                l3_local += (uint64_t)local;
                l3_remote += (uint64_t)remote;
            }
        }
        dzero(node_count*in_dim, Y);
        if (interactive) {putchar('.'); fflush(stdout);}
    }

    double bandwidth = bytes / min_time;
    double flops_per_sec = (double) kernel.flops / min_time;
    double intensity = flops_per_sec / bandwidth;
    if (interactive) printf("\r\033[K%s: %s\n", kernel.name, kernel.desc);
    // else printf("%s: %s\n", kernel.name, kernel.desc);
    printf("    %.5fs, %.2f MFlops/s, %.2f MBytes/s, %.2f flop/byte, "
           "%.2f MB(L3-local), %.2f MB(L3-remote)\n",
           min_time, flops_per_sec/1e6, bandwidth/1e6, intensity,
           (double)l3_local/1e6, (double)l3_remote/1e6);

}

size_t count_nodes_with_indegree(size_t node_count, size_t edge_count, EdgeIndex *edges)
{
    size_t indegree_count = 0;
    for (size_t i = 0; i < node_count; i++) {
        for (size_t j = 0; j < edge_count; j++) {
            if (i == EDGE_DST(edges, j)) {
                indegree_count++;
                break;
            }
        }
    }
    return indegree_count;
}

bool kernel_enabled(const char *name)
{
    (void)name;
    return true;
#if 0
    if (kernel_filter.count == 0) return true;

    for (size_t i = 0; i < kernel_filter.count; i++) {
        const char *filter = kernel_filter.items[i];

        // Handle comma-separated values
        if (strchr(filter, ',')) {
            char *copy = strdup(filter);
            char *p = copy, *tok;
            while ((tok = strsep(&p, ","))) {
                if (*tok && strcmp(tok, name) == 0) {
                    free(copy);
                    return true;
                }
            }
            free(copy);
        } else if (strcmp(filter, name) == 0) {
            return true;
        }
    }
    return false;
#endif
}

int main()
{
    int omp_num_threads = omp_get_max_threads();
    thread_counters = malloc(omp_num_threads * sizeof(cache_counter_t));
    if (!thread_counters) {
        fprintf(stderr, "ERROR: Could not allocate thread_counters\n");
    }
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_counters[tid] = cache_counter_init();
    }

    Dataset *data = load_arxiv_dataset();
    const size_t edge_count = data->num_edges;
    const size_t node_count = data->num_inputs;
    const size_t in_dim = data->num_features;
    const size_t indegree_count = count_nodes_with_indegree(node_count, edge_count, &data->edges);

    COO coo_edges = {.data=data->edges.data};

    CRS crs_edges = {
        .row_ptr=malloc((node_count+1)*sizeof(*crs_edges.row_ptr)),
        .col_idx=malloc(edge_count*sizeof(*crs_edges.col_idx)),
    };
    coo2crs(node_count, edge_count, &coo_edges, &crs_edges);

    CCS ccs_edges = {
        .col_ptr=malloc((node_count+1)*sizeof(*ccs_edges.col_ptr)),
        .row_idx=malloc(edge_count*sizeof(*ccs_edges.row_idx)),
    };
    coo2ccs(node_count, edge_count, &coo_edges, &ccs_edges);

    CCS ccs_edges_sorted = {
        .col_ptr=malloc((node_count+1)*sizeof(*ccs_edges.col_ptr)),
        .row_idx=malloc(edge_count*sizeof(*ccs_edges.row_idx)),
    };
    coo2ccs_sorted(node_count, edge_count, &coo_edges, &ccs_edges_sorted);

    printf("Computing reference\n");
    double *ref = malloc(node_count*data->num_features*sizeof(*ref));
    dzero(node_count*data->num_features, ref);
    coo_v1(node_count, edge_count, in_dim, &coo_edges, data->inputs, in_dim, ref, in_dim);

    double *agg = malloc(node_count*data->num_features*sizeof(*agg));
    dzero(node_count*data->num_features, agg);


    AggregateKernel kernels[] = {
        NEW_KERNEL(coo_v1, &coo_edges, "TODO: desc.", (edge_count + indegree_count) * in_dim + indegree_count),
        NEW_KERNEL(crs_v1, &crs_edges, "TODO: desc.", (edge_count + indegree_count) * in_dim + indegree_count),
        NEW_KERNEL(ccs_v1, &ccs_edges, "TODO: desc.", 2ULL * edge_count * in_dim + indegree_count),
        NEW_KERNEL(ccs_v2, &ccs_edges, "TODO: desc.", 2ULL * edge_count * in_dim + indegree_count),
        NEW_KERNEL(ccs_v3, &ccs_edges, "TODO: desc.", (edge_count + indegree_count) * in_dim + indegree_count),
        NEW_KERNEL(ccs_v4, &ccs_edges, "TODO: desc.", (edge_count + indegree_count) * in_dim + indegree_count),
    };

#ifndef SKIP_VALIDATE
    // Validate kernels
    bool any_invalid = false;
    printf("Validate kernels\n");
    for (size_t i = 0; i < ARRAY_LEN(kernels); i++) {
        if (!kernel_enabled(kernels[i].name)) continue;
        kernels[i].fn(node_count, edge_count, in_dim, kernels[i].edge_index, data->inputs, in_dim, agg, in_dim);
        if (!is_valid(node_count*data->num_features, agg, ref)) {
            any_invalid = true;
            fprintf(stderr, "Result from '%s' implementation doesn't match reference\n", kernels[i].name);
        }
        dzero(node_count*data->num_features, agg);
    }
    if (any_invalid) abort();
#endif

    for (size_t i = 0; i < ARRAY_LEN(kernels); i++) {
        if (!kernel_enabled(kernels[i].name)) continue;
        run_benchmark(node_count, edge_count, in_dim, data->inputs, in_dim, agg, in_dim, kernels[i]);
    }
}
