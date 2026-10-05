#include "sparsegraph.h"

#include <errno.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "core.h"

static void alloc_coo(SparseGraph *graph);
static void free_coo(SparseGraph *graph);
static void load_coo(SparseGraph *graph);
static void ideg_coo(const int64_t *endpoints, Real *inv_degree, SparseGraph *graph);

static void alloc_cs(SparseGraph *graph);
static void free_cs(SparseGraph *graph);
static void load_cs(SparseGraph *graph);
static void ideg_cs(const int64_t *indptr, Real *inv_degree, SparseGraph *graph);

SparseGraph* sparsegraph_load(int64_t node_count, int64_t edge_count, bool is_undirected, const char *edge_path, SparseFormat format)
{
    SparseGraph *graph = ALLOC_OR_DIE(calloc(1, sizeof(*graph)));

    graph->node_count     = node_count;
    graph->edge_count     = edge_count;
    graph->format         = format;
    graph->undirected     = is_undirected;
    graph->edge_path      = ALLOC_OR_DIE(strdup(edge_path));
    graph->inv_in_degree  = ALLOC_OR_DIE(alloc_local(node_count * sizeof(Real)));
    graph->inv_out_degree = is_undirected ? graph->inv_in_degree
                                          : ALLOC_OR_DIE(alloc_local(node_count * sizeof(Real)));

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        graph->inv_in_degree[n] = REAL(0.0);
        if (!is_undirected)
            graph->inv_out_degree[n] = REAL(0.0);
    }

    bool has_cs  = format & SPARSE_FORMAT_CS;
    bool has_coo = format & SPARSE_FORMAT_COO;
    if (has_cs)
    {
        alloc_cs(graph);
        load_cs(graph);
    }
    if (has_coo)
    {
        alloc_coo(graph);
        load_coo(graph);
    }

   // Prioritize ideg for CSR/CSC over COO, as it's much faster
   if (has_cs)
   {
       ideg_cs(graph->ptr_csr, graph->inv_out_degree, graph);
       if (!graph->undirected) ideg_cs(graph->ptr_csc, graph->inv_in_degree, graph);
   }
   else if (has_coo)
   {
       ideg_coo(graph->src, graph->inv_out_degree, graph);
       if (!graph->undirected) ideg_coo(graph->dst, graph->inv_in_degree, graph);
   }

    return graph;
}

void sparsegraph_free(SparseGraph **graph)
{
    if (!*graph) return;

    bool has_cs  = (*graph)->format & SPARSE_FORMAT_CS;
    bool has_coo = (*graph)->format & SPARSE_FORMAT_COO;

    if (has_cs)  free_cs(*graph);
    if (has_coo) free_coo(*graph);

    free((*graph)->inv_in_degree);
    if (!(*graph)->undirected) free((*graph)->inv_out_degree);
    free((*graph)->edge_path);
    free(*graph);
    *graph = NULL;
}

static void alloc_coo(SparseGraph *graph)
{
    graph->src = ALLOC_OR_DIE(alloc_shared(graph->edge_count * sizeof(*graph->src)));
    graph->dst = ALLOC_OR_DIE(alloc_shared(graph->edge_count * sizeof(*graph->dst)));
}

static void free_coo(SparseGraph *graph)
{
    if (!graph) return;
    free(graph->src); graph->src = NULL;
    free(graph->dst); graph->dst = NULL;
}

static void load_coo(SparseGraph *graph)
{
    MmapInfo info = map_file(graph->edge_path, PROT_READ, MAP_PRIVATE | MAP_POPULATE);
    int64_t *data = (int64_t*)info.data;

    int64_t *src = data;
    int64_t *dst = data + graph->edge_count;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < graph->edge_count; i++)
    {
        graph->src[i] = src[i];
        graph->dst[i] = dst[i];
    }
    unmap_file(&info);
}

static void ideg_coo(const int64_t *endpoints, Real *inv_degree, SparseGraph *graph)
{
    int64_t node_count = graph->node_count;
    int64_t edge_count = graph->edge_count;
    int64_t *degree_count = ALLOC_OR_DIE(alloc_shared(node_count * sizeof(*degree_count)));
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < node_count; i++)
    {
        degree_count[i] = 0;
        inv_degree[i] = REAL(0.0);
    }

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < edge_count; i++)
    {
#pragma omp atomic
        degree_count[endpoints[i]] += 1;
    }

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < node_count; i++)
    {
        inv_degree[i] = REAL(1.0) / degree_count[i];
    }

    free(degree_count);
}

static void alloc_cs(SparseGraph *graph)
{
    graph->ptr_csr = ALLOC_OR_DIE(alloc_local((graph->node_count+1) * sizeof(*graph->ptr_csc)));
    graph->idx_csr = ALLOC_OR_DIE(alloc_shared(graph->edge_count * sizeof(*graph->idx_csr)));
    graph->ptr_csc = ALLOC_OR_DIE(alloc_local((graph->node_count+1) * sizeof(*graph->ptr_csc)));
    graph->idx_csc = ALLOC_OR_DIE(alloc_shared(graph->edge_count * sizeof(*graph->idx_csc)));
}

static void free_cs(SparseGraph *graph)
{
    if (!graph) return;
    free(graph->ptr_csr); graph->ptr_csr = NULL;
    free(graph->idx_csr); graph->idx_csr = NULL;
    free(graph->ptr_csc); graph->ptr_csc = NULL;
    free(graph->idx_csc); graph->idx_csc = NULL;
}

static void load_cs(SparseGraph *graph)
{
    int64_t *row_pos = ALLOC_OR_DIE(malloc(graph->node_count * sizeof(*row_pos)));
    int64_t *col_pos = ALLOC_OR_DIE(malloc(graph->node_count * sizeof(*col_pos)));

    MmapInfo info = map_file(graph->edge_path, PROT_READ, MAP_PRIVATE | MAP_POPULATE);
    int64_t *data = (int64_t*)info.data;

    int64_t *src = data;
    int64_t *dst = data + graph->edge_count;
    int64_t sum_csr = 0, sum_csc = 0;
#pragma omp parallel
    {
#pragma omp for
        for (int64_t i = 0; i < (graph->node_count+1); i++)
        {
            graph->ptr_csr[i] = 0;
            graph->ptr_csc[i] = 0;
        }

#pragma omp for
        for (int64_t i = 0; i < graph->edge_count; i++)
        {
#pragma omp atomic
            graph->ptr_csr[src[i]+1]++;
#pragma omp atomic
            graph->ptr_csc[dst[i]+1]++;
        }

#pragma omp for simd reduction(inscan, +:sum_csr, sum_csc)
        for(int64_t i = 1; i <= graph->node_count; i++)
        {
            sum_csr += graph->ptr_csr[i];
            sum_csc += graph->ptr_csc[i];

#pragma omp scan inclusive(sum_csr, sum_csc)
            graph->ptr_csr[i] = sum_csr;
            graph->ptr_csc[i] = sum_csc;
        }

#pragma omp for
        for (int64_t i = 0; i < graph->node_count; i++)
        {
            row_pos[i] = graph->ptr_csr[i];
            col_pos[i] = graph->ptr_csc[i];
        }

#pragma omp for
        for (int64_t i = 0; i < graph->edge_count; i++)
        {
            int64_t r_idx, c_idx;
#pragma omp atomic capture
            r_idx = row_pos[src[i]]++;
#pragma omp atomic capture
            c_idx = col_pos[dst[i]]++;

            graph->idx_csr[r_idx] = dst[i];
            graph->idx_csc[c_idx] = src[i];
        }
    }

    free(row_pos);
    free(col_pos);
    unmap_file(&info);
}

static void ideg_cs(const int64_t *indptr, Real *inv_degree, SparseGraph *graph)
{
    int64_t node_count = graph->node_count;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < node_count; i++)
    {
        inv_degree[i] = REAL(1.0) / (indptr[i+1] - indptr[i]);
    }
}
