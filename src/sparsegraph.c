#include "sparsegraph.h"

#include <errno.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "core.h"

static void sparsegraph_alloc_coo(SparseGraph *graph)
{
    graph->src = ALLOC_OR_DIE(cache_aligned_alloc(graph->edge_count * sizeof(*graph->src)));
    graph->dst = ALLOC_OR_DIE(cache_aligned_alloc(graph->edge_count * sizeof(*graph->dst)));
}

static void sparsegraph_free_coo(SparseGraph *graph)
{
    if (!graph) return;
    free(graph->src);
    free(graph->dst);
}

static void sparsegraph_alloc_cs(SparseGraph *graph)
{
    graph->ptr_csr = ALLOC_OR_DIE(cache_aligned_alloc((graph->node_count+1) * sizeof(*graph->ptr_csc)));
    graph->idx_csr = ALLOC_OR_DIE(cache_aligned_alloc(graph->edge_count * sizeof(*graph->idx_csr)));
    graph->ptr_csc = ALLOC_OR_DIE(cache_aligned_alloc((graph->node_count+1) * sizeof(*graph->ptr_csc)));
    graph->idx_csc = ALLOC_OR_DIE(cache_aligned_alloc(graph->edge_count * sizeof(*graph->idx_csc)));
}

static void sparsegraph_free_cs(SparseGraph *graph)
{
    if (!graph) return;
    free(graph->ptr_csr);
    free(graph->idx_csr);
    free(graph->ptr_csc);
    free(graph->idx_csc);
}

static void sparsegraph_load_coo(SparseGraph *graph)
{
    MmapInfo info = map_file(graph->edge_path, PROT_READ, MAP_PRIVATE | MAP_POPULATE);
    int64_t *data = (int64_t*)info.data;
    size_t count = info.bytes / sizeof(*data);

    int64_t *src = data;
    int64_t *dst = data + graph->edge_count;
#pragma omp parallel for
    for (int64_t i = 0; i < graph->edge_count; i++)
    {
        graph->src[i] = src[i];
        graph->dst[i] = dst[i];
    }
    unmap_file(&info);
}

static void sparsegraph_load_cs(SparseGraph *graph)
{
    int64_t *row_pos = ALLOC_OR_DIE(malloc(graph->node_count * sizeof(*row_pos)));
    int64_t *col_pos = ALLOC_OR_DIE(malloc(graph->node_count * sizeof(*col_pos)));

    MmapInfo info = map_file(graph->edge_path, PROT_READ, MAP_PRIVATE | MAP_POPULATE);
    int64_t *data = (int64_t*)info.data;
    size_t count = info.bytes / sizeof(*data);

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
            #pragma omp scan exclusive(sum_csr, sum_csc)
            graph->ptr_csr[i] += sum_csr;

            sum_csc += graph->ptr_csc[i];
            graph->ptr_csc[i] += sum_csc;
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

SparseGraph* sparsegraph_alloc(int64_t node_count, int64_t edge_count, bool is_undirected, const char *edge_path, SparseFormat initial_format)
{
    SparseGraph *graph = ALLOC_OR_DIE(calloc(1, sizeof(*graph)));

    graph->node_count = node_count;
    graph->edge_count = edge_count;
    graph->undirected = is_undirected;
    graph->edge_path = ALLOC_OR_DIE(strdup(edge_path));
    graph->inv_in_degree = ALLOC_OR_DIE(cache_aligned_alloc(node_count * sizeof(*graph->inv_in_degree)));
    graph->inv_out_degree = is_undirected ? graph->inv_in_degree
                                          : ALLOC_OR_DIE(cache_aligned_alloc(node_count * sizeof(*graph->inv_out_degree)));
    sparsegraph_update_format_alloc(graph, initial_format);

    return graph;
}

void sparsegraph_load(SparseGraph *graph)
{
    if (graph->format & SPARSE_COO) sparsegraph_load_coo(graph);
    if (graph->format & SPARSE_CS)  sparsegraph_load_cs(graph);
}

void sparsegraph_update_format_alloc(SparseGraph *graph, SparseFormat format)
{
    SparseFormat to_add  = format & ~graph->format;
    SparseFormat to_free = graph->format & ~format;

    if (to_add & SPARSE_CS)   sparsegraph_alloc_cs(graph);
    if (to_add & SPARSE_COO)  sparsegraph_alloc_coo(graph);
    if (to_free & SPARSE_COO) sparsegraph_free_coo(graph);
    if (to_free & SPARSE_CS)  sparsegraph_free_cs(graph);

    graph->format = format;
}

void sparsegraph_free(SparseGraph **graph)
{
    if (!*graph) return;
    free((*graph)->inv_in_degree);
    free((*graph)->inv_out_degree);
    free((*graph)->src);
    free((*graph)->dst);
    free((*graph)->ptr_csr);
    free((*graph)->idx_csr);
    free((*graph)->ptr_csc);
    free((*graph)->idx_csc);
    free(*graph);
    *graph = NULL;
}
