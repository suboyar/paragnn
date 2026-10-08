#include <stdint.h>

#include "core.h"
#include "flow.h"
#include "sparsegraph.h"

void coo_v1_touch(Real *restrict x_in, Real *restrict x_neigh, int64_t in_dim, SparseGraph *graph, FlowDirection flow)
{
    int64_t edge_count = graph->edge_count;
    int64_t node_count = graph->node_count;

    int64_t *restrict roots, *restrict neighbors;
    if (flow == SOURCE_TO_TARGET)
    {
        roots = graph->dst;
        neighbors = graph->src;
    }
    else // flow == TARGET_TO_SOURCE
    {
        roots = graph->src;
        neighbors = graph->dst;
    }

#pragma omp parallel for
    for (size_t e = 0; e < edge_count; e++)
    {
        int64_t root = roots[e];
        int64_t neighbor = neighbors[e];

        Real *x_in_ptr = &x_in[neighbor * in_dim];
        memset(x_in_ptr, 0, in_dim * sizeof(*x_in_ptr));
        Real *x_neigh_ptr = &x_neigh[root * in_dim];
        memset(x_neigh_ptr, 0, in_dim * sizeof(*x_neigh_ptr));
    }

#pragma omp parallel for
    for (size_t n = 0; n < node_count; n++)
    {
        int64_t root = roots[n];
        int64_t neighbor = neighbors[n];

        Real *x_in_ptr = &x_in[neighbor * in_dim];
        memset(x_in_ptr, 0, in_dim * sizeof(*x_in_ptr));
        Real *x_neigh_ptr = &x_neigh[root * in_dim];
        memset(x_neigh_ptr, 0, in_dim * sizeof(*x_neigh_ptr));
    }
}

void coo_v1(const Real *restrict x_in, Real *restrict x_neigh, int64_t in_dim, SparseGraph *graph, FlowDirection flow)
{
    int64_t edge_count = graph->edge_count;

    int64_t *restrict roots, *restrict neighbors;
    const Real *restrict inv_degree;
    if (flow == SOURCE_TO_TARGET)
    {
        roots = graph->dst;
        neighbors = graph->src;
        inv_degree = graph->inv_in_degree;
    }
    else // flow == TARGET_TO_SOURCE
    {
        roots = graph->src;
        neighbors = graph->dst;
        inv_degree = graph->inv_out_degree;
    }

#pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < edge_count; e++)
    {
        int64_t root = roots[e];
        int64_t neighbor = neighbors[e];
        const Real scale = inv_degree[root];

        const Real *x_in_ptr = &x_in[neighbor * in_dim];
        Real *x_neigh_ptr = &x_neigh[root * in_dim];

        for (int64_t d = 0; d < in_dim; d++)
        {
            Real scaled_val = x_in_ptr[d] * scale;
#pragma omp atomic
            x_neigh_ptr[d] += scaled_val;
        }
    }
}
