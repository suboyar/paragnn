#include <stdint.h>
#include "layers.h"

static void scale_by_inv_degree_coo(SageLayer *l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;

    const Real *restrict inv_degree;
    if (l->flow == SOURCE_TO_TARGET) // e.g. src (citer) aggregates from dst (cited)
    {
        inv_degree = l->edges.inv_in_degree;
    }
    else // flow == TARGET_TO_SOURCE // e.g. dst (cited) aggregates from src (citer)
    {
        inv_degree = l->edges.inv_out_degree;
    }

    Real *restrict grad_scatter = l->grad_scatter;
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        Real scale   = inv_degree[i];
        Real *gs_row = &grad_scatter[i * in_dim];
#pragma omp simd
        for (int64_t j = 0; j < in_dim; j++)
        {
            gs_row[j] *= scale;
        }
    }
}

static void scale_by_inv_degree_csx(SageLayer *l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict ptr;
    if (l->flow == SOURCE_TO_TARGET)
    {
        ptr = l->edges.ptr_csc;
    }
    else
    {
        ptr = l->edges.ptr_csr;
    }

    Real *restrict grad_scatter = l->grad_scatter;
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        Real scale = 0.0;
        int64_t degree = ptr[i+1] - ptr[i];
        if (degree != 0) scale = REAL(1.0) / degree;
        Real *gs_row = &grad_scatter[i * in_dim];
#pragma omp simd
        for (int64_t j = 0; j < in_dim; j++)
        {
            gs_row[j] *= scale;
        }
    }
}

static void scatter_coo(SageLayer *l)
{
    int64_t num_edges = l->num_edges;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict nodes, *restrict peers;
    if (l->flow == SOURCE_TO_TARGET) // e.g. src (citer) aggregates from dst (cited)
    {
        nodes = l->edges.dst;
        peers = l->edges.src;
    }
    else // flow == TARGET_TO_SOURCE // e.g. dst (cited) aggregates from src (citer)
    {
        nodes = l->edges.src;
        peers = l->edges.dst;
    }

    const Real *restrict grad_scatter = l->grad_scatter;
    Real       *restrict grad_input   = l->grad_input;

#pragma omp parallel for
    for (int64_t e = 0; e < num_edges; e++)
    {
        const Real *gs_row = &grad_scatter[nodes[e] * in_dim];
        Real       *gi_row = &grad_input[peers[e] * in_dim];

        for (int64_t i = 0; i < in_dim; i++)
        {
#pragma omp atomic
            gi_row[i] += gs_row[i];
        }
    }
}

void scatter_csx(SageLayer *l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict ptr, *restrict idx;
    if (l->flow == SOURCE_TO_TARGET)
    {
        ptr = l->edges.ptr_csr;
        idx = l->edges.idx_csr;
    }
    else // flow == TARGET_TO_SOURCE
    {
        ptr = l->edges.ptr_csc;
        idx = l->edges.idx_csc;
    }

    const Real *restrict grad_scatter = l->grad_scatter;
    Real       *restrict grad_input   = l->grad_input;

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        Real *gi_row = &grad_input[i * in_dim];
        for (int64_t j = ptr[i]; j < ptr[i+1]; j++)
        {
            const Real *gs_row = &grad_scatter[idx[j] * in_dim];
            for (int64_t k = 0; k < l->in_dim; k++)
            {
                gi_row[k] += gs_row[k];
            }
        }
    }
}

void grad_mean_aggregate(SageLayer *l)
{
    if (l->edges.format == EDGE_COO)
    {
        scale_by_inv_degree_coo(l);
        scatter_coo(l);
    }
    else // format == EDGE_CSX
    {
        scale_by_inv_degree_csx(l);
        scatter_csx(l);
    }
}
