#include <stdint.h>
#include "layers.h"

void scale_by_inv_degree(SageLayer *l)
{
#pragma omp for
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        Real s = l->edges.inv_in_degree[i]; // Assumes SOURCE_TO_TARGET
        Real *restrict gs = &l->grad_scatter[i * l->in_dim];
#pragma omp simd
        for (size_t j = 0; j < l->in_dim; j++)
        {
            gs[j] *= s;
        }
    }
}

void scatter_coo(uint32_t *nodes, uint32_t *peers, SageLayer *l)
{
#pragma omp for
    for (size_t e = 0; e < l->num_edges; e++)
    {
        Real *restrict gi       = &l->grad_input[peers[e] * l->in_dim];
        const Real *restrict gs = &l->grad_scatter[nodes[e] * l->in_dim];

        for (size_t i = 0; i < l->in_dim; i++)
        {
#pragma omp atomic
            gi[i] += gs[i];
        }
    }
}

void scale_by_inv_degree_parallel(SageLayer *l)
{
#pragma omp parallel
    scale_by_inv_degree(l);
}

void scatter_coo_parallel(uint32_t *nodes, uint32_t *peers, SageLayer *l)
{
#pragma omp parallel
    scatter_coo(nodes, peers, l);
}
