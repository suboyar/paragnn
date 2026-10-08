#include <stdint.h>

#include "core.h"
#include "flow.h"
#include "sparsegraph.h"

void cs_v2_touch(Real *restrict x_in, Real *restrict x_neigh, int64_t in_dim, SparseGraph *graph)
{
    int64_t node_count = graph->node_count;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        memset(&x_in[n * in_dim], 0, in_dim * sizeof(Real));
        memset(&x_neigh[n * in_dim], 0, in_dim * sizeof(Real));
    }
}

void cs_v2(const Real *restrict x_in, Real *restrict x_neigh, int64_t in_dim, SparseGraph *graph, FlowDirection flow)
{
    int64_t node_count = graph->node_count;

    const int64_t *restrict indptr, *restrict indices;
    const Real *restrict inv_degree;
    if (flow == SOURCE_TO_TARGET)
    {
        indptr = graph->ptr_csc;
        indices = graph->idx_csc;
        inv_degree = graph->inv_in_degree;
    }
    else // flow == TARGET_TO_SOURCE
    {
        indptr = graph->ptr_csr;
        indices = graph->idx_csr;
        inv_degree = graph->inv_out_degree;
    }

#if !defined(CACHE_INV_DEGREE)
    (void)inv_degree;
#endif

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        int64_t start = indptr[n];
        int64_t end = indptr[n+1];
        int64_t degree = end - start;
        if (degree == 0) continue;

        Real *x_neigh_ptr = &x_neigh[n*in_dim];

#if defined(CACHE_INV_DEGREE)
        Real scale = inv_degree[n];
#else
        Real scale = REAL(1.0) / degree;
#endif

        int64_t e = start;
        for (; e + 3 < end; e += 4)
        {
            const Real *x_in_ptr0 = &x_in[indices[e+0] * in_dim];
            const Real *x_in_ptr1 = &x_in[indices[e+1] * in_dim];
            const Real *x_in_ptr2 = &x_in[indices[e+2] * in_dim];
            const Real *x_in_ptr3 = &x_in[indices[e+3] * in_dim];

#pragma omp simd
            for (int64_t d = 0; d < in_dim; d++)
                x_neigh_ptr[d] += (x_in_ptr0[d] + x_in_ptr1[d] + x_in_ptr2[d] + x_in_ptr3[d]) * scale;
        }

        // Remainder
        for (; e < end; e++)
        {
            const Real *x_in_ptr = &x_in[indices[e] * in_dim];
#pragma omp simd
            for (int64_t d = 0; d < in_dim; d++)
                x_neigh_ptr[d] += x_in_ptr[d] * scale;
        }
    }
}
