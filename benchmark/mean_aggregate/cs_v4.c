#include <stdint.h>

#include <omp.h>

#include "core.h"
#include "flow.h"
#include "sparsegraph.h"
#include "vreg.h"


void cs_v4_touch(Real *restrict x_in, Real *restrict x_neigh, int64_t in_dim, SparseGraph *graph)
{
    int64_t node_count = graph->node_count;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        memset(&x_in[n * in_dim], 0, in_dim * sizeof(Real));
        memset(&x_neigh[n * in_dim], 0, in_dim * sizeof(Real));
    }
}

#define UNROLL_FACTOR (NUM_REGS / 2)
#define REM_FACTOR    (32 / N_VEC)

static inline __attribute__((always_inline)) void microkernel_spmm(
    int nv_tile,
    int64_t start, int64_t end,
    const int64_t *restrict indices,
    const Real *restrict x_in, int64_t in_dim,
    Real *restrict out_row,
    VReal vscale,
    bool use_stream);

void cs_v4(const Real *restrict x_in, Real *restrict x_neigh, int64_t in_dim,
           SparseGraph *graph, FlowDirection flow)
{
    int64_t node_count = graph->node_count;

    const int64_t *restrict indptr  = (flow == SOURCE_TO_TARGET) ? graph->ptr_csc : graph->ptr_csr;
    const int64_t *restrict indices = (flow == SOURCE_TO_TARGET) ? graph->idx_csc : graph->idx_csr;
#if defined(CACHE_INV_DEGREE)
    const Real *restrict inv_degree = (flow == SOURCE_TO_TARGET) ? graph->inv_in_degree : graph->inv_out_degree;
#endif

    if (LIKELY(in_dim == 256))
    {
#pragma omp parallel for schedule(dynamic, 64)
        for (int64_t n = 0; n < node_count; n++)
        {
            int64_t start = indptr[n];
            int64_t end   = indptr[n + 1];
            int64_t degree = end - start;
            Real *restrict out_row = &x_neigh[n * in_dim];

            if (UNLIKELY(degree == 0)) continue;

#if defined(CACHE_INV_DEGREE)
            Real scale = inv_degree[n];
#else
            Real scale = REAL(1.0) / degree;
#endif
            VReal vscale = vrbcast(scale);

            PRAGMA_UNROLL(4)
            for (int64_t j = 0; j < 256; j += UNROLL_FACTOR * N_VEC)
            {
                microkernel_spmm(UNROLL_FACTOR, start, end, indices,
                                 &x_in[j], 256, &out_row[j], vscale, true);
            }

        }
        return;
    }
    else
    {
#pragma omp parallel for schedule(dynamic, 64)
        for (int64_t n = 0; n < node_count; n++)
        {
            int64_t start = indptr[n];
            int64_t end   = indptr[n + 1];
            int64_t degree = end - start;
            Real *restrict out_row = &x_neigh[n * in_dim];

            if (UNLIKELY(degree == 0)) continue;

#if defined(CACHE_INV_DEGREE)
            Real scale = inv_degree[n];
#else
            Real scale = REAL(1.0) / degree;
#endif
            VReal vscale = vrbcast(scale);

            int64_t j = 0;

            // Full register-tile chunks
            for (; j + UNROLL_FACTOR * N_VEC <= in_dim; j += UNROLL_FACTOR * N_VEC)
            {
                microkernel_spmm(UNROLL_FACTOR, start, end, indices,
                                 &x_in[j], in_dim, &out_row[j], vscale, false);
            }

            // Medium vector-tile chunks
            for (; j + REM_FACTOR * N_VEC <= in_dim; j += REM_FACTOR * N_VEC)
            {
                microkernel_spmm(REM_FACTOR, start, end, indices,
                                 &x_in[j], in_dim, &out_row[j], vscale, false);
            }

            // Single-vector chunks
            for (; j + N_VEC <= in_dim; j += N_VEC)
            {
                microkernel_spmm(1, start, end, indices,
                                 &x_in[j], in_dim, &out_row[j], vscale, false);
            }

            // Scalar fringe
            for (; j < in_dim; j++)
            {
                Real sum_scalar = (Real)0.0;
                for (int64_t k = start; k < end; k++)
                    sum_scalar += x_in[indices[k] * in_dim + j];
                out_row[j] = sum_scalar * scale;
            }
        }
    }
}

static inline __attribute__((always_inline)) void microkernel_spmm(
    int nv_tile,
    int64_t start, int64_t end,
    const int64_t *restrict indices,
    const Real *restrict x_in, int64_t in_dim,
    Real *restrict out_row,
    VReal vscale,
    bool use_stream)
{
    VReal sum[UNROLL_FACTOR];

    PRAGMA_UNROLL(UNROLL_FACTOR)
    for (int iv = 0; iv < nv_tile; iv++)
        sum[iv] = vrbcast((Real)0.0);

    int64_t k = start;
    for (; k + 3 < end; k += 4)
    {
        const Real *p0 = &x_in[indices[k + 0] * in_dim];
        const Real *p1 = &x_in[indices[k + 1] * in_dim];
        const Real *p2 = &x_in[indices[k + 2] * in_dim];
        const Real *p3 = &x_in[indices[k + 3] * in_dim];

        PRAGMA_UNROLL(UNROLL_FACTOR)
        for (int iv = 0; iv < nv_tile; iv++)
        {
            VReal v01 = vrload_u(p0 + iv * N_VEC) + vrload_u(p1 + iv * N_VEC);
            VReal v23 = vrload_u(p2 + iv * N_VEC) + vrload_u(p3 + iv * N_VEC);
            sum[iv] += v01 + v23;
        }
    }

    for (; k < end; k++)
    {
        const Real *p0 = &x_in[indices[k] * in_dim];

        PRAGMA_UNROLL(UNROLL_FACTOR)
        for (int iv = 0; iv < nv_tile; iv++)
            sum[iv] += vrload_u(p0 + iv * N_VEC);
    }

    if (use_stream)
    {
        PRAGMA_UNROLL(UNROLL_FACTOR)
        for (int iv = 0; iv < nv_tile; iv++)
            stream_vrstore(out_row + iv * N_VEC, sum[iv] * vscale);
    }
    else
    {
        PRAGMA_UNROLL(UNROLL_FACTOR)
        for (int iv = 0; iv < nv_tile; iv++)
            vrstore_u(out_row + iv * N_VEC, sum[iv] * vscale);
    }
}

#undef UNROLL_FACTOR
#undef REM_FACTOR
