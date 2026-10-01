#include <stdint.h>
#include <string.h>

#include "core.h"
#include "layers.h"
#include "matmul_naive.h"
#include "timer.h"

#if defined(SAGECONV_NAIVE_IMPL)
#  define GEMM_NN(M,N,K,a,A,lda,B,ldb,b,C,ldc) matmul(MatmulNoTrans, MatmulNoTrans, (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_NT(M,N,K,a,A,lda,B,ldb,b,C,ldc) matmul(MatmulNoTrans, MatmulTrans,   (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_TN(M,N,K,a,A,lda,B,ldb,b,C,ldc) matmul(MatmulTrans,   MatmulNoTrans, (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#elif defined(SAGECONV_TUNED_IMPL)
#  define GEMM_NN(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_NT(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_TN(M,N,K,a,A,lda,B,ldb,b,C,ldc) outer_tn((M),(N),(K),(A),(lda),(B),(ldb),(C),(ldc)) // assumes alpha=1.0, beta=0.0
#else // SAGECONV_BLAS_IMPL
#  define GEMM_NN(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_NT(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_TN(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#endif

static void sage_mean_aggregate_coo(SageLayer *l)
{
    TIMER_FUNC();

    int64_t  node_count  = l->node_count;
    int64_t  edge_count  = l->edge_count;
    int64_t  in_dim     = l->in_dim;

    const int64_t *restrict nodes, *restrict peers;
    const Real *restrict inv_degree;
    if (l->flow == SOURCE_TO_TARGET) // e.g. src (citer) aggregates from dst (cited)
    {
        nodes = l->graph->dst;
        peers = l->graph->src;
        inv_degree = l->graph->inv_in_degree;
    }
    else // flow == TARGET_TO_SOURCE // e.g. dst (cited) aggregates from src (citer)
    {
        nodes = l->graph->src;
        peers = l->graph->dst;
        inv_degree = l->graph->inv_out_degree;
    }

    const Real *restrict x_in = l->x_in;
    Real       *restrict x_neigh = l->x_neigh;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real *x_neigh_ptr = &x_neigh[n*in_dim];
        memset(x_neigh_ptr, 0, in_dim * sizeof(*x_neigh_ptr));

        Real scale = inv_degree[n];
        if (scale == REAL(0.0)) continue;

        for (int64_t e = 0; e < edge_count; e++)
        {
            if (n == nodes[e])
            {
                const Real *x_in_ptr = &x_in[peers[e]*in_dim];
#pragma omp simd
                for (int64_t d = 0; d < in_dim; d++)
                    x_neigh_ptr[d] += x_in_ptr[d] * scale;
            }
        }
    }
}

static void sage_mean_aggregate_cs(SageLayer *l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict ptr, *restrict idx;
    if (l->flow == SOURCE_TO_TARGET)
    {
        ptr = l->graph->ptr_csc;
        idx = l->graph->idx_csc;
    }
    else // flow == TARGET_TO_SOURCE
    {
        ptr = l->graph->ptr_csr;
        idx = l->graph->idx_csr;
    }

    const Real *restrict x_in = l->x_in;
    Real       *restrict x_neigh   = l->x_neigh;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real *x_neigh_ptr = &x_neigh[n*in_dim];
        memset(x_neigh_ptr, 0, in_dim * sizeof(*x_neigh_ptr));

        // TODO: compare it with using inv_degree directly
        int64_t degree = ptr[n+1] - ptr[n];
        if (degree == 0) continue; // since we memset x_neigh_ptr with 0 by default we can just skip here
        Real scale = (Real)1.0 / degree;
        for (int64_t e = ptr[n]; e < ptr[n+1]; e++)
        {
            const Real *x_in_ptr = &x_in[idx[e]*in_dim];
            for (int64_t d = 0; d < in_dim; d++)
                x_neigh_ptr[d] += x_in_ptr[d] * scale;
        }
    }
}

void sageconv(SageLayer *const l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t in_dim    = l->in_dim;
    int64_t out_dim   = l->out_dim;
    int64_t W_stride  = l->W_stride;

    // x_out = x_in @ W_self
    TIMER_BLOCK("x_out_self", {
            GEMM_NN(node_count, out_dim, in_dim,
                    1.0,
                    l->x_in,   in_dim,
                    l->W_self, W_stride,
                    0.0,
                    l->x_out,  out_dim);
        });

#if defined(SPARSE_COO)
        sage_mean_aggregate_coo(l);
#else
        sage_mean_aggregate_cs(l);
#endif

    // x_out += x_neigh @ W_neighagg
    TIMER_BLOCK("x_out_neigh", {
            GEMM_NN(node_count, out_dim, in_dim,
                    1.0,
                    l->x_neigh, in_dim,
                    l->W_neigh, W_stride,
                    1.0,
                    l->x_out,   out_dim);
        });
}

static void scale_by_inv_degree_coo(SageLayer *l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t in_dim    = l->in_dim;

    const Real *restrict inv_degree;
    if (l->flow == SOURCE_TO_TARGET) // e.g. src (citer) aggregates from dst (cited)
    {
        inv_degree = l->graph->inv_in_degree;
    }
    else // flow == TARGET_TO_SOURCE // e.g. dst (cited) aggregates from src (citer)
    {
        inv_degree = l->graph->inv_out_degree;
    }

    Real *restrict dx_scatter = l->dx_scatter;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        const Real scale     = inv_degree[n];
        Real *dx_scatter_ptr = &dx_scatter[n*in_dim];
#pragma omp simd
        for (int64_t d = 0; d < in_dim; d++)
            dx_scatter_ptr[d] *= scale;
    }
}

static void scale_by_inv_degree_csx(SageLayer *l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict ptr;
    if (l->flow == SOURCE_TO_TARGET)
    {
        ptr = l->graph->ptr_csc;
    }
    else
    {
        ptr = l->graph->ptr_csr;
    }

    Real *restrict dx_scatter = l->dx_scatter;
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real scale = 0.0;
        int64_t degree = ptr[n+1] - ptr[n];
        if (degree != 0) scale = REAL(1.0) / degree;
        Real *dx_scatter_ptr = &dx_scatter[n*in_dim];
#pragma omp simd
        for (int64_t d = 0; d < in_dim; d++)
            dx_scatter_ptr[d] *= scale;
    }
}

static void scatter_coo(SageLayer *l)
{
    TIMER_FUNC();

    int64_t edge_count = l->edge_count;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict nodes, *restrict peers;
    if (l->flow == SOURCE_TO_TARGET) // e.g. src (citer) aggregates from dst (cited)
    {
        nodes = l->graph->dst;
        peers = l->graph->src;
    }
    else // flow == TARGET_TO_SOURCE // e.g. dst (cited) aggregates from src (citer)
    {
        nodes = l->graph->src;
        peers = l->graph->dst;
    }

    const Real *restrict dx_scatter = l->dx_scatter;
    Real       *restrict dx_in      = l->dx_in;

#pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < edge_count; e++)
    {
        const Real *dx_scatter_ptr = &dx_scatter[nodes[e]*in_dim];
        Real       *dx_in_ptr      = &dx_in[peers[e]*in_dim];

        for (int64_t i = 0; i < in_dim; i++)
        {
#pragma omp atomic
            dx_in_ptr[i] += dx_scatter_ptr[i];
        }
    }
}

static void scatter_csx(SageLayer *l)
{
    TIMER_FUNC();

    int64_t node_count = l->node_count;
    int64_t in_dim    = l->in_dim;

    const int64_t *restrict ptr, *restrict idx;
    if (l->flow == SOURCE_TO_TARGET)
    {
        ptr = l->graph->ptr_csr;
        idx = l->graph->idx_csr;
    }
    else // flow == TARGET_TO_SOURCE
    {
        ptr = l->graph->ptr_csc;
        idx = l->graph->idx_csc;
    }

    const Real *restrict dx_scatter = l->dx_scatter;
    Real       *restrict dx_in   = l->dx_in;

#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < node_count; n++)
    {
        Real *dx_in_ptr = &dx_in[n*in_dim];
        for (int64_t e = ptr[n]; e < ptr[n+1]; e++)
        {
            const Real *dx_scatter_ptr = &dx_scatter[idx[e]*in_dim];
            for (int64_t d = 0; d < l->in_dim; d++)
                dx_in_ptr[d] += dx_scatter_ptr[d];
        }
    }
}

void grad_sageconv(SageLayer *l)
{
    TIMER_FUNC();

    // dx_Wroot = x_in^T @ dx_output
    TIMER_BLOCK("dW_self", {
            GEMM_TN(l->in_dim, l->out_dim, l->node_count,
                    1.0,
                    l->x_in,    l->in_dim,
                    l->dx_out,  l->out_dim,
                    0.0,
                    l->dW_self, l->W_stride);
        });

    // dW_neigh = x_neigh^T @ dx_output
    TIMER_BLOCK("dW_neigh", {
            GEMM_TN(l->in_dim, l->out_dim, l->node_count,
                    1.0,
                    l->x_neigh,  l->in_dim,
                    l->dx_out,   l->out_dim,
                    0.0,
                    l->dW_neigh, l->W_stride);
        });

    // dx_in  = dx_out @ W_self^T
    TIMER_BLOCK("dx_in", {
            GEMM_NT(l->node_count, l->in_dim, l->out_dim,
                    1.0,
                    l->dx_out, l->out_dim,
                    l->W_self, l->W_stride,
                    0.0,
                    l->dx_in,  l->in_dim);
        });

    // grad_scatter = grad_output @ Wagg^T
    TIMER_BLOCK("dx_scatter", {
            GEMM_NT(l->node_count, l->in_dim, l->out_dim,
                    1.0,
                    l->dx_out,     l->out_dim,
                    l->W_neigh,    l->W_stride,
                    0.0,
                    l->dx_scatter, l->in_dim);
        });

#if defined(SPARSE_COO)
        scale_by_inv_degree_coo(l);
        scatter_coo(l);
#else
        scale_by_inv_degree_csx(l);
        scatter_csx(l);
#endif
}
