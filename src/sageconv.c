#include <stdint.h>

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
#else // BLAS
#  define GEMM_NN(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_NT(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#  define GEMM_TN(M,N,K,a,A,lda,B,ldb,b,C,ldc) cblas_rgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, (M),(N),(K),(a),(A),(lda),(B),(ldb),(b),(C),(ldc))
#endif

static void sage_mean_aggregate_coo(const int64_t *restrict nodes, const int64_t *restrict peers, SageLayer *l)
{
    int64_t  num_nodes  = l->num_nodes;
    int64_t  num_edges  = l->num_edges;
    int64_t  in_dim     = l->in_dim;

    const Real *restrict inv_degree = l->edges.inv_degree;
    const Real *restrict input = l->input;
    Real       *restrict agg = l->agg;

#pragma omp parallel
    {
#pragma omp for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            Real *agg_row = &agg[i * in_dim];
            real_zero_out(agg_row, in_dim);

            Real scale = inv_degree[i];
            if (scale == REAL(0.0)) continue;

            for (int64_t e = 0; e < num_edges; e++)
            {
                if (i == nodes[e])
                {
                    const Real *in_row = &input[peers[e] * in_dim];
#pragma omp simd
                    for (int64_t j = 0; j < in_dim; j++)
                    {
                        agg_row[j] += in_row[j] * scale;
                    }
                }
            }
        }
    }
}

static void sage_mean_aggregate_cs(const int64_t *restrict ptr, const int64_t *restrict idx, SageLayer *l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;

    const Real *restrict input = l->input;
    Real       *restrict agg   = l->agg;

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        Real *agg_row = &agg[i*in_dim];

        // TODO: compare it with using inv_degree directly
        int64_t degree = ptr[i+1] - ptr[i];
        if (degree == 0) continue;
        Real scale = (Real)1.0 / degree;

        for (int64_t j = ptr[i]; j < ptr[i+1]; j++)
        {
            const Real *in_row = &input[idx[j] * in_dim];
            for (int64_t k = 0; k < in_dim; k++)
            {
                agg_row[k] += in_row[k] * scale;
            }
        }
    }
}

static void sage_mean_aggregate(SageLayer *const l)
{
    TIMER_FUNC();

    if (l->flow == FLOW_NONE)
    {
        ERROR("SageLayer is missing flow direction");
    }

    switch (l->edges.format)
    {
    case EDGE_COO:
    {
        if (l->flow == SOURCE_TO_TARGET) // e.g. src (citer) aggregates from dst (cited)
        {
            sage_mean_aggregate_coo(l->edges.dst, l->edges.src, l);
        }
        else if (l->flow == TARGET_TO_SOURCE) // e.g. dst (cited) aggregates from src (citer)
        {
            sage_mean_aggregate_coo(l->edges.src, l->edges.dst, l);
        }
        break;
    }
    case EDGE_COMPRESSED:
        sage_mean_aggregate_cs(l->edges.ptr, l->edges.idx, l);
        break;
    default:
        ERROR("Unknown EdgeFormat was given: %d", l->edges.format);
    }
}

void sageconv(SageLayer *const l)
{
    TIMER_FUNC();

    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;
    int64_t out_dim   = l->out_dim;

    // output = input @ Wroot
    TIMER_BLOCK("root", {
            GEMM_NN(num_nodes, out_dim, in_dim,
                    1.0,
                    l->input, in_dim,
                    l->Wroot, out_dim,
                    0.0,
                    l->output,out_dim);
        });

    sage_mean_aggregate(l);

    // output += agg @ Wagg
    TIMER_BLOCK("neoigh", {
            GEMM_NN(num_nodes, out_dim, in_dim,
                    1.0,
                    l->agg,    in_dim,
                    l->Wagg,   out_dim,
                    1.0,
                    l->output, out_dim);
        });
}

static void scale_by_inv_degree(SageLayer *l)
{
    int64_t num_nodes = l->num_nodes;
    int64_t in_dim    = l->in_dim;

    const Real *restrict inv_degree   = l->edges.inv_degree;
    Real       *restrict grad_scatter = l->grad_scatter;

#pragma omp for
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

static void scatter_coo(int64_t *nodes, int64_t *peers, SageLayer *l)
{

    const Real *restrict grad_scatter = l->grad_scatter;
    Real       *restrict grad_input   = l->grad_input;

#pragma omp for
    for (int64_t e = 0; e < l->num_edges; e++)
    {
        const Real *gs_row = &grad_scatter[nodes[e] * l->in_dim];
        Real       *gi_row = &grad_input[peers[e] * l->in_dim];

        for (int64_t i = 0; i < l->in_dim; i++)
        {
#pragma omp atomic
            gi_row[i] += gs_row[i];
        }
    }
}

void grad_sageconv(SageLayer *l)
{
    TIMER_FUNC();

    // grad_Wroot = input^T @ grad_output
    TIMER_BLOCK("dWroot", {
            GEMM_TN(l->in_dim, l->out_dim, l->num_nodes,
                    1.0,
                    l->input,       l->in_dim,
                    l->grad_output, l->out_dim,
                    0.0,
                    l->grad_Wroot,  l->out_dim);
        });

    // grad_Wagg = agg^T @ grad_output
    TIMER_BLOCK("dWagg", {
            GEMM_TN(l->in_dim, l->out_dim, l->num_nodes,
                    1.0,
                    l->agg,         l->in_dim,
                    l->grad_output, l->out_dim,
                    0.0,
                    l->grad_Wagg,   l->out_dim);
        });

    // grad_input  = grad_output @ Wroot^T
    TIMER_BLOCK("dinput", {
            GEMM_NT(l->num_nodes, l->in_dim, l->out_dim,
                    1.0,
                    l->grad_output, l->out_dim,
                    l->Wroot,       l->out_dim,
                    0.0,
                    l->grad_input,  l->in_dim);
        });

    // grad_scatter = grad_output @ Wagg^T
    TIMER_BLOCK("grad_scatter_matmul", {
            GEMM_NT(l->num_nodes, l->in_dim, l->out_dim,
                    1.0,
                    l->grad_output,  l->out_dim,
                    l->Wagg,         l->out_dim,
                    0.0,
                    l->grad_scatter, l->in_dim);
        });

    int64_t *nodes, *peers;
    if (l->flow == SOURCE_TO_TARGET) // e.g. src (citer) aggregates from dst (cited)
    {
        nodes = l->edges.dst;
        peers = l->edges.src;
    }
    else if (l->flow == TARGET_TO_SOURCE) // e.g. dst (cited) aggregates from src (citer)
    {
        nodes = l->edges.src;
        peers = l->edges.dst;
    }
    else
    {
        ERROR("SageLayer is missing flow direction");
    }

    double t = omp_get_wtime();
#pragma omp parallel
    {
        scale_by_inv_degree(l);
        scatter_coo(nodes, peers, l);
    }
    timer_record("grad_aggregate", omp_get_wtime() - t, NULL);
}
