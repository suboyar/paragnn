#include "outer_tn_kernel.h"
#include "../grad_mean_aggregate.h"

typedef void (*outer_fn)(int64_t, int64_t, int64_t, const Real*, int64_t, const Real*, int64_t, Real*, int64_t);

static void outer_tn_kernel_impl(SageLayer *l, outer_fn kernel)
{
    kernel(l->in_dim, l->out_dim, l->num_nodes,
           l->input,       l->in_dim,
           l->grad_output, l->out_dim,
           l->grad_Wroot,  l->out_dim);
}

static void grad_sageconv_impl(SageLayer *l, outer_fn kernel)
{
    // grad_Wroot = input^T @ grad_output
    kernel(l->in_dim, l->out_dim, l->num_nodes,
           l->input,       l->in_dim,
           l->grad_output, l->out_dim,
           l->grad_Wroot,  l->out_dim);

    // grad_Wagg = agg^T @ grad_output
    kernel(l->in_dim, l->out_dim, l->num_nodes,
           l->agg,         l->in_dim,
           l->grad_output, l->out_dim,
           l->grad_Wagg,   l->out_dim);

    // grad_input = grad_output @ Wroot^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output, l->out_dim,
                l->Wroot,       l->out_dim,
                0.0,
                l->grad_input,  l->in_dim);

    // grad_scatter = grad_output @ Wagg^T
    cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output,  l->out_dim,
                l->Wagg,         l->out_dim,
                0.0,
                l->grad_scatter, l->in_dim);

    grad_mean_aggregate(l);
}

#define DEFINE_WRAPPERS(N) \
    void outer_tn_kernel_v##N(SageLayer *l) { outer_tn_kernel_impl(l, outer_tn_v##N); } \
    void grad_sageconv_v##N(SageLayer *l)   { grad_sageconv_impl(l, outer_tn_v##N); }

DEFINE_WRAPPERS(1)
DEFINE_WRAPPERS(2)
DEFINE_WRAPPERS(3)
DEFINE_WRAPPERS(4)
DEFINE_WRAPPERS(5)
DEFINE_WRAPPERS(6)
DEFINE_WRAPPERS(7)
