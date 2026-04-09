#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ARM_FEATURE_SIMD32) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cblas.h>

#include "core.h"
#include "cache_counter.h"
#include "dataset.h"
#include "layers.h"
#include "timer.h"

static cache_counter_t* thread_counters = NULL;


// SOURCE_TO_TARGET
void scatter_grad(SageLayer *l)
{
#pragma omp parallel for
    for (size_t n = 0; n < l->num_nodes; n++) // aka dst
    {
        const Real *restrict go = &l->grad_output[n * l->out_dim];
        Real *restrict gs = &l->grad_scatter[n * l->in_dim];
        const Real s = l->edges.inv_in_degree[n];

        for (size_t i = 0; i < l->in_dim; i++)
        {
            const Real *restrict wa = &l->Wagg[i * l->out_dim];
            Real sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < l->out_dim; j++)
            {
                sum += go[j] * wa[j];
            }
            gs[i] = sum * s;
        }
    }

#pragma omp parallel for
    for (size_t e = 0; e < l->num_edges; e++)
    {
        uint32_t src = l->edges.src[e];
        uint32_t dst = l->edges.dst[e];
        Real *restrict gi  = &l->grad_input[src * l->in_dim];
        const Real *restrict gs = &l->grad_scatter[dst * l->in_dim];

        for (size_t i = 0; i < l->in_dim; i++)
        {
#pragma omp atomic
            gi[i] += gs[i];
        }
    }
}

static void scatter_coo(uint32_t *nodes, uint32_t *peers, SageLayer *l)
{
#pragma omp parallel for
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

void sageconv_backward_v1(SageLayer *const l)
{
    // grad_Wroot = input^T @ grad_output
    // input:       num_nodes x in_dim
    // grad_output: num_nodes x out_dim
    // grad_Wroot:  in_dim x out_dim
#pragma omp parallel for
    for (uint32_t i = 0; i < l->in_dim; i++)
    {
        for (uint32_t j = 0; j < l->out_dim; j++)
        {
            Real sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (uint32_t k = 0; k < l->num_nodes; k++)
            {
                sum += l->input[k*l->in_dim+i] * l->grad_output[k*l->out_dim+j];
            }
            l->grad_Wroot[i*l->out_dim+j] = sum;
        }
    }

    // grad_Wagg = agg^T @ grad_output
    // agg:         num_nodes x in_dim
    // grad_output: num_nodes x out_dim
    // grad_Wagg:   in_dim x out_dim
#pragma omp parallel for
    for (uint32_t i = 0; i < l->in_dim; i++)
    {
        for (uint32_t j = 0; j < l->out_dim; j++)
        {
            Real sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (uint32_t k = 0; k < l->num_nodes; k++)
            {
                sum += l->agg[k*l->in_dim+i] * l->grad_output[k*l->out_dim+j];
            }
            l->grad_Wagg[i*l->out_dim+j] = sum;
        }
    }

    // grad_input = grad_output @ Wroot^T
    // grad_output: num_nodes x out_dim
    // Wroot:       in_dim x out_dim
    // grad_input:  num_nodes x in_dim
#pragma omp parallel for
    for (uint32_t i = 0; i < l->num_nodes; i++)
    {
        for (uint32_t j = 0; j < l->in_dim; j++)
        {
            Real sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (uint32_t k = 0; k < l->out_dim; k++)
            {
                sum += l->grad_output[i*l->out_dim+k] * l->Wroot[j*l->out_dim+k];
            }
            l->grad_input[i*l->in_dim+j] = sum;
        }
    }

    scatter_grad(l);
}

// Cblas implementation
void sageconv_backward_v2(SageLayer *const l)
{
    cblas_dgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                l->in_dim, l->out_dim, l->num_nodes,
                1.0,
                l->input,      l->in_dim,
                l->grad_output, l->out_dim,
                0.0,
                l->grad_Wroot, l->out_dim);

    cblas_dgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                l->in_dim, l->out_dim, l->num_nodes,
                1.0,
                l->agg,        l->in_dim,
                l->grad_output, l->out_dim,
                0.0,
                l->grad_Wagg,  l->out_dim);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output, l->out_dim,
                l->Wroot,       l->out_dim,
                0.0,
                l->grad_input,  l->in_dim);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                l->num_nodes, l->in_dim, l->out_dim,
                1.0,
                l->grad_output,  l->out_dim,
                l->Wagg,         l->out_dim,
                0.0,
                l->grad_scatter, l->in_dim);

    // Scalling by inverse degree
#pragma omp parallel for
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

    scatter_coo(l->edges.dst, l->edges.src, l);
}

void sageconv_backward_v3(SageLayer *const l)
{
    size_t in_dim  = l->in_dim;      // 256, 512, 1024
    size_t out_dim = l->out_dim;     // 256, 512, 1024
    size_t num_nodes = l->num_nodes; // 1'166'243, 2'449'029, 111'059'956

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dW = &l->tls_dW[tid * 2 * wt_size];
        real_zero_out(tid_dW, 2 * wt_size);
        const Real *restrict Wroot = l->Wroot;
        const Real *restrict Wagg = l->Wagg;

#pragma omp for
        for (size_t n = 0; n < num_nodes; n++)
        {
            const Real *restrict go = &l->grad_output[n * out_dim];
            const Real *restrict in_row = &l->input[n * in_dim];
            const Real *restrict ag_row = &l->agg[n * in_dim];
            Real *restrict gi = &l->grad_input[n * in_dim];
            Real *restrict gs = &l->grad_scatter[n * in_dim];

            for (size_t i = 0; i < in_dim; i++)
            {
                const Real *restrict wr = &Wroot[i * out_dim];
                const Real *restrict wa = &Wagg[i * out_dim];
                register Real xi = in_row[i];
                register Real ai = ag_row[i];
                Real *restrict dwr = &tid_dW[i * 2 * out_dim];
                Real *restrict dwa = dwr + out_dim;

                Real si = 0.0, ss = 0.0;
#pragma omp simd reduction(+:si,ss)
                for (size_t j = 0; j < out_dim; j++)
                {
                    Real gj = go[j];
                    si += go[j] * wr[j];
                    ss += go[j] * wa[j];
                    dwr[j] += xi * gj;
                    dwa[j] += ai * gj;
                }
                gi[i] = si;
                gs[i] = ss;
            }
        }

        const size_t thread_stride = 2 * wt_size;
        const size_t row_stride    = 2 * out_dim;
#pragma omp barrier
#pragma omp for
        for (size_t row = 0; row < in_dim; row++)
        {
            Real *restrict dWr_out = &l->grad_Wroot[row * out_dim];
            Real *restrict dWa_out = &l->grad_Wagg [row * out_dim];

#pragma omp simd
            for (size_t j = 0; j < out_dim; j++)
            {
                Real wr = 0.0, wa = 0.0;
#pragma GCC unroll 4
                for (int t = 0; t < nthreads; t++)
                {
                    Real *td = &l->tls_dW[t * thread_stride + row * row_stride];
                    wr += td[j];
                    wa += td[out_dim + j];
                }
                dWr_out[j] = wr;
                dWa_out[j] = wa;
            }
        }
    }

    // Scalling by inverse degree
#pragma omp parallel for
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

    scatter_coo(l->edges.dst, l->edges.src, l);
}

#define STRINGIFY(x) #x
#define PRAGMA_UNROLL(n) _Pragma(STRINGIFY(GCC unroll n))

/*
 * For some cpus architecture GCC doesn't define a architecture-specific macros e.g. __neoverse_v2__.
 * Hence, when compiling for these architectures, one needs to manually define them at compile time:
 * -D__neoverse_v2__
 */

#if !defined(SAGE_NODE_BLOCK)
    #if defined(__znver3__) || defined(__neoverse_v2__) || defined(__sapphirerapids__)
        #define SAGE_NODE_BLOCK 8
    #elif defined(__znver4__)
        #error "Optimal __znver4__ SAGE_NODE_BLOCK value needs to be found"
    #elif defined(__znver2__)
        #define SAGE_NODE_BLOCK 4
    #elif defined(__thunderx2__)
        #define SAGE_NODE_BLOCK 16
    #elif defined(__kunpeng_920__)
        #error "Optimal __kunpeng_920__ SAGE_NODE_BLOCK value needs to be found"
    #elif defined(__icelake_server__) // habanaq
        #error "Optimal __icelake_server__ SAGE_NODE_BLOCK value needs to be found"
    #else
        #define SAGE_NODE_BLOCK 2
    #endif
#endif

#ifndef SAGE_NODE_BLOCK // To stop compiler from cascading error even the #error above is triggered
    #define SAGE_NODE_BLOCK 2
#endif

void sageconv_backward_v4(SageLayer *const l)
{
    size_t in_dim    = l->in_dim;
    size_t out_dim   = l->out_dim;
    size_t num_nodes = l->num_nodes;

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    const uint32_t num_nodes_aligned = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dW = &l->tls_dW[tid * 2 * wt_size];
        real_zero_out(tid_dW, 2 * wt_size);
        const Real *restrict Wroot = l->Wroot;
        const Real *restrict Wagg = l->Wagg;

#pragma omp for nowait
        for (uint32_t n = 0; n < num_nodes_aligned; n+=SAGE_NODE_BLOCK)
        {
            const Real *go[SAGE_NODE_BLOCK], *in_r[SAGE_NODE_BLOCK], *ag_r[SAGE_NODE_BLOCK];
            Real       *gi[SAGE_NODE_BLOCK], *gs[SAGE_NODE_BLOCK];

            PRAGMA_UNROLL(SAGE_NODE_BLOCK)
            for (int u = 0; u < SAGE_NODE_BLOCK; u++)
            {
                size_t nx = n + u;
                go[u]   = &l->grad_output[nx * out_dim];
                in_r[u] = &l->input[nx * in_dim];
                ag_r[u] = &l->agg[nx * in_dim];
                gi[u]   = &l->grad_input[nx * in_dim];
                gs[u]   = &l->grad_scatter[nx * in_dim];
            }

            for (size_t i = 0; i < in_dim; i++)
            {
                Real x[SAGE_NODE_BLOCK], a[SAGE_NODE_BLOCK], si[SAGE_NODE_BLOCK], ss[SAGE_NODE_BLOCK];
                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    x[u] = in_r[u][i];
                    a[u] = ag_r[u][i];
                    si[u] = 0.0;
                    ss[u] = 0.0;
                }

                const Real *restrict wr_row  = &Wroot[i * out_dim];
                const Real *restrict wa_row  = &Wagg[i * out_dim];
                Real *restrict dwr = &tid_dW[i * 2 * out_dim];
                Real *restrict dwa = dwr + out_dim;

#pragma omp simd
                for (size_t j = 0; j < out_dim; j++)
                {
                    Real wr = wr_row[j];
                    Real wa = wa_row[j];
                    Real dwr_acc = 0.0, dwa_acc = 0.0;
                    PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                    {
                        Real g = go[u][j];
                        si[u]   += g * wr;
                        ss[u]   += g * wa;
                        dwr_acc += x[u] * g;
                        dwa_acc += a[u] * g;
                    }
                    dwr[j] += dwr_acc;
                    dwa[j] += dwa_acc;
                }
                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    gi[u][i] = si[u];
                    gs[u][i] = ss[u];
                }
            }
        }

#pragma omp for nowait
        for (uint32_t n = num_nodes_aligned; n < num_nodes; n++)
        {
            const Real *restrict go = &l->grad_output[n * out_dim];
            const Real *restrict in_row = &l->input[n * in_dim];
            const Real *restrict ag_row = &l->agg[n * in_dim];
            Real *restrict gi = &l->grad_input[n * in_dim];
            Real *restrict gs = &l->grad_scatter[n * in_dim];

            for (size_t i = 0; i < in_dim; i++)
            {
                const Real *restrict wr = &Wroot[i * out_dim];
                const Real *restrict wa = &Wagg[i * out_dim];
                Real *restrict dwr = &tid_dW[i * 2 * out_dim];
                Real *restrict dwa = dwr + out_dim;
                Real xi = in_row[i], ai = ag_row[i];
                Real si = 0.0, ss = 0.0;
#pragma omp simd reduction(+:si,ss)
                for (size_t j = 0; j < out_dim; j++)
                {
                    Real gj = go[j];
                    si     += gj * wr[j];
                    ss     += gj * wa[j];
                    dwr[j] += xi * gj;
                    dwa[j] += ai * gj;
                }
                gi[i] = si;
                gs[i] = ss;
            }
        }

        const size_t thread_stride = 2 * wt_size;
        const size_t row_stride    = 2 * out_dim;
#pragma omp barrier
#pragma omp for
        for (size_t row = 0; row < in_dim; row++)
        {
            Real *restrict dWr_out = &l->grad_Wroot[row * out_dim];
            Real *restrict dWa_out = &l->grad_Wagg [row * out_dim];

#pragma omp simd
            for (size_t j = 0; j < out_dim; j++)
            {
                Real wr = 0.0, wa = 0.0;
#pragma GCC unroll 4
                for (int t = 0; t < nthreads; t++)
                {
                    Real *td = &l->tls_dW[t * thread_stride + row * row_stride];
                    wr += td[j];
                    wa += td[out_dim + j];
                }
                dWr_out[j] = wr;
                dWa_out[j] = wa;
            }
        }
    }

    // Scalling by inverse degree
#pragma omp parallel for
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

    scatter_coo(l->edges.dst, l->edges.src, l);
}


//
// This performed worse than the version that doesn't tile in_dim
// (i.e. sageconv_backward_v4), 0.80s vs 0.14s. I have left this here
// to be as a reminder to my self on what I have tried.
//

#if 0
#ifndef I_TILE
#define I_TILE 4 // Optimal for rome16q
#endif // I_TILE

void sageconv_backward_vX(SageLayer *const l)
{
    size_t in_dim    = l->in_dim;
    size_t out_dim   = l->out_dim;
    size_t num_nodes = l->num_nodes;

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    real_zero_out(l->tls_dWroot, nthreads * wt_size);
    real_zero_out(l->tls_dWagg,  nthreads * wt_size);

    const uint32_t num_nodes_aligned = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dWroot = &l->tls_dWroot[tid * wt_size];
        Real *restrict tid_dWagg  = &l->tls_dWagg [tid * wt_size];
        const Real *restrict Wroot = l->Wroot;


#pragma omp for nowait
        for (uint32_t n = 0; n < num_nodes_aligned; n+=SAGE_NODE_BLOCK)
        {
            const Real *go[SAGE_NODE_BLOCK], *in_r[SAGE_NODE_BLOCK], *ag_r[SAGE_NODE_BLOCK];
            Real       *gi[SAGE_NODE_BLOCK];

            PRAGMA_UNROLL(SAGE_NODE_BLOCK)
            for (int u = 0; u < SAGE_NODE_BLOCK; u++)
            {
                size_t nx = n + u;
                go[u]   = &l->grad_output[nx * out_dim];
                in_r[u] = &l->input[nx * in_dim];
                ag_r[u] = &l->agg[nx * in_dim];
                gi[u]   = &l->grad_input[nx * in_dim];
            }



            for (size_t ib = 0; ib < in_dim; ib += I_TILE)
            {
                size_t ii_end = MIN(ib + I_TILE, in_dim);

                Real x[SAGE_NODE_BLOCK][I_TILE], a[SAGE_NODE_BLOCK][I_TILE], s[SAGE_NODE_BLOCK][I_TILE];

                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    for (size_t ii = ib; ii < ii_end; ii++)
                    {
                        x[u][ii - ib] = in_r[u][ii];
                        a[u][ii - ib] = ag_r[u][ii];
                        s[u][ii - ib] = 0.0;
                    }
                }

#pragma omp simd
                for (size_t j = 0; j < out_dim; j++)
                {
                    Real g[SAGE_NODE_BLOCK];
                    PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                    {
                        g[u] = go[u][j];
                    }

                    for (size_t ii = ib; ii < ii_end; ii++)
                    {
                        size_t ti = ii - ib;
                        Real w = Wroot[ii * out_dim + j];
                        PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                        for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                        {
                            s[u][ti]                    += g[u] * w;
                            tid_dWroot[ii * out_dim + j] += x[u][ti] * g[u];
                            tid_dWagg [ii * out_dim + j] += a[u][ti] * g[u];
                        }
                    }
                }

                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    for (size_t ii = ib; ii < ii_end; ii++)
                    {
                        gi[u][ii] = s[u][ii - ib];
                    }
                }
            }
        }


#pragma omp for nowait
        for (uint32_t n = num_nodes_aligned; n < num_nodes; n++)
        {
            const Real *restrict go = &l->grad_output[n * out_dim];
            const Real *restrict in_row = &l->input[n * in_dim];
            const Real *restrict ag_row = &l->agg[n * in_dim];
            Real *restrict gi = &l->grad_input[n * in_dim];

            for (size_t i = 0; i < in_dim; i++)
            {
                const Real *restrict wr  = &Wroot[i * out_dim];
                Real *restrict dwr = &tid_dWroot[i * out_dim];
                Real *restrict dwa = &tid_dWagg [i * out_dim];
                Real xi = in_row[i], ai = ag_row[i];
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (size_t j = 0; j < out_dim; j++)
                {
                    Real gj = go[j];
                    sum    += gj * wr[j];
                    dwr[j] += xi * gj;
                    dwa[j] += ai * gj;
                }
                gi[i] = sum;
            }
        }

#pragma omp barrier

#pragma omp for
        for (size_t i = 0; i < wt_size; i++)
        {
            Real wr = 0.0, wa = 0.0;
#pragma GCC unroll 4
            for (int t = 0; t < nthreads; t++)
            {
                wr += l->tls_dWroot[t * wt_size + i];
                wa += l->tls_dWagg [t * wt_size + i];
            }
            l->grad_Wroot[i] = wr;
            l->grad_Wagg[i]  = wa;
        }
    }

    // Scalling by inverse degree
#pragma omp parallel for
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

    scatter_coo(l->edges.dst, l->edges.src, l);
}
#endif

//
// This performed about the same as the non J_TILE version
// (i.e. sageconv_backward_v4), for when J_TILE>=256 and in_dim=256,
// and when J_TILE<256, it performed worse. Meaning tilling over
// in_dim did not provide any benefit. I have left this here to be as
// a reminder to my self on what I have tried.
//

#if 0
#ifndef J_TILE
#define J_TILE 4
#endif // I_TILE

void sageconv_backward_vXX(SageLayer *const l)
{
    size_t in_dim    = l->in_dim;
    size_t out_dim   = l->out_dim;
    size_t num_nodes = l->num_nodes;

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    real_zero_out(l->tls_dWroot, nthreads * wt_size);
    real_zero_out(l->tls_dWagg,  nthreads * wt_size);

    const uint32_t num_nodes_aligned = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dWroot = &l->tls_dWroot[tid * wt_size];
        Real *restrict tid_dWagg  = &l->tls_dWagg [tid * wt_size];
        const Real *restrict Wroot = l->Wroot;


#pragma omp for nowait
        for (uint32_t n = 0; n < num_nodes_aligned; n+=SAGE_NODE_BLOCK)
        {
            const Real *go[SAGE_NODE_BLOCK], *in_r[SAGE_NODE_BLOCK], *ag_r[SAGE_NODE_BLOCK];
            Real       *gi[SAGE_NODE_BLOCK];

            PRAGMA_UNROLL(SAGE_NODE_BLOCK)
            for (int u = 0; u < SAGE_NODE_BLOCK; u++)
            {
                size_t nx = n + u;
                go[u]   = &l->grad_output[nx * out_dim];
                in_r[u] = &l->input[nx * in_dim];
                ag_r[u] = &l->agg[nx * in_dim];
                gi[u]   = &l->grad_input[nx * in_dim];
            }

            for (size_t jb = 0; jb < out_dim; jb += J_TILE)
            {
                size_t j_end = MIN(jb + J_TILE, out_dim);

                for (size_t i = 0; i < in_dim; i++)
                {
                    Real x[SAGE_NODE_BLOCK], a[SAGE_NODE_BLOCK], s[SAGE_NODE_BLOCK];
                    PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                    {
                        x[u] = in_r[u][i];
                        a[u] = ag_r[u][i];
                        s[u] = 0.0;
                    }

                    const Real *restrict wr  = &Wroot[i * out_dim];
                    Real       *restrict dwr = &tid_dWroot[i * out_dim];
                    Real       *restrict dwa = &tid_dWagg[i * out_dim];


#pragma omp simd
                    for (size_t j = jb; j < j_end; j++)
                    {
                        Real w = wr[j];
                        Real dwr_acc = 0.0, dwa_acc = 0.0;
                        PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                        for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                        {
                            Real g = go[u][j];
                            s[u]    += g * w;
                            dwr_acc += x[u] * g;
                            dwa_acc += a[u] * g;
                        }
                        dwr[j] += dwr_acc;
                        dwa[j] += dwa_acc;
                    }
                    PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                    {
                        gi[u][i] += s[u];
                    }
                }
            }
        }

#pragma omp for nowait
        for (uint32_t n = num_nodes_aligned; n < num_nodes; n++)
        {
            const Real *restrict go = &l->grad_output[n * out_dim];
            const Real *restrict in_row = &l->input[n * in_dim];
            const Real *restrict ag_row = &l->agg[n * in_dim];
            Real *restrict gi = &l->grad_input[n * in_dim];

            for (size_t i = 0; i < in_dim; i++)
            {
                const Real *restrict wr  = &Wroot[i * out_dim];
                Real *restrict dwr = &tid_dWroot[i * out_dim];
                Real *restrict dwa = &tid_dWagg [i * out_dim];
                Real xi = in_row[i], ai = ag_row[i];
                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (size_t j = 0; j < out_dim; j++)
                {
                    Real gj = go[j];
                    sum    += gj * wr[j];
                    dwr[j] += xi * gj;
                    dwa[j] += ai * gj;
                }
                gi[i] = sum;
            }
        }

#pragma omp barrier

#pragma omp for
        for (size_t i = 0; i < wt_size; i++)
        {
            Real wr = 0.0, wa = 0.0;
#pragma GCC unroll 4
            for (int t = 0; t < nthreads; t++)
            {
                wr += l->tls_dWroot[t * wt_size + i];
                wa += l->tls_dWagg [t * wt_size + i];
            }
            l->grad_Wroot[i] = wr;
            l->grad_Wagg[i]  = wa;
        }
    }

    // Scalling by inverse degree
#pragma omp parallel for
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

    scatter_coo(l->edges.dst, l->edges.src, l);
}
#endif

static inline void fill_uniform(Real *restrict x, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = (Real)rand() / (Real)RAND_MAX;
    }
}

static inline bool real_eq(Real a, Real b)
{
    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    const Real abs_tol = 1e-9;
    const Real rel_tol = 1e-6;

    Real abs_diff = real_fabs(a - b);
    Real abs_max = real_fmax(fabs(a), fabs(b));
    if (abs_diff > abs_tol && abs_diff > rel_tol * abs_max)
    {
        return false;
    }
    return true;
}

static bool is_valid(Real *x, Real *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (!real_eq(x[i], y[i])) return false;
    }

    return true;
}

typedef void (*fptr)(SageLayer *const l);

typedef struct {
    fptr func;
    const char *name;
} BenchFunc;

#define BENCH_FUNC(fn) { .func = &(fn), .name = #fn }

#if defined(SKIP_VALID)
static bool skip_validation = true;
#else
static bool skip_validation = false;
#endif
int main(void)
{
    srand(0);
    thread_counters = cache_counter_init_all();

    bool to_symmetric = true;
    Dataset *ds = dataset_load("arxiv", "./data", EDGE_COO, to_symmetric);
    // Dataset *ds_train = dataset_split(ds, SPLIT_TRAIN);
    // ds = ds_train;
    printf("num nodes: %u\n", ds->num_nodes);

    // Sentetic data
    size_t in_dim = 256, out_dim = 256;
    Real *input = cache_aligned_alloc(ds->num_nodes * in_dim * sizeof(Real));
    fill_uniform(input, ds->num_nodes * in_dim);
    Real *grad_output = cache_aligned_alloc(ds->num_nodes * out_dim * sizeof(Real));
    fill_uniform(grad_output, ds->num_nodes * out_dim);

    SageLayer *l = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, in_dim, out_dim, SOURCE_TO_TARGET);
    l->input = input;
    l->grad_output = grad_output;

    BenchFunc funcs[] = {
#if defined(FULL_TEST)
        // BENCH_FUNC(sageconv_backward_v1),
        BENCH_FUNC(sageconv_backward_v2),
        BENCH_FUNC(sageconv_backward_v3),
#endif
        BENCH_FUNC(sageconv_backward_v4),
        // BENCH_FUNC(sageconv_backward_v5),
    };

    if (!skip_validation)
    {
        // compute reference
        if (isatty(STDOUT_FILENO))
        {
            printf("Reference:");
            fflush(stdout);
        }
        sageconv_backward_v2(l);
        Real *ref_gwr = cache_aligned_alloc(in_dim * out_dim * sizeof(Real));
        Real *ref_gwa = cache_aligned_alloc(in_dim * out_dim * sizeof(Real));
        Real *ref_gi = cache_aligned_alloc(l->num_nodes * in_dim * sizeof(Real));
        memcpy(ref_gwr, l->grad_Wroot, in_dim * out_dim * sizeof(Real));
        memcpy(ref_gwa, l->grad_Wagg, in_dim * out_dim * sizeof(Real));
        memcpy(ref_gi, l->grad_input, l->num_nodes * in_dim * sizeof(Real));
        if (isatty(STDOUT_FILENO)) printf(" ok\n");

        // validation
        for (size_t i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++)
        {
            if (isatty(STDOUT_FILENO))
            {
                printf("\r\033[KValidating: %s", funcs[i].name);
                fflush(stdout);
            }

            real_zero_out(l->grad_Wroot, in_dim * out_dim);
            real_zero_out(l->grad_Wagg, in_dim * out_dim);
            real_zero_out(l->grad_input, l->num_nodes * in_dim);
            real_zero_out(l->grad_scatter, l->num_nodes * in_dim);

            funcs[i].func(l);
            if(!is_valid(l->grad_Wroot, ref_gwr, in_dim * out_dim))
            {
                printf("\r\033[K");
                ERROR("grad_Wroot doesn't match the reference (%s)", funcs[i].name);
            }
            if(!is_valid(l->grad_Wagg, ref_gwa, in_dim * out_dim))
            {
                printf("\r\033[K");
                ERROR("grad_Wagg doesn't match the reference (%s)", funcs[i].name);
            }
            if(!is_valid(l->grad_input, ref_gi, l->num_nodes * in_dim))
            {
                printf("\r\033[K");
                ERROR("grad_input doesn't match the reference (%s)", funcs[i].name);
            }
        }
        if (isatty(STDOUT_FILENO)) printf("\r\033[KValidating: ok\n");
    }
    else printf("Skiping validation\n");

    for (size_t i = 0; i < sizeof(funcs)/sizeof(funcs[0]); i++)
    {
        // warm up
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KWarmup: %s", funcs[i].name);
            fflush(stdout);
        }

        double min_time = DBL_MAX;
        uint64_t bytes = 0, l3_local = 0, l3_remote = 0; // 0 initialized to silent -Wmaybe-uninitialized
        // 4 GEMMs (2*N*I*O each) + inv_degree scale (N*I) + scatter (E*I)
        uint64_t flops = 8UL * l->num_nodes * in_dim * out_dim + l->num_nodes * in_dim + l->num_edges * in_dim;

        for (size_t j = 0; j < 10; j++)
        {
            if (isatty(STDOUT_FILENO))
            {
                printf(".");
                fflush(stdout);
            }

            real_zero_out(l->grad_Wroot, in_dim * out_dim);
            real_zero_out(l->grad_Wagg, in_dim * out_dim);
            real_zero_out(l->grad_input, ds->num_nodes * in_dim);
            real_zero_out(l->grad_scatter, ds->num_nodes * in_dim);
            funcs[i].func(l);
        }

        // Run
        if (isatty(STDOUT_FILENO))
        {
            printf("\r\033[KRun: %s", funcs[i].name);
            fflush(stdout);
        }
        for (size_t j = 0; j < 100; j++)
        {
            if (isatty(STDOUT_FILENO))
            {
                printf(".");
                fflush(stdout);
            }

            real_zero_out(l->grad_Wroot, in_dim * out_dim);
            real_zero_out(l->grad_Wagg, in_dim * out_dim);
            real_zero_out(l->grad_input, ds->num_nodes * in_dim);

            timer_enable();
            cache_counter_start_all(thread_counters);
            double t0 = omp_get_wtime();
            funcs[i].func(l);
            double elapsed = omp_get_wtime()-t0;
            cache_counter_stop_all(thread_counters);
            timer_record(funcs[i].name, elapsed, NULL);
            timer_disable();
            if (elapsed < min_time)
            {
                min_time = elapsed;
                bytes = 0, l3_local = 0, l3_remote = 0;
                for (int tid = 0; tid < omp_get_max_threads(); tid++)
                {
                    bytes += cache_counter_get_bytes_loaded(&thread_counters[tid]);
                    long long local = 0, remote = 0;
                    cache_counter_get_cache_misses(&thread_counters[tid], &local, &remote);
                    l3_local += (uint64_t)local;
                    l3_remote += (uint64_t)remote;
                }
            }
        }
        timer_enable();
        timer_record_counters(funcs[i].name, flops, l3_local, l3_remote, bytes);
        timer_disable();
    }

    if (isatty(STDOUT_FILENO))
    {
        printf("\r\033[K");
        fflush(stdout);
    }

    timer_print();
    timer_export_csv("stdout");
    cache_counter_close_all(thread_counters);
}
