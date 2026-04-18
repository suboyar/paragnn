#include <omp.h>
#include <stddef.h>

#include "core.h"
#include "layers.h"
#include "sageconv_backward_common.h"

void sageconv_backward_fused_v1(SageLayer *const l)
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

        scale_by_inv_degree(l);
        scatter_coo(l->edges.dst, l->edges.src, l);
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
}

/*
 * For some cpus architecture GCC doesn't define a architecture-specific macros e.g. __neoverse_v2__.
 * Hence, when compiling for these architectures, one needs to manually define them at compile time:
 * -D__neoverse_v2__
 */

#if !defined(FUSED_NB)
    // genoaxq
    #if defined(__znver4__) // TBD
        #define FUSED_NB 4
    // milanq, fpgaq
    #elif defined(__znver3__) // milanq prefers 8, fpgaq prefers 4; difference is negligible
        #define FUSED_NB 4
    // rome16q
    #elif defined(__znver2__)
        #define FUSED_NB 4
    // defq
    #elif defined(__znver1__)
        #define FUSED_NB 8
    // xeonmaxq
    #elif defined(__sapphirerapids__) // TBD
        #error "Optimal __sapphirerapids__ FUSED_NB value needs to be found"
    // habanaq
    #elif defined(__icelake_server__)
        #define FUSED_NB 8
    // gh200q
    #elif defined(__neoverse_v2__)
        #define FUSED_NB 8
    // armq
    #elif defined(__thunderx2__)
        #define FUSED_NB 8
    // huaq
    #elif defined(__kunpeng_920__)
        #define FUSED_NB 32
    #else
        #define FUSED_NB 2
    #endif
#endif

#ifndef FUSED_NB // To stop compiler from cascading error even when the #error above is triggered
    #define FUSED_NB 2
#endif

void sageconv_backward_fused_v2(SageLayer *const l)
{
    size_t in_dim    = l->in_dim;    // 256, 512, 1024
    size_t out_dim   = l->out_dim;   // 256, 512, 1024
    size_t num_nodes = l->num_nodes; // 1'166'243, 2'449'029, 111'059'956

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    const uint32_t num_nodes_tiled = (num_nodes / FUSED_NB) * FUSED_NB;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dW = &l->tls_dW[tid * 2 * wt_size];
        real_zero_out(tid_dW, 2 * wt_size);
        const Real *restrict Wroot = l->Wroot;
        const Real *restrict Wagg = l->Wagg;

#pragma omp for nowait
        for (uint32_t n = 0; n < num_nodes_tiled; n+=FUSED_NB)
        {
            const Real *go[FUSED_NB], *in_r[FUSED_NB], *ag_r[FUSED_NB];
            Real       *gi[FUSED_NB], *gs[FUSED_NB];

            PRAGMA_UNROLL(FUSED_NB)
            for (int u = 0; u < FUSED_NB; u++)
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
                Real x[FUSED_NB], a[FUSED_NB], si[FUSED_NB], ss[FUSED_NB];
                PRAGMA_UNROLL(FUSED_NB)
                for (int u = 0; u < FUSED_NB; u++)
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
                    PRAGMA_UNROLL(FUSED_NB)
                    for (int u = 0; u < FUSED_NB; u++)
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
                PRAGMA_UNROLL(FUSED_NB)
                for (int u = 0; u < FUSED_NB; u++)
                {
                    gi[u][i] = si[u];
                    gs[u][i] = ss[u];
                }
            }
        }

#pragma omp for nowait
        for (uint32_t n = num_nodes_tiled; n < num_nodes; n++)
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

        scale_by_inv_degree(l);
        scatter_coo(l->edges.dst, l->edges.src, l);
    }
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

void sageconv_backward_fused_vX(SageLayer *const l)
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

    scatter_coo_impl(l->edges.dst, l->edges.src, l);
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
#define J_TILE 256
#endif // J_TILE

void sageconv_backward_fused_vXX(SageLayer *const l)
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

#pragma omp for nowait
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


        // Scalling by inverse degree
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

        scatter_coo_impl(l->edges.dst, l->edges.src, l);
    }
}
#endif
