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

void scatter_grad(SageLayer *l)
{
#pragma omp parallel for
    for (size_t n = 0; n < l->num_nodes; n++) // aka dst
    {
        const Real *restrict go = &l->grad_output[n * l->out_dim];
        Real *restrict gs = &l->grad_scatter[n * l->in_dim];
        const Real s = l->mean_scale[n];

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

    // scatter_grad(l);
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
}

void sageconv_backward_v3(SageLayer *const l)
{
    size_t in_dim  = l->in_dim;      // 256, 512, 1024
    size_t out_dim = l->out_dim;     // 256, 512, 1024
    size_t num_nodes = l->num_nodes; // 1'166'243, 2'449'029, 111'059'956

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    real_zero_out(l->tls_dWroot, nthreads * wt_size);
    real_zero_out(l->tls_dWagg, nthreads * wt_size);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dWroot = &l->tls_dWroot[tid * wt_size];
        Real *restrict tid_dWagg  = &l->tls_dWagg[tid * wt_size];
        const Real * restrict Wroot = l->Wroot;

#pragma omp for
        for (size_t n = 0; n < num_nodes; n++)
        {
            const Real * restrict go = &l->grad_output[n * out_dim];
            const Real * restrict in_row = &l->input[n * in_dim];
            const Real * restrict ag_row = &l->agg[n * in_dim];
            Real * restrict gi = &l->grad_input[n * in_dim];

            for (size_t i = 0; i < in_dim; i++)
            {
                const Real * restrict wr = &Wroot[i * out_dim];
                register Real xi = in_row[i];
                register Real ai = ag_row[i];
                Real * restrict dwr = &tid_dWroot[i * out_dim];
                Real * restrict dwa = &tid_dWagg[i * out_dim];

                Real sum = 0.0;
#pragma omp simd reduction(+:sum)
                for (size_t j = 0; j < out_dim; j++)
                {
                    Real gj = go[j];
                    dwr[j] += xi * gj;
                    dwa[j] += ai * gj;
                    sum += go[j] * wr[j];
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
                wa += l->tls_dWagg[t * wt_size + i];
            }
            l->grad_Wroot[i] = wr;
            l->grad_Wagg[i]  = wa;

        }
    }
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

#ifndef SAGE_NODE_BLOCK         // To stop compiler from cascading error #error is hit above
    #define SAGE_NODE_BLOCK 2
#endif

void sageconv_backward_v4(SageLayer *const l)
{
    size_t in_dim    = l->in_dim;
    size_t out_dim   = l->out_dim;
    size_t num_nodes = l->num_nodes;

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    real_zero_out(l->tls_dWroot, nthreads * wt_size);
    real_zero_out(l->tls_dWagg,  nthreads * wt_size);

    const size_t num_nodes_aligned = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dWroot = &l->tls_dWroot[tid * wt_size];
        Real *restrict tid_dWagg  = &l->tls_dWagg [tid * wt_size];
        const Real *restrict Wroot = l->Wroot;

#pragma omp for nowait
        for (size_t n = 0; n < num_nodes_aligned; n+=SAGE_NODE_BLOCK)
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
                for (size_t j = 0; j < out_dim; j++)
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
                    gi[u][i] = s[u];
                }
            }
        }

#pragma omp single nowait
        for (size_t n = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK; n < num_nodes; n++)
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
}

static inline Real hsum_avx(__m256r v)
{
#if defined(USE_DOUBLE)
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);                    // [0+2, 1+3]
    __m128d hi64 = _mm_unpackhi_pd(lo, lo);      // [1+3, 1+3]
    return _mm_cvtsd_f64(_mm_add_sd(lo, hi64));  // (0+2)+(1+3)
#else
    __m128 lo = _mm256_castps256_128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);           // 4 floats
    __m128 shuf = _mm_movehdup_ps(lo); // [1,1,3,3]
    lo = _mm_add_ps(lo, shuf);         // [0+1, -, 2+3, -]
    shuf = _mm_movehl_ps(shuf, lo);    // [2+3, -, -, -]
    return _mm_cvtss_f32(_mm_add_ss(lo, shuf));
#endif
}

void sageconv_backward_v6(SageLayer *const l)
{
    size_t in_dim    = l->in_dim;
    size_t out_dim   = l->out_dim;
    size_t num_nodes = l->num_nodes;

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    real_zero_out(l->tls_dWroot, nthreads * wt_size);
    real_zero_out(l->tls_dWagg,  nthreads * wt_size);

    const size_t num_nodes_aligned = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dWroot = &l->tls_dWroot[tid * wt_size];
        Real *restrict tid_dWagg  = &l->tls_dWagg [tid * wt_size];
        const Real *restrict Wroot = l->Wroot;

#pragma omp for nowait
        for (size_t n = 0; n < num_nodes_aligned; n+=SAGE_NODE_BLOCK)
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

                __m256r vs[SAGE_NODE_BLOCK];
                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    vs[u] = _mm256_setzero();
                }

                const Real *restrict wr  = &Wroot[i * out_dim];
                Real       *restrict dwr = &tid_dWroot[i * out_dim];
                Real       *restrict dwa = &tid_dWagg[i * out_dim];

                size_t j = 0;
                for (; j + (SIMD_LANES-1) < out_dim; j += SIMD_LANES)

                {
                    __m256r vw   = _mm256_loadu(&wr[j]);
                    __m256r vdwr = _mm256_loadu(&dwr[j]);
                    __m256r vdwa = _mm256_loadu(&dwa[j]);

                    PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                    {
                        __m256r vg = _mm256_loadu(&go[u][j]);
                        vs[u] = _mm256_fmadd(vg, vw, vs[u]);   // s[u] += g * w
                        vdwr  = _mm256_fmadd(_mm256_set1(x[u]), vg, vdwr);  // dwr  += x[u]*g
                        vdwa  = _mm256_fmadd(_mm256_set1(a[u]), vg, vdwa);  // dwa  += a[u]*g
                    }

                    _mm256_storeu(&dwr[j], vdwr);
                    _mm256_storeu(&dwa[j], vdwa);
                }

                for (; j < out_dim; j++) {
                    Real w = wr[j];
                    PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                    {
                        Real g = go[u][j];
                        s[u]   += g * w;
                        dwr[j] += x[u] * g;
                        dwa[j] += a[u] * g;
                    }
                }

                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    gi[u][i] = hsum_avx(vs[u]) + s[u];
                }
            }
        }

#pragma omp single nowait
        for (size_t n = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK; n < num_nodes; n++)
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
}

void sageconv_backward_v7(SageLayer *const l)
{
    size_t in_dim    = l->in_dim;
    size_t out_dim   = l->out_dim;
    size_t num_nodes = l->num_nodes;

    int nthreads = omp_get_max_threads();
    size_t wt_size = in_dim * out_dim;

    real_zero_out(l->tls_dWroot, nthreads * wt_size);
    real_zero_out(l->tls_dWagg,  nthreads * wt_size);

    const size_t num_nodes_aligned = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real *restrict tid_dWroot = &l->tls_dWroot[tid * wt_size];
        Real *restrict tid_dWagg  = &l->tls_dWagg [tid * wt_size];
        const Real *restrict Wroot = l->Wroot;

#pragma omp for nowait
        for (size_t n = 0; n < num_nodes_aligned; n+=SAGE_NODE_BLOCK)
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

                __m256r vs[SAGE_NODE_BLOCK];
                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    vs[u] = _mm256_setzero();
                }

                const Real *restrict wr  = &Wroot[i * out_dim];
                Real       *restrict dwr = &tid_dWroot[i * out_dim];
                Real       *restrict dwa = &tid_dWagg[i * out_dim];

                size_t j = 0;
                for (; j + (SIMD_LANES-1) < out_dim; j += SIMD_LANES)

                {
                    __m256r vw = _mm256_loadu(&wr[j]);

                    // Two independent chains per accumulator
                    __m256r vdwr0 = _mm256_loadu(&dwr[j]);
                    __m256r vdwr1 = _mm256_setzero();
                    __m256r vdwa0 = _mm256_loadu(&dwa[j]);
                    __m256r vdwa1 = _mm256_setzero();

                    PRAGMA_UNROLL(SAGE_NODE_BLOCK/2)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u+=2)
                    {
                        __m256r vg0 = _mm256_loadu(&go[u  ][j]);
                        __m256r vg1 = _mm256_loadu(&go[u+1][j]);

                        vs[u  ] = _mm256_fmadd(vg0, vw, vs[u  ]);
                        vs[u+1] = _mm256_fmadd(vg1, vw, vs[u+1]);
                        vdwr0 = _mm256_fmadd(_mm256_set1(x[u  ]), vg0, vdwr0);
                        vdwr1 = _mm256_fmadd(_mm256_set1(x[u+1]), vg1, vdwr1);
                        vdwa0 = _mm256_fmadd(_mm256_set1(a[u  ]), vg0, vdwa0);
                        vdwa1 = _mm256_fmadd(_mm256_set1(a[u+1]), vg1, vdwa1);
                    }

                    _mm256_storeu(&dwr[j], _mm256_add(vdwr0, vdwr1));
                    _mm256_storeu(&dwa[j], _mm256_add(vdwa0, vdwa1));
                }

                for (; j < out_dim; j++) {
                    Real w = wr[j];
                    PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                    for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                    {
                        Real g = go[u][j];
                        s[u]   += g * w;
                        dwr[j] += x[u] * g;
                        dwa[j] += a[u] * g;
                    }
                }

                PRAGMA_UNROLL(SAGE_NODE_BLOCK)
                for (int u = 0; u < SAGE_NODE_BLOCK; u++)
                {
                    gi[u][i] = hsum_avx(vs[u]) + s[u];
                }
            }
        }

#pragma omp single nowait
        for (size_t n = (num_nodes / SAGE_NODE_BLOCK) * SAGE_NODE_BLOCK; n < num_nodes; n++)
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
}

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

    SageLayer *l = sage_layer_create(ds->num_nodes, ds->num_edges, ds->edges, in_dim, out_dim);
    l->input = input;
    l->grad_output = grad_output;

    BenchFunc funcs[] = {
        // BENCH_FUNC(sageconv_backward_v1),
#if defined(FULL_TEST)
        BENCH_FUNC(sageconv_backward_v2),
        BENCH_FUNC(sageconv_backward_v3_1),
        BENCH_FUNC(sageconv_backward_v3),
        BENCH_FUNC(sageconv_backward_v4),
#endif
        BENCH_FUNC(sageconv_backward_v5),
        BENCH_FUNC(sageconv_backward_v6),
        BENCH_FUNC(sageconv_backward_v7),
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
        memcpy(ref_gi, l->grad_input, ds->num_nodes * in_dim * sizeof(Real));
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
            real_zero_out(l->grad_input, ds->num_nodes * in_dim);

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
            if(!is_valid(l->grad_input, ref_gi, ds->num_nodes * in_dim))
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
        uint64_t bytes = 0, l3_local = 0, l3_remote = 0;

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
            cache_counter_close_all(thread_counters);
            timer_record(funcs[i].name, elapsed, NULL);
            timer_disable();
            if (elapsed < min_time)
            {
                min_time = elapsed;
                bytes = 0, l3_local = 0, l3_remote = 0;
                for (int tid = 0; tid < omp_get_max_threads(); tid++) {
                    bytes += cache_counter_get_bytes_loaded(&thread_counters[tid]);
                    long long local = 0, remote = 0;
                    cache_counter_get_cache_misses(&thread_counters[tid], &local, &remote);
                    l3_local += (uint64_t)local;
                    l3_remote += (uint64_t)remote;
                }
            }
        }
    }

    if (isatty(STDOUT_FILENO))
    {
        printf("\r\033[K");
        fflush(stdout);
    }

    timer_print();
    timer_export_csv("stdout");
}
