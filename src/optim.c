#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>

#include "optim.h"
#include "core.h"
#include "layers.h"
#include "timer.h"

// SGD optimizer

static inline void sgd_step(SGD *restrict sgd, Real *restrict param, const Real *restrict grad, int64_t n)
{
#pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; i++)
    {
        param[i] -= sgd->lr * grad[i];
    }
}

void sgd_update(SGD *sgd, SageNet *net)
{
    for (int64_t i = 0; i < net->layer_count; i++)
    {
        Layer layer = net->layers[i];
        if (layer.type == LAYER_SAGE)
        {
            SageLayer *l = (SageLayer*)layer.ctx;
            sgd_step(sgd, l->W_self, l->dW_self, l->in_dim * l->W_stride);
            sgd_step(sgd, l->W_neigh, l->dW_neigh, l->in_dim * l->W_stride);
        }
    }
}

SGD* sgd_create(Real lr)
{
    SGD *sgd = ALLOC_OR_DIE(malloc(sizeof(*sgd)));
    sgd->kind = OPTIM_SGD;
    sgd->lr = lr;
    return sgd;
}

void sgd_free(SGD **sgd)
{
    free(*sgd);
    *sgd = NULL;
}

// ADAM optimizer

static void adam_step(AdamState *restrict s, Real *restrict param, const Real *restrict grad, int64_t n)
{
    s->t++;
    s->beta1_t *= s->beta1;
    s->beta2_t *= s->beta2;

    const Real step_size = s->lr / (REAL(1.0) - s->beta1_t);
    const Real bc2 = real_sqrt(REAL(1.0) - s->beta2_t);

#pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; i++)
    {
        const Real g = grad[i];
        s->m[i] = s->beta1 * s->m[i] + s->beta1_comp * g;
        s->v[i] = s->beta2 * s->v[i] + s->beta2_comp * g * g;
        param[i] -= step_size * s->m[i] / ((real_sqrt(s->v[i]) / bc2) + s->epsilon);
    }
}

static AdamState *adam_state_create(int64_t n, Real lr)
{
    Real beta1 = REAL(0.9), beta2 = REAL(0.999);
    AdamState *state  = ALLOC_OR_DIE(malloc(sizeof(*state)));
    *state = (AdamState) {
        .t          = 0,
        .lr         = lr,
        .beta1      = beta1,
        .beta2      = beta2,
        .epsilon    = REAL(1e-8),
        .beta1_comp = REAL(1.0) - beta1,
        .beta2_comp = REAL(1.0) - beta2,
        .beta1_t    = REAL(1.0),
        .beta2_t    = REAL(1.0),
        .m          = ALLOC_OR_DIE(alloc_local(n*sizeof(Real))),
        .v          = ALLOC_OR_DIE(alloc_local(n*sizeof(Real))),
    };

#pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; i++)
    {
        state->m[i] = REAL(0.0);
        state->v[i] = REAL(0.0);
    }

    return state;
}

Adam* adam_create(SageNet *net, Real lr)
{
    Adam *adam = ALLOC_OR_DIE(malloc(sizeof(*adam)));
    adam->kind = OPTIM_ADAM;
    int64_t count = 0;
    for (int64_t i = 0; i < net->layer_count; i++)
    {
        switch (net->layers[i].type)
        {
        case LAYER_SAGE:   count += 2; break;  // W_self, W_neigh
        default: break;
        }
    }

    adam->num_states = count;
    adam->states = ALLOC_OR_DIE(malloc(count * sizeof(*adam->states)));

    // Allocate a state for each parameter matrix
    int64_t s = 0;
    for (int64_t i = 0; i < net->layer_count; i++) {
        Layer layer = net->layers[i];
        if (layer.type == LAYER_SAGE)
        {
            SageLayer *l = (SageLayer*)layer.ctx;
            adam->states[s++] = adam_state_create(l->in_dim * l->W_stride, lr);
            adam->states[s++] = adam_state_create(l->in_dim  * l->W_stride,  lr);
        }
    }

    return adam;
}

void adam_update(Adam *adam, SageNet *net)
{
    int64_t s = 0;
    for (int64_t i = 0; i < net->layer_count; i++)
    {
        Layer layer = net->layers[i];
        if (layer.type == LAYER_SAGE)
        {
            SageLayer *l = (SageLayer*)layer.ctx;
            adam_step(adam->states[s++], l->W_self, l->dW_self, l->in_dim * l->W_stride);
            adam_step(adam->states[s++], l->W_neigh, l->dW_neigh, l->in_dim * l->W_stride);
        }
    }
}

static void adam_state_free(AdamState **s)
{
    if (!s) return;
    free((*s)->m); (*s)->m = NULL;
    free((*s)->v); (*s)->v = NULL;
    free(*s);
    *s = NULL;
}

void adam_free(Adam **adam)
{
    for (int64_t i = 0; i < (*adam)->num_states; i++)
    {
        adam_state_free(&(*adam)->states[i]);
    }
    free((*adam)->states); (*adam)->states = NULL;
    free(*adam);
    *adam = NULL;
}

// General interface
Optim *optim_create(OptimKind kind, SageNet *net, Real lr)
{
    Optim *optim = NULL;
    switch(kind)
    {
    case OPTIM_SGD:
        optim = (Optim*)sgd_create(lr);
        break;
    case OPTIM_ADAM:
        optim = (Optim*)adam_create(net, lr);
        break;
    default:
        ERROR("Unknown optim kind: %d", kind);
    }

    return optim;
}

void optim_update(Optim *optim, SageNet *net)
{
    TIMER_FUNC();
    OptimKind kind = *(OptimKind*)optim;
    switch(kind)
    {
    case OPTIM_SGD:
        sgd_update((SGD*)optim, net);
        break;
    case OPTIM_ADAM:
        adam_update((Adam*)optim, net);
        break;
    default:
        ERROR("Unknown optim kind: %d", kind);
    }
}

void optim_free(Optim **optim)
{
    if (!optim || !*optim) return;

    OptimKind kind = *(OptimKind*)(*optim);
    switch(kind)
    {
    case OPTIM_SGD:
        sgd_free((SGD**)optim);
        break;
    case OPTIM_ADAM:
        adam_free((Adam**)optim);
        break;
    default:
        ERROR("Unknown optim kind: %d", kind);
    }
}
