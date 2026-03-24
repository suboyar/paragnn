#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>

#include "optim.h"
#include "core.h"
#include "layers.h"
#include "timer.h"

// SGD optimizer

static inline
void sgd_step(SGD *restrict sgd, double *restrict param, const double *restrict grad, size_t n)
{
#pragma omp parallel for simd
    for (size_t i = 0; i < n; i++)
    {
        param[i] -= sgd->lr * grad[i];
    }
}

void sgd_update(SGD *sgd, SageNet *net)
{
    TIMER_FUNC();

    for (size_t i = 0; i < net->num_layers; i++)
    {
        Layer layer = net->layers[i];
        if (layer.type == LAYER_SAGE)
        {
            SageLayer *l = (SageLayer*)layer.ctx;
            sgd_step(sgd, l->Wroot, l->grad_Wroot, l->in_dim * l->out_dim);
            sgd_step(sgd, l->Wagg, l->grad_Wagg, l->in_dim * l->out_dim);
        }

        else if (layer.type == LAYER_LINEAR)
        {
            LinearLayer *l = (LinearLayer*)layer.ctx;
            sgd_step(sgd, l->W, l->grad_W, l->in_dim * l->out_dim);
            sgd_step(sgd, l->bias, l->grad_bias, l->out_dim);
        }
    }
}

SGD* sgd_create(double lr)
{
    SGD *sgd = malloc(sizeof(*sgd));
    if (!sgd) ERROR("Could not allocate SGD");
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

// __attribute__((optimize("unroll-loops")))
static void adam_step(AdamState *restrict s, double *restrict param, const double *restrict grad, size_t n)
{
    s->t++;
    s->beta1_t *= s->beta1;
    s->beta2_t *= s->beta2;
    const double bc = sqrt(1.0 - s->beta2_t) / (1.0 - s->beta1_t);
    const double lr_t = s->lr * bc;
#pragma omp parallel for simd
    for (size_t i = 0; i < n; i++)
    {
        const double g = grad[i];
        s->m[i] = s->beta1 * s->m[i] + s->beta1_comp * g;
        s->v[i] = s->beta2 * s->v[i] + s->beta2_comp * g * g;
        param[i] -=  lr_t * s->m[i] / (sqrt(s->v[i]) + s->epsilon);
    }
}

static void adam_state_reset(AdamState *s, size_t n, double lr)
{
    memset(s->m, 0, n * sizeof(*s->m));
    memset(s->v, 0, n * sizeof(*s->v));
    s->t          = 0;
    s->lr         = lr;
    s->beta1      = 0.9;
    s->beta2      = 0.999;
    s->epsilon    = 1e-8;
    s->beta1_comp = 1.0 - s->beta1;
    s->beta2_comp = 1.0 - s->beta2;
    s->beta1_t    = 1.0;
    s->beta2_t    = 1.0;
}

static AdamState *adam_state_create(size_t n, double lr)
{
    AdamState *s  = malloc(sizeof(*s));
    if (!s) ERROR("Could not allocate AdamState");
    s->kind       = OPTIM_ADAM;
    s->m          = cache_aligned_alloc(n*sizeof(double));
    s->v          = cache_aligned_alloc(n*sizeof(double));
    adam_state_reset(s, n, lr);

    // First touch
    double *dummy_grad = calloc(n, sizeof(*dummy_grad));
    double *dummy_param = calloc(n, sizeof(*dummy_param));
    if (!dummy_grad || !dummy_param) ERROR("Could not allocate dummy arrays");
    adam_step(s, dummy_param, dummy_grad, n);
    free(dummy_grad);
    free(dummy_param);

    // Reset state back to clean
    adam_state_reset(s, n, lr);

    return s;
}

Adam* adam_create(SageNet *net, double lr)
{
    Adam *adam = malloc(sizeof(*adam));
    if (!adam) ERROR("Could not allocate Adam");

    size_t count = 0;
    for (size_t i = 0; i < net->num_layers; i++)
    {
        switch (net->layers[i].type)
        {
        case LAYER_SAGE:   count += 2; break;  // Wroot, Wagg
        case LAYER_LINEAR: count += 2; break;  // W, bias
        default: break;
        }
    }

    adam->num_states = count;
    adam->states = malloc(count * sizeof(*adam->states));

    // Allocate a state for each parameter matrix
    size_t s = 0;
    for (size_t i = 0; i < net->num_layers; i++) {
        Layer layer = net->layers[i];
        if (layer.type == LAYER_SAGE)
        {
            SageLayer *l = (SageLayer*)layer.ctx;
            adam->states[s++] = adam_state_create(l->in_dim * l->out_dim, lr);
            adam->states[s++] = adam_state_create(l->in_dim  * l->out_dim,  lr);
        }
        else if (layer.type == LAYER_LINEAR)
        {
            LinearLayer *l = (LinearLayer*)layer.ctx;
            adam->states[s++] = adam_state_create(l->in_dim * l->out_dim,    lr);
            adam->states[s++] = adam_state_create(l->out_dim, lr);
        }
    }

    return adam;
}

void adam_update(Adam *adam, SageNet *net)
{
    size_t s = 0;
    for (size_t i = 0; i < net->num_layers; i++)
    {
        Layer layer = net->layers[i];
        if (layer.type == LAYER_SAGE)
        {
            SageLayer *l = (SageLayer*)layer.ctx;
            adam_step(adam->states[s++], l->Wroot, l->grad_Wroot, l->in_dim * l->out_dim);
            adam_step(adam->states[s++], l->Wagg, l->grad_Wagg, l->in_dim * l->out_dim);
        }
        else if (layer.type == LAYER_LINEAR)
        {
            LinearLayer *l = (LinearLayer*)layer.ctx;
            adam_step(adam->states[s++], l->W, l->grad_W, l->in_dim * l->out_dim);
            adam_step(adam->states[s++], l->bias, l->grad_bias, l->out_dim);
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
    for (size_t i = 0; i < (*adam)->num_states; i++)
    {
        adam_state_free(&(*adam)->states[i]);
    }
    free((*adam)->states); (*adam)->states = NULL;
    free(*adam);
    *adam = NULL;
}

// General interface
Optim *optim_create(OptimKind kind, SageNet *net, double lr)
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

void optim_update(Optim *optim, OptimKind kind, SageNet *net)
{
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

void optim_free(Optim **optim, OptimKind kind)
{
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
