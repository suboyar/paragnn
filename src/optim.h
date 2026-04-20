#ifndef OPTIM_H
#define OPTIM_H

#include "core.h"
#include "layers.h"

typedef enum {
    OPTIM_SGD,
    OPTIM_ADAM,
} OptimKind;

typedef void Optim;
Optim *optim_create(OptimKind kind, SageNet *net, Real lr);
void optim_update(Optim *optim, OptimKind kind, SageNet *net);
void optim_free(Optim **optim, OptimKind kind);

typedef struct {
    OptimKind kind;
    Real    lr;
} SGD;

SGD* sgd_create(Real lr);
void sgd_update(SGD *sgd, SageNet *net);
void sgd_free(SGD **sgd);

typedef struct {
    OptimKind  kind;
    Real    *m;               // first moment
    Real    *v;               // second moment
    int64_t  t;               // timestep
    Real     lr;
    Real     beta1;
    Real     beta2;
    Real     epsilon;
    Real     beta1_comp;      // 1 - beta1
    Real     beta2_comp;      // 1 - beta2
    Real     beta1_t;         // beta1^t, updated each step
    Real     beta2_t;         // beta2^t, updated each step
} AdamState;

typedef struct {
    OptimKind   kind;
    int64_t     num_states;
    AdamState **states;
} Adam;

Adam* adam_create(SageNet *net, Real lr);
void adam_update(Adam *adam, SageNet *net);
void adam_free(Adam **adam);

#endif // OPTIM_H
