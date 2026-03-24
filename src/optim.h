#ifndef OPTIM_H
#define OPTIM_H

#include "layers.h"

typedef enum {
    OPTIM_SGD,
    OPTIM_ADAM,
} OptimKind;

typedef void Optim;
Optim *optim_create(OptimKind kind, SageNet *net, double lr);
void optim_update(Optim *optim, OptimKind kind, SageNet *net);
void optim_free(Optim **optim, OptimKind kind);

typedef struct {
    OptimKind kind;
    double    lr;
} SGD;

SGD* sgd_create(double lr);
void sgd_update(SGD *sgd, SageNet *net);
void sgd_free(SGD **sgd);

typedef struct {
    OptimKind  kind;
    double    *m;               // first moment
    double    *v;               // second moment
    size_t     t;               // timestep
    double     lr;
    double     beta1;
    double     beta2;
    double     epsilon;
    double     beta1_comp;      // 1 - beta1
    double     beta2_comp;      // 1 - beta2
    double     beta1_t;         // beta1^t, updated each step
    double     beta2_t;         // beta2^t, updated each step
} AdamState;

typedef struct {
    OptimKind   kind;
    size_t            num_states;
    AdamState       **states;
} Adam;

Adam* adam_create(SageNet *net, double lr);
void adam_update(Adam *adam, SageNet *net);
void adam_free(Adam **adam);

#endif // OPTIM_H
