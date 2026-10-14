/*
 * ParaGNN
 * Copyright (C) 2026  Suleyman Boyar
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
void optim_update(Optim *optim, SageNet *net);
void optim_free(Optim **optim);

typedef struct {
    OptimKind kind;
    Real    lr;
} SGD;

SGD* sgd_create(Real lr);
void sgd_update(SGD *sgd, SageNet *net);
void sgd_free(SGD **sgd);

typedef struct {
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
