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

#ifndef NN_H
#define NN_H

#include <stdint.h>

#include "layers.h"

void relu(ReluLayer* const l);
void l2norm(L2NormLayer* const l);
void logsoftmax(LogSoftmaxLayer* const l);

Real nll_loss(LogSoftmaxLayer *l, const int64_t *restrict y);
Real accuracy(const LogSoftmaxLayer *l, const int64_t *restrict y);

void grad_logsoftmax_nll(LogSoftmaxLayer *const l, const int64_t *restrict y);
void grad_l2norm(L2NormLayer* const l);
void grad_relu(ReluLayer* const l);

#endif // NN_H
