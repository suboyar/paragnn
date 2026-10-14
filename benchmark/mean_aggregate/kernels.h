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

#ifndef KERNELS_H
#define KERNELS_H

#define KERNEL_TOUCH_ARGS                       \
    Real *restrict in_states,                   \
    Real *restrict agg_states,                  \
    int64_t dim,                                \
    SparseGraph *graph,                         \
    FlowDirection flow

#define KERNEL_ARGS                             \
    const Real *restrict in_states,             \
    Real *restrict agg_states,                  \
    int64_t dim,                                \
    SparseGraph *graph,                         \
    FlowDirection flow

typedef void (*TouchFunc)(KERNEL_TOUCH_ARGS);
typedef void (*KernelFunc)(KERNEL_ARGS);

void coo_v1_touch(KERNEL_TOUCH_ARGS);
void coo_v1(KERNEL_ARGS);

void cs_v1_touch(KERNEL_TOUCH_ARGS);
void cs_v1(KERNEL_ARGS);

void cs_v2_touch(KERNEL_TOUCH_ARGS);
void cs_v2(KERNEL_ARGS);

void cs_v3_touch(KERNEL_TOUCH_ARGS);
void cs_v3(KERNEL_ARGS);

void cs_v4_touch(KERNEL_TOUCH_ARGS);
void cs_v4(KERNEL_ARGS);

#endif // KERNELS_H
