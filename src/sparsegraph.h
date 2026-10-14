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

#ifndef EDGES_H
#define EDGES_H

#include <stdint.h>
#include <stdbool.h>

#include "core.h"

typedef enum {
    SPARSE_FORMAT_COO = 1 << 0,
    SPARSE_FORMAT_CS  = 1 << 1, // Compressed Sparse (CSR & CSC)
    SPARSE_FORMAT_ALL = SPARSE_FORMAT_COO | SPARSE_FORMAT_CS
} SparseFormat;

typedef struct {
    SparseFormat format;
    int64_t node_count;
    int64_t edge_count;
    bool undirected;
    char *edge_path;

    // COO
    int64_t *src;
    int64_t *dst;

    // Orientation SageLayer.flow determens how they are used:
    //   SOURCE_TO_TARGET:
    //     Forward CSC: ptr_csc[target] -> sources
    //     Backward CSR: ptr_csr[source] -> targets
    //   TARGET_TO_SOURCE:
    //     Forward CSR: ptr_csr[source] -> targets
    //     Backward CSC: ptr_csc[target] -> sources

    // CSR + CSC
    int64_t *ptr_csr;  // [num_nodes + 1]
    int64_t *idx_csr;  // [num_edges]
    int64_t *ptr_csc;  // [num_nodes + 1]
    int64_t *idx_csc;  // [num_edges]

    // Which degree (inward or outward) to use is determined by SageLayer.flow:
    //   SOURCE_TO_TARGET: inv_in_degree[v]  = 1/deg_in(v)
    //   TARGET_TO_SOURCE: inv_out_degree[v] = 1/deg_out(v)
    Real    *inv_in_degree;
    Real    *inv_out_degree;
} SparseGraph;

SparseGraph* sparsegraph_load(int64_t node_count, int64_t edge_count, bool is_undirected, const char *edge_path, SparseFormat initial_format);
void sparsegraph_free(SparseGraph **graph);

#endif // EDGES_H
