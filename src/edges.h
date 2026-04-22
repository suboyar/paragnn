#ifndef EDGES_H
#define EDGES_H

#include <stdint.h>

#include "core.h"

typedef enum {
    EDGE_COO,
    EDGE_CSX,
} EdgeFormat;

typedef struct {
    EdgeFormat format;

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

    uint8_t *self_loop; // O(1) lookup for node self-loops, NULL if none exist
    // Which degree (inward or outward) this holds is determined by SageLayer.flow:
    //   SOURCE_TO_TARGET: inv_in_degree[v] = 1/deg_in(v)
    //   TARGET_TO_SOURCE: inv_out_degree[v] = 1/deg_out(v)
    Real    *inv_in_degree;
    Real    *inv_out_degree;

    // Statistics
    float avg_self_loop;
    float avg_degree;
} Edges;

#endif // EDGES_H
