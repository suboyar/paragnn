#ifndef EDGES_H
#define EDGES_H

#include <stdint.h>

#include "core.h"

typedef enum {
    EDGE_COO,
    EDGE_COMPRESSED,
} EdgeFormat;

typedef struct {
    EdgeFormat format;
    union {
        struct { // EDGE_COO
            int64_t *src;
            int64_t *dst;
        };
        // EDGE_COMPRESSED orientation is determined by SageLayer.flow:
        //   SOURCE_TO_TARGET: ptr groups by target, idx = sources (CSC-like)
        //   TARGET_TO_SOURCE: ptr groups by source, idx = targets (CSR-like)
        struct { // EDGE_COMPRESSED
            int64_t *ptr;  // [num_nodes + 1]
            int64_t *idx;  // [num_edges]
        };
    };
    uint8_t *self_loop; // O(1) lookup for node self-loops, NULL if none exist
    // Which degree (inward or outward) this holds is determined by SageLayer.flow:
    //   SOURCE_TO_TARGET: inv_degree[v] = 1/deg_in(v)
    //   TARGET_TO_SOURCE: inv_degree[v] = 1/deg_out(v)
    Real    *inv_degree;

    // Statistics
    float avg_self_loop;
    float avg_degree;
} Edges;

#endif // EDGES_H
