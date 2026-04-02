#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdbool.h>
#include <stdint.h>

#include "core.h"

typedef enum {
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLIT_TEST,
} Split;

typedef enum {
    EDGE_COO,
    EDGE_CSR,
    EDGE_CSC,
} EdgeFormat;

typedef struct {
    EdgeFormat format;
    union {
        struct {
            uint32_t *src;
            uint32_t *dst;
        };
        struct {
            uint32_t *row_ptr;  // [num_nodes + 1]
            uint32_t *col_idx;  // [num_edges]
        };
        struct {
            uint32_t *col_ptr;  // [num_nodes + 1]
            uint32_t *row_idx;  // [num_edges]
        };
    };
    uint8_t *self_loop;      // O(1) lookup for node self-loops, NULL if none exist
    Real    *inv_in_degree;  // 1/deg_in(v), for source_to_target aggregation
    Real    *inv_out_degree; // 1/deg_out(v), for target_to_source aggregation

    // Statistics
    float   avg_self_loop;
    float   avg_degree;
} Edges;

typedef struct {
    char *path;
    uint32_t num_nodes;
    uint32_t num_features;
    uint32_t num_classes;
    uint32_t num_edges;
    double   *nodes;            // Node features with shape [num_nodes, num_node_features]
    uint32_t *labels;           // Labels to each node [num_nodes]
    Edges edges;
} Dataset;

EdgeFormat parse_edge_format(const char* str);
Dataset* dataset_load(char const* dataset, char const* data_dir, EdgeFormat format, bool to_symmetric);
Dataset *dataset_split(Dataset *base, Split split);
void dataset_free(Dataset **ds);

#endif // GRAPH_H_
