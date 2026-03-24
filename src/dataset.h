#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdbool.h>
#include <stdint.h>

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
} Edges;

typedef struct {
    uint32_t num_nodes;
    uint32_t num_features;
    uint32_t num_classes;
    uint32_t num_edges;
    double   *nodes;            // Node features with shape [num_nodes, num_node_features]
    uint32_t *labels;           // Labels to each node [num_nodes]
    Edges edges;
} Dataset;

EdgeFormat parse_edge_format(const char* str);
Dataset* dataset_load_arxiv(EdgeFormat format, bool to_symmetric);
Dataset *dataset_split(Dataset *src, Split split);
void dataset_free(Dataset **ds);

#endif // GRAPH_H_
