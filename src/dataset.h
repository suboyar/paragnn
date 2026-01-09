#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdint.h>

#include "matrix.h"

#ifndef SRC_NODE
#define SRC_NODE 0
#endif
#ifndef DST_NODE
#define DST_NODE 1
#endif

#ifdef USE_CSR_EDGE
    #define EDGE_ROW_START(e, i)  ((e)->row_ptr[i])
    #define EDGE_ROW_END(e, i)    ((e)->row_ptr[(i) + 1])
    #define EDGE_COL(e, k)        ((e)->col_idx[k])
#else
    #define EDGE_SRC(e, k)        ((e)->data[2 * (k) + 0])
    #define EDGE_DST(e, k)        ((e)->data[2 * (k) + 1])
#endif

typedef struct {
    uint32_t num_nodes;
    uint32_t num_edges;
#ifdef USE_CSR_EDGE
    uint32_t *row_ptr;
    uint32_t *col_idx;
    double *val;
#else
    uint32_t *data; // COO format stored as [src0, dst0, src1, dst1, ...]
#endif
} EdgeIndex;

typedef struct {
    size_t start;
    size_t end;
} Range;

typedef struct {
    Range node;
    Range edge;
} Slice;

typedef struct {
    uint32_t num_edges;
    uint32_t num_inputs;
    uint32_t num_features;
    uint32_t num_classes;
    double *inputs; // Node features with shape [num_nodes, num_node_features]
    uint32_t *labels; // Labels to each node [num_nodes]
    EdgeIndex edges;

    Slice train;
    Slice valid;
    Slice test;
    Slice full;
} Dataset;

Dataset* load_arxiv_dataset();
void destroy_dataset(Dataset *ds);

#endif // GRAPH_H_
