#ifndef DATASET_H_
#define DATASET_H_

#include <stdbool.h>
#include <stdint.h>

#include "core.h"
#include "dataset_info.h"
#include "edges.h"
#include "flow.h"

typedef enum {
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLIT_TEST,
} Split;

typedef struct {
    char *path;
    int64_t  num_nodes;
    int64_t  num_features;
    int64_t  num_classes;
    int64_t  num_edges;
    Real    *nodes;            // Node features with shape [num_nodes, num_node_features]
    int64_t *labels;           // Labels to each node [num_nodes]
    Edges    edges;
} Dataset;

EdgeFormat parse_edge_format(const char* str);
Dataset* dataset_load(DatasetKind dataset, char const* data_dir, EdgeFormat format, bool to_symmetric, FlowDirection flow);
Dataset *dataset_split(Dataset *base, Split split, FlowDirection flow);
void dataset_free(Dataset **ds);

#endif // DATASET_H_
