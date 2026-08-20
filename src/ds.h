#ifndef DATASET_H_
#define DATASET_H_

#include <stdbool.h>
#include <stdint.h>

#include "core.h"
#include "dsinfo.h"
#include "sparsegraph.h"

typedef enum {
    SPLIT_NONE = 0,
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLIT_TEST,
} Split;

typedef struct {
    char        *label_path;
    char        *feat_path;
    Split        split;
    const DatasetInfo *info;
    // TODO: rename num_* -> *_count
    int64_t      num_nodes;
    int64_t      num_features;
    int64_t      num_classes;
    int64_t      num_edges;
    // TODO: rename nodes -> xs and lables -> ys
    Real        *nodes;            // Node features with shape [num_nodes, num_node_features]
    int64_t     *labels;           // Labels to each node [num_nodes]
    SparseGraph *graph;
} Dataset;

Dataset* dataset_alloc(DatasetKind dskind, char const *root, SparseFormat format, Split split);
void dataset_load(Dataset *ds);
void dataset_free(Dataset **ds);

#endif // DATASET_H_
