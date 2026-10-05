#ifndef DS_H
#define DS_H

#include <stdbool.h>
#include <stdint.h>

#include "core.h"
#include "dsinfo.h"
#include "sparsegraph.h"

typedef struct {
    char              *label_path;
    char              *feat_path;
    const DatasetInfo *info;
    int64_t            node_count;
    int64_t            feature_count;
    int64_t            class_count;
    int64_t            edge_count;
    Real              *x;       // Node features with shape [num_nodes, num_node_features]
    int64_t           *y;       // Labels to each node [num_nodes]
    int64_t           *y_train; // Labels to each node [num_nodes]
    int64_t           *y_valid; // Labels to each node [num_nodes]
    int64_t           *y_test;  // Labels to each node [num_nodes]
    SparseGraph       *graph;
} Dataset;

Dataset* dataset_load(DatasetKind dskind, char const *root, SparseFormat format);
void dataset_free(Dataset **ds);

#endif // DS_H
