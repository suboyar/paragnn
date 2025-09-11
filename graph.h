#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdint.h>

#include "matrix.h"

#define EDGE_AT(g, i, j) (g)->edge_index[(i)*(g)->num_edges + (j)]

typedef struct {
    size_t num_edges;
    size_t num_nodes;
    size_t num_node_features;
    size_t num_label_classes;
    size_t *edge_index; // Graph connectivity in COO format with shape [2, num_edges]
    matrix_t *x; // Node feature matrix with shape [num_nodes, num_node_features]
    matrix_t *y; // node-level targets of shape [num_nodes, num_label_classes] (One-hot encoding?)
    // Optional members
    size_t *node_year;
} graph_t;

#endif // GRAPH_H_
