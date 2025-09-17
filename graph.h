#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdint.h>

#include "matrix.h"

#define EDGE_AT(g, v, u) (g)->edge_index[(u)*(g)->num_edges + (v)]

#ifdef ROW_MAJOR
    #define GRAPH_AT(m, node, feat) MAT_AT(m, node, feat)
    #define GRAPH_NODES(m)              ((m)->height)
    #define GRAPH_FEATURES(m)           ((m)->width)
#else
    #define GRAPH_AT(m, node, feat)  MAT_AT(m, feat, node)
    #define GRAPH_NODES(m)               ((m)->width)
    #define GRAPH_FEATURES(m)            ((m)->height)
#endif

#ifndef NDEBUG
    #define GRAPH_BOUNDS_CHECK(m, n, f) do {                                       \
            if ((n) >= GRAPH_NODES((m))) {                                         \
                fprintf(stderr, "%s:%d: error: Node index %zu out of bounds\n",    \
                        __FILE__, __LINE__, (size_t)(n));                          \
                abort();                                                           \
            }                                                                      \
            if ((f) >= GRAPH_FEATURES((m))) {                                      \
                fprintf(stderr, "%s:%d: error: Feature index %zu out of bounds\n", \
                        __FILE__, __LINE__, (size_t)(f));                          \
                abort();                                                           \
            }                                                                      \
        } while(0)
#else
    #define GRAPH_BOUNDS_CHECK(m, n, f) (void)(0)
#endif

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
