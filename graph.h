#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdint.h>

#include "matrix.h"

#define SRC_NODE 0
#define DST_NODE 1

#define EDGE_AT(g, edge, end) (g)->edge_index[(end)*(g)->num_edges + (edge)]
#define GRAPH_NODES(m)    ((m)->height)
#define GRAPH_FEATURES(m) ((m)->width)

#ifndef NDEBUG
#define EDGE_BOUNDS_CHECK(g, v, u) do {                                                           \
        if (EDGE_AT((g), (v), (u)) >= (g)->num_nodes) {                                           \
            fprintf(stderr, "Edge %zu endpoint %zu points to out-of-bounds node %zu (max %zu)\n", \
                    (size_t)(v), (size_t)(u), (size_t)EDGE_AT((g), (v), (u)), (g)->num_nodes);    \
            abort();                                                                              \
        }                                                                                         \
    }                                                                                             \
    while(0);
#else
    #define EDGE_BOUNDS_CHECK(g, v, u) (void)(0)
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

graph_t* load_graph();

void destroy_graph(graph_t *g);

#endif // GRAPH_H_
