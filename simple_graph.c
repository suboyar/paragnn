#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "matrix.h"

// Creates a 3-node directed graph where each node connects to the other two.
// Nodes have features [1.0, 2.0, 3.0] and belong to classes [0, 1, 2] respectively.

void load_simple_data(graph_t *graph)
{
    printf("Loading simple graph dataset\n");

    graph->num_edges = 6;
    graph->num_nodes = 3;
    graph->num_node_features = 1;
    graph->num_label_classes = 3;

    size_t edges[] = {
        0, 0, 1, 1, 2, 2,
        1, 2, 0, 2, 0, 1
    };
    // graph->edge_index = malloc(sizeof(edges)); // Graph connectivity in COO format with shape [2, num_edges]
    // memcpy(graph->edge_index, edges, sizeof(edges));
    graph->edge_index = malloc(2 * graph->num_edges * sizeof(graph->edge_index)); // Graph connectivity in COO format with shape [2, num_edges]
    memcpy(graph->edge_index, edges, 2 * graph->num_edges * sizeof(graph->edge_index));

    graph->x = mat_create(graph->num_nodes, graph->num_node_features); // Node feature matrix with shape [num_nodes, num_node_features]
    mat_set(graph->x, 0, 0, 1.0);
    mat_set(graph->x, 1, 0, 2.0);
    mat_set(graph->x, 2, 0, 3.0);

    graph->y = mat_create(graph->num_nodes, graph->num_label_classes);  // node-level targets of shape [num_nodes, num_label_classes]
    mat_set(graph->y, 0, 0, 1);
    mat_set(graph->y, 1, 1, 1);
    mat_set(graph->y, 2, 2, 1);
}

// void load_simple_data(graph_t *graph)
// {
//     printf("Loading simple graph dataset\n");

//     graph->num_edges = 4;
//     graph->num_nodes = 3;
//     graph->num_node_features = 1;
//     graph->num_label_classes = 2;

//     size_t edges[] = {
//         1, 2, 2, 1,     // source nodes
//         0, 0, 1, 2      // target nodes
//     };

//     graph->edge_index = malloc(2 * graph->num_edges * sizeof(graph->edge_index)); // Graph connectivity in COO format with shape [2, num_edges]
//     memcpy(graph->edge_index, edges, 2 * graph->num_edges * sizeof(graph->edge_index));

//     graph->x = mat_create(graph->num_nodes, graph->num_node_features); // Node feature matrix with shape [num_nodes, num_node_features]
//     mat_set(graph->x, 0, 0, 1.0);
//     mat_set(graph->x, 1, 0, 2.0);
//     mat_set(graph->x, 2, 0, 3.0);

//     graph->y = mat_create(graph->num_nodes, graph->num_label_classes);  // node-level targets of shape [num_nodes, num_label_classes]
//     mat_set(graph->y, 0, 0, 1.0);
//     mat_set(graph->y, 1, 1, 1.0);
//     mat_set(graph->y, 2, 0, 1.0);

// }
