#include <stdlib.h>

#include "graph.h"

#if TESTGRAPH

graph_t* load_graph()
{
    printf("Loading simple graph dataset\n");

    graph_t *g = malloc(sizeof(*g));

    g->num_edges = 4;
    g->num_nodes = 3;
    g->num_node_features = 1;
    g->num_label_classes = 2;

    size_t edges[] = {
        0, 0, 1, 2,
        1, 2, 2, 0,
    };
    // g->edge_index = malloc(sizeof(edges)); // Graph connectivity in COO format with shape [2, num_edges]
    // memcpy(g->edge_index, edges, sizeof(edges));
    g->edge_index = malloc(2 * g->num_edges * sizeof(g->edge_index)); // Graph connectivity in COO format with shape [2, num_edges]
    memcpy(g->edge_index, edges, 2 * g->num_edges * sizeof(g->edge_index));

    g->x = MAT_CREATE(g->num_nodes, g->num_node_features); // Node feature matrix with shape [num_nodes, num_node_features]
    mat_set(g->x, 0, 0, 1.0);
    mat_set(g->x, 1, 0, 2.0);
    mat_set(g->x, 2, 0, 3.0);

    g->y = MAT_CREATE(g->num_nodes, g->num_label_classes);  // node-level targets of shape [num_nodes, num_label_classes]
    mat_set(g->y, 0, 0, 1);
    mat_set(g->y, 1, 1, 1);
    mat_set(g->y, 2, 0, 1);

    return g;
}


#else // ogb-arxiv

#define NUM_NODES 169343
#define NUM_EDGES 1166243
#define NUM_LABELS 40
#define NUM_FEATURES 128

#ifndef PROCESSED_PATH
    #define PROCESSED_PATH "./dataset/arxiv/processed/"
#endif

graph_t* load_graph()
{
    printf("Loading ogb-arxiv from %s\n", PROCESSED_PATH);
    Nob_String_Builder sb = {0};
    char *p;
    const char *end;

    graph_t *g = malloc(sizeof(*g));
    g->x = mat_create(NUM_NODES, NUM_FEATURES);
    g->y = mat_create(NUM_NODES, NUM_LABELS);
    g->edge_index = malloc(2*NUM_EDGES * sizeof(*g->edge_index));
    g->node_year  = malloc(NUM_NODES * sizeof(*g->edge_index));

    // Edges
    nob_read_entire_file(PROCESSED_PATH"edge.csv", &sb);
    p = sb.items;
    end = sb.items + sb.count;

    for (size_t i = 0; i < NUM_EDGES && p < end; i++) {
        // Parse first node v
        size_t src = strtoull(p, &p, 10);
        if (p < end && *p == ',') p++; // Skip comma

        // Parse second node u
        size_t dst = strtoull(p, &p, 10);
        if (p < end && *p == '\n') p++; // Skip newline

        EDGE_AT(g, i, SRC_NODE) = src;
        EDGE_AT(g, i, DST_NODE) = dst;
    }
    sb.count = 0;

    // Features
    nob_read_entire_file(PROCESSED_PATH"node-feat.csv", &sb);
    p = sb.items;
    end = sb.items + sb.count;

    for (size_t i = 0; i < NUM_NODES && p < end; i++) {
        for (size_t j = 0; j < NUM_FEATURES && p < end; j++) {
            MAT_AT(g->x, i, j) = strtod(p, &p);
            if (p < end && *p == ',') p++; // Skip comma
        }

        if (p < end && *p == '\n') p++; // Skip newline
    }
    sb.count = 0;

    // Labels
    nob_read_entire_file(PROCESSED_PATH"node-label.csv", &sb);
    p = sb.items;
    end = sb.items + sb.count;

    for (size_t i = 0; i < NUM_NODES && p < end; i++) {
        for (size_t j = 0; j < NUM_LABELS; j++) {
            MAT_AT(g->y, i, j) = 0;
        }
        size_t label = strtoull(p, &p, 10);
        MAT_AT(g->y, i, label) = 1.0;
        if (p < end && *p == '\n') p++; // Skip newline
    }
    sb.count = 0;

    // Node year
    nob_read_entire_file(PROCESSED_PATH"node_year.csv", &sb);
    p = sb.items;
    end = sb.items + sb.count;

    for (size_t i = 0; i < NUM_NODES && p < end; i++) {
        g->node_year[i] = strtoull(p, &p, 10);
        if (p < end && *p == '\n') p++; // Skip newline
    }
    sb.count = 0;

    nob_sb_free(sb);
    return g;
}


#endif // TESTGRAPH




void destroy_graph(graph_t *g)
{
    free(g->edge_index);
    mat_destroy(g->x);
    mat_destroy(g->y);
    // Optional members
    if (g->node_year != NULL) free(g->node_year);

    free(g);
}
