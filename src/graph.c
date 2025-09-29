#include <stdlib.h>
#include <stdint.h>

#include "graph.h"

static graph_t* init_graph(size_t num_nodes, size_t num_edges, size_t num_features, size_t num_labels)
{
    graph_t* g = malloc(sizeof(*g));

    g->num_edges = 0;   // Edges needs to be counted one by one while partitioning the graph into train/valid/test splits
    g->num_nodes = num_nodes;
    g->num_node_features = num_features;
    g->num_label_classes = num_labels;

    g->edge_index = malloc(2*num_edges * sizeof(*g->edge_index)); // Graph connectivity in COO format with shape [2, num_edges]
    g->node_year  = malloc(num_nodes * sizeof(*g->edge_index));

    g->x = mat_create(num_nodes, num_features);
    g->y = mat_create(num_nodes, num_labels);

    return g;
}


#ifndef USE_OGB_ARXIV

#define NUM_NODES 6
#define NUM_EDGES 6
#define NUM_LABELS 2
#define NUM_FEATURES 3

// Split sizes for test graph
#define NUM_TRAIN_NODES 2
#define NUM_VALID_NODES 2
#define NUM_TEST_NODES 2

#define NUM_TRAIN_EDGES 2
#define NUM_VALID_EDGES 2
#define NUM_TEST_EDGES 2

graph_t* load_graph()
{
    printf("Loading TEST graph from - %zu nodes\n", (size_t)NUM_NODES);

    graph_t* g = init_graph(NUM_NODES, NUM_EDGES, NUM_FEATURES, NUM_LABELS);

    // Edges: creating a graph where each pair of nodes in each split is connected
    size_t edges[] = {
        0, 1, 2, 3, 4, 5,  // source nodes
        1, 0, 3, 2, 5, 4,  // destination nodes
    };

    memcpy(g->edge_index, edges, 2 * NUM_EDGES * sizeof(*g->edge_index));

    // Train nodes (0, 1)
    mat_set(g->x, 0, 0, 1.0); mat_set(g->x, 0, 1, 0.5); mat_set(g->x, 0, 2, 2.0);
    mat_set(g->x, 1, 0, 1.5); mat_set(g->x, 1, 1, 0.8); mat_set(g->x, 1, 2, 2.2);

    // Valid nodes (2, 3)
    mat_set(g->x, 2, 0, 2.0); mat_set(g->x, 2, 1, 1.0); mat_set(g->x, 2, 2, 1.8);
    mat_set(g->x, 3, 0, 2.3); mat_set(g->x, 3, 1, 1.2); mat_set(g->x, 3, 2, 1.5);

    // Test nodes (4, 5)
    mat_set(g->x, 4, 0, 3.0); mat_set(g->x, 4, 1, 1.5); mat_set(g->x, 4, 2, 1.0);
    mat_set(g->x, 5, 0, 3.2); mat_set(g->x, 5, 1, 1.8); mat_set(g->x, 5, 2, 0.8);

    // Train labels
    mat_set(g->y, 0, 0, 1.0); mat_set(g->y, 0, 1, 0.0); // class 0
    mat_set(g->y, 1, 0, 0.0); mat_set(g->y, 1, 1, 1.0); // class 1

    // Valid labels
    mat_set(g->y, 2, 0, 1.0); mat_set(g->y, 2, 1, 0.0); // class 0
    mat_set(g->y, 3, 0, 0.0); mat_set(g->y, 3, 1, 1.0); // class 1

    // Test labels
    mat_set(g->y, 4, 0, 0.0); mat_set(g->y, 4, 1, 1.0); // class 1
    mat_set(g->y, 5, 0, 1.0); mat_set(g->y, 5, 1, 0.0); // class 0

    // Set node years for completeness
    g->node_year[0] = 2018; g->node_year[1] = 2019;
    g->node_year[2] = 2019; g->node_year[3] = 2020;
    g->node_year[4] = 2020; g->node_year[5] = 2021;

    return g;
}

void load_split_graph(graph_t** train_graph, graph_t** valid_graph, graph_t** test_graph)
{
    printf("Loading TEST graph split from - ");
    printf("Train: %zu nodes | Valid: %zu nodes | Test: %zu nodes\n",
           (size_t)NUM_TRAIN_NODES, (size_t)NUM_VALID_NODES, (size_t)NUM_TEST_NODES);

    // Initialize the three subgraphs
    *train_graph = init_graph(NUM_TRAIN_NODES, NUM_TRAIN_EDGES, NUM_FEATURES, NUM_LABELS);
    *valid_graph = init_graph(NUM_VALID_NODES, NUM_VALID_EDGES, NUM_FEATURES, NUM_LABELS);
    *test_graph = init_graph(NUM_TEST_NODES, NUM_TEST_EDGES, NUM_FEATURES, NUM_LABELS);

    // Train subgraph (nodes 0, 1 -> remapped to 0, 1)
    // Edges: (0,1), (1,0)
    EDGE_AT(*train_graph, 0, SRC_NODE) = 0; EDGE_AT(*train_graph, 0, DST_NODE) = 1;
    EDGE_AT(*train_graph, 1, SRC_NODE) = 1; EDGE_AT(*train_graph, 1, DST_NODE) = 0;

    // Train features
    mat_set((*train_graph)->x, 0, 0, 1.0); mat_set((*train_graph)->x, 0, 1, 0.5); mat_set((*train_graph)->x, 0, 2, 2.0);
    mat_set((*train_graph)->x, 1, 0, 1.5); mat_set((*train_graph)->x, 1, 1, 0.8); mat_set((*train_graph)->x, 1, 2, 2.2);

    // Train labels
    mat_set((*train_graph)->y, 0, 0, 1.0); mat_set((*train_graph)->y, 0, 1, 0.0);
    mat_set((*train_graph)->y, 1, 0, 0.0); mat_set((*train_graph)->y, 1, 1, 1.0);

    // Train node years
    (*train_graph)->node_year[0] = 2018;
    (*train_graph)->node_year[1] = 2019;

    // Valid subgraph (nodes 2, 3 -> remapped to 0, 1)
    // Edges: (0,1), (1,0) (remapped from (2,3), (3,2))
    EDGE_AT(*valid_graph, 0, SRC_NODE) = 0; EDGE_AT(*valid_graph, 0, DST_NODE) = 1;
    EDGE_AT(*valid_graph, 1, SRC_NODE) = 1; EDGE_AT(*valid_graph, 1, DST_NODE) = 0;

    // Valid features
    mat_set((*valid_graph)->x, 0, 0, 2.0); mat_set((*valid_graph)->x, 0, 1, 1.0); mat_set((*valid_graph)->x, 0, 2, 1.8);
    mat_set((*valid_graph)->x, 1, 0, 2.3); mat_set((*valid_graph)->x, 1, 1, 1.2); mat_set((*valid_graph)->x, 1, 2, 1.5);

    // Valid labels
    mat_set((*valid_graph)->y, 0, 0, 1.0); mat_set((*valid_graph)->y, 0, 1, 0.0);
    mat_set((*valid_graph)->y, 1, 0, 0.0); mat_set((*valid_graph)->y, 1, 1, 1.0);

    // Valid node years
    (*valid_graph)->node_year[0] = 2019;
    (*valid_graph)->node_year[1] = 2020;

    // Test subgraph (nodes 4, 5 -> remapped to 0, 1)
    // Edges: (0,1), (1,0) (remapped from (4,5), (5,4))
    EDGE_AT(*test_graph, 0, SRC_NODE) = 0; EDGE_AT(*test_graph, 0, DST_NODE) = 1;
    EDGE_AT(*test_graph, 1, SRC_NODE) = 1; EDGE_AT(*test_graph, 1, DST_NODE) = 0;

    // Test features
    mat_set((*test_graph)->x, 0, 0, 3.0); mat_set((*test_graph)->x, 0, 1, 1.5); mat_set((*test_graph)->x, 0, 2, 1.0);
    mat_set((*test_graph)->x, 1, 0, 3.2); mat_set((*test_graph)->x, 1, 1, 1.8); mat_set((*test_graph)->x, 1, 2, 0.8);

    // Test labels
    mat_set((*test_graph)->y, 0, 0, 0.0); mat_set((*test_graph)->y, 0, 1, 1.0);
    mat_set((*test_graph)->y, 1, 0, 1.0); mat_set((*test_graph)->y, 1, 1, 0.0);

    // Test node years
    (*test_graph)->node_year[0] = 2020;
    (*test_graph)->node_year[1] = 2021;
}

#else // ogb-arxiv

// Then number of nodes and edges for each split has been pre-counted

#define NUM_NODES 169343
#define NUM_TRAIN_NODES 90941
#define NUM_VALID_NODES 29799
#define NUM_TEST_NODES 48603

// A ~60% reduction of number of edges
#define NUM_EDGES 1166243
#define NUM_TRAIN_EDGES 374839
#define NUM_VALID_EDGES 29119
#define NUM_TEST_EDGES 60403

#define NUM_LABELS 40
#define NUM_FEATURES 128

#ifndef PROCESSED_PATH
    #define PROCESSED_PATH "./dataset/arxiv/processed/"
#endif

graph_t* load_graph()
{
    printf("Loading ogb-arxiv graph from [%s] - %zu nodes", PROCESSED_PATH, (size_t)NUM_NODES);
    Nob_String_Builder sb = {0};
    char *p;
    const char *end;

    graph_t *g = init_graph(NUM_NODES, NUM_EDGES, NUM_FEATURES, NUM_LABELS);

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

typedef enum {
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    INVALID_SPLIT
} graph_split_t;

void create_split_mapping(graph_split_t* split_map, size_t* train_split_idx, size_t* valid_split_idx, size_t* test_split_idx)
{
    // Initialize all as invalid
    for (size_t i = 0; i < NUM_NODES; i++) {
        split_map[i] = INVALID_SPLIT;
    }

    // Mark train nodes
    for (size_t i = 0; i < NUM_TRAIN_NODES; i++) {
        split_map[train_split_idx[i]] = TRAIN_SPLIT;
    }

    // Mark valid nodes
    for (size_t i = 0; i < NUM_VALID_NODES; i++) {
        split_map[valid_split_idx[i]] = VALID_SPLIT;
    }

    // Mark test nodes
    for (size_t i = 0; i < NUM_TEST_NODES; i++) {
        split_map[test_split_idx[i]] = TEST_SPLIT;
    }
}

static inline void parse_split_data(const char* path, size_t* split_idx, size_t split_size, Nob_String_Builder* sb)
{
    nob_read_entire_file(path, sb);
    char* p = sb->items;
    const char* end = sb->items + sb->count;

    for (size_t i = 0; i < split_size && p < end; i++) {
        // Parse first node v
        split_idx[i] = strtoull(p, &p, 10);
        if (p < end && *p == '\n') p++; // Skip newline
    }
    sb->count = 0;
}

static inline void create_node_mappings(size_t* split_map, size_t* split_idx, size_t split_size) {
    for (size_t i = 0; i < NUM_NODES; i++) {
        split_map[i] = SIZE_MAX; // Invalid indices
    }

    for (size_t i = 0; i < split_size; i++) {
        split_map[split_idx[i]] = i;
    }
}

void load_split_graph(graph_t** train_graph, graph_t** valid_graph, graph_t** test_graph)
{
    printf("Loading ogb-arxiv graph split from [%s] - ", PROCESSED_PATH);
    printf("Train: %zu nodes | Valid: %zu nodes | Test: %zu nodes\n",
           (size_t)NUM_TRAIN_NODES, (size_t)NUM_VALID_NODES, (size_t)NUM_TEST_NODES);

    // Allocate some
    graph_split_t* split_map = malloc(NUM_NODES * sizeof(*split_map));

    size_t* train_split_idx = malloc(NUM_TRAIN_NODES * sizeof(*train_split_idx));
    size_t* valid_split_idx = malloc(NUM_VALID_NODES * sizeof(*valid_split_idx));
    size_t* test_split_idx  = malloc(NUM_TEST_NODES * sizeof(*test_split_idx));

    // A sparse mapping
    size_t* train_split_map = malloc(NUM_NODES * sizeof(*train_split_map));
    size_t* valid_split_map = malloc(NUM_NODES * sizeof(*valid_split_map));
    size_t* test_split_map  = malloc(NUM_NODES * sizeof(*test_split_map));

    Nob_String_Builder sb = {0};

    // Load the split data and parse them to size_t
    parse_split_data(PROCESSED_PATH"train.csv", train_split_idx, NUM_TRAIN_NODES, &sb);
    parse_split_data(PROCESSED_PATH"valid.csv", valid_split_idx, NUM_VALID_NODES, &sb);
    parse_split_data(PROCESSED_PATH"test.csv",  test_split_idx,  NUM_TEST_NODES,  &sb);

    // Node Mapping
    create_split_mapping(split_map, train_split_idx, valid_split_idx, test_split_idx);

    // Create ...
    create_node_mappings(train_split_map, train_split_idx, NUM_TRAIN_NODES);
    create_node_mappings(valid_split_map, valid_split_idx, NUM_VALID_NODES);
    create_node_mappings(test_split_map,  test_split_idx,  NUM_TEST_NODES);

    // Initilize the graphs
    *train_graph = init_graph(NUM_TRAIN_NODES, NUM_TRAIN_EDGES, NUM_FEATURES, NUM_LABELS);
    *valid_graph = init_graph(NUM_VALID_NODES, NUM_VALID_EDGES, NUM_FEATURES, NUM_LABELS);
    *test_graph = init_graph(NUM_TEST_NODES, NUM_TEST_EDGES, NUM_FEATURES, NUM_LABELS);

    char *p;
    const char *end;

    // Edges
    nob_read_entire_file(PROCESSED_PATH"edge.csv", &sb);
    p = sb.items;
    end = sb.items + sb.count;

    size_t train_edge_idx = 0;
    size_t valid_edge_idx = 0;
    size_t test_edge_idx = 0;
    for (size_t i = 0; i < NUM_EDGES && p < end; i++) {
        // Parse first node v
        size_t src = strtoull(p, &p, 10);
        if (p < end && *p == ',') p++; // Skip comma

        // Parse second node u
        size_t dst = strtoull(p, &p, 10);
        if (p < end && *p == '\n') p++; // Skip newline

        graph_split_t src_split = split_map[src];
        graph_split_t dst_split = split_map[dst];
        graph_split_t split = (src_split == dst_split) ? src_split : INVALID_SPLIT;

        switch(split) {
        case TRAIN_SPLIT:
            EDGE_AT(*train_graph, train_edge_idx, SRC_NODE) = train_split_map[src];
            EDGE_AT(*train_graph, train_edge_idx, DST_NODE) = train_split_map[dst];
            train_edge_idx++;
            break;
        case VALID_SPLIT:
            EDGE_AT(*valid_graph, valid_edge_idx, SRC_NODE) = valid_split_map[src];
            EDGE_AT(*valid_graph, valid_edge_idx, DST_NODE) = valid_split_map[dst];
            valid_edge_idx++;
            break;
        case TEST_SPLIT:
            EDGE_AT(*test_graph, test_edge_idx, SRC_NODE) = test_split_map[src];
            EDGE_AT(*test_graph, test_edge_idx, DST_NODE) = test_split_map[dst];
            test_edge_idx++;
            break;
        default:
            break;
        }
    }
    sb.count = 0;

    // Features
    nob_read_entire_file(PROCESSED_PATH"node-feat.csv", &sb);
    p = sb.items;
    end = sb.items + sb.count;

    for (size_t i = 0; i < NUM_NODES && p < end; i++) {
        graph_t* sub_graph = NULL;
        size_t idx;
        graph_split_t split = split_map[i];

        switch (split) {
        case TRAIN_SPLIT:
            sub_graph = *train_graph;
            idx = train_split_map[i];
            break;
        case VALID_SPLIT:
            sub_graph = *valid_graph;
            idx = valid_split_map[i];
            break;
        case TEST_SPLIT:
            sub_graph = *test_graph;
            idx = test_split_map[i];
            break;
        default:
            assert(false && "Found invalid graph");
        }

        for (size_t j = 0; j < NUM_FEATURES && p < end; j++) {
            MAT_AT(sub_graph->x, idx, j) = strtod(p, &p);
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
        graph_t* sub_graph = NULL;
        size_t idx;
        graph_split_t split = split_map[i];

        switch (split) {
        case TRAIN_SPLIT:
            sub_graph = *train_graph;
            idx = train_split_map[i];
            break;
        case VALID_SPLIT:
            sub_graph = *valid_graph;
            idx = valid_split_map[i];
            break;
        case TEST_SPLIT:
            sub_graph = *test_graph;
            idx = test_split_map[i];
            break;
        default:
            assert(false && "Found invalid graph");
        }

        for (size_t j = 0; j < NUM_LABELS; j++) {
            MAT_AT(sub_graph->y, idx, j) = 0;
        }
        size_t label = strtoull(p, &p, 10);
        MAT_AT(sub_graph->y, idx, label) = 1.0;
        if (p < end && *p == '\n') p++; // Skip newline
    }
    sb.count = 0;

    // Node year
    nob_read_entire_file(PROCESSED_PATH"node_year.csv", &sb);
    p = sb.items;
    end = sb.items + sb.count;

    for (size_t i = 0; i < NUM_NODES && p < end; i++) {
        graph_t* sub_graph = NULL;
        size_t idx;
        graph_split_t split = split_map[i];

        switch (split) {
        case TRAIN_SPLIT:
            sub_graph = *train_graph;
            idx = train_split_map[i];
            break;
        case VALID_SPLIT:
            sub_graph = *valid_graph;
            idx = valid_split_map[i];
            break;
        case TEST_SPLIT:
            sub_graph = *test_graph;
            idx = test_split_map[i];
            break;
        default:
            assert(false && "Found invalid graph");
        }

        sub_graph->node_year[idx] = strtoull(p, &p, 10);
        if (p < end && *p == '\n') p++; // Skip newline
    }
    sb.count = 0;


    free(split_map);

    free(train_split_idx);
    free(valid_split_idx);
    free(test_split_idx);

    free(train_split_map);
    free(valid_split_map);
    free(test_split_map);

    nob_sb_free(sb);
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
