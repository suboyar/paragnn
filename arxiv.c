#include <stdio.h>
#include <zlib.h>

#include "nob.h"

#include "graph.h"

#define ERROR(fmt, ...) do { \
    fprintf(stderr, "%s:%d: ERROR: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    abort(); \
} while(0)

#define DATASET_PATH "./arxiv"

Nob_String_Builder sb = {0};

#define IDX(i, j, width) ((i) * (width) + (j))

#define CHUNK 0x1000 // 4kb window size
Nob_String_View read_gz(char* file_path)
{
    gzFile file = gzopen(file_path, "rb");
    if (!file) { ERROR("gzopen of '%s' failed: %s", file_path, strerror(errno)); }

    sb.count = 0;
    char buffer[CHUNK];
    while (1) {
        int bytes_read = gzread(file, buffer, CHUNK);
        if (bytes_read <= 0) {
            break;
        }

        nob_sb_append_buf(&sb, buffer, bytes_read);
    }

    int err = 0;
    const char *error_string;
    if (!gzeof(file)) {
        error_string = gzerror(file, &err);
    }

    gzclose(file);

    if (err){
        ERROR("gzread of '%s' failed: %s", file_path, error_string);
    }

    nob_sb_append_null(&sb);
    return nob_sb_to_sv(sb);
}

void load_arxiv_data(graph_t *arxiv)
{
    printf("Loading arxiv dataset\n");

    Nob_String_View sv = {0};
    sv = read_gz(DATASET_PATH"/raw/num-node-list.csv.gz");

    Nob_String_View num_node_sv = nob_sv_chop_by_delim(&sv, '\n');
    const char *num_node_cstr = nob_temp_sv_to_cstr(num_node_sv);
    arxiv->num_nodes = atol(num_node_cstr);

#if ENABLE_NODE_LIMIT
    // Limit nodes for testing purposes
    const size_t max_nodes = 100;
    if (arxiv->num_nodes > max_nodes) {
        fprintf(stderr, "DEBUG: Limiting nodes from %zu to %zu for testing\n",
                arxiv->num_nodes, max_nodes);
        arxiv->num_nodes = max_nodes;
    }
#endif

    sv = read_gz(DATASET_PATH"/raw/num-edge-list.csv.gz");
    Nob_String_View num_edge_sv = nob_sv_chop_by_delim(&sv, '\n');
    const char *num_edge_cstr = nob_temp_sv_to_cstr(num_edge_sv);
    arxiv->num_edges = atol(num_edge_cstr);

    sv = read_gz(DATASET_PATH"/raw/node_year.csv.gz");
    arxiv->node_year = malloc(arxiv->num_nodes * sizeof(*arxiv->node_year));
    if (arxiv->node_year == NULL) { ERROR("Failed to allocate memory for node years"); }

    for (size_t i = 0; i < arxiv->num_nodes && sv.count > 0; i++) {
        nob_temp_reset();
        Nob_String_View year_sv = nob_sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (year_sv.data[year_sv.count-1] == '\0') {
            ERROR("Empty line found in node_year");
        }

        const char *year_cstr = nob_temp_sv_to_cstr(year_sv);
        assert(strcmp(year_cstr, "") != 0);
        arxiv->node_year[i] = atol(year_cstr);
    }

    // TODO: Programmatically find the number of labels by counting unique
    // labels in ogbn_arxiv/raw/node-label.csv.gz
    arxiv->num_label_classes = 40;
    arxiv->y = mat_create(arxiv->num_nodes, arxiv->num_label_classes);
    if (arxiv->y == NULL) { ERROR("Failed to allocate memory for labels"); }
    sv = read_gz(DATASET_PATH"/raw/node-label.csv.gz");

    for (size_t i = 0; i < arxiv->num_nodes && sv.count > 0; i++) {
        Nob_String_View label_sv = nob_sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (label_sv.data[label_sv.count-1] == '\0') {
            ERROR("Empty line found in node_label");
        }

        const char *label_cstr = nob_temp_sv_to_cstr(label_sv);
        assert(strcmp(label_cstr, "") != 0);
        int label_index = atoi(label_cstr); // max 40 classes in ogbn-arxiv

        if ((size_t)label_index >= arxiv->num_label_classes) {
            ERROR("Label class index overflow: got %d, expected < %zu",
                        label_index, arxiv->num_label_classes-1);
        }

        MAT_AT(arxiv->y, i, label_index) = 1.0; // Everything else should have been inited to 0.0 by calloc
        nob_temp_reset();
    }

    // TODO: Programatically find the number of features through counting features
    // in the first line of ogbn_arxiv/raw/node-feat.csv.gz
    arxiv->num_node_features = 128;
    arxiv->x = mat_create(arxiv->num_nodes, arxiv->num_node_features);
    if (arxiv->x == NULL) { ERROR("Failed to allocate memory for features"); }
    sv = read_gz(DATASET_PATH"/raw/node-feat.csv.gz");

    for (size_t i = 0; i < arxiv->num_nodes && sv.count > 0; i++) {
        Nob_String_View feats_sv = nob_sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (feats_sv.data[feats_sv.count-1] == '\0') {
            ERROR("Empty line found in node_feature at line %zu", i+1);
        }

        for (size_t j = 0; feats_sv.count > 0; j++) {
            Nob_String_View feat_sv = nob_sv_chop_by_delim(&feats_sv, ',');
            const char *feat_cstr = nob_temp_sv_to_cstr(feat_sv);
            assert(strcmp(feat_cstr, "") != 0);
            MAT_AT(arxiv->x, i, j) = atof(feat_cstr);
            if (j >= arxiv->num_node_features) {
                ERROR("Feature index overflow: got %zu, expected < %zu",
                            j, arxiv->num_node_features);
            }
        }
        nob_temp_reset();
    }

    arxiv->edge_index = malloc(2 * arxiv->num_edges * sizeof(*arxiv->edge_index));
    if (arxiv->edge_index == NULL) { ERROR("Failed to allocate memory for edges"); }
    sv = read_gz(DATASET_PATH"/raw/edge.csv.gz");

    for (size_t i = 0; i < arxiv->num_edges && sv.count > 0; i++) {
        Nob_String_View edges_sv = nob_sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (edges_sv.data[edges_sv.count-1] == '\0') {
            ERROR("Empty line found in edge at line %zu", i+1);
        }

        // uint32_t v, u;
        for (size_t j = 0; edges_sv.count > 0; j++) {
            Nob_String_View edge_sv = nob_sv_chop_by_delim(&edges_sv, ',');
            const char *edge_cstr = nob_temp_sv_to_cstr(edge_sv);

            if (strcmp(edge_cstr, "") == 0) {
                ERROR("Empty edge string at position %zu in line %zu", j, i+1);
            }

            arxiv->edge_index[IDX(j, i, arxiv->num_edges)] = atol(edge_cstr);

            if (j >= 2) {
                ERROR("Too many edges: got %zu on line %zu, expected exactly 2",
                            j+1, i+1);
            }
        }
        nob_temp_reset();
    }
    nob_temp_reset();
    nob_sb_free(sb);
}
