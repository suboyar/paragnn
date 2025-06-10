#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <zlib.h>

#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#include "nob.h"

#define ERROR_PRINT(format, ...)                                        \
  fprintf(stderr, "[ERROR] %s:%d - " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define FATAL_ERROR(format, ...)                \
  do {                                          \
    ERROR_PRINT(format, ##__VA_ARGS__);         \
    exit(EXIT_FAILURE);                         \
  } while(0)

#define IDX(i, j, width) ((i) * (width) + (j))
#define IDX3(i, j, k, height, width) ((i) * (height) * (width) + (j) * (width) + (k))

typedef struct {
    double *data;
    size_t width;
    size_t height;
    size_t capacity;
} matrix_t;

typedef struct {
    size_t num_edges;
    size_t num_nodes;
    size_t num_node_features;
    size_t num_label_classes;
    uint32_t *edge_index; // Graph connectivity in COO format with shape [2, num_edges]
    uint32_t *node_year;
    double   *x; // Node feature matrix with shape [num_nodes, num_node_features]
    uint32_t *y; // node-level targets of shape [num_nodes, num_label_classes] or graph-level targets of shape [1, num_label_classes]
} ogb_arxiv_t;


#define CHUNK 0x1000 // 4kb window size

String_Builder sb = {0};

String_View read_gz(char* file_path)
{
    gzFile file = gzopen(file_path, "rb");
    if (!file) { FATAL_ERROR("gzopen of '%s' failed: %s", file_path, strerror(errno)); }


    sb.count = 0;
    char buffer[CHUNK];
    while (1) {
        int bytes_read = gzread(file, buffer, CHUNK);
        if (bytes_read <= 0) {
            break;
        }

        sb_append_buf(&sb, buffer, bytes_read);
    }

    int err = 0;
    const char *error_string;
    if (!gzeof(file)) {
        error_string = gzerror(file, &err);
    }

    gzclose(file);

    if (err){
        FATAL_ERROR("gzread of '%s' failed: %s", file_path, error_string);
    }

    sb_append_null(&sb);
    return sb_to_sv(sb);
}

#define RAW_PATH "/home/sboyar/D1/dataset/ogb/data/ogbn_arxiv/raw"

void load_data(ogb_arxiv_t *arxiv)
{
    String_View sv = {0};

    sv = read_gz(RAW_PATH"/num-node-list.csv.gz");
    String_View num_node_sv = sv_chop_by_delim(&sv, '\n');
    const char *num_node_cstr = temp_sv_to_cstr(num_node_sv);
    arxiv->num_nodes = atol(num_node_cstr);

    sv = read_gz(RAW_PATH"/num-edge-list.csv.gz");
    String_View num_edge_sv = sv_chop_by_delim(&sv, '\n');
    const char *num_edge_cstr = temp_sv_to_cstr(num_edge_sv);
    arxiv->num_edges = atol(num_edge_cstr);

    sv = read_gz(RAW_PATH"/node_year.csv.gz");
    arxiv->node_year = malloc(arxiv->num_nodes * sizeof(*arxiv->node_year));
    if (!arxiv->node_year) { FATAL_ERROR("Failed to allocate memory for node years"); }
    for (size_t i = 0; sv.count > 0; i++) {
        temp_reset();
        String_View year_sv = sv_chop_by_delim(&sv, '\n');
        // Jump over empty lines if any
        if (year_sv.data[year_sv.count-1] == '\0') { continue; }

        const char *year_cstr = temp_sv_to_cstr(year_sv);
        assert(strcmp(year_cstr, "") != 0);
        arxiv->node_year[i] = atol(year_cstr);
        assert (i < arxiv->num_nodes);
    }

    // TODO: Programatically find the number of labels through counting labels in
    // the first line of ogbn_arxiv/raw/node-label.csv.gz
    arxiv->num_label_classes = 1;
    arxiv->y = malloc(arxiv->num_nodes * arxiv->num_label_classes * sizeof(*arxiv->y));
    if (!arxiv->y) { FATAL_ERROR("Failed to allocate memory for labels"); }
    sv = read_gz(RAW_PATH"/node-label.csv.gz");

    for (size_t i = 0; sv.count > 0; i++) {
        String_View labels_sv = sv_chop_by_delim(&sv, '\n');
        // Jump over empty lines if any
        if (labels_sv.data[labels_sv.count-1] == '\0') { continue; }

        for (size_t j = 0; labels_sv.count > 0; j++) {
            temp_reset();
            String_View label_sv = sv_chop_by_delim(&labels_sv, ',');
            const char *label_cstr = temp_sv_to_cstr(label_sv);
            assert(strcmp(label_cstr, "") != 0);
            arxiv->y[IDX(i, j, arxiv->num_label_classes)] = atol(label_cstr);
            assert (j < arxiv->num_label_classes);
        }
        assert (i < arxiv->num_nodes);
    }

    // TODO: Programatically find the number of features through counting features
    // in the first line of ogbn_arxiv/raw/node-feat.csv.gz
    arxiv->num_node_features = 128;
    arxiv->x = malloc(arxiv->num_nodes * arxiv->num_node_features * sizeof(*arxiv->x));
    if (!arxiv->x) { FATAL_ERROR("Failed to allocate memory for features"); }
    sv = read_gz(RAW_PATH"/node-feat.csv.gz");

    for (size_t i = 0; sv.count > 0; i++) {
        String_View feats_sv = sv_chop_by_delim(&sv, '\n');
        // Jump over empty lines if any
        if (feats_sv.data[feats_sv.count-1] == '\0') { continue; }

        for (size_t j = 0; feats_sv.count > 0; j++) {
            temp_reset();
            String_View feat_sv = sv_chop_by_delim(&feats_sv, ',');
            const char *feat_cstr = temp_sv_to_cstr(feat_sv);
            assert(strcmp(feat_cstr, "") != 0);
            arxiv->x[IDX(i, j, arxiv->num_node_features)] = atof(feat_cstr);
            assert(j < arxiv->num_node_features);
        }
        assert(i < arxiv->num_nodes);
    }

    arxiv->edge_index = malloc(2 * arxiv->num_edges * sizeof(*arxiv->edge_index));
    if (!arxiv->edge_index) { FATAL_ERROR("Failed to allocate memory for edges"); }
    sv = read_gz(RAW_PATH"/edge.csv.gz");

    for (size_t i = 0; sv.count > 0; i++) {
        String_View edges_sv = sv_chop_by_delim(&sv, '\n');
        // Jump over empty lines if any
        if (edges_sv.data[edges_sv.count-1] == '\0') { continue; }

        // uint32_t v, u;
        for (size_t j = 0; edges_sv.count > 0; j++) {
            temp_reset();
            String_View edge_sv = sv_chop_by_delim(&edges_sv, ',');
            const char *edge_cstr = temp_sv_to_cstr(edge_sv);
            assert(strcmp(edge_cstr, "") != 0);
            arxiv->edge_index[IDX(j, i, arxiv->num_edges)] = atol(edge_cstr);

            assert(j < 2);
        }
        assert(i < arxiv->num_edges);
    }
}

matrix_t* matrix_create(size_t height, size_t width)
{
    matrix_t *mat = malloc(sizeof(matrix_t));
    if (!mat) return NULL;

    mat->data = calloc(height * width, sizeof(double));
    if (!mat->data) {
        free(mat);
        return NULL;
    }

    mat->height = height;
    mat->width = width;
    mat->capacity = height * width;
    return mat;
}

int main(void)
{
    int K = 2;
    uint8_t sample_size = 5;
    ogb_arxiv_t arxiv = {0};
    printf("Loading data\n");
    load_data(&arxiv);
    matrix_t **h = malloc((K+1) * sizeof(matrix_t*));
    for (int k = 0; k <= K; k++) {
        h[k] = matrix_create(arxiv.num_nodes, arxiv.num_node_features);
    }
    memcpy(h[0]->data, arxiv.x, h[0]->capacity * sizeof(double));

    assert(h[0]->data[10] == arxiv.x[10]);
    assert(h[0]->data[30] == arxiv.x[30]);
    assert(h[0]->data[100] == arxiv.x[100]);

    size_t *neighbor_ids = malloc(sample_size * sizeof(*neighbor_ids));
    matrix_t *neighbor_agg = matrix_create(1, arxiv.num_node_features);
    matrix_t *concat_features = matrix_create(1, 2 * arxiv.num_node_features);
    size_t input_dim = 2 * arxiv.num_node_features;
    size_t output_dim = arxiv.num_node_features;
    matrix_t *W = matrix_create(input_dim, output_dim);

    for (int k = 1; k <= K; k++) {
        printf("Staring k=%d\n", k);
        for (size_t v = 0; v < arxiv.num_nodes; v++) {
            uint8_t neigh_count = 0;

            // Find neighbors of count sample size
            for (size_t edge = 0; edge < arxiv.num_edges && neigh_count < sample_size; edge++) {
                if (arxiv.edge_index[IDX(0, edge, arxiv.num_edges)] == v) {
                    neighbor_ids[neigh_count++] = arxiv.edge_index[IDX(1, edge, arxiv.num_edges)];
                } else if (arxiv.edge_index[IDX(1, edge, arxiv.num_edges)] == v) {
                    neighbor_ids[neigh_count++] = arxiv.edge_index[IDX(0, edge, arxiv.num_edges)];
                }
            }

            // Aggrigate with mean
            size_t u = neighbor_ids[0];
            memcpy(neighbor_agg->data, &h[k-1]->data[IDX(u, 0, arxiv.num_node_features)], arxiv.num_node_features * sizeof(double));
            for (size_t i = 1; i < neigh_count; i++) {
                u = neighbor_ids[i];
                for (size_t x = 0; x < arxiv.num_node_features; x++) {
                    neighbor_agg->data[x] += h[k-1]->data[IDX(u, x, arxiv.num_node_features)];
                }
            }
            for (size_t x = 0; x < arxiv.num_node_features; x++) {
                neighbor_agg->data[x] /= sample_size;
            }

            // Copy self features
            for (size_t x = 0; x < arxiv.num_node_features; x++) {
                concat_features->data[x] = h[k-1]->data[IDX(v, x, arxiv.num_node_features)];
            }

            // Copy aggregated neighbor features
            for (size_t x = 0; x < arxiv.num_node_features; x++) {
                concat_features->data[arxiv.num_node_features + x] = neighbor_agg->data[x];
            }

            // Matrix multiplication
            for (size_t out_idx = 0; out_idx < output_dim; out_idx++) {
                double sum = 0.0;
                for (size_t in_idx = 0; in_idx < 2 * arxiv.num_node_features; in_idx++) {
                    sum += concat_features->data[in_idx] * W->data[IDX(in_idx, out_idx, arxiv.num_node_features)];
                }

                // ReLU activation
                h[k]->data[IDX(v, out_idx, arxiv.num_node_features)] = fmax(0.0, sum);
            }

            if (v > 0 && v % 10000 == 0) {
                printf("Finished %zu / %zu nodes\n", v, arxiv.num_nodes);
            }

        }

        for (size_t v = 0; v < arxiv.num_nodes; v++) {
            double norm = 0.0;
            for (size_t x = 0; x < arxiv.num_node_features; x++) {
                double val = h[K]->data[IDX(v, x, arxiv.num_node_features)];
                norm += val * val;
            }
            norm = sqrt(norm);

            // if (norm > 1e-8) {
            for (size_t x = 0; x < arxiv.num_node_features; x++) {
                h[k]->data[IDX(v, x, arxiv.num_node_features)] /= norm;
            }
            // }
        }
    }

    free(arxiv.node_year);
    free(arxiv.edge_index);
    free(arxiv.x);
    free(arxiv.y);
    sb_free(sb);
    return 0;
}


// TODO: Initilize weight matrix with random values (with a seed set)
// TODO: Implement gradient descent training
// TODO: Configurable layer dimensions
// TODO: Use CRS format for edges
// TODO: Clean up all allocated memory
