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

#define todo(str, ...) FATAL_ERROR("TODO: %s", str, ##__VA_ARGS__)

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
    matrix_t *x; // Node feature matrix with shape [num_nodes, num_node_features]
    matrix_t *y; // node-level targets of shape [num_nodes, num_label_classes] or graph-level targets of shape [1, num_label_classes]
} ogb_arxiv_t;

size_t input_dim;
size_t output_dim;
size_t hidden_layer_size;
size_t hidden_layer_count;
size_t K;
size_t sample_size;

#define CHUNK 0x1000 // 4kb window size

String_Builder sb = {0};

matrix_t* matrix_create(size_t height, size_t width)
{
    matrix_t *mat = malloc(sizeof(matrix_t));
    if (!mat) return NULL;

    mat->data = calloc(height * width, sizeof(*mat->data));
    if (!mat->data) {
        free(mat);
        return NULL;
    }

    mat->height = height;
    mat->width = width;
    mat->capacity = height * width;
    return mat;
}

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

#define RAW_PATH "/home/sboyar/D1/dataset/ogbn_arxiv/raw/"

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
    if (arxiv->node_year == NULL) { FATAL_ERROR("Failed to allocate memory for node years"); }
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
    arxiv->y = matrix_create(arxiv->num_nodes, arxiv->num_label_classes);
    // arxiv->y = malloc(arxiv->num_nodes * arxiv->num_label_classes * sizeof(*arxiv->y));
    if (arxiv->y == NULL) { FATAL_ERROR("Failed to allocate memory for labels"); }
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
            arxiv->y->data[IDX(i, j, arxiv->num_label_classes)] = atol(label_cstr);
            assert (j < arxiv->num_label_classes);
        }
        assert (i < arxiv->num_nodes);
    }

    // TODO: Programatically find the number of features through counting features
    // in the first line of ogbn_arxiv/raw/node-feat.csv.gz
    arxiv->num_node_features = 128;
    arxiv->x = matrix_create(arxiv->num_nodes, arxiv->num_node_features);
    // arxiv->x = malloc(arxiv->num_nodes * arxiv->num_node_features * sizeof(*arxiv->x));
    if (arxiv->x == NULL) { FATAL_ERROR("Failed to allocate memory for features"); }
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
            arxiv->x->data[IDX(i, j, arxiv->num_node_features)] = atof(feat_cstr);
            assert(j < arxiv->num_node_features);
        }
        assert(i < arxiv->num_nodes);
    }

    arxiv->edge_index = malloc(2 * arxiv->num_edges * sizeof(*arxiv->edge_index));
    if (arxiv->edge_index == NULL) { FATAL_ERROR("Failed to allocate memory for edges"); }
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

matrix_t *sage_layer(matrix_t *input, matrix_t *W, ogb_arxiv_t *arxiv)
{
    matrix_t *output = matrix_create(arxiv->num_nodes, arxiv->num_node_features);

    size_t *neighbor_ids = malloc(sample_size * sizeof(*neighbor_ids));
    matrix_t *neighbor_agg = matrix_create(1, arxiv->num_node_features);
    matrix_t *concat_features = matrix_create(1, 2 * arxiv->num_node_features);

    for (size_t v = 0; v < arxiv->num_nodes; v++) {
        uint8_t neigh_count = 0;

        // find neighbors of count sample size
        for (size_t edge = 0; edge < arxiv->num_edges && neigh_count < sample_size; edge++) {
            if (arxiv->edge_index[IDX(0, edge, arxiv->num_edges)] == v) {
                neighbor_ids[neigh_count++] = arxiv->edge_index[IDX(1, edge, arxiv->num_edges)];
            } else if (arxiv->edge_index[IDX(1, edge, arxiv->num_edges)] == v) {
                neighbor_ids[neigh_count++] = arxiv->edge_index[IDX(0, edge, arxiv->num_edges)];
            }
        }

        if (neigh_count > 0) {
            // aggrigate with mean
            size_t u = neighbor_ids[0];
            memcpy(neighbor_agg->data, &input->data[IDX(u, 0, arxiv->num_node_features)], arxiv->num_node_features * sizeof(double));
            for (size_t i = 1; i < neigh_count; i++) {
                u = neighbor_ids[i];
                for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
                    neighbor_agg->data[feat_idx] += input->data[IDX(u, feat_idx, arxiv->num_node_features)];
                }
            }
            for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
                neighbor_agg->data[feat_idx] /= neigh_count;
            }
        }

        // copy self features
        for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
            concat_features->data[feat_idx] = input->data[IDX(v, feat_idx, arxiv->num_node_features)];
        }

        // copy aggregated neighbor features
        for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
            concat_features->data[arxiv->num_node_features + feat_idx] = neighbor_agg->data[feat_idx];
        }

        // matrix multiplication
        for (size_t out_idx = 0; out_idx < output_dim; out_idx++) {
            double sum = 0.0;
            for (size_t in_idx = 0; in_idx < input_dim; in_idx++) {
                sum += concat_features->data[in_idx] * W->data[IDX(in_idx, out_idx, arxiv->num_node_features)];
            }

            // relu activation
            output->data[IDX(v, out_idx, arxiv->num_node_features)] = fmax(0.0, sum);
        }

#ifndef ndebug
        if (v > 0 && v % 10000 == 0) {
            printf("finished %zu / %zu nodes\n", v, arxiv->num_nodes);
        }
#endif
    }

    for (size_t v = 0; v < arxiv->num_nodes; v++) {
        double norm = 0.0;
        for (size_t x = 0; x < arxiv->num_node_features; x++) {
            double val = output->data[IDX(v, x, arxiv->num_node_features)];
            norm += val * val;
        }
        norm = sqrt(norm);

        if (norm > 1e-8) {
            for (size_t x = 0; x < arxiv->num_node_features; x++) {
                output->data[IDX(v, x, arxiv->num_node_features)] /= norm;
            }
        }
    }

    free(neighbor_ids);
    free(neighbor_agg);
    free(concat_features);

    return output;
}

matrix_t *linear_layer(matrix_t *input, matrix_t *W, ogb_arxiv_t *arxiv)
{
    todo("Implement linear_layer");
}

void backward()
{
    todo("Implement backward");
}

int main(void)
{
    ogb_arxiv_t arxiv = {0};
    printf("Loading data\n");
    load_data(&arxiv);

    input_dim = arxiv.num_node_features;
    output_dim = arxiv.num_label_classes;
    hidden_layer_size = 256;
    K = 1;
    sample_size = 10;

    matrix_t **W = malloc(K+1 * sizeof(matrix_t*));
    W[0] = matrix_create(input_dim, hidden_layer_size);
    // W[1] = matrix_create(hidden_layer_size, hidden_layer_size);
    W[1] = matrix_create(hidden_layer_size, output_dim);

    size_t max_epoch = 1;
    for (size_t epoch = 1; epoch <= max_epoch; epoch++) {
        matrix_t *x = arxiv.x;
        for (size_t k = 1; k <= K; k++) {
            printf("Staring k=%zu\n", k);
            x = sage_layer(x, W[k-1], &arxiv);
        }
        matrix_t *z = linear_layer(x, W[K], &arxiv);

        backward();
        break;
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
// TODO: Add bias
