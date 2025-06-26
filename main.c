#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <zlib.h>

#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#include "nob.h"

#include "matrix.h"

#define ERROR_PRINT(format, ...)                                        \
  fprintf(stderr, "[ERROR] %s:%d - " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define FATAL_ERROR(format, ...)                \
  do {                                          \
    ERROR_PRINT(format, ##__VA_ARGS__);         \
    exit(EXIT_FAILURE);                         \
  } while(0)

#define todo(str, ...) FATAL_ERROR("TODO: %s", str, ##__VA_ARGS__)

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

size_t K;
size_t sample_size;

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

    // TODO: Programmatically find the number of labels by counting unique
    // labels in ogbn_arxiv/raw/node-label.csv.gz

    arxiv->num_label_classes = 40;
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

void relu(matrix_t *in, matrix_t *out)
{
    assert(in->height == out->height);
    assert(in->width == out->width);

    for (size_t i = 0; i < in->height; i++) {
        for (size_t j = 0; j < in->width; j++) {
            out->data[IDX(i, j, out->width)] = fmax(0.0, in->data[IDX(i, j, in->width)]);
        }
    }
}

/*
  https://stackoverflow.com/a/61570752:
  def log_softmax(x):
      c = x.max()
      logsumexp = np.log(np.exp(x - c).sum())
      return x - c - logsumexp
*/
void log_softmax(matrix_t* in, matrix_t *out)
{
    assert(in->height == out->height);
    assert(in->width == out->width);

    for (size_t i = 0; i < in->height; i++) {
        double max_x = in->data[IDX(i, 0, in->width)];
        for (size_t j = 0; j < in->width; j++) {
            double x = in->data[IDX(i, j, in->width)];
            if (x > max_x) max_x = x;
        }


        double sum = 0.0;
        for (size_t j = 0; j < in->width; j++) {
            double x = in->data[IDX(i, j, in->width)];
            sum += exp(x - max_x);
        }

        double log_sum = log(sum);
        for (size_t j = 0; j < in->width; j++) {
            double x = in->data[IDX(i, j, in->width)];
            out->data[IDX(i, j, out->width)] = x - max_x - log_sum;
        }
    }
}

// Applies an affine linear transformation to the incoming data: y = x @ A^T + b
void linear_layer(matrix_t *in, matrix_t *weight, matrix_t *bias, matrix_t *out)
{
    // Transposing weight will give correct inner dimensions
    assert(in->height == out->height);
    assert(in->width == weight->width);
    assert(out->width == weight->height);

    // Computes out = in @ weight^T
    dot_ex(in, weight, out, false, true);

    if (bias) {
        assert(bias->height == 1);
        assert(out->width == bias->width);

        for (size_t i = 0; i < out->height; i++) {
            for (size_t j = 0; j < out->width; j++) {
                out->data[IDX(i, j, out->width)] += bias->data[j];
            }
        }
    }
}

void sage_layer(matrix_t *in, matrix_t *weight, matrix_t *bias, matrix_t *out, ogb_arxiv_t *arxiv)
{

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
            memcpy(neighbor_agg->data, &in->data[IDX(u, 0, arxiv->num_node_features)], arxiv->num_node_features * sizeof(double));
            for (size_t i = 1; i < neigh_count; i++) {
                u = neighbor_ids[i];
                for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
                    neighbor_agg->data[feat_idx] += in->data[IDX(u, feat_idx, arxiv->num_node_features)];
                }
            }
            for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
                neighbor_agg->data[feat_idx] /= neigh_count;
            }
        }

        // TODO: Divide the linear transformation calculation instead of doing
        // concatenation linear transformation, and add them together at the end

        // copy self features
        for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
            concat_features->data[feat_idx] = in->data[IDX(v, feat_idx, arxiv->num_node_features)];
        }

        // copy aggregated neighbor features
        for (size_t feat_idx = 0; feat_idx < arxiv->num_node_features; feat_idx++) {
            concat_features->data[arxiv->num_node_features + feat_idx] = neighbor_agg->data[feat_idx];
        }

        matrix_t row;
        row.height = 1;
        row.width = out->width;
        row.data = matrix_row(out, v);
        // asm("int $3");
        linear_layer(concat_features, weight, bias, &row);

#ifndef ndebug
        if (v > 0 && v % 10000 == 0) {
            printf("finished %zu / %zu nodes\n", v, arxiv->num_nodes);
        }
#endif
    }

    // L2 Normalization
    for (size_t v = 0; v < arxiv->num_nodes; v++) {
        double norm = 0.0;
        for (size_t x = 0; x < out->width; x++) {
            double val = out->data[IDX(v, x, out->width)];
            norm += val * val;
        }
        norm = sqrt(norm);

        if (norm > 1e-8) {
            for (size_t x = 0; x < out->width; x++) {
                out->data[IDX(v, x, out->width)] /= norm;
            }
        }
    }

    free(neighbor_ids);
    matrix_destroy(neighbor_agg);
    matrix_destroy(concat_features);
}

/*
 * For each training sample i:
 *   1. Get the true class: true_class = targets[i]
 *   2. Get the predicted log probability: pred_log_prob = log_probs[i][true_class]
 *   3. Get the class weight: weight = class_weights[true_class]
 *   4. Compute loss: loss_i = -weight × pred_log_prob
**/
void nll_loss()
{
    todo("Implement nll_loss");
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

    size_t input_dim = arxiv.num_node_features;  // 128
    size_t output_dim = arxiv.num_label_classes; // 40
    size_t hidden_layer_size = 256;
    K = 1;
    sample_size = 3;

    // Weights will be transposed when feed through linear transformation, hence
    // the reverse shape
    matrix_t *W1 = matrix_create(hidden_layer_size, arxiv.num_node_features*2); // times 2 because of concatenation
    matrix_t *W2 = matrix_create(arxiv.num_label_classes, hidden_layer_size);
    matrix_fill(W1, 0.1);
    matrix_fill(W2, 0.1);

    // Make it small for testing purposes
    arxiv.num_nodes = 100;

    matrix_t *x = arxiv.x;
    matrix_t *bias = NULL;
    matrix_t *z1 = matrix_create(arxiv.num_nodes, hidden_layer_size);
    matrix_t *a1 = matrix_create(arxiv.num_nodes, hidden_layer_size);
    matrix_t *z2 = matrix_create(arxiv.num_nodes, arxiv.num_label_classes);
    matrix_t *y = matrix_create(arxiv.num_nodes, arxiv.num_label_classes);

    FILE *f = fopen("output.log", "w");

    size_t max_epoch = 1;
    for (size_t epoch = 1; epoch <= max_epoch; epoch++) {
        sage_layer(x, W1, bias, z1, &arxiv);
        printf("sage_layer completed\n");
        fmatrix_print(f, z1, "z1");
        fprintf(f, "\n");

        relu(z1, a1);
        printf("sage_layer relu\n");
        fmatrix_print(f, a1, "a1");
        fprintf(f, "\n");

        linear_layer(a1, W2, bias, z2);
        printf("sage_layer linear_layer\n");
        fmatrix_print(f, z2, "z2");
        fprintf(f, "\n");

        log_softmax(z2, y);
        printf("sage_layer log_softmax\n");
        fmatrix_print(f, y, "y");
        fprintf(f, "\n");

        break;
    }

    fclose(f);
    matrix_destroy(W1);
    matrix_destroy(W2);
    matrix_destroy(z1);
    matrix_destroy(a1);
    matrix_destroy(z2);
    matrix_destroy(y);
    matrix_destroy(arxiv.x);
    matrix_destroy(arxiv.y);
    free(arxiv.node_year);
    free(arxiv.edge_index);
    sb_free(sb);
    return 0;
}

// TODO: Create a small test node (CRITICAL)
// TODO: Initilize weight matrix with random values (with a seed set)
// TODO: Implement gradient descent training
// TODO: Configurable layer dimensions
// TODO: Use CRS format for edges
// TODO: Clean up all allocated memory
// TODO: Add bias
// TODO: Split up dataset according to RAW_PATH/../split/time/{test.csv.gz,train.csv.gz,valid.csv.gz} which are indexes
// TODO: Xavier Initialization for weight matrices
