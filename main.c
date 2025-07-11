#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <zlib.h>

#define NOB_IMPLEMENTATION
#define NOB_TEMP_CAPACITY (100*1024*1024) // 100MB
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
    matrix_t *y; // node-level targets of shape [num_nodes, num_label_classes]
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

#if ENABLE_NODE_LIMIT
    // Limit nodes for testing purposes
    const size_t max_nodes = 100;
    if (arxiv->num_nodes > max_nodes) {
        fprintf(stderr, "DEBUG: Limiting nodes from %zu to %zu for testing\n",
                arxiv->num_nodes, max_nodes);
        arxiv->num_nodes = max_nodes;
    }
#endif

    sv = read_gz(RAW_PATH"/num-edge-list.csv.gz");
    String_View num_edge_sv = sv_chop_by_delim(&sv, '\n');
    const char *num_edge_cstr = temp_sv_to_cstr(num_edge_sv);
    arxiv->num_edges = atol(num_edge_cstr);

    sv = read_gz(RAW_PATH"/node_year.csv.gz");
    arxiv->node_year = malloc(arxiv->num_nodes * sizeof(*arxiv->node_year));
    if (arxiv->node_year == NULL) { FATAL_ERROR("Failed to allocate memory for node years"); }

    for (size_t i = 0; i < arxiv->num_nodes && sv.count > 0; i++) {
        temp_reset();
        String_View year_sv = sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (year_sv.data[year_sv.count-1] == '\0') {
            FATAL_ERROR("Empty line found in node_year");
        }

        const char *year_cstr = temp_sv_to_cstr(year_sv);
        assert(strcmp(year_cstr, "") != 0);
        arxiv->node_year[i] = atol(year_cstr);
    }

    // TODO: Programmatically find the number of labels by counting unique
    // labels in ogbn_arxiv/raw/node-label.csv.gz
    arxiv->num_label_classes = 40;
    arxiv->y = matrix_create(arxiv->num_nodes, arxiv->num_label_classes);
    if (arxiv->y == NULL) { FATAL_ERROR("Failed to allocate memory for labels"); }
    sv = read_gz(RAW_PATH"/node-label.csv.gz");

    for (size_t i = 0; i < arxiv->num_nodes && sv.count > 0; i++) {
        String_View label_sv = sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (label_sv.data[label_sv.count-1] == '\0') {
            FATAL_ERROR("Empty line found in node_label");
        }
        if (i % 10000 == 0) {
            temp_reset();
        }

        const char *label_cstr = temp_sv_to_cstr(label_sv);
        assert(strcmp(label_cstr, "") != 0);
        int label_index = atoi(label_cstr); // max 40 classes in ogbn-arxiv

        if ((size_t)label_index >= arxiv->num_label_classes) {
            FATAL_ERROR("Label class index overflow: got %d, expected < %zu",
                        label_index, arxiv->num_label_classes-1);
        }

        arxiv->y->data[IDX(i, label_index, arxiv->num_label_classes)] = 1.0; // Everything else should have been inited to 0.0 by calloc
    }

    // TODO: Programatically find the number of features through counting features
    // in the first line of ogbn_arxiv/raw/node-feat.csv.gz
    arxiv->num_node_features = 128;
    arxiv->x = matrix_create(arxiv->num_nodes, arxiv->num_node_features);
    if (arxiv->x == NULL) { FATAL_ERROR("Failed to allocate memory for features"); }
    sv = read_gz(RAW_PATH"/node-feat.csv.gz");

    for (size_t i = 0; i < arxiv->num_nodes && sv.count > 0; i++) {
        String_View feats_sv = sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (feats_sv.data[feats_sv.count-1] == '\0') {
            FATAL_ERROR("Empty line found in node_feature at line %zu", i+1);
        }

        for (size_t j = 0; feats_sv.count > 0; j++) {
            String_View feat_sv = sv_chop_by_delim(&feats_sv, ',');
            const char *feat_cstr = temp_sv_to_cstr(feat_sv);
            assert(strcmp(feat_cstr, "") != 0);
            arxiv->x->data[IDX(i, j, arxiv->num_node_features)] = atof(feat_cstr);
            if (j >= arxiv->num_node_features) {
                FATAL_ERROR("Feature index overflow: got %zu, expected < %zu",
                            j, arxiv->num_node_features);
            }
        }
    }

    arxiv->edge_index = malloc(2 * arxiv->num_edges * sizeof(*arxiv->edge_index));
    if (arxiv->edge_index == NULL) { FATAL_ERROR("Failed to allocate memory for edges"); }
    sv = read_gz(RAW_PATH"/edge.csv.gz");

    for (size_t i = 0; i < arxiv->num_edges && sv.count > 0; i++) {
        String_View edges_sv = sv_chop_by_delim(&sv, '\n');
        // Err if empty line is found while reading
        if (edges_sv.data[edges_sv.count-1] == '\0') {
            FATAL_ERROR("Empty line found in edge at line %zu", i+1);
        }

        // uint32_t v, u;
        for (size_t j = 0; edges_sv.count > 0; j++) {
            temp_reset();
            String_View edge_sv = sv_chop_by_delim(&edges_sv, ',');
            const char *edge_cstr = temp_sv_to_cstr(edge_sv);

            if (strcmp(edge_cstr, "") == 0) {
                FATAL_ERROR("Empty edge string at position %zu in line %zu", j, i+1);
            }

            arxiv->edge_index[IDX(j, i, arxiv->num_edges)] = atol(edge_cstr);

            if (j >= 2) {
                FATAL_ERROR("Too many edges: got %zu on line %zu, expected exactly 2",
                            j+1, i+1);
            }
        }
    }
    sb_free(sb);
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

        // aggrigate with mean
        size_t u = neighbor_ids[0];
        if (u >= arxiv->num_nodes) { continue; }
        if (neigh_count > 0) {
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

#define NLL_LOSS_REDUCTION_MEAN
double nll_loss(matrix_t *pred, matrix_t *target)
{
    assert(pred->height == target->height);
    assert(pred->width == target->width);

    double* L = malloc(target->height * sizeof(*L));
	for (size_t v = 0; v < target->height; v++) {
        size_t c;
        for (c = 0; c < target->width; c++) {
            double class = target->data[IDX(v,c,target->width)];
            if (class == 1.0) {
                break;
            }
        }
        // printf("Node %zu class %zu pred: %f\n", v, c, );
        assert(c < target->width && "No true class was found");
        double logits = pred->data[IDX(v, c, pred->width)];
        L[v] = -logits;
    }

    size_t N = target->height;
    double sum = 0.0;
    for (size_t l = 0; l < N; l++) {
        sum += L[l];
    }
#ifdef NLL_LOSS_REDUCTION_MEAN
    printf("reduction mean\n");
    return sum / N;
#else
    printf("reduction sum\n");
    return sum;
#endif
}

void nll_loss_backward()
{
    todo("Implement nll_loss_backward");
}

/*
 * Since output = log_softmax(x), we can recover softmax(x) = exp(output)
 * without needing the original input x.
 *
 * def log_softmax_backward(grad_output, output):
 *     softmax_vals = torch.exp(output)  # recover softmax(x) from log_softmax(x)
 *     return grad_output - softmax_vals * torch.sum(grad_output)
 */
void log_softmax_backward(matrix_t *grad_output, matrix_t *output)
{
    todo("Implement log_softmax_backward");
}

void linear_backward()
{
    todo("Implement linear_backward");
}

void relu_backward()
{
    todo("Implement relu_backward");
}

void l2_normalize_backward()
{
    todo("Implement relu_backward");
}

/*
 * Gradients w.r.t. the aggregation function parameters
 * Gradients w.r.t. the weight matrices W^k
 * Gradients w.r.t. the node's self features
 * Gradients w.r.t. the L2 normalization step
 */
void sage_backward()
{

}


// TODO: Check out getrusage from <sys/resource.h>
void print_memory_usage()
{
    FILE* file = fopen("/proc/self/status", "r");
    char line[128];

    if (file) {
        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                printf("Memory usage: %s", line);
                break;
            }
        }
        fclose(file);
    }
}

int main(void)
{
    ogb_arxiv_t arxiv = {0};
    printf("Loading data\n");
    load_data(&arxiv);

    printf("After loading dataset:\n");
    print_memory_usage();

    size_t input_dim = arxiv.num_node_features;  // 128
    size_t output_dim = arxiv.num_label_classes; // 40
    size_t hidden_layer_size = 256;
    K = 1;
    sample_size = 3;

    // Weights will be transposed when feed through linear transformation, hence
    // the reverse shape
    matrix_t *W1 = matrix_create(hidden_layer_size, arxiv.num_node_features*2); // times 2 because of concatenation
    matrix_t *W2 = matrix_create(arxiv.num_label_classes, hidden_layer_size);
    matrix_fill(W1, 0.4);
    matrix_fill(W2, 0.8);

    printf("After initializing only weights:\n");
    print_memory_usage();

    matrix_t *x = arxiv.x;
    matrix_t *bias = NULL;
    matrix_t *z1 = matrix_create(arxiv.num_nodes, hidden_layer_size);
    matrix_t *a1 = matrix_create(arxiv.num_nodes, hidden_layer_size);
    matrix_t *z2 = matrix_create(arxiv.num_nodes, arxiv.num_label_classes);
    matrix_t *y = matrix_create(arxiv.num_nodes, arxiv.num_label_classes);

    printf("After initializing matrices:\n");
    print_memory_usage();

    FILE *f = fopen("output.log", "w");

    size_t max_epoch = 1;
    for (size_t epoch = 1; epoch <= max_epoch; epoch++) {
        sage_layer(x, W1, bias, z1, &arxiv);
        printf("sage_layer completed\n");

        relu(z1, a1);
        printf("sage_layer relu\n");

        linear_layer(a1, W2, bias, z2);
        printf("sage_layer linear_layer\n");

        log_softmax(z2, y);
        printf("log_softmax\n");

        double loss = nll_loss(y, arxiv.y);
        printf("Loss: %f\n", loss);

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
