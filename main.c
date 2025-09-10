#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <zlib.h>

#define NOB_IMPLEMENTATION
#include "nob.h"
#include "matrix.h"
#include "graph.h"

#define USE_SIMPLE_GRAPH    // Comment out to use arxiv

#ifdef USE_SIMPLE_GRAPH
#include "simple_graph.h"
#define load_data load_simple_data
#else
#include "arxiv.h"
#define load_data load_arxiv_data
#endif

#define ERROR(fmt, ...) do { \
    fprintf(stderr, "%s:%d: ERROR: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    abort(); \
} while(0)

size_t K;
size_t sample_size;

void relu(matrix_t* in, matrix_t* out)
{
    nob_log(NOB_INFO, "relu...");

    assert(in->height == out->height);
    assert(in->width == out->width);

    for (size_t i = 0; i < in->height; i++) {
        for (size_t j = 0; j < in->width; j++) {
            out->data[IDX(i, j, out->width)] = fmax(0.0, in->data[IDX(i, j, in->width)]);
        }
    }

    nob_log(NOB_INFO, "relu ok");
}

/*
  https://stackoverflow.com/a/61570752:
*/
void log_softmax(matrix_t* in, matrix_t* out)
{
    nob_log(NOB_INFO, "log_softmax...");

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

    nob_log(NOB_INFO, "log_softmax ok");
}

// Applies an affine linear transformation to the incoming data: y = x @ A^T + b
void linear_layer(matrix_t* in, matrix_t* weight, matrix_t* bias, matrix_t* out)
{
    nob_log(NOB_INFO, "linear_layer...");

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
    nob_log(NOB_INFO, "linear_layer ok");
}

void sage_layer(matrix_t* in, matrix_t* weight, matrix_t* bias, matrix_t* out, graph_t* arxiv)
{
    nob_log(NOB_INFO, "sage_layer...");

    size_t *neighbor_ids = malloc(sample_size * sizeof(*neighbor_ids));
    matrix_t* neighbor_agg = mat_create(1, arxiv->num_node_features);
    matrix_t* concat_features = mat_create(1, 2 * arxiv->num_node_features);


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
                    size_t idx = IDX(u, feat_idx, arxiv->num_node_features);
                    neighbor_agg->data[feat_idx] += in->data[idx];
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
        row.data = mat_row(out, v);
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
    mat_destroy(neighbor_agg);
    mat_destroy(concat_features);

    nob_log(NOB_INFO, "sage_layer ok");
}

#define NLL_LOSS_REDUCTION_MEAN
double nll_loss(matrix_t* pred, matrix_t* target)
{
    nob_log(NOB_INFO, "nll_loss...");

    MAT_ASSERT(pred, target);

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
    nob_log(NOB_INFO, "nll_loss(mean): ok");
    return sum / N;
#else
    nob_log(NOB_INFO, "nll_loss(sum): ok");
    return sum;
#endif
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
void cross_entropy_backward(matrix_t *yhat, matrix_t *y, matrix_t *grad_out)
{
    nob_log(NOB_INFO, "cross_entropy_backward...");

    MAT_ASSERT(yhat, y);
    MAT_ASSERT(yhat, grad_out);

    size_t height = yhat->height, width = yhat->width;
    for (size_t v = 0; v < height; v++) {
        for (size_t i = 0; i < width; i++) {
            MAT_AT(grad_out, v, i) = exp(MAT_AT(yhat, v, i));
        }

        for (size_t c = 0; c < width; c++) {
            if (MAT_AT(y, v, c) == 1.0) { //  True class
                MAT_AT(grad_out, v, c) = MAT_AT(grad_out, v, c) - 1;
                break;
            }
        }
    }

    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}

void linear_weight_backward(matrix_t* h, matrix_t* grad_out, matrix_t* grad_in)
{
    nob_log(NOB_INFO, "linear_weight_backward...");

    assert(grad_out->width == h->width);
    assert(grad_out->height == grad_in->width);
    assert(grad_in->height == h->height);

    dot_ex(grad_in, h, grad_out, true, false);

    nob_log(NOB_INFO, "linear_weight_backward: ok");
}

void linear_h_backward(matrix_t* W, matrix_t* grad_out, matrix_t* grad_in)
{
    nob_log(NOB_INFO, "linear_h_backward...");

    assert(grad_out->height == grad_in->height);
    assert(grad_out->width == W->width);
    assert(grad_in->width == W->height);

    dot(grad_in, W, grad_out);
    nob_log(NOB_INFO, "linear_h_backward: ok");
}

void relu_backward(matrix_t* grad_input, matrix_t* grad_output, matrix_t* output)
{
    assert(output->height == grad_output->height);
    assert(output->width == grad_output->width);
    assert(output->height == grad_input->height);
    assert(output->width == grad_input->width);

    // TODO: Check if one of these are better:
    // - grad_input->data[IDX] = (output->data[IDX] > 0.0f) * grad_output->data[IDX];
    // - grad_input->data[IDX] = grad_output->data[IDX] * fmaxf(0.0f, copysignf(1.0f, output->data[IDX]));

    for (size_t i = 0; i < output->height; i++) {
        for (size_t j = 0; j < output->width; j++) {
            grad_input->data[IDX(i, j, grad_input->width)] = 0; //  f'(x) x <= 0
            if (output->data[IDX(i, j, output->width)]) {       //  f'(x) x > 0
                grad_input->data[IDX(i, j, grad_input->width)] = grad_output->data[IDX(i, j, grad_output->width)];
            }
        }
    }
}

void l2_normalize_backward()
{
    NOB_TODO("Implement relu_backward");
}

void sage_backward()
{
    NOB_TODO("Implement sage_backward");
}


void update_weights(matrix_t* W, matrix_t* grad_W, size_t V)
{
    assert(W->height == grad_W->height);
    assert(W->width == grad_W->width);

    float V_recip = 1/V;
    size_t height = W->height, width = W->width;
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            MAT_AT(W, i, j) -= MAT_AT(grad_W, i, j) * V_recip;
        }
    }
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
    srand(0);
    // srand(time(NULL));

    // nob_minimal_log_level = NOB_NO_LOGS;

    graph_t arxiv = {0};
    load_data(&arxiv);

    // printf("After loading dataset:\n");
    // print_memory_usage();

    size_t input_dim = arxiv.num_node_features;  // 128
    size_t output_dim = arxiv.num_label_classes; // 40
    size_t hidden_layer_size = 5;
    K = 1;
    sample_size = 2;

    // Weights will be transposed when feed through linear transformation, hence
    // the reverse shape
    matrix_t* W1 = mat_create(hidden_layer_size, arxiv.num_node_features*2); // times 2 because of concatenation
    matrix_t* W2 = mat_create(arxiv.num_label_classes, hidden_layer_size);
    mat_rand(W1, -1.0, 1.0);
    mat_rand(W2, -1.0, 1.0);

    // printf("After initializing only weights:\n");
    // print_memory_usage();

    matrix_t* x = arxiv.x;
    matrix_t* y = arxiv.y;
    matrix_t* bias = NULL;
    matrix_t* h1 = mat_create(arxiv.num_nodes, hidden_layer_size); // Hidden layer outcome from GraphSAGE
    matrix_t* logits = mat_create(arxiv.num_nodes, arxiv.num_label_classes);
    matrix_t* yhat = mat_create(arxiv.num_nodes, arxiv.num_label_classes);

    matrix_t* grad_logits = mat_create(arxiv.num_nodes, arxiv.num_label_classes);
    matrix_t* grad_W2 = mat_create(arxiv.num_label_classes, hidden_layer_size);
    matrix_t* grad_bias = grad_logits; // dC/dBias = dC/dLogits
    matrix_t* grad_h1 = mat_create(arxiv.num_nodes, hidden_layer_size);
    matrix_t* grad_W1 = mat_create(hidden_layer_size, arxiv.num_node_features*2); // times 2 because of concatenation

    // printf("After initializing matrices:\n");
    // print_memory_usage();

    FILE *f = fopen("output.log", "w");

    size_t max_epoch = 20;
    for (size_t epoch = 1; epoch <= max_epoch; epoch++) {
        sage_layer(x, W1, bias, h1, &arxiv);
        linear_layer(h1, W2, bias, logits);
        log_softmax(logits, yhat);
        double loss = nll_loss(yhat, arxiv.y);
        printf("Loss: %f\n", loss);

        cross_entropy_backward(yhat, arxiv.y, grad_logits);
        linear_weight_backward(h1, grad_W2, grad_logits);
        linear_h_backward(W2, grad_h1, grad_logits);

        break;

        update_weights(W2, grad_W2, arxiv.num_nodes);
        // update_bias(bias, grad_bias, arxiv.num_nodes);
    }

    fclose(f);
    mat_destroy(W1);
    mat_destroy(W2);
    mat_destroy(x);
    mat_destroy(y);
    // mat_destroy(bias);
    mat_destroy(h1);
    mat_destroy(logits);
    mat_destroy(yhat);
    mat_destroy(grad_logits);
    mat_destroy(grad_h1);
    // mat_destroy(grad_bias);
    mat_destroy(grad_W2);
    mat_destroy(grad_W1);
    free(arxiv.node_year);
    free(arxiv.edge_index);
    return 0;
}

// TODO: Implement gradient descent training
// TODO: Configurable layer dimensions
// TODO: Use CRS format for edges
// TODO: Clean up all allocated memory
// TODO: Add bias
// TODO: Split up dataset according to DATASET_PATH/split/time/{test.csv.gz,train.csv.gz,valid.csv.gz} which are indexes
// TODO: Xavier Initialization for weight matrices
