#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <zlib.h>

#include "matrix.h"
#include "graph.h"
#include "print.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

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
    assert(in->height == out->height);
    assert(in->width == out->width);

    for (size_t i = 0; i < in->height; i++) {
        for (size_t j = 0; j < in->width; j++) {
            out->data[IDX(i, j, out->width)] = fmax(0.0, in->data[IDX(i, j, in->width)]);
        }
    }
    nob_log(NOB_INFO, "relu: ok");
}

/*
  https://stackoverflow.com/a/61570752:
*/
void log_softmax(matrix_t* in, matrix_t* out)
{
    MAT_ASSERT(in, out);

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
    nob_log(NOB_INFO, "log_softmax: ok");
}

// Applies an affine linear transformation to the incoming data: y = x @ A^T + b
void linear_layer(matrix_t* in, matrix_t* weight, matrix_t* bias, matrix_t* out)
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
                MAT_AT(out, i, j) += MAT_AT(bias, 0, j);
            }
        }
    }
    nob_log(NOB_INFO, "linear_layer: ok");
}

#define SAMPLE_SIZE 10 // XXX: Placeholder unitl I implement proper neighbor samplig
void aggregate(size_t v, graph_t *G, matrix_t *in, double *agg)
{
    size_t *adj = malloc(SAMPLE_SIZE * sizeof(*adj));

    size_t neigh_count = 0;

    // NOTE: This neighbor finding approach collects neighbors from both directions
    // (incoming and outgoing edges). For directed graphs like ogbn-arxiv, this is
    // the intended behavior to aggregate information from both:
    //
    // - Papers that node v cites (outgoing edges: v -> u1)
    // - Papers that cite node v (incoming edges: u0 -> v)
    //
    // However, for undirected graphs, this would cause double counting since
    // each undirected edge (i,j) appears as both (i,j) and (j,i) in the edge
    // list.
    //
    // TODO: Make this behavior configurable based on graph type
    // - Directed graphs: Keep bidirectional collection (current behavior)
    // - Undirected graphs: Use deduplication or PyG's to_symmetric approach
    //
    // Find neighbors of count sample size
    for (size_t edge = 0; edge < G->num_edges && neigh_count < SAMPLE_SIZE; edge++) {
        size_t u0 = EDGE_AT(G, edge, 0);
        size_t u1 = EDGE_AT(G, edge, 1);
        if (v == u0) {  // Paper v cites u1
            // printf("%zu=u0: adj[%zu]=%zu\n", v, neigh_count, u1);
            adj[neigh_count++] = u1;
        } else if (v == u1) {   // Paper u0 cites v
            // printf("%pzu=u1: adj[%zu]=%zu\n", v, neigh_count, u0);
            adj[neigh_count++] = u0;
        }
    }

    // Aggregate with mean
    if (neigh_count == 0) {
        goto exit;
    }

    // printf("node %zu: ", v);
    // print_zuarr(adj, neigh_count);

    // for (size_t i = 0; i < neigh_count; i++) {
    //     size_t u = adj[i];
    //     assert(u < G->num_nodes && nob_temp_sprintf("Neighbor adj[%zu] points to out-of-bounds value %zu", i, u));
    //     printf("%zu = ", u);
    //     print_farr(&MAT_ROW(in, u), G->num_node_features);
    // }

    size_t u = adj[0];
    assert(u < G->num_nodes && nob_temp_sprintf("Neighbor adj[0] points to out-of-bounds value %zu", u));

    // printf("MAT_ROW(in, %zu) = ", u);
    // print_farr(&(MAT_ROW(in, u)), G->num_node_features);

    // Move the first neighbor features to memory space for aggregation
    memcpy(agg, &MAT_ROW(in, u), G->num_node_features * sizeof(*agg));
    // print_farr(agg, G->num_node_features);
    for (size_t i = 1; i < neigh_count; i++) {
        u = adj[i];
        assert(u < G->num_nodes && nob_temp_sprintf("Neighbor adj[%zu] points to out-of-bounds value %zu", i, u));

        for (size_t feat_idx = 0; feat_idx < G->num_node_features; feat_idx++) {
            agg[feat_idx] += MAT_AT(in, u, feat_idx);
        }
    }

    // printf("agg before norm = ");
    // print_farr(agg, G->num_node_features);

    float neigh_count_recp = (float)1/neigh_count;
    for (size_t feat_idx = 0; feat_idx < G->num_node_features; feat_idx++) {
        agg[feat_idx] *= neigh_count_recp;
    }

    // printf("agg after norm = ");
    // print_farr(agg, G->num_node_features);

exit:
    free(adj);
}


void sage_layer(matrix_t* in, matrix_t* W, matrix_t* bias, matrix_t* out, graph_t* G)
{
    // MAT_SPEC(in);
    // MAT_SPEC(W);
    // MAT_SPEC(out);

    double *agg = malloc(G->num_node_features*sizeof(*agg));
    matrix_t *concat_features = mat_create(1, 2 * G->num_node_features);

    for (size_t v = 0; v < G->num_nodes; v++) {
        aggregate(v, G, in, agg);

        // TODO: Divide the linear transformation instead of doing concatenation
        // linear transformation, and add them together at the end.
        // y = W*concat(x, agg()) => y = Wl*x + Wr*agg(), where W = [Wl | Wr]

        // copy self features
        for (size_t feat_idx = 0; feat_idx < G->num_node_features; feat_idx++) {
            concat_features->data[feat_idx] = in->data[IDX(v, feat_idx, G->num_node_features)];
        }

        // copy aggregated neighbor features
        for (size_t feat_idx = 0; feat_idx < G->num_node_features; feat_idx++) {
            concat_features->data[G->num_node_features + feat_idx] = agg[feat_idx];
        }

        matrix_t row;
        row.height = 1;
        row.width = out->width;
        row.data = &MAT_ROW(out, v);

        linear_layer(concat_features, W, bias, &row);

#ifndef NDEBUG
        if (v > 0 && v % 10000 == 0) {
            printf("finished %zu / %zu nodes\n", v, G->num_nodes);
        }
#endif
    }

    free(agg);
    mat_destroy(concat_features);

    nob_log(NOB_INFO, "sage_layer: ok");
}

void l2_normalization(matrix_t *out, matrix_t *in, graph_t *G)
{
    MAT_ASSERT(out, in);

    for (size_t v = 0; v < G->num_nodes; v++) {
        double norm = 0.0;
        for (size_t x = 0; x < in->width; x++) {
            double val = MAT_AT(in, v, x);
            norm += val * val;
        }
        double norm_recp = 1/sqrt(norm);

        if (norm > 1e-8) {
            for (size_t x = 0; x < in->width; x++) {
                MAT_AT(out, v, x) = MAT_AT(in, v, x) * norm_recp;
            }
        }
    }
    nob_log(NOB_INFO, "l2_normalization: ok");
}


#define NLL_LOSS_REDUCTION_MEAN
double nll_loss(matrix_t* pred, matrix_t* target)
{
    MAT_ASSERT(pred, target);

    double* L = malloc(target->height * sizeof(*L));
	for (size_t v = 0; v < target->height; v++) {
        size_t c;
        for (c = 0; c < target->width; c++) {
            double class = MAT_AT(target, v, c);
            if (class == 1.0) {
                break;
            }
        }
        // printf("Node %zu class %zu pred: %f\n", v, c, );
        assert(c < target->width && "No true class was found");
        double logits = MAT_AT(pred, v, c);
        L[v] = -logits;
    }

    size_t N = target->height;
    double sum = 0.0;
    for (size_t l = 0; l < N; l++) {
        sum += L[l];
    }
    free(L);
#ifdef NLL_LOSS_REDUCTION_MEAN
    nob_log(NOB_INFO, "nll_loss(mean): ok");
    return sum / N;
#else

    nob_log(NOB_INFO, "nll_loss(sum): ok");
    return sum;
#endif
}

// Computes gradient flow from both NLLLoss and LogSoftmax.
void cross_entropy_backward(matrix_t *grad_out, matrix_t *yhat, matrix_t *y)
{
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

void linear_weight_backward(matrix_t *grad_in, matrix_t *grad_out, matrix_t *lin_in)
{
    dot_ex(grad_in, lin_in, grad_out, true, false);
    nob_log(NOB_INFO, "linear_weight_backward: ok");
}

void linear_h_backward(matrix_t* grad_in, matrix_t* grad_out, matrix_t* W)
{
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

    float V_recip = (float) 1/V;
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

    nob_minimal_log_level = NOB_NO_LOGS;

    graph_t G = {0};
    load_data(&G);

    // printf("After loading dataset:\n");
    // print_memory_usage();

    size_t input_dim = G.num_node_features;  // 128
    size_t output_dim = G.num_label_classes; // 40
    size_t hidden_layer_size = 5;
    K = 1;

    // Weights will be transposed when feed through linear transformation, hence
    // the reverse shape
    matrix_t* W1 = mat_create(hidden_layer_size, G.num_node_features*2); // times 2 because of concatenation
    matrix_t* W2 = mat_create(G.num_label_classes, hidden_layer_size);
    mat_rand(W1, -1.0, 1.0);
    mat_rand(W2, -1.0, 1.0);

    // printf("After initializing only weights:\n");
    // print_memory_usage();

    matrix_t* x = G.x;
    matrix_t* y = G.y;
    matrix_t* bias = NULL;
    matrix_t* h1 = mat_create(G.num_nodes, hidden_layer_size); // Hidden layer outcome from GraphSAGE
    matrix_t* logits = mat_create(G.num_nodes, G.num_label_classes);
    matrix_t* yhat = mat_create(G.num_nodes, G.num_label_classes);

    matrix_t* grad_logits = mat_create(G.num_nodes, G.num_label_classes);
    matrix_t* grad_W2 = mat_create(G.num_label_classes, hidden_layer_size); // grad_out
    matrix_t* grad_bias = grad_logits; // dC/dBias = dC/dLogits
    matrix_t* grad_h1 = mat_create(G.num_nodes, hidden_layer_size);
    matrix_t* grad_W1 = mat_create(hidden_layer_size, G.num_node_features*2); // times 2 because of concatenation

    // printf("After initializing matrices:\n");
    // print_memory_usage();

    FILE *f = fopen("output.log", "w");

    size_t max_epoch = 20;
    for (size_t epoch = 1; epoch <= max_epoch; epoch++) {
        sage_layer(x, W1, bias, h1, &G);
        relu(h1, h1);
        l2_normalization(h1, h1, &G);

        linear_layer(h1, W2, bias, logits);
        log_softmax(logits, yhat);

        double loss = nll_loss(yhat, y);
        printf("Loss: %f\n", loss);

        cross_entropy_backward(grad_logits, yhat, y);
        linear_weight_backward(grad_logits, grad_W2, h1);

        linear_h_backward(grad_logits, grad_h1, W2);

        update_weights(W2, grad_W2, G.num_nodes);
    }

    fclose(f);
    mat_destroy(W1);
    mat_destroy(W2);
    mat_destroy(h1);
    mat_destroy(logits);
    mat_destroy(yhat);
    mat_destroy(G.x);
    mat_destroy(G.y);
    free(G.node_year);
    free(G.edge_index);
    return 0;
}

// TODO: Implement gradient descent training
// TODO: Configurable layer dimensions
// TODO: Use CRS format for edges
// TODO: Clean up all allocated memory
// TODO: Add bias
// TODO: Split up dataset according to DATASET_PATH/split/time/{test.csv.gz,train.csv.gz,valid.csv.gz} which are indexes
// TODO: Xavier Initialization for weight matrices
// TODO: Make a global memory pool that for internal use
