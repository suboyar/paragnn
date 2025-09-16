#ifndef GNN_H
#define GNN_H

#include <stddef.h>

#include "matrix.h"
#include "layers.h"
#include "graph.h"

#ifdef NEWWAY

void sage_conv(SageLayer *l, graph_t *g);

#else
// Forward propagation
void sage_conv(matrix_t *in, matrix_t *Wl, matrix_t *Wr, matrix_t *agg, matrix_t *out, graph_t *g);
void relu(matrix_t* in, matrix_t* out);
void l2_normalization(matrix_t *in, matrix_t *out, graph_t *G);
void linear_layer(matrix_t* in, matrix_t* weight, matrix_t* bias, matrix_t* out);
void log_softmax(matrix_t* in, matrix_t* out);
double nll_loss(matrix_t* pred, matrix_t* target);

// Backpropagation
void update_weights(matrix_t* W, matrix_t* grad_W, size_t V);
void update_sage_weights(matrix_t* W, matrix_t* grad_W, size_t V);
void sage_conv_backward(matrix_t *grad_in, matrix_t *h_relu, matrix_t *h, matrix_t *agg, matrix_t *grad_Wl, matrix_t *grad_Wr, graph_t *g);
void relu_backward(matrix_t* grad_in, matrix_t* h, matrix_t* grad_out);
void l2_normalization_backward(matrix_t *grad_in, matrix_t *h_relu, matrix_t *h_l2, matrix_t *grad_out);
void linear_weight_backward(matrix_t *grad_in, matrix_t *lin_in, matrix_t *grad_out);
void linear_h_backward(matrix_t* grad_in, matrix_t* W, matrix_t* grad_out);
void cross_entropy_backward(matrix_t *grad_out, matrix_t *yhat, matrix_t *y);

#endif // NEWWAY

#endif // GNN_H
