#ifndef GNN_H
#define GNN_H

#include <stddef.h>

#include "matrix.h"
#include "layers.h"
#include "graph.h"

void sageconv(SageLayer* const l, graph_t* const g);
void relu(ReluLayer* const l);
void normalize(NormalizeLayer* const l);
void linear(LinearLayer* const l);
void logsoft(LogSoftLayer* const l);
double nll_loss(matrix_t* const yhat, matrix_t* const y);
double accuracy(matrix_t* const yhat, matrix_t* const y);

void cross_entropy_backward(LogSoftLayer* const l, matrix_t* const y);
void linear_backward(LinearLayer* const l);
void normalize_backward(NormalizeLayer* const l);
void relu_backward(ReluLayer* const l);
void sageconv_backward(SageLayer* const l, graph_t* const g);

void update_linear_weights(LinearLayer* const l, float lr);
void update_sageconv_weights(SageLayer* const l, float lr);

#endif // GNN_H
