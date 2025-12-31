#ifndef GNN_H
#define GNN_H

#include <stddef.h>

#include "matrix.h"
#include "layers.h"
#include "graph.h"

void sageconv(SageLayer* const l, graph_t* const g);
void relu(ReluLayer* const l);
void normalize(L2NormLayer* const l);
void linear(LinearLayer* const l);
void logsoft(LogSoftLayer* const l);
double nll_loss(Matrix* const yhat, Matrix* const y);
double accuracy(Matrix* const yhat, Matrix* const y);

void cross_entropy_backward(LogSoftLayer *const l, Matrix *const y);
void linear_backward(LinearLayer* const l);
void normalize_backward(L2NormLayer* const l);
void relu_backward(ReluLayer* const l);
void sageconv_backward(SageLayer* const l, graph_t* const g);

void linear_layer_update_weights(LinearLayer* const l, float lr);
void sage_layer_update_weights(SageLayer* const l, float lr);

void sage_layer_zero_gradients(SageLayer* l);
void relu_layer_zero_gradients(ReluLayer* l);
void normalize_layer_zero_gradients(L2NormLayer* l);
void linear_layer_zero_gradients(LinearLayer* l);
void logsoft_layer_zero_gradients(LogSoftLayer* l);
void sage_net_zero_gradients(SageNet* net);

#endif // GNN_H
