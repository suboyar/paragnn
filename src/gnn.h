#ifndef GNN_H
#define GNN_H

#include <stddef.h>

#include "matrix.h"
#include "layers.h"
#include "graph.h"

void sageconv(SageLayer* const l);
void relu(ReluLayer* const l);
void normalize(L2NormLayer* const l);
void linear(LinearLayer* const l);
void logsoft(LogSoftLayer* const l);
double nll_loss(Matrix *const pred, uint32_t *labels, Slice slice);
double accuracy(Matrix *const pred, uint32_t *labels, uint32_t num_classes, Slice slice);

void cross_entropy_backward(LogSoftLayer *const l, Slice slice);
void linear_backward(LinearLayer* const l, Slice slice);
void normalize_backward(L2NormLayer* const l, Slice slice);
void relu_backward(ReluLayer* const l, Slice slice);
void sageconv_backward(SageLayer* const l, Slice slice);

void linear_layer_update_weights(LinearLayer* const l, float lr, Slice slice);
void sage_layer_update_weights(SageLayer* const l, float lr, Slice slice);

void sage_layer_zero_gradients(SageLayer* l);
void relu_layer_zero_gradients(ReluLayer* l);
void normalize_layer_zero_gradients(L2NormLayer* l);
void linear_layer_zero_gradients(LinearLayer* l);
void logsoft_layer_zero_gradients(LogSoftLayer* l);
void sage_net_zero_gradients(SageNet* net);

#endif // GNN_H
