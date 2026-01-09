#ifndef GNN_H
#define GNN_H

#include <stddef.h>

#include "matrix.h"
#include "layers.h"
#include "dataset.h"

void sageconv(SageLayer* const l, Range nodes, Range edges);
void relu(ReluLayer* const l, Range nodes);
void normalize(L2NormLayer* const l, Range nodes);
void linear(LinearLayer* const l, Range nodes);
void logsoft(LogSoftLayer* const l, Range nodes);
double nll_loss(Matrix *const pred, uint32_t *labels, Range nodes);
double accuracy(Matrix *const pred, uint32_t *labels, uint32_t num_classes, Range nodes);

void cross_entropy_backward(LogSoftLayer *const l, Range nodes);
void linear_backward(LinearLayer* const l, Range nodes);
void normalize_backward(L2NormLayer* const l, Range nodes);
void relu_backward(ReluLayer* const l, Range nodes);
void sageconv_backward(SageLayer *const l, Range nodes, Range edges);

void sage_layer_update_weights(SageLayer* const l, float lr);
void linear_layer_update_weights(LinearLayer* const l, float lr);

void sage_layer_zero_gradients(SageLayer* l);
void relu_layer_zero_gradients(ReluLayer* l);
void normalize_layer_zero_gradients(L2NormLayer* l);
void linear_layer_zero_gradients(LinearLayer* l);
void logsoft_layer_zero_gradients(LogSoftLayer* l);
void sage_net_zero_gradients(SageNet* net);

#endif // GNN_H
