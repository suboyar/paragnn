#ifndef GNN_H
#define GNN_H

#include <stddef.h>

#include "matrix.h"
#include "layers.h"
#include "dataset.h"

void sageconv(SageLayer* const l, Dataset *ds);
void relu(ReluLayer* const l, Dataset *ds);
void normalize(L2NormLayer* const l, Dataset *ds);
void linear(LinearLayer* const l, Dataset *ds);
void logsoft(LogSoftLayer* const l, Dataset *ds);
double nll_loss(Matrix *const pred, uint32_t *labels);
double accuracy(Matrix *const pred, uint32_t *labels, uint32_t num_classes);

void cross_entropy_backward(LogSoftLayer *const l, Dataset *ds);
void linear_backward(LinearLayer* const l, Dataset *ds);
void normalize_backward(L2NormLayer* const l, Dataset *ds);
void relu_backward(ReluLayer* const l, Dataset *ds);
void sageconv_backward(SageLayer *const l, Dataset *ds);

void sage_layer_update_weights(SageLayer* const l, float lr, Dataset *ds);
void linear_layer_update_weights(LinearLayer* const l, float lr, Dataset *ds);

void sage_layer_zero_gradients(SageLayer* l, Dataset *ds);
void relu_layer_zero_gradients(ReluLayer* l, Dataset *ds);
void normalize_layer_zero_gradients(L2NormLayer* l, Dataset *ds);
void linear_layer_zero_gradients(LinearLayer* l, Dataset *ds);
void logsoft_layer_zero_gradients(LogSoftLayer* l, Dataset *ds);
void sage_net_zero_gradients(SageNet* net);

#endif // GNN_H
