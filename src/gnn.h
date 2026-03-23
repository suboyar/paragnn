#ifndef GNN_H
#define GNN_H

#include <stddef.h>

#include "layers.h"

void sageconv(SageLayer* const l);
void relu(ReluLayer* const l);
void normalize(L2NormLayer* const l);
void linear(LinearLayer* const l);
void logsoft(LogSoftLayer* const l);

float nll_loss(LogSoftLayer *l, const uint32_t *labels);
float accuracy(const LogSoftLayer *l, const uint32_t *labels);

void cross_entropy_backward(LogSoftLayer *const l, uint32_t *labels);
void linear_backward(LinearLayer* const l);
void normalize_backward(L2NormLayer* const l);
void relu_backward(ReluLayer* const l);
void sageconv_backward(SageLayer *const l);

void sage_layer_zero_gradients(SageLayer* l);
void relu_layer_zero_gradients(ReluLayer* l);
void normalize_layer_zero_gradients(L2NormLayer* l);
void linear_layer_zero_gradients(LinearLayer* l);
void logsoft_layer_zero_gradients(LogSoftLayer* l);
void sage_net_zero_gradients(SageNet* net);

#endif // GNN_H
