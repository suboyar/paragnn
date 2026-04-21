#ifndef NN_H
#define NN_H

#include <stdint.h>

#include "layers.h"

void relu(ReluLayer* const l);
void l2norm(L2NormLayer* const l);
void linear(LinearLayer* const l);
void logsoftmax(LogSoftmaxLayer* const l);

Real nll_loss(LogSoftmaxLayer *l, const int64_t *labels);
Real accuracy(const LogSoftmaxLayer *l, const int64_t *labels);

void grad_cross_entropy(LogSoftmaxLayer *const l, int64_t *labels);
void grad_linear(LinearLayer* const l);
void grad_l2norm(L2NormLayer* const l);
void grad_relu(ReluLayer* const l);

void relu_layer_zero_gradients(ReluLayer* l);
void normalize_layer_zero_gradients(L2NormLayer* l);
void linear_layer_zero_gradients(LinearLayer* l);
void logsoft_layer_zero_gradients(LogSoftmaxLayer* l);

#endif // NN_H
