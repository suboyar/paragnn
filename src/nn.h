#ifndef NN_H
#define NN_H

#include <stdint.h>

#include "layers.h"

void relu(ReluLayer* const l);
void l2norm(L2NormLayer* const l);
void logsoftmax(LogSoftmaxLayer* const l);

Real nll_loss(LogSoftmaxLayer *l, const int64_t *labels);
Real accuracy(const LogSoftmaxLayer *l, const int64_t *labels);

void grad_logsoftmax_nll(LogSoftmaxLayer *const l, int64_t *labels);
void grad_l2norm(L2NormLayer* const l);
void grad_relu(ReluLayer* const l);

#endif // NN_H
