#ifndef SAGECONV_H
#define SAGECONV_H

#include "layers.h"

void sageconv(SageLayer* const l);
void grad_sageconv(SageLayer *const l);
void sage_layer_zero_gradients(SageLayer* l);

#endif // SAGECONV_H
