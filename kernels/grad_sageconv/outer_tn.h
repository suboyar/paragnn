#ifndef OUTER_TN_KERNEL_H
#define OUTER_TN_KERNEL_H

#include "core.h"
#include "layers.h"

// Outer TN V3
void outer_tn_v3_touch(SageLayer *l);
void outer_tn_v3(SageLayer *l);
void grad_sageconv_outer_tn_v3(SageLayer *l);

#endif // OUTER_TN_KERNEL_H
