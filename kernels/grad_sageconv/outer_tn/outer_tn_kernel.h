#ifndef OUTER_TN_KERNEL_H
#define OUTER_TN_KERNEL_H

#include "core.h"
#include "layers.h"

#define DECLARE_OUTER_TN(N)                                             \
    void outer_tn_v##N(int64_t, int64_t, int64_t, const Real*, int64_t, const Real*, int64_t, Real*, int64_t); \
    void outer_tn_kernel_v##N(SageLayer *l);                            \
    void grad_sageconv_v##N(SageLayer *l);

DECLARE_OUTER_TN(1)
DECLARE_OUTER_TN(2)
DECLARE_OUTER_TN(3)

#endif // OUTER_TN_KERNEL_H
