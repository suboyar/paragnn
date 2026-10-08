#ifndef KERNELS_H
#define KERNELS_H

#define KERNEL_TOUCH_ARGS                       \
    Real *restrict in_states,                   \
    Real *restrict agg_states,                  \
    int64_t dim,                                \
    SparseGraph *graph,                         \
    FlowDirection flow

#define KERNEL_ARGS                             \
    const Real *restrict in_states,             \
    Real *restrict agg_states,                  \
    int64_t dim,                                \
    SparseGraph *graph,                         \
    FlowDirection flow

typedef void (*TouchFunc)(KERNEL_TOUCH_ARGS);
typedef void (*KernelFunc)(KERNEL_ARGS);

void coo_v1_touch(KERNEL_TOUCH_ARGS);
void coo_v1(KERNEL_ARGS);

void cs_v1_touch(KERNEL_TOUCH_ARGS);
void cs_v1(KERNEL_ARGS);

void cs_v2_touch(KERNEL_TOUCH_ARGS);
void cs_v2(KERNEL_ARGS);

void cs_v3_touch(KERNEL_TOUCH_ARGS);
void cs_v3(KERNEL_ARGS);

void cs_v4_touch(KERNEL_TOUCH_ARGS);
void cs_v4(KERNEL_ARGS);

#endif // KERNELS_H
