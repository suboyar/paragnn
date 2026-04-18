#ifndef SAGECONV_BACKWARD_H
#define SAGECONV_BACKWARD_H

#include <stdint.h>
#include "layers.h"

void scale_by_inv_degree(SageLayer *l);
void scatter_coo(uint32_t *nodes, uint32_t *peers, SageLayer *l);
void scale_by_inv_degree_parallel(SageLayer *l);
void scatter_coo_parallel(uint32_t *nodes, uint32_t *peers, SageLayer *l);

#endif // SAGECONV_BACKWARD_H
