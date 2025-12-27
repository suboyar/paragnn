#ifndef LINALG_H
#define LINALG_H

typedef enum LINALG_TRANSPOSE {
    LinalgNoTrans = 0,
    LinalgTrans = 1,
} LINALG_TRANSPOSE;

#include "gemm.h"
#include "axpy.h"

#endif // LINALG_H
