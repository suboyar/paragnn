#ifndef LINALG_H
#define LINALG_H

typedef enum LINALG_TRANSPOSE {
    LinalgNoTrans = 0,
    LinalgTrans = 1,
} LINALG_TRANSPOSE;

// Level 1
#include "axpy.h"
#include "copy.h"
#include "scal.h"

// Level 3
#include "gemm.h"

#endif // LINALG_H
