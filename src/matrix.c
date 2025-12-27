#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "matrix.h"
#include "perf.h"

#include "../nob.h"

static long get_cache_line_size()
{
    long size;

    size = sysconf(_SC_LEVEL4_CACHE_LINESIZE);
    if (size > 0) return size;

    size = sysconf(_SC_LEVEL3_CACHE_LINESIZE);
    if (size > 0) return size;

    size = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
    if (size > 0) return size;

    size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (size > 0) return size;

#pragma omp single
    fprintf(stderr, "warning: Cache line size unavailable via sysconf(), using default of 64 bytes\n");

    return 64;
}

Matrix* matrix_create(size_t M, size_t N)
{
    Matrix *m = malloc(sizeof(*m));
    if (!m){
        ERROR("Could not allocate the matrix struct on heap");
    }

    size_t alignment = get_cache_line_size();
    size_t size = M * N * sizeof(*m->data);
    size_t padded_size = (size + alignment - 1) & ~(alignment - 1);

    m->data = aligned_alloc(alignment, padded_size);
    if (!m->data) {
        ERROR("Could not allocate data for the matrix");
    }

    m->M = M;
    m->N = N;
    m->stride = N;

#pragma omp parallel for
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            m->data[i*m->stride+j] = 0.0;
        }
    }

    return m;
}

void matrix_destroy(Matrix *m)
{
    free(m->data);
    free(m);
}

void matrix_zero(Matrix *m)
{
#pragma omp parallel for
    for (size_t i = 0; i < m->M; i++) {
        for (size_t j = 0; j < m->N; j++) {
            m->data[i*m->stride+j] = 0.0;
        }
    }

}

void matrix_fill_random(Matrix *m, double low, double high)
{
    double scale = (high - low);
    // OpenMP can't be used here as rand() isn't thread-safe, variants that might
    // be of interest are srand48_r or random_r. This can be looked at closed iff
    // fill_uniform becomes a bottleneck.
    for (size_t i = 0; i < m->M; i++) {
        for (size_t j = 0; j < m->N; j++) {
            m->data[i*m->stride+j] = ((double)rand() / RAND_MAX) * scale + low;
        }
    }
}

void matrix_dgemm(enum LINALG_TRANSPOSE TransA,
                  enum LINALG_TRANSPOSE TransB,
                  double alpha,
                  const Matrix *A,
                  const Matrix *B,
                  double beta,
                  Matrix *C) {

    size_t M = (TransA==LinalgNoTrans) ? A->M : A->N;
    size_t K = (TransA==LinalgNoTrans) ? A->N : A->M;
    size_t N = (TransB==LinalgNoTrans) ? B->N : B->M;
    dgemm(M, N, K,
          TransA,
          TransB,
          alpha,
          A->data, A->stride,
          B->data, B->stride,
          beta,
          C->data, C->stride);
}

void matrix_spec(Matrix* m, const char* name)
{
    if (m != NULL) {
        printf("%s (%zux%zu)\n", name, m->M, m->N);
    } else {
        printf("%s (nil)\n", name);
    }
}

void matrix_print(Matrix* m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m->rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m->cols; ++j) {
            printf("%f ", m->data[i*m->stride+j]);
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}

const char* matrix_shape(Matrix* m)
{
    if (m == NULL ) {
        return "(nil)";
    }
    return nob_temp_sprintf("(%zux%zu)", m->M, m->N);

}
