// This is left as reference on how CSR in TARGET_TO_SOURCE would look like
void csr_v1(size_t node_count, size_t edge_count, size_t in_dim, void *edge_index,
            double *restrict X, size_t ldx, double *restrict Y, size_t ldy)
{
    (void)edge_count;
    CRS *edges = (CRS*)edge_index;

#pragma omp parallel for
    for (size_t i = 0; i < node_count; i++) {
        size_t degree = 0;
        double *y = Y + i * ldy;

        for (size_t src = 0; src < node_count; src++) {
            for (size_t j = edges->row_ptr[src]; j < edges->row_ptr[src+1]; j++) {
                uint32_t dst = edges->col_idx[j];
                if (dst == i) {
                    degree++;
                    double *x = X + src * ldx;
                    for (size_t k = 0; k < in_dim; k++) {
                        y[k] += x[k];
                    }
                }
            }
        }

        if (degree == 0) continue;
        double scale = 1.0 / degree;

        for (size_t k = 0; k < in_dim; k++) {
            y[k] *= scale;
        }
    }
}
