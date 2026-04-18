#ifndef SAGECONV_BACKWARD_GEMM_H
#define SAGECONV_BACKWARD_GEMM_H

void sageconv_backward_gemm_tn_v1(SageLayer *l);
void sageconv_backward_gemm_tn_v2(SageLayer *l);
void sageconv_backward_gemm_tn_v3(SageLayer *l);
void sageconv_backward_gemm_tn_v4(SageLayer *l);
void sageconv_backward_gemm_tn_blas(SageLayer *l);

#endif // SAGECONV_BACKWARD_GEMM_H
