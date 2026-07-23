#ifndef GRAD_SAGECONV_GEMM_H
#define GRAD_SAGECONV_GEMM_H

void grad_sageconv_gemm_tn_v1(SageLayer *l);
void grad_sageconv_gemm_tn_v2(SageLayer *l);
void grad_sageconv_gemm_tn_v3(SageLayer *l);
void grad_sageconv_gemm_tn_v4(SageLayer *l);
void grad_sageconv_gemm_tn_blas(SageLayer *l);

#endif // GRAD_SAGECONV_GEMM_H
