#ifndef OUTER_TN_PARAMS_h
#define OUTER_TN_PARAMS_h

#if defined(TARGET_CPU_XEONMAX9480)        /* xeonmaxq */
    #define KC 448
    #define NC 1024
    #define K_UNROLL 8

#elif defined(TARGET_CPU_XEON8360Y)        /* habanaq  */
    #define KC 448
    #define NC 656
    #define K_UNROLL 8

#elif TARGET_CPU_XEON6960P                 /* h200q    */
    #define KC 448
    #define NC 1024
    #define K_UNROLL 8

#elif defined(TARGET_CPU_EPYC7601)         /* defq     */
    #define KC 320
    #define NC 304
    #define K_UNROLL 1

#elif defined(TARGET_CPU_EPYC7302P)        /* rome16q  */
    #define KC 320
    #define NC 304
    #define K_UNROLL 1

#elif defined(TARGET_CPU_EPYC7413)         /* fpgaq    */
    #define KC 384
    #define NC 256
    #define K_UNROLL 2

#elif defined(TARGET_CPU_EPYC7763)         /* milanq   */
    #define KC 384
    #define NC 256
    #define K_UNROLL 2

#elif defined(TARGET_CPU_EPYC9684X)        /* genoaxq  */
    #define KC 256
    #define NC 768
    #define K_UNROLL 8

#elif defined(TARGET_CPU_THUNDERX2)        /* armq     */
    #define KC 512
    #define NC 96
    #define K_UNROLL 4

#elif defined(TARGET_CPU_KUNPENG920)       /* huaq     */
    #define KC 512
    #define NC 192
    #define K_UNROLL 4

#elif defined(TARGET_CPU_NEOVERSEV2)       /* gh200q   */
    #define KC 512
    #define NC 384
    #define K_UNROLL 4

#else                                      /* fallback */
    #define KC 256
    #define NC 48
    #define K_UNROLL 1
#endif

#endif // OUTER_TN_PARAMS_H
