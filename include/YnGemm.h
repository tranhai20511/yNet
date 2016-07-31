#ifndef YNGEMM_H
#define YNGEMM_H

#include "../YnStd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
void YnGemm(int TA,
        int TB,
        int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float BETA,
        float *C,
        int ldc)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnGemmRandomMatrix(int rows,
        int cols)
YN_ALSWAY_INLINE;

YN_FINAL
void YnGemmTimeRandomMatrix(int TA,
        int TB,
        int m,
        int k,
        int n)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNGEMM_H */
