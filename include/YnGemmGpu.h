#ifndef YNGEMMGPU_H
#define YNGEMMGPU_H

#include "YnGemm.h"

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
void YnGemmGpu(int TA,
        int TB,
        int M,
        int N,
        int K,
        float ALPHA,
        float * A_gpu,
        int lda,
        float *B_gpu,
        int ldb,
        float BETA,
        float *C_gpu,
        int ldc)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNGEMMGPU_H */
