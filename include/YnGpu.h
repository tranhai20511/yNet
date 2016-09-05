#ifndef YNGPU_H
#define YNGPU_H

#include "YnStd.h"

#ifdef YN_GPU
#include "YnCudaGpu.h"
#endif

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
int YnCudaGpuIndexGet(void)
YN_ALSWAY_INLINE;

YN_FINAL
void YnCudaGpuIndexSet(int index)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNGPU_H */
