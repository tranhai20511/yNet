#ifndef YNLAYERSOFTMAXGPU_H
#define YNLAYERSOFTMAXGPU_H

#include "../YnLayerSoftmax.h"
#include "../YnCudaGpu.h"
#include "../YnBlasGpu.h"
#include "../YnImageGpu.h"

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
void YnLayerSoftmaxGpuForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerSoftmaxGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerSoftmaxGpuPull(tYnLayer layer)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNLAYERSOFTMAXGPU_H */
