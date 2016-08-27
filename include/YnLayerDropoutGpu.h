#ifndef YNLAYERDROPOUTGPU_H
#define YNLAYERDROPOUTGPU_H

#include "../YnLayerDetection.h"
#include "../YnCudaGpu.h"
#include "../YnActivationGpu.h"
#include "../YnBlasGpu.h"

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
void YnLayerDetectionGpuForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDetectionGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERDROPOUTGPU_H */
