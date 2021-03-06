#ifndef YNLAYERDROPOUTGPU_H
#define YNLAYERDROPOUTGPU_H

#include "YnLayerDropout.h"
#include "YnCudaGpu.h"
#include "YnActivationGpu.h"
#include "YnBlasGpu.h"

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
void YnLayerDropoutGpuForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDropoutGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERDROPOUTGPU_H */
