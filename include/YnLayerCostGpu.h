#ifndef YNLAYERCOSTGPU_H
#define YNLAYERCOSTGPU_H

#include "../YnLayerCost.h"

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
void YnLayerCostGpuForward(tYnLayer * layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerCostGpuBackward(tYnLayer * layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERCOSTGPU_H */
