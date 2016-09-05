#ifndef YNLAYERCONNECTEDGPU_H
#define YNLAYERCONNECTEDGPU_H

#include "YnLayerConnected.h"

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
void YnLayerConnectedGpuForward(tYnLayer * layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConnectedGpuBackward(tYnLayer * layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConnectedGpuUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConnectedGpuPush(tYnLayer * layer)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConnectedGpuPull(tYnLayer * layer)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERCONNECTEDGPU_H */
