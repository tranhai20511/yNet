#ifndef YNLAYERAVGPOOLGPU_H
#define YNLAYERAVGPOOLGPU_H

#include "../YnLayerAvgPool.h"

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
void YnLayerAvgPoolGpuForward(tYnLayer * layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerAvgPoolGpuBackward(tYnLayer * layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERAVGPOOLGPU_H */
