#ifndef YNACTIVATIONLAYERGPU_H
#define YNACTIVATIONLAYERGPU_H

#include "../YnActivationLayer.h"
#include "../YnLayer.h"
#include "../YnNetworkGpu.h"

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

/*
 * Init layer
 */
YN_FINAL
eYnRetCode YnActivationLayerGpuInit(tYnLayer * layer,
        int32 batchNum,
        int32 inputNum,
        eYnActivationType activation)
YN_ALSWAY_INLINE;
/*
 * Forward layer
 */
YN_FINAL
eYnRetCode YnActivationLayerGpuForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

/*
 * Backward layer
 */
YN_FINAL
eYnRetCode YnActivationLayerGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNACTIVATIONLAYERGPU_H */
