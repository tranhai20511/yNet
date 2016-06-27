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
YN_FINAL eYnRetCode YnActivationLayerGpuInit(tYnLayer * layer,
        uint32 batchNum,
        uint32 inputNum,
        eYnActivationType activation);
/*
 * Forward layer
 */
YN_FINAL eYnRetCode YnActivationLayerGpuForward(tYnLayer layer,
        tYnNetworkState netState);

/*
 * Backward layer
 */
YN_FINAL eYnRetCode YnActivationLayerGpuBackward(tYnLayer layer,
        tYnNetworkState netState);


#ifdef __cplusplus
}
#endif
