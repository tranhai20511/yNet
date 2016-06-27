#ifndef YNACTIVATIONLAYER_H
#define YNACTIVATIONLAYER_H

#include "../YnActivation.h"
#include "../YnLayer.h"
#include "../YnNetwork.h"


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
YN_FINAL eYnRetCode YnActivationLayerInit(tYnLayer * layer,
        uint32 batchNum,
        uint32 inputNum,
        eYnActivationType activation);

/*
 * Init layer
 */
YN_FINAL eYnRetCode YnActivationLayerForward(tYnLayer * layer,
        tYnNetworkState* netState);


#ifdef __cplusplus
}
#endif
