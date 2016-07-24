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
YN_FINAL
eYnRetCode YnActivationLayerInit(tYnLayer * layer,
        int32 batchNum,
        int32 inputNum,
        eYnActivationType activation)
YN_ALSWAY_INLINE;

/*
 * Forward layer
 */
YN_FINAL
eYnRetCode YnActivationLayerForward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;

/*
 * Backward layer
 */
YN_FINAL
eYnRetCode YnActivationLayerBackward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNACTIVATIONLAYER_H */
