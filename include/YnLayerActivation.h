#ifndef YNLAYERACTIVATION_H
#define YNLAYERACTIVATION_H

#include "YnActivation.h"
#include "YnLayer.h"
#include "YnNetwork.h"


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
tYnLayer YnLayerActivationMake(int32 batchNum,
        int32 inputNum,
        eYnActivationType activation)
YN_ALSWAY_INLINE;

/*
 * Forward layer
 */
YN_FINAL
void YnLayerActivationForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

/*
 * Backward layer
 */
YN_FINAL
void YnLayerActivationBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNLAYERACTIVATION_H */
