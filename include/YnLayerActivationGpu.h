#ifndef YNLAYERACTIVATIONGPU_H
#define YNLAYERACTIVATIONGPU_H

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
 * Forward layer
 */
YN_FINAL
void YnLayerActivationGpuForward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;

/*
 * Backward layer
 */
YN_FINAL
void YnLayerActivationGpuBackward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNLAYERACTIVATIONGPU_H */
