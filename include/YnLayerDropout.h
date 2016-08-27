#ifndef YNLAYERDROPOUT_H
#define YNLAYERDROPOUT_H

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
YN_FINAL
tYnLayer YnLayerDropoutMake(int batchNum,
        int inputs,
        float probability)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDropoutForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDropoutBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDropoutResize(tYnLayer layer,
        int inputs)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERDROPOUT_H */
