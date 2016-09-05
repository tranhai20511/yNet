#ifndef YNLAYERDETECTION_H
#define YNLAYERDETECTION_H

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
YN_FINAL
tYnLayer YnLayerDetectionMake(int batchNum,
        int inputs,
        int num,
        int side,
        int classes,
        int coords,
        int rescore)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDetectionForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDetectionBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERDETECTION_H */
