#ifndef YNLAYERAVGPOOL_H
#define YNLAYERAVGPOOL_H

#include "../YnImage.h"
#include "../YnCuda.h"
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
tYnLayer YnLayerAvgpoolMake(int32 batchNum,
        int32 width,
        int32 height,
        int32 channel)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerAvgpoolResize(tYnLayer * layer,
        int32 width,
        int32 height)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerAvgpoolForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerAvgpoolBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNLAYERAVGPOOL_H */
