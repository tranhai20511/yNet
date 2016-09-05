#ifndef YNLAYERCROP_H
#define YNLAYERCROP_H

#include "YnLayer.h"
#include "YnNetwork.h"
#include "YnImage.h"

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
tYnLayer YnLayerCropMake(int32 batchNum,
        int32 height,
        int32 width,
        int32 channel,
        int32 cropHeight,
        int32 cropWidth,
        int32 flip,
        float angle,
        float saturation,
        float exposure)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerCropForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerCropImageGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerCropResize(tYnLayer * layer,
        int32 width,
        int32 height)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERCROP_H */
