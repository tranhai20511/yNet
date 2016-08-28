#ifndef YNLAYERMAXPOOL_H
#define YNLAYERMAXPOOL_H

#include "../YnLayer.h"
#include "../YnNetwork.h"
#include "../YnImage.h"

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
tYnLayer YnLayerMaxpoolMake(int batchNum,
        int height,
        int width,
        int channel,
        int size,
        int stride)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerMaxpoolForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerMaxpoolBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerMaxpoolResize(tYnLayer * layer,
        int width,
        int height)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerMaxpoolImageGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerMaxpoolGradientGet(tYnLayer layer)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERMAXPOOL_H */

