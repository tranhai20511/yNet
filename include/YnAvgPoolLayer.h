#ifndef YNAVGPOOLLAYER_H
#define YNAVGPOOLLAYER_H

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

/*
 * Get image in avgPool layer
 */
YN_FINAL
eYnRetCode YnAvgPoolLayerImageGet(tYnLayer * layer,
        tYnImage * image)
YN_ALSWAY_INLINE;

/*
 * Init layer
 */
YN_FINAL
eYnRetCode YnAvgPoolLayerInit(tYnLayer * layer,
        int32 batchNum,
        int32 width,
        int32 height,
        int32 channel)
YN_ALSWAY_INLINE;

/*
 * Init layer
 */
YN_FINAL
eYnRetCode YnAvgPoolLayerResize(tYnLayer * layer,
        int32 width,
        int32 height)
YN_ALSWAY_INLINE;

/*
 * Forward layer
 */
YN_FINAL
eYnRetCode YnAvgPoolLayerForward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;

/*
 * Backward layer
 */
YN_FINAL
eYnRetCode YnAvgPoolLayerBackward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNAVGPOOLLAYER_H */
