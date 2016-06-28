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
YN_FINAL eYnRetCode YnAvgPoolLayerImageGet(tYnLayer * layer,
        tYnImage * image);

/*
 * Init layer
 */
YN_FINAL eYnRetCode YnAvgPoolLayerInit(tYnLayer * layer,
        uint32 batchNum,
        uint32 width,
        uint32 height,
        uint32 channel);

/*
 * Init layer
 */
YN_FINAL eYnRetCode YnAvgPoolLayerResize(tYnLayer * layer,
        uint32 width,
        uint32 height);

/*
 * Forward layer
 */
YN_FINAL eYnRetCode YnAvgPoolLayerForward(tYnLayer * layer,
        tYnNetworkState* netState);

/*
 * Backward layer
 */
YN_FINAL eYnRetCode YnAvgPoolLayerBackward(tYnLayer * layer,
        tYnNetworkState* netState);


#ifdef __cplusplus
}
#endif
