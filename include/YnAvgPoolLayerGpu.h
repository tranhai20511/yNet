#ifndef YNAVGPOOLLAYERGPU_H
#define YNAVGPOOLLAYERGPU_H

#include "../YnAvgPoolLayer.h"

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
YN_FINAL eYnRetCode YnAvgPoolLayerGpuInit(tYnLayer * layer,
        uint32 batchNum,
        uint32 width,
        uint32 height,
        uint32 channel);

/*
 * Forward layer
 */
YN_FINAL eYnRetCode YnAvgPoolLayerGpuForward(tYnLayer * layer,
        tYnNetworkState* netState);

/*
 * Backward layer
 */
YN_FINAL eYnRetCode YnAvgPoolLayerGpuBackward(tYnLayer * layer,
        tYnNetworkState* netState);


#ifdef __cplusplus
}
#endif
