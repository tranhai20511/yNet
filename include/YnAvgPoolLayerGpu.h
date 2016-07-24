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
YN_FINAL
eYnRetCode YnAvgPoolLayerGpuInit(tYnLayer * layer,
        int32 batchNum,
        int32 width,
        int32 height,
        int32 channel)
YN_ALSWAY_INLINE;

/*
 * Forward layer
 */
YN_FINAL
eYnRetCode YnAvgPoolLayerGpuForward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;

/*
 * Backward layer
 */
YN_FINAL
eYnRetCode YnAvgPoolLayerGpuBackward(tYnLayer * layer,
        tYnNetworkState* netState)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNAVGPOOLLAYERGPU_H */
