#ifndef YNLAYERCOST_H
#define YNLAYERCOST_H

#include "YnLayer.h"
#include "YnNetwork.h"
#include "YnActivation.h"

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
tYnLayer YnLayerCostMake(int32 batchNum,
        int32 inputNum,
        eYnLayerCostType costType,
        float scale)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerCostForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerCostBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
eYnLayerCostType YnLayerCostStringToType(char * string)
YN_ALSWAY_INLINE;

YN_FINAL
char * YnLayerCostTypeToString(eYnLayerCostType type)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerCostResize(tYnLayer * layer,
        int input)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERCOST_H */
