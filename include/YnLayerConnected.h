#ifndef YNLAYERCONNECTED_H
#define YNLAYERCONNECTED_H

#include "../YnLayer.h"
#include "../YnNetwork.h"
#include "../YnActivation.h"

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
tYnLayer YnLayerConnectedMake(int32 batchNum,
        int32 inputNum,
        int32 outputNum,
        eYnActivationType activation,
        int32 batchNormalize)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConnectedForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConnectedBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConnectedUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERCONNECTED_H */
