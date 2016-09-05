#ifndef YNLAYERSOFTMAX_H
#define YNLAYERSOFTMAX_H

#include "YnLayer.h"
#include "YnNetwork.h"
#include "YnBlas.h"
#include "YnGpu.h"

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
tYnLayer YnLayerSoftmaxMake(int batchNum,
        int inputs,
        int groups)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerSoftmaxArray(float * input,
        int num,
        float temp,
        float * output)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerSoftmaxForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerSoftmaxBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERSOFTMAX_H */

