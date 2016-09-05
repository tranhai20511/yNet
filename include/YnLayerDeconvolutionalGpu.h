#ifndef YNLAYERDECONVOLUTIONALGPU_H
#define YNLAYERDECONVOLUTIONALGPU_H

#include "YnLayerDeconvolutional.h"

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
void YnLayerDeconvolutionalGpuForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalGpuUpdate(tYnLayer layer,
        float learningRate,
        float momentum,
        float decay)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalGpuPush(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalGpuPull(tYnLayer layer)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERDECONVOLUTIONALGPU_H */
