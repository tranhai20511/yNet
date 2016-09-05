#ifndef YNLAYERCONVOLUTIONALGPU_H
#define YNLAYERCONVOLUTIONALGPU_H

#include "YnLayerConvolutional.h"

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
void YnLayerConvolutionalGpuForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalGpuUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalGpuPush(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalGpuPull(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalGpuAddBias(float * output,
        float * biases,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalGpuBackwardBias(float * biasUpdates,
        float * delta,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERCONVOLUTIONALGPU_H */
