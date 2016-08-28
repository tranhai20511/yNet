#ifndef YNNETWORKGPU_H
#define YNNETWORKGPU_H

#include "../YnNetwork.h"
#include "../YnLayerCropGpu.h"
#include "../YnLayerConnectedGpu.h"
#include "../YnLayerConvolutionalGpu.h"
#include "../YnLayerActivationGpu.h"
#include "../YnLayerAvgpoolGpu.h"
#include "../YnLayerDeconvolutionalGpu.h"
#include "../YnLayerDetectionGpu.h"
#include "../YnLayerMaxpoolGpu.h"
#include "../YnLayerCostGpu.h"
#include "../YnLayerSoftmaxGpu.h"
#include "../YnLayerDropoutGpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Typedef */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
void YnNetworkGpuForward(tYnNetwork net,
        tYnNetworkState state)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkGpuBackward(tYnNetwork net,
        tYnNetworkState state)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkGpuUpdate(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkGpuTrainDatum(tYnNetwork net,
        float *x,
        float *y)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnNetworkGpuOutputLayerGet(tYnNetwork net,
        int i)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnNetworkGpuOutputGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnNetworkGpuPredict(tYnNetwork net,
        float *input)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNNETWORKGPU_H */
