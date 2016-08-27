//	File        :   YnLayerCostGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   15-08-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "../include/YnLayerCostGpu.h"
#include "../include/YnCudaGpu.h"
#include "../include/YnBlasGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnLayerCostGpuForward(tYnLayer * layer,
        tYnNetworkState netState)
{
    if (!netState.truth)
        return;

    if (layer.costType == cYnLayerCostMasked)
        YnBlasGpuArrayMaskValueSet(netState.input, layer.batch * layer.inputs, YN_CUS_NUM, netState.truth);

    if (layer.costType == cYnLayerCostSmooth)
        YnBlasGpuGradientSmoothL1(netState.input, netState.truth, layer.deltaGpu, layer.batch * layer.inputs);
    else
    {
        YnBlasGpuArrayCopyValueSet(layer.deltaGpu, netState.truth, layer.batch * layer.inputs, 1, 1);
        YnBlasGpuArrayAxpyValueSet(layer.deltaGpu, netState.input, layer.batch * layer.inputs, 1, 1, -1);
    }

    YnCudaArrayPullFromGpu(layer.deltaGpu, layer.delta, layer.batch*layer.inputs);
    *(layer.output) = YnBlasArrayDotValueSet(layer.delta, layer.delta, layer.batch * layer.inputs, 1, 1);
}

void YnLayerCostGpuBackward(tYnLayer * layer,
        tYnNetworkState netState)
{
    YnBlasGpuArrayAxpyValueSet(netState.delta, layer.deltaGpu, layer.batch * layer.inputs, 1, 1, layer.scale);
}
