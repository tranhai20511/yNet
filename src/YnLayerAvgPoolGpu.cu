//	File        :   YnLayerAvgPoolGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   30-07-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "../include/YnLayerAvgPoolGpu.h"
#include "../include/YnCudaGpu.h"
}

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_GPU_GLOBAL void YnLayerAvgPoolGpuKernelForward(int num,
        int width,
        int height,
        int channel,
        float * input,
        float * output)
{
    int k;
    int b;
    int i;
    int out_index;
    int in_index;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (id >= num)
        return;

    k = id % channel;
    id /= channel;
    b = id;

    out_index = (k + channel * b);
    output[out_index] = 0;
    for (i = 0; i < width * height; i ++)
    {
        in_index = i + height * width * (k + b * channel);
        output[out_index] += input[in_index];
    }

    output[out_index] /= width * height;
}

YN_GPU_GLOBAL void YnLayerAvgPoolGpuKernelBackward(int num,
        int width,
        int height,
        int channel,
        float * inDelta,
        float * outDelta)
{
    int k;
    int b;
    int i;
    int out_index;
    int in_index;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (id >= num)
        return;

    k = id % channel;
    id /= channel;
    b = id;

    out_index = (k + channel * b);
    for (i = 0; i < width * height; i ++)
    {
        in_index = i + height * width * (k + b * channel);
        inDelta[in_index] += outDelta[out_index] / (width * height);
    }
}

YN_EXTERN_C
void YnLayerAvgPoolGpuForward(tYnLayer * layer,
        tYnNetworkState netState)
{
    YnBlasGpuArrayCopyValueSet(layer.output, netState.input, layer.outputs * layer.batch, 1, 1);
    YnActivationGpuOutputArrayCal(layer.output, layer.outputs * layer.batch, layer.activation);
}

YN_EXTERN_C
void YnLayerAvgPoolGpuBackward(tYnLayer * layer,
        tYnNetworkState netState)
{
    YnActivationGpuGradientArrayCal(layer.output, layer.outputs * layer.batch, layer.activation, layer.delta);
    YnBlasGpuArrayCopyValueSet(netState.delta, layer.delta, layer.outputs * layer.batch, 1, 1);
}
