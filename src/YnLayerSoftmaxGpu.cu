//	File        :   YnLayerSoftmaxGpu.cu
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerSoftmaxGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_GPU_GLOBAL void _YnLayerSoftmaxGpuForward(int n,
        int batch,
        float *input,
        float temp,
        float *output)
{
    int i;
    int val;
    float sum = 0;
    float largest = -INFINITY;

    int b = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (b >= batch)
        return;

    for (i = 0; i < n; i ++)
    {
        val = input[i + b * n];
        largest = (val > largest) ? val : largest;
    }

    for (i = 0; i < n; i ++)
    {
        sum += exp(input[i + b * n] / temp - largest / temp);
    }

    sum = (sum != 0) ? largest / temp + log(sum) : largest - 100;

    for (i = 0; i < n; i ++)
    {
        output[i + b * n] = exp(input[i + b * n] / temp-sum);
    }
}

YN_EXTERN_C
void YnLayerSoftmaxGpuForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int inputs = layer.inputs / layer.groups;
    int batch = layer.batch * layer.groups;

    _YnLayerSoftmaxGpuForward<<<YnCudaGridSize(batch), YN_GPU_NUM_THREADS_IN_BLOCK>>>(inputs, batch, state.input, layer.temperature, layer.outputGpu);
    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnLayerSoftmaxGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    YnBlasGpuArrayAxpyValueSet(netState.delta, layer.deltaGpu, layer.batch * layer.inputs, 1, 1, 1);
}

YN_EXTERN_C
void YnLayerSoftmaxGpuPull(tYnLayer layer)
{
    YnCudaArrayPullFromGpu(layer.outputGpu, layer.output, layer.inputs * layer.batch);
}
