//	File        :   YnLayerDropoutayerGpu.cu
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerDropoutGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_GPU_GLOBAL void _YnDropout(float *input,
        int size,
        float *rand,
        float prob,
        float scale)
{
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if(id < size)
        input[id] = (rand[id] < prob) ? 0 : input[id] * scale;
}

void YnLayerDropoutGpuForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int size;
    if (!netState.train)
        return;

    size = layer.inputs * layer.batch;
    YnCudaRandomArray(layer.randGpu, size);

    _YnDropout<<<YnCudaGridSize(size), YN_GPU_NUM_THREADS_IN_BLOCK>>>(state.input, size, layer.randGpu, layer.probability, layer.scale);
    YnCudaCheckError(cudaPeekAtLastError());
}

void YnLayerDropoutGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int size;
    if (!netState.delta)
        return;

    size = layer.inputs * layer.batch;

    _YnDropout<<<YnCudaGridSize(size), YN_GPU_NUM_THREADS_IN_BLOCK>>>(state.delta, size, layer.randGpu, layer.probability, layer.scale);
    YnCudaCheckError(cudaPeekAtLastError());
}
