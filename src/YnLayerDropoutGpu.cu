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
    if (!state.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(state.input, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

void YnLayerDropoutGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    if(!state.delta) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(state.delta, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}
