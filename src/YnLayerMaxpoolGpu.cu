//	File        :   YnLayerMaxpoolGpu.cu
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerMaxpoolGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_GPU_GLOBAL void _YnLayerMaxpoolGpuForward(int n,
        int in_h,
        int in_w,
        int in_c,
        int stride,
        int size,
        float *input,
        float *output,
        int *indexes)
{
    int i;
    int j;
    int k;
    int b;
    int w_offset;
    int h_offset;
    int out_index;
    float max;
    int max_i;
    int l, m;
    int cur_h;
    int cur_w;
    int index;
    int valid;
    float val;
    int h = (in_h - 1) / stride + 1;
    int w = (in_w - 1) / stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    j = id % w;
    id /= w;
    i = id % h;
    id /= h;
    k = id % c;
    id /= c;
    b = id;

    w_offset = (- size - 1) / 2 + 1;
    h_offset = (- size - 1) / 2 + 1;

    out_index = j + w * (i + h * (k + c * b));
    max = -INFINITY;
    max_i = -1;

    for (l = 0; l < size; l ++)
    {
        for (m = 0; m < size; m ++)
        {
            cur_h = h_offset + i * stride + l;
            cur_w = w_offset + j * stride + m;
            index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
            valid = ((cur_h >= 0) && (cur_h < in_h) && (cur_w >= 0) && (cur_w < in_w));
            val = (valid != 0) ? input[index] : - INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }

    output[out_index] = max;
    indexes[out_index] = max_i;
}

YN_GPU_GLOBAL void _YnLayerMaxpoolGpuBackward(int n,
        int in_h,
        int in_w,
        int in_c,
        int stride,
        int size,
        float *delta,
        float *prev_delta,
        int *indexes)
{
    int index;
    int j;
    int i;
    int k;
    int b;
    int w_offset;
    int h_offset;
    int l, m;
    int out_w;
    int out_h;
    int out_index;
    int valid;

    int h = (in_h - 1) / stride + 1;
    int w = (in_w - 1) / stride + 1;
    int c = in_c;

    float d = 0;
    int area = (size - 1) / stride;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n)
        return;

    index = id;
    j = id % in_w;
    id /= in_w;
    i = id % in_h;
    id /= in_h;
    k = id % in_c;
    id /= in_c;
    b = id;

    w_offset = (-size-1)/2 + 1;
    h_offset = (-size-1)/2 + 1;

    for (l = -area; l < area+1; l ++)
    {
        for (m = -area; m < area+1; m ++)
        {
            out_w = (j - w_offset) / stride + m;
            out_h = (i - h_offset) / stride + l;
            out_index = out_w + w * (out_h + h * (k + c * b));
            valid = ((out_w >= 0) && (out_w < w) && (out_h >= 0) && (out_h < h));
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }

    prev_delta[index] += d;
}

YN_EXTERN_C
void YnLayerMaxpoolGpuForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;

    size_t n = h*w*c*layer.batch;

    _YnLayerMaxpoolGpuForward<<<YnCudaGridSize(n), YN_GPU_NUM_THREADS_IN_BLOCK>>>(n,
            layer.h,
            layer.w,
            layer.c,
            layer.stride,
            layer.size,
            state.input,
            layer.outputGpu,
            layer.indexesGpu);
    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnLayerMaxpoolGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;

    _YnLayerMaxpoolGpuBackward<<<YnCudaGridSize(n), YN_GPU_NUM_THREADS_IN_BLOCK>>>(n,
            layer.h,
            layer.w,
            layer.c,
            layer.stride,
            layer.size,
            layer.deltaGpu,
            state.delta,
            layer.indexesGpu);
    YnCudaCheckError(cudaPeekAtLastError());
}
