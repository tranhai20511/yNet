//	File        :   YnLayerDeconvolutionalGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   25-08-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "../include/YnLayerDeconvolutionalGpu.h"
#include "../include/YnLayerConvolutionalGpu.h"
#include "../include/YnCudaGpu.h"
#include "../include/YnGemmGpu.h"
#include "../include/YnBlasGpu.h"
#include "../include/YnImageGpu.h"
#include "../include/YnUtil.h"
}

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_EXTERN_C
void YnLayerDeconvolutionalGpuForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;

    int m = layer.size*layer.size*layer.n;
    int n = layer.h*layer.w;
    int k = layer.c;

    fill_ongpu(layer.outputs*layer.batch, 0, layer.output_gpu, 1);

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.filters_gpu;
        float *b = state.input + i*layer.c*layer.h*layer.w;
        float *c = layer.col_image_gpu;

        gemm_ongpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_ongpu(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.output_gpu+i*layer.n*size);
    }
    add_bias_gpu(layer.output_gpu, layer.biases_gpu, layer.batch, layer.n, size);
    activate_array(layer.output_gpu, layer.batch*layer.n*size, layer.activation);
}

YN_EXTERN_C
void YnLayerDeconvolutionalGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    float alpha = 1./layer.batch;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;
    int i;

    gradient_array(layer.output_gpu, size*layer.n*layer.batch, layer.activation, layer.delta_gpu);
    backward_bias(layer.bias_updates_gpu, layer.delta, layer.batch, layer.n, size);

    if(state.delta) memset(state.delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));

    for(i = 0; i < layer.batch; ++i){
        int m = layer.c;
        int n = layer.size*layer.size*layer.n;
        int k = layer.h*layer.w;

        float *a = state.input + i*m*n;
        float *b = layer.col_image_gpu;
        float *c = layer.filter_updates_gpu;

        im2col_ongpu(layer.delta_gpu + i*layer.n*size, layer.n, out_h, out_w,
                layer.size, layer.stride, 0, b);
        gemm_ongpu(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(state.delta){
            int m = layer.c;
            int n = layer.h*layer.w;
            int k = layer.size*layer.size*layer.n;

            float *a = layer.filters_gpu;
            float *b = layer.col_image_gpu;
            float *c = state.delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

YN_EXTERN_C
void YnLayerDeconvolutionalGpuUpdate(tYnLayer layer,
        float learningRate,
        float momentum,
        float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(size, -decay, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, learning_rate, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, momentum, layer.filter_updates_gpu, 1);
}

YN_EXTERN_C
void YnLayerDeconvolutionalGpuPush(tYnLayer layer)
{
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

YN_EXTERN_C
void YnLayerDeconvolutionalGpuPull(tYnLayer layer)
{
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

