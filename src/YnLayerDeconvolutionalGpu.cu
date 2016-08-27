//	File        :   YnLayerDeconvolutionalGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   02-08-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "../include/YnLayerDeconvolutionalGpu.h"
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
        tYnNetworkState state)
{
    int i;
    float *a;
    float *b;
    float *c;
    int out_h = YnLayerDeconvolutionalOutHeightGet(layer);
    int out_w = YnLayerDeconvolutionalOutWidthGet(layer);
    int size = out_h*out_w;

    int m = layer.size*layer.size*layer.n;
    int n = layer.h*layer.w;
    int k = layer.c;

    YnBlasGpuArrayFillValueSet(layer.outputGpu, layer.outputs * layer.batch, 1, 0);

    for (i = 0; i < layer.batch; i ++)
    {
        a = layer.filtersGpu;
        b = state.input + i * layer.c * layer.h * layer.w;
        c = layer.colImageGpu;

        YnGemmGpu(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

        YnImageGpuCol2Image(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.outputGpu + i * layer.n * size);
    }

    YnBlasGpuBiasAdd(layer.outputGpu, layer.biasesGpu, layer.batch, layer.n, size);
    YnActivationGpuOutputArrayCal(layer.outputGpu, layer.batch * layer.n * size, layer.activation);
}

void YnLayerDeconvolutionalGpuBackward(tYnLayer layer,
        tYnNetworkState state)
{
    int m;
    int n;
    int k;
    float *a;
    float *b;
    float *c;
    float alpha = 1. / layer.batch;
    int out_h = YnLayerDeconvolutionalOutHeightGet(layer);
    int out_w = YnLayerDeconvolutionalOutWidthGet(layer);
    int size = out_h * out_w;
    int i;

    YnActivationGradientArrayCal(layer.outputGpu, size*layer.n*layer.batch, layer.activation, layer.deltaGpu);
    YnBlasArrayBackwardBias(layer.biasUpdatesGpu, layer.delta, layer.batch, layer.n, size);

    if (state.delta)
        memset(state.delta, 0, layer.batch * layer.h * layer.w * layer.c * sizeof(float));

    for (i = 0; i < layer.batch; i ++)
    {
        m = layer.c;
        n = layer.size*layer.size*layer.n;
        k = layer.h*layer.w;

        a = state.input + i*m*n;
        b = layer.colImageGpu;
        c = layer.filterUpdatesGpu;

        YnImageGpuImage2Col(layer.deltaGpu + i * layer.n * size, layer.n, out_h, out_w,
                layer.size, layer.stride, 0, b);
        YnGemmGpu(0, 1, m, n, k, alpha, a, k ,b ,k ,1 ,c , n);

        if (state.delta)
        {
            m = layer.c;
            n = layer.h*layer.w;
            k = layer.size*layer.size*layer.n;

            a = layer.filtersGpu;
            b = layer.colImageGpu;
            c = state.delta + i * n * m;

            YnGemm(0, 0, m, n, k, 1, a, k, b ,n ,1 ,c , n);
        }
    }
}

void YnLayerDeconvolutionalGpuPull(tYnLayer layer)
{
    YnCudaArrayPullFromGpu(layer.filtersGpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    YnCudaArrayPullFromGpu(layer.biasesGpu, layer.biases, layer.n);
    YnCudaArrayPullFromGpu(layer.filterUpdatesGpu, layer.filterUpdates, layer.c * layer.n * layer.size * layer.size);
    YnCudaArrayPullFromGpu(layer.biasUpdatesGpu, layer.biasUpdates, layer.n);
}

void YnLayerDeconvolutionalGpuPush(tYnLayer layer)
{
    YnCudaArrayPushToGpu(layer.filtersGpu, layer.filters, layer.c * layer.n * layer.size * layer.size);
    YnCudaArrayPushToGpu(layer.biasesGpu, layer.biases, layer.n);
    YnCudaArrayPushToGpu(layer.filterUpdatesGpu, layer.filterUpdates, layer.c * layer.n * layer.size * layer.size);
    YnCudaArrayPushToGpu(layer.biasUpdatesGpu, layer.biasUpdates, layer.n);
}

void YnLayerDeconvolutionalGpuUpdate(tYnLayer layer,
        int batch,
        float learning_rate,
        float momentum,
        float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    YnBlasGpuArrayAxpyValueSet(layer.biasesGpu, layer.biasUpdatesGpu, layer.n, 1, 1, learning_rate);
    YnBlasGpuArrayScaleValueSet(layer.biasUpdatesGpu, layer.n, 1, momentum);

    YnBlasGpuArrayAxpyValueSet(layer.filterUpdatesGpu, layer.filtersGpu, size, 1, 1, - decay);
    YnBlasGpuArrayAxpyValueSet(layer.filtersGpu, layer.filterUpdatesGpu, size, 1, 1, learning_rate);
    YnBlasGpuArrayScaleValueSet(layer.filterUpdatesGpu, size, 1, momentum);
}

