//	File        :   YnLayerConvolutionalGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   02-08-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
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
YN_GPU_GLOBAL void _YnBinarizeFilters(float *filters,
        int num,
        int size,
        float *binary)
{
    int i = 0;
    float mean = 0;
    int f = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (f >= num)
        return;

    for (i = 0; i < size; i ++)
    {
        mean += abs(filters[f * size + i]);
    }

    mean = mean / size;

    for (i = 0; i < size; i ++)
    {
        binary[f * size + i] = (filters[f * size + i] > 0) ? mean : (- mean);
    }
}

YN_EXTERN_C YN_STATIC
void YnBinarizeFilters(float *filters,
        int num,
        int size,
        float *mean)
{
    _YnBinarizeFilters<<<YnCudaGridSize(num), YNGpu_NUM_THREADS_IN_BLOCK>>>(filters, num, size, mean);
    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C YN_STATIC
void YnBinarySwap(tYnLayer layer)
{
    float *swap = layer.filtersGpu;
    layer.filtersGpu = layer.binaryFiltersGpu;
    layer.binaryFiltersGpu = swap;
}

YN_EXTERN_C
void YnLayerConvolutionalGpuForward(tYnLayer layer,
        tYnNetworkState state)
{
    int i;
    float * a;
    float * b;
    float * c;
    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = YnLayerConvolutionalOutHeightGet(layer) * YnLayerConvolutionalOutWidthGet(layer);

    YnBlasGpuArrayFillValueSet(layer.outputGpu, layer.outputs * layer.batch, 1, 0);

    if (layer.binary)
    {
        YnBinarizeFilters(layer.filtersGpu,
                layer.n,
                layer.c * layer.size * layer.size,
                layer.binaryFiltersGpu);
        YnBinarySwap(layer);
    }

    for (i = 0; i < layer.batch; i ++)
    {
        YnImageGpuImage2Col(state.input + i * layer.c * layer.h * layer.w,
                layer.c,
                layer.h,
                layer.w,
                layer.size,
                layer.stride,
                layer.pad,
                layer.colImageGpu);

        a = layer.filtersGpu;
        b = layer.colImageGpu;
        c = layer.outputGpu;
        YnGemmGpu(0, 0, m, n, k, 1., a, k, b, n, 1., c + i * m * n, n);
    }

    if (layer.batchNormalize)
    {
        if (state.train)
        {
            YnBlasGpuFastArrayMeanCal(layer.outputGpu,
                    layer.batch,
                    layer.n,
                    layer.outH*layer.outW,
                    layer.meanGpu);

            YnBlasGpuFastArrayVarianceCal(layer.outputGpu,
                    layer.meanGpu,
                    layer.batch,
                    layer.n,
                    layer.outH * layer.outW,
                    layer.varianceGpu);

            YnBlasGpuArrayScaleValueSet(layer.rollingMeanGpu,
                    layer.n,
                    1,
                    .95);
            YnBlasGpuArrayAxpyValueSet(layer.rollingMeanGpu,
                    layer.meanGpu,
                    layer.n,
                    1,
                    1,
                    .05);
            YnBlasGpuArrayScaleValueSet(layer.rollingVarianceGpu,
                    layer.n,
                    1,
                    .95);
            YnBlasGpuArrayAxpyValueSet(layer.rollingVarianceGpu,
                    layer.varianceGpu,
                    layer.n,
                    1,
                    1,
                    .05);

            YnBlasGpuArrayCopyValueSet(layer.xGpu,
                    layer.outputGpu,
                    layer.outputs * layer.batch,
                    1,
                    1);

            YnBlasGpuArrayNormalizeCal(layer.outputGpu,
                    layer.meanGpu,
                    layer.varianceGpu,
                    layer.batch,
                    layer.n,
                    layer.outH * layer.outW);

            YnBlasGpuArrayCopyValueSet(layer.xNormGpu,
                    layer.outputGpu,
                    layer.outputs * layer.batch,
                    1,
                    1);
        }
        else
        {
            YnBlasGpuArrayNormalizeCal(layer.outputGpu,
                    layer.rollingMeanGpu,
                    layer.rollingVarianceGpu,
                    layer.batch,
                    layer.n,
                    layer.outH * layer.outW);
        }

        YnBlasGpuBiasScale(layer.outputGpu,
                layer.scalesGpu,
                layer.batch,
                layer.n,
                layer.outH * layer.outW);
    }

    YnBlasGpuBiasAdd(layer.outputGpu, layer.biasesGpu, layer.batch, layer.n, n);

    YnActivationGpuOutputArrayCal(layer.outputGpu, m * n * layer.batch, layer.activation);

    if (layer.binary)
        YnBinarySwap(layer);
}

void YnLayerConvolutionalGpuBackward(tYnLayer layer,
        tYnNetworkState state)
{
    int i;
    float * a;
    float * b;
    float * c;
    int m = layer.n;
    int n = layer.size * layer.size * layer.c;
    int k = YnLayerConvolutionalOutHeightGet(layer) * YnLayerConvolutionalOutWidthGet(layer);

    YnActivationGpuGradientArrayCal(layer.outputGpu,
            m * k * layer.batch,
            layer.activation,
            layer.deltaGpu);

    YnBlasGpuBiasBackward(layer.biasUpdatesGpu, layer.deltaGpu, layer.batch, layer.n, k);

    if (layer.batchNormalize)
    {
        YnBlasGpuBackwardScale(layer.xNormGpu,
                layer.deltaGpu,
                layer.batch,
                layer.n,
                layer.outW * layer.outH,
                layer.scaleUpdatesGpu);

        YnBlasGpuBiasScale(layer.deltaGpu,
                layer.scalesGpu,
                layer.batch,
                layer.n,
                layer.outH * layer.outW);

        YnBlasGpuFastArrayMeanGradientCal(layer.deltaGpu,
                layer.varianceGpu,
                layer.batch,
                layer.n,
                layer.outW * layer.outH,
                layer.meanDeltaGpu);

        YnBlasGpuFastArrayVarianceGradientCal(layer.xGpu,
                layer.deltaGpu,
                layer.meanGpu,
                layer.varianceGpu,
                layer.batch,
                layer.n,
                layer.outW * layer.outH,
                layer.varianceDeltaGpu);

        YnBlasGpuArrayNormalizeGradientCal(layer.xGpu,
                layer.meanGpu,
                layer.varianceGpu,
                layer.meanDeltaGpu,
                layer.varianceDeltaGpu,
                layer.batch,
                layer.n,
                layer.outW * layer.outH,
                layer.deltaGpu);
    }

    for (i = 0; i < layer.batch; i ++)
    {
        a = layer.deltaGpu;
        b = layer.colImageGpu;
        c = layer.filterUpdatesGpu;

        YnImageGpuImage2Col(state.input + i * layer.c * layer.h * layer.w,
                layer.c,
                layer.h,
                layer.w,
                layer.size,
                layer.stride,
                layer.pad,
                layer.colImageGpu);

        YnGemmGpu(0, 1, m, n, k, 1, a + i * m * k, k, b, k, 1, c, n);

        if (state.delta)
        {
            if (layer.binary)
                YnBinarySwap(layer);

            a = layer.filtersGpu;
            b = layer.deltaGpu;
            c = layer.colImageGpu;

            YnGemmGpu(1, 0, n, k, m, 1, a, n, b + i * k * m, k, 0, c, k);

            YnImageGpuCol2Image(layer.colImageGpu,
                    layer.c,
                    layer.h,
                    layer.w,
                    layer.size,
                    layer.stride,
                    layer.pad,
                    state.delta + i * layer.c * layer.h * layer.w);

            if (layer.binary)
                YnBinarySwap(layer);
        }
    }
}

void YnLayerConvolutionalGpuPull(tYnLayer layer)
{
    YnCudaArrayPullFromGpu(layer.filtersGpu,
            layer.filters,
            layer.c * layer.n * layer.size * layer.size);

    YnCudaArrayPullFromGpu(layer.biasesGpu,
            layer.biases,
            layer.n);

    YnCudaArrayPullFromGpu(layer.filterUpdatesGpu,
            layer.filterUpdates,
            layer.c * layer.n * layer.size * layer.size);

    YnCudaArrayPullFromGpu(layer.biasUpdatesGpu,
            layer.biasUpdates,
            layer.n);

    if (layer.batchNormalize)
    {
        YnCudaArrayPullFromGpu(layer.scalesGpu, layer.scales, layer.n);
        YnCudaArrayPullFromGpu(layer.rollingMeanGpu, layer.rollingMean, layer.n);
        YnCudaArrayPullFromGpu(layer.rollingVarianceGpu, layer.rollingVariance, layer.n);
    }
}

void YnLayerConvolutionalGpuPush(tYnLayer layer)
{
    YnCudaArrayPushToGpu(layer.filtersGpu,
            layer.filters,
            layer.c * layer.n * layer.size * layer.size);

    YnCudaArrayPushToGpu(layer.biasesGpu,
            layer.biases,
            layer.n);

    YnCudaArrayPushToGpu(layer.filterUpdatesGpu,
            layer.filterUpdates,
            layer.c * layer.n * layer.size * layer.size);

    YnCudaArrayPushToGpu(layer.biasUpdatesGpu,
            layer.biasUpdates,
            layer.n);

    if (layer.batchNormalize)
    {
        YnCudaArrayPushToGpu(layer.scalesGpu, layer.scales, layer.n);
        YnCudaArrayPushToGpu(layer.rollingMeanGpu, layer.rollingMean, layer.n);
        YnCudaArrayPushToGpu(layer.rollingVarianceGpu, layer.rollingVariance, layer.n);
    }
}

void YnLayerConvolutionalGpuUpdate(tYnLayer layer,
        int batch,
        float learning_rate,
        float momentum,
        float decay)
{
    int size = layer.size * layer.size * layer.c * layer.n;

    YnBlasGpuArrayAxpyValueSet(layer.biasesGpu, layer.biasUpdatesGpu, layer.n, 1, 1, learning_rate / batch);
    YnBlasGpuArrayScaleValueSet(layer.biasUpdatesGpu, layer.n, 1, momentum);

    YnBlasGpuArrayAxpyValueSet(layer.scalesGpu, layer.scaleUpdatesGpu, layer.n, 1, 1, learning_rate / batch);
    YnBlasGpuArrayScaleValueSet(layer.scaleUpdatesGpu, layer.n, 1, momentum);

    YnBlasGpuArrayAxpyValueSet(layer.filterUpdatesGpu, layer.filtersGpu, size, 1, 1, - decay * batch);
    YnBlasGpuArrayAxpyValueSet(layer.filtersGpu, layer.filterUpdatesGpu, size, 1, 1, learning_rate / batch);
    YnBlasGpuArrayScaleValueSet(layer.filterUpdatesGpu, size, 1, momentum);
}

