//	File        :   YnLayerConnectedGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   31-07-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "../include/YnLayerConnectedGpu.h"
#include "../include/YnCudaGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnLayerConnectedGpuPull(tYnLayer layer)
{
    YnCudaArrayPullFromGpu(layer.weightsGpu, layer.weights, layer.inputs * layer.outputs);
    YnCudaArrayPullFromGpu(layer.biasesGpu, layer.biases, layer.outputs);
    YnCudaArrayPullFromGpu(layer.weightUpdatesGpu, layer.weightUpdates, layer.inputs * layer.outputs);
    YnCudaArrayPullFromGpu(layer.biasUpdatesGpu, layer.biasUpdates, layer.outputs);

    if (layer.batchNormalize)
    {
        YnCudaArrayPullFromGpu(layer.scalesGpu, layer.scales, layer.outputs);
        YnCudaArrayPullFromGpu(layer.rollingMeanGpu, layer.rollingMean, layer.outputs);
        YnCudaArrayPullFromGpu(layer.rollingVarianceGpu, layer.rollingVariance, layer.outputs);
    }
}

void YnLayerConnectedGpuPush(tYnLayer layer)
{
    YnCudaArrayPushToGpu(layer.weightsGpu, layer.weights, layer.inputs * layer.outputs);
    YnCudaArrayPushToGpu(layer.biasesGpu, layer.biases, layer.outputs);
    YnCudaArrayPushToGpu(layer.weightUpdatesGpu, layer.weightUpdates, layer.inputs * layer.outputs);
    YnCudaArrayPushToGpu(layer.biasUpdatesGpu, layer.biasUpdates, layer.outputs);

    if (layer.batchNormalize)
    {
        YnCudaArrayPushToGpu(layer.scalesGpu, layer.scales, layer.outputs);
        YnCudaArrayPushToGpu(layer.rollingMeanGpu, layer.rollingMean, layer.outputs);
        YnCudaArrayPushToGpu(layer.rollingVarianceGpu, layer.rollingVariance, layer.outputs);
    }
}

void YnLayerConnectedGpuUpdate(tYnLayer layer,
        int batch,
        float learning_rate,
        float momentum,
        float decay)
{
    YnBlasGpuArrayAxpyValueSet(layer.biasesGpu, layer.biasUpdatesGpu, layer.outputs, 1, 1, learning_rate / batch);
    YnBlasGpuArrayScaleValueSet(layer.biasUpdatesGpu, layer.outputs, 1, momentum);

    if (layer.batchNormalize)
    {
        YnBlasGpuArrayAxpyValueSet(layer.scalesGpu, layer.scaleUpdatesGpu, layer.outputs, 1, 1, learning_rate / batch);
        YnBlasGpuArrayScaleValueSet(layer.scaleUpdatesGpu, layer.outputs, 1, momentum);
    }

    YnBlasGpuArrayAxpyValueSet(layer.weightUpdatesGpu, layer.weightsGpu, layer.inputs * layer.outputs, 1, 1, -decay * batch);
    YnBlasGpuArrayAxpyValueSet(layer.weightsGpu, layer.weightUpdatesGpu, layer.inputs * layer.outputs, 1, 1, learning_rate / batch);
    YnBlasGpuArrayScaleValueSet(layer.weightUpdatesGpu, layer.inputs * layer.outputs, 1, momentum);
}

void YnLayerConnectedGpuForward(tYnLayer layer,
        tYnNetworkState state)
{
    int i;
    int m;
    int k;
    int n;
    float * a;
    float * b;
    float * c;

    YnBlasGpuArrayFillValueSet(layer.outputGpu, layer.outputs * layer.batch, 1, 0);

    m = layer.batch;
    k = layer.inputs;
    n = layer.outputs;
    a = state.input;
    b = layer.weightsGpu;
    c = layer.outputGpu;
    YnGemmGpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    if (layer.batchNormalize)
    {
        if (state.train)
        {
            YnBlasGpuFastArrayMeanCal(layer.outputGpu, layer.batch, layer.outputs, 1, layer.meanGpu);
            YnBlasGpuFastArrayVarianceCal(layer.outputGpu, layer.meanGpu, layer.batch, layer.outputs, 1, layer.varianceGpu);

            YnBlasGpuArrayScaleValueSet(layer.rollingMeanGpu, layer.outputs, 1, .95);
            YnBlasGpuArrayAxpyValueSet(layer.rollingMeanGpu, layer.meanGpu, layer.outputs, 1, 1, .05);
            YnBlasGpuArrayScaleValueSet(layer.rollingVarianceGpu, layer.outputs, 1, .95);
            YnBlasGpuArrayAxpyValueSet(layer.rollingVarianceGpu, layer.varianceGpu, layer.outputs, 1, 1, .05);

            YnBlasGpuArrayCopyValueSet(layer.xGpu, layer.outputGpu, layer.outputs * layer.batch, 1, 1);
            YnBlasGpuArrayNormalizeCal(layer.outputGpu, layer.meanGpu, layer.varianceGpu, layer.batch, layer.outputs, 1);
            YnBlasGpuArrayCopyValueSet(layer.xNormGpu, layer.outputGpu, layer.outputs * layer.batch, 1, 1);
        }
        else
        {
            YnBlasGpuArrayNormalizeCal(layer.outputGpu, layer.rollingMeanGpu, layer.rollingVarianceGpu, layer.batch, layer.outputs, 1);
        }

        YnBlasGpuScaleBias(layer.outputGpu, layer.scalesGpu, layer.batch, layer.outputs, 1);
    }

    for (i = 0; i < layer.batch; i ++)
    {
        YnBlasGpuArrayAxpyValueSet(layer.outputGpu + i * layer.outputs, layer.biasesGpu, layer.outputs, 1, 1, 1);
    }

    YnActivationGpuOutputArrayCal(layer.outputGpu, layer.outputs * layer.batch, layer.activation);

}

void YnLayerConnectedGpuBackward(tYnLayer layer,
        tYnNetworkState state)
{
    int i;
    int m;
    int k;
    int n;
    float * a;
    float * b;
    float * c;

    YnActivationGpuGradientArrayCal(layer.outputGpu, layer.outputs * layer.batch, layer.activation, layer.deltaGpu);

    for (i = 0; i < layer.batch; i ++)
    {
        YnBlasGpuArrayAxpyValueSet(layer.biasUpdatesGpu, layer.deltaGpu + i * layer.outputs, layer.outputs, 1, 1, 1);
    }

    if (layer.batchNormalize)
    {
        YnBlasGpuBackwardScale(layer.xNormGpu, layer.deltaGpu, layer.batch, layer.outputs, 1, layer.scaleUpdatesGpu);

        YnBlasGpuScaleBias(layer.deltaGpu, layer.scalesGpu, layer.batch, layer.outputs, 1);

        YnBlasGpuFastArrayMeanGradientCal(layer.deltaGpu, layer.varianceGpu, layer.batch, layer.outputs, 1, layer.meanDeltaGpu);
        YnBlasGpuFastArrayVarianceGradientCal(layer.xGpu, layer.deltaGpu, layer.meanGpu, layer.varianceGpu, layer.batch, layer.outputs, 1, layer.varianceDeltaGpu);
        YnBlasGpuArrayNormalizeGradientCal(layer.xGpu, layer.meanGpu, layer.varianceGpu, layer.meanDeltaGpu, layer.varianceDeltaGpu, layer.batch, layer.outputs, 1, layer.deltaGpu);
    }

    m = layer.outputs;
    k = layer.batch;
    n = layer.inputs;
    a = layer.deltaGpu;
    b = state.input;
    c = layer.weightUpdatesGpu;
    YnGemmGpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = layer.batch;
    k = layer.outputs;
    n = layer.inputs;

    a = layer.deltaGpu;
    b = layer.weightsGpu;
    c = state.delta;

    if (c)
        YnGemmGpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}
