//	File        :   YnLayerConnected.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   31-07-2016
//	Author      :   haittt

#include "../include/YnLayerConnected.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
tYnLayer YnLayerConnectedMake(int32 batchNum,
        int32 inputNum,
        int32 outputNum,
        eYnActivationType activation,
        int32 batchNormalize)
{
    int i;
    float scale;
    tYnLayer layer = {0};

    layer.type = cYnLayerConnected;
    layer.inputs = inputNum;
    layer.outputs = outputNum;
    layer.batch = batchNum;
    layer.batchNormalize = batchNormalize;

    layer.output = calloc(batchNum * outputNum, sizeof(float));
    layer.delta = calloc(batchNum * outputNum, sizeof(float));

    layer.weightUpdates = calloc(inputNum * outputNum, sizeof(float));
    layer.biasUpdates = calloc(outputNum, sizeof(float));

    layer.weights = calloc(outputNum * inputNum, sizeof(float));
    layer.biases = calloc(outputNum, sizeof(float));

    scale = sqrt(2. / inputNum);

    for(i = 0; i < outputNum * inputNum; i ++)
    {
        layer.weights[i] = scale * YnUtilRandomUniformNum(-1, 1);
    }

    for(i = 0; i < outputNum; i ++)
    {
        layer.biases[i] = scale;
    }

    if (batchNormalize)
    {
        layer.scales = calloc(outputNum, sizeof(float));
        layer.scaleUpdates = calloc(outputNum, sizeof(float));

        for(i = 0; i < outputNum; ++i)
        {
            layer.scales[i] = 1;
        }

        layer.mean = calloc(outputNum, sizeof(float));
        layer.meanDelta = calloc(outputNum, sizeof(float));
        layer.variance = calloc(outputNum, sizeof(float));
        layer.varianceDelta = calloc(outputNum, sizeof(float));

        layer.rollingMean = calloc(outputNum, sizeof(float));
        layer.rollingVariance = calloc(outputNum, sizeof(float));

        layer.x = calloc(batchNum * outputNum, sizeof(float));
        layer.xNorm = calloc(batchNum * outputNum, sizeof(float));
    }

#ifdef YN_GPU
    layer.weightsGpu = cuda_make_array(layer.weights, outputNum * inputNum);
    layer.biasesGpu = cuda_make_array(layer.biases, outputNum);

    layer.weightUpdatesGpu = cuda_make_array(layer.weightUpdates, outputNum * inputNum);
    layer.biasUpdatesGpu = cuda_make_array(layer.biasUpdates, outputNum);

    layer.outputGpu = cuda_make_array(layer.output, outputNum * batchNum);
    layer.deltaGpu = cuda_make_array(layer.delta, outputNum * batchNum);

    if(batchNormalize)
    {
        layer.scalesGpu = cuda_make_array(layer.scales, outputNum);
        layer.scaleUpdatesGpu = cuda_make_array(layer.scaleUpdates, outputNum);

        layer.meanGpu = cuda_make_array(layer.mean, outputNum);
        layer.varianceGpu = cuda_make_array(layer.variance, outputNum);

        layer.rollingMeanGpu = cuda_make_array(layer.mean, outputNum);
        layer.rollingVarianceGpu = cuda_make_array(layer.variance, outputNum);

        layer.meanDeltaGpu = cuda_make_array(layer.mean, outputNum);
        layer.varianceDeltaGpu = cuda_make_array(layer.variance, outputNum);

        layer.xGpu = cuda_make_array(layer.output, layer.batch * outputNum);
        layer.xNormGpu = cuda_make_array(layer.output, layer.batch * outputNum);
    }
#endif

    layer.activation = activation;
    fprintf(stderr, "Connected Layer: %d inputs, %d outputs\n", inputNum, outputNum);

    return layer;
}

void YnLayerConnectedUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
{
    YnBlasArrayAxpyValueSet(layer.biases, layer.biasUpdates, layer.outputs, 1, 1, learningRate / batch);
    YnBlasArrayScaleValueSet(layer.biasUpdates, layer.outputs, 1, momentum);

    if(layer.batchNormalize)
    {
        YnBlasArrayAxpyValueSet(layer.scales, layer.scaleUpdates, layer.outputs, 1, 1, learningRate / batch);
        YnBlasArrayScaleValueSet(layer.scaleUpdates, layer.outputs, 1, momentum);
    }

    YnBlasArrayAxpyValueSet(layer.weightUpdates, layer.weights, layer.inputs * layer.outputs, 1, 1, -decay * batch);
    YnBlasArrayAxpyValueSet(layer.weights, layer.weightUpdates, layer.inputs * layer.outputs, 1, 1, learningRate / batch);
    YnBlasArrayScaleValueSet(layer.weightUpdates, layer.inputs * layer.outputs, 1, momentum);
}

void YnLayerConnectedForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    int m;
    int k;
    int n;
    float *a;
    float *b;
    float *c;

    YnBlasArrayFillValueSet(layer.output, layer.outputs * layer.batch, 1, 0);

    m = layer.batch;
    k = layer.inputs;
    n = layer.outputs;
    a = netState.input;
    b = layer.weights;
    c = layer.output;
    YnGemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    if(layer.batchNormalize)
    {
        if(netState.train)
        {
            YnBlasArrayMeanCal(layer.output, layer.batch, layer.outputs, 1, layer.mean);
            YnBlasArrayVarianceCal(layer.output, layer.mean, layer.batch, layer.outputs, 1, layer.variance);

            YnBlasArrayScaleValueSet(layer.rollingMean, layer.outputs, 1, .95);
            YnBlasArrayAxpyValueSet(layer.rollingMean, layer.mean, layer.outputs, 1, 1, .05);

            YnBlasArrayScaleValueSet(layer.rollingVariance, layer.outputs, 1, .95);
            YnBlasArrayAxpyValueSet(layer.rollingVariance, layer.variance, layer.outputs, 1, 1, .05);

            YnBlasArrayCopyValueSet(layer.x, layer.output, layer.outputs * layer.batch, 1, 1);
            YnBlasArrayNormalizeCal(layer.output, layer.mean, layer.variance, layer.batch, layer.outputs, 1);
            YnBlasArrayCopyValueSet(layer.xNorm, layer.output, layer.outputs * layer.batch, 1, 1);
        }
        else
        {
            YnBlasArrayNormalizeCal(layer.output, layer.rollingMean, layer.rollingVariance, layer.batch, layer.outputs, 1);
        }

        YnLayerConvolutionalScaleBias(layer.output, layer.scales, layer.batch, layer.outputs, 1);
    }

    for(i = 0; i < layer.batch; i ++)
    {
        YnBlasArrayAxpyValueSet(layer.output + i*layer.outputs, layer.biases, layer.outputs, 1, 1, 1);
    }

    YnActivationOutputArrayCal(layer.output, layer.outputs * layer.batch, layer.activation);
}

void YnLayerConnectedBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    int m;
    int k;
    int n;
    float *a;
    float *b;
    float *c;

    YnActivationGradientArrayCal(layer.output, layer.outputs * layer.batch, layer.activation, layer.delta);

    for(i = 0; i < layer.batch; i ++)
    {
        YnBlasArrayAxpyValueSet(layer.biasUpdates, layer.delta + i*layer.outputs, layer.outputs, 1, 1, 1);
    }

    if(layer.batchNormalize)
    {
        YnLayerConvolutionalBackwardScale(layer.xNorm, layer.delta, layer.batch, layer.outputs, 1, layer.scaleUpdates);
        YnLayerConvolutionalScaleBias(layer.delta, layer.scales, layer.batch, layer.outputs, 1);

        YnLayerConvolutionalMeanDelta(layer.delta, layer.variance, layer.batch, layer.outputs, 1, layer.meanDelta);
        YnLayerConvolutionalVarianceDelta(layer.x, layer.delta, layer.mean, layer.variance, layer.batch, layer.outputs, 1, layer.varianceDelta);
        YnLayerConvolutionalNormalizeDelta(layer.x, layer.mean, layer.variance, layer.meanDelta, layer.varianceDelta, layer.batch, layer.outputs, 1, layer.delta);
    }

    m = layer.outputs;
    k = layer.batch;
    n = layer.inputs;
    a = layer.delta;
    b = netState.input;
    c = layer.weightUpdates;
    YnGemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = layer.batch;
    k = layer.outputs;
    n = layer.inputs;

    a = layer.delta;
    b = layer.weights;
    c = netState.delta;

    if (c)
        YnGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}
