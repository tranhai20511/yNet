//	File        :   YnNetworkGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnNetworkGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnNetworkGpuForward(tYnNetwork net,
        tYnNetworkState state)
{
    int i;
    tYnLayer layer;

    for (i = 0; i < net.n; i ++)
    {
        state.index = i;
        layer = net.layers[i];
        if (layer.deltaGpu)
        {
            YnBlasGpuArrayFillValueSet(layer.deltaGpu, layer.outputs * layer.batch, 1, 0)
        }

        switch(layer.type)
        {
            case cYnLayerConvolutional:
                YnLayerConvolutionalGpuForward(layer, state);
                break;
            case cYnLayerActive:
                YnLayerActivationGpuForward(layer, state);
                break;
            case cYnLayerLocal:
                /*YnLayerLocalGpuForward(layer, state);*/
                break;
            case cYnLayerDeconvolutional:
                YnLayerDeconvolutionalGpuForward(layer, state);
                break;
            case cYnLayerConnected:
                YnLayerConnectedGpuForward(layer, state);
                break;
            case cYnLayerRnn:
                /*YnLayerRnnGpuForward(layer, state);*/
                break;
            case cYnLayerMaxpool:
                YnLayerMaxpoolGpuForward(layer, state);
                break;
            case cYnLayerAvgpool:
                YnLayerAvgpoolGpuForward(layer, state);
                break;
            case cYnLayerSoftmax:
                YnLayerSoftmaxGpuForward(layer, state);
                break;
            case cYnLayerDetection:
                YnLayerDetectionGpuForward(layer, state);
                break;
            case cYnLayerDropout:
                YnLayerDropoutGpuForward(layer, state);
                break;
            case cYnLayerCrop:
                YnLayerCropGpuForward(layer, state);
                break;
            case cYnLayerCost:
                YnLayerCostGpuForward(layer, state);
                break;
            case cYnLayerRoute:
                /*YnLayerRouteGpuForward(layer, state);*/
                break;
            case cYnLayerShortcut:
                /*YnLayerShortcutGpuForward(layer, state);*/
                break;
            case cYnLayerNormalization:
                /*YnLayerNormalizationGpuForward(layer, state);*/
                break;
            default:
                break;
        }

        state.input = layer.outputGpu;
    }
}

void YnNetworkGpuBackward(tYnNetwork net,
        tYnNetworkState state)
{
    int i;
    tYnLayer layer;
    tYnLayer prev;
    float * original_input = state.input;
    float * original_delta = state.delta;

    for (i = net.n - 1; i >= 0; i --)
    {
        state.index = i;
        layer = net.layers[i];

        if (i == 0)
        {
            state.input = original_input;
            state.delta = original_delta;
        }
        else
        {
            prev = net.layers[i - 1];
            state.input = prev.outputGpu;
            state.delta = prev.deltaGpu;
        }

        switch(layer.type)
        {
            case cYnLayerConvolutional:
                YnLayerConvolutionalGpuBackward(layer, state);
                break;
            case cYnLayerActive:
                YnLayerActivationGpuBackward(layer, state);
                break;
            case cYnLayerLocal:
                YnLayerLocalGpuBackward(layer, state);
                break;
            case cYnLayerDeconvolutional:
                YnLayerDeconvolutionalGpuBackward(layer, state);
                break;
            case cYnLayerConnected:
                YnLayerConnectedGpuBackward(layer, state);
                break;
            case cYnLayerRnn:
                /*YnLayerRnnGpuBackward(layer, state);*/
                break;
            case cYnLayerMaxpool:
                YnLayerMaxpoolGpuBackward(layer, state);
                break;
            case cYnLayerAvgpool:
                YnLayerAvgpoolGpuBackward(layer, state);
                break;
            case cYnLayerSoftmax:
                YnLayerSoftmaxGpuBackward(layer, state);
                break;
            case cYnLayerDetection:
                YnLayerDetectionGpuBackward(layer, state);
                break;
            case cYnLayerDropout:
                YnLayerDropoutGpuBackward(layer, state);
                break;
            case cYnLayerCost:
                YnLayerCostGpuBackward(layer, state);
                break;
            case cYnLayerRoute:
                /*YnLayerRouteGpuBackward(layer, state);*/
                break;
            case cYnLayerShortcut:
                /*YnLayerShortcutGpuBackward(layer, state);*/
                break;
            case cYnLayerNormalization:
                /*YnLayerNormalizationGpuBackward(layer, state);*/
                break;
            default:
                break;
        }
    }
}

void YnNetworkGpuUpdate(tYnNetwork net)
{
    int i;
    tYnLayer layer;
    int updateBatch = net.batch * net.subdivisions;
    float rate = YnNetworkCurrentRateget(net);

    for (i = 0; i < net.n; i ++)
    {
        layer = net.layers[i];
        if (layer.type == cYnLayerConvolutional)
        {
            YnLayerConvolutionalGpuUpdate(layer, updateBatch, rate, net.momentum, net.decay);
        }
        else if (layer.type == cYnLayerDeconvolutional)
        {
            YnLayerDeconvolutionalGpuUpdate(layer, rate, net.momentum, net.decay);
        }
        else if (layer.type == cYnLayerConnected)
        {
            YnLayerConnectedGpuUpdate(layer, updateBatch, rate, net.momentum, net.decay);
        }
        else if (layer.type == cYnLayerRnn)
        {
            /*YnLayerRnnGpuUpdate(layer, updateBatch, rate, net.momentum, net.decay);*/
        }
        else if (layer.type == cYnLayerLocal)
        {
            /*YnLayerLocalGpuUpdate(layer, updateBatch, rate, net.momentum, net.decay);*/
        }
    }
}

float YnNetworkGpuTrainDatum(tYnNetwork net,
        float *x,
        float *y)
{
    tYnNetworkState state;
    int x_size;
    int y_size;
    float error;

    state.index = 0;
    state.net = net;

    x_size = YnNetworkInputSizeGet(net) * net.batch;
    y_size = YnNetworkOutputSizeGet(net) * net.batch;

    if (net.layers[net.n - 1].type == cYnLayerDetection)
        y_size = net.layers[net.n - 1].truths * net.batch;

    if (!*net.inputGpu)
    {
        *net.inputGpu = YnCudaMakeArray(x, x_size);
        *net.truthGpu = YnCudaMakeArray(y, y_size);
    }
    else
    {
        YnCudaArrayPushToGpu(*net.inputGpu, x, x_size);
        YnCudaArrayPushToGpu(*net.truthGpu, y, y_size);
    }

    state.input = *net.inputGpu;
    state.delta = 0;
    state.truth = *net.truthGpu;
    state.train = 1;
    YnNetworkGpuForward(net, state);
    YnNetworkGpuBackward(net, state);

    error = YnNetworkCostGet(net);
    if (((*net.seen) / net.batch) % net.subdivisions == 0)
        YnNetworkGpuUpdate(net);

    return error;
}

float * YnNetworkGpuOutputLayerGet(tYnNetwork net,
        int i)
{
    tYnLayer layer = net.layers[i];
    YnCudaArrayPullFromGpu(layer.outputGpu, layer.output, layer.outputs * layer.batch);
    return layer.output;
}

float * YnNetworkGpuOutputGet(tYnNetwork net)
{
    int i;

    for (i = net.n-1; i > 0; i --)
        if (net.layers[i].type != cYnLayerCost)
            break;

    return YnNetworkGpuOutputLayerGet(net, i);
}

float * YnNetworkGpuPredict(tYnNetwork net,
        float *input)
{
    tYnNetworkState state;
    float *out;
    int size = YnNetworkInputSizeGet(net) * net.batch;

    state.index = 0;
    state.net = net;
    state.input = YnCudaMakeArray(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    YnNetworkGpuForward(net, state);
    out = YnNetworkGpuOutputGet(net);

    YnCudaFreeArray(state.input);

    return out;
}
