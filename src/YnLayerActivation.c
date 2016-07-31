//	File        :   YnLayerActivation.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   30-07-2016
//	Author      :   haittt

#include "../include/YnLayerActivation.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
tYnLayer YnLayerActivationMake(int32 batchNum,
        int32 inputNum,
        eYnActivationType activation)
{
    tYnLayer layer = {0};

    layer.type = cYnLayerActive;
    layer.inputs = inputNum;
    layer.outputs = inputNum;
    layer.batch = batchNum;

    layer.output = calloc(batchNum * inputNum, sizeof(float *));
    layer.delta = calloc(batchNum * inputNum, sizeof(float *));

#ifdef YN_GPU
    layer.outputGpu = YnCudaMakeArray(layer.output, inputNum * batchNum);
    layer.deltaGpu = YnCudaMakeArray(layer.delta, inputNum * batchNum);
#endif

    layer.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputNum);

    return layer;
}

void YnLayerActivationForward(tYnLayer * layer,
        tYnNetworkState netState)
{
    YnBlasArrayCopyValueSet(layer.output, netState.input, layer.outputs * layer.batch, 1, 1);
    YnActivationOutputArrayCal(layer.output, layer.outputs * layer.batch, layer.activation);
}

void YnLayerActivationBackward(tYnLayer * layer,
        tYnNetworkState netState)
{
    YnActivationGradientArrayCal(layer.output, layer.outputs * layer.batch, layer.activation, layer.delta);
    YnBlasArrayCopyValueSet(netState.delta, layer.delta, layer.outputs * layer.batch, 1, 1);
}

