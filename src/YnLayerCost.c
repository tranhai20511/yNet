//	File        :   YnLayerCost.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   15-08-2016
//	Author      :   haittt

#include "../include/YnLayerCost.h"
#include "../include/YnCuda.h"
#include "../include/YnBlas.h"
#include "../include/YnUtil.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
eYnLayerCostType YnLayerCostStringToType(char * string)
{
    if (strcmp(string, "sse") == 0)
        return cYnLayerCostSse;
    if (strcmp(string, "masked") == 0)
        return cYnLayerCostMasked;
    if (strcmp(string, "smooth") == 0)
        return cYnLayerCostSmooth;

    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", string);
    return cYnLayerCostSse;
}

char * YnLayerCostTypeToString(eYnLayerCostType type)
{
    switch(type)
    {
        case cYnLayerCostSse:
            return "sse";
        case cYnLayerCostMasked:
            return "masked";
        case cYnLayerCostSmooth:
            return "smooth";
    }

    return "sse";
}

tYnLayer YnLayerCostMake(int32 batchNum,
        int32 inputNum,
        eYnLayerCostType costType,
        float scale)
{
    fprintf(stderr, "Cost Layer: %d inputs\n", inputNum);
    tYnLayer layer = {0};
    layer.type = cYnLayerCost;

    layer.scale = scale;
    layer.batch = batchNum;
    layer.inputs = inputNum;
    layer.outputs = inputNum;
    layer.costType = costType;
    layer.delta = calloc(inputNum * batchNum, sizeof(float));
    layer.output = calloc(1, sizeof(float));

#ifdef YN_GPU
    layer.deltaGpu = YnCudaMakeArray(layer.delta, inputNum * batchNum);
#endif

    return layer;
}

void YnLayerCostResize(tYnLayer layer,
        int inputNum)
{
    layer->inputs = inputNum;
    layer->outputs = inputNum;
    layer->delta = realloc(layer->delta, inputNum * layer->batch * sizeof(float));

#ifdef YN_GPU
    YnCudaFreeArray(layer->deltaGpu);
    layer->deltaGpu = YnCudaMakeArray(layer->delta, inputNum * layer->batch);
#endif
}

void YnLayerCostForward(tYnLayer layer,
        tYnNetworkState netState)
{
    if (!netState.truth)
        return;

    if (layer.costType == cYnLayerCostMasked)
    {
        int i;
        for (i = 0; i < layer.batch * layer.inputs; i ++)
        {
            if (netState.truth[i] == YN_CUS_NUM)
                netState.input[i] = YN_CUS_NUM;
        }
    }

    if (layer.costType == cYnLayerCostSmooth)
    {
        YnBlasGradientSmoothL1(netState.input, netState.truth, layer.delta, layer.batch * layer.inputs);
    }
    else
    {
        YnBlasArrayCopyValueSet(layer.delta, netState.truth, layer.batch * layer.inputs, 1, 1);
        YnBlasArrayAxpyValueSet(layer.delta, netState.input, layer.batch * layer.inputs, 1, 1, -1);
    }

    *(layer.output) = YnBlasArrayDotValueSet(layer.delta, layer.delta, layer.batch * layer.inputs, 1, 1);
}

void YnLayerCostBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    YnBlasArrayAxpyValueSet(netState.delta, layer.delta, layer.batch * layer.inputs, 1, 1, layer.scale);
}
