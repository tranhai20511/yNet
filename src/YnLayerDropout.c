//	File        :   YnLayerDropout.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerDropout.h"
#include "../include/YnUtil.h"
#include "../include/YnCuda.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
tYnLayer YnLayerDropoutMake(int batchNum,
        int inputs,
        float probability)
{
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
    tYnLayer layer = {0};

    layer.type = cYnLayerDropout;
    layer.probability = probability;
    layer.inputs = inputs;
    layer.outputs = inputs;
    layer.batch = batchNum;
    layer.rand = calloc(inputs * batchNum, sizeof(float));
    layer.scale = 1. / (1.-probability);

#ifdef YN_GPU
    layer.randGpu = cuda_make_array(layer.rand, inputs *  batchNum);
#endif

    return layer;
}

void YnLayerDropoutForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    float r;

    if (!netState.train)
        return;

    for (i = 0; i < layer.batch * layer.inputs; i ++)
    {
        r = YnUtilRandomUniformNum(0, 1);
        layer.rand[i] = r;

        if (r < layer.probability)
            netState.input[i] = 0;
        else
            netState.input[i] *= layer.scale;
    }
}

void YnLayerDropoutBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    float r;

    if (!netState.delta)
        return;

    for (i = 0; i < layer.batch * layer.inputs; i ++)
    {
        r = layer.rand[i];

        if (r < layer.probability)
            netState.delta[i] = 0;
        else
            netState.delta[i] *= layer.scale;
    }
}

void YnLayerDropoutResize(tYnLayer layer,
        int inputs)
{
    layer->rand = realloc(layer->rand, layer->inputs * layer->batch * sizeof(float));

#ifdef YN_GPU
    YnCudaFreeArray(layer->randGpu);
    layer->randGpu = YnCudaMakeArray(layer->rand, inputs * layer->batch);
#endif
}
