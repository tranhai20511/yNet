//	File        :   YnLayerAvgpool.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   30-07-2016
//	Author      :   haittt

#include "../include/YnLayerAvgpool.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
tYnLayer YnLayerAvgpoolMake(int32 batchNum,
        int32 width,
        int32 height,
        int32 channel)
{
    tYnLayer layer = {0};
    int outputSize;

    fprintf(stderr, "Avgpool Layer: %d x %d x %d image\n", width, height, channel);

    layer.type = cYnLayerAvgpool;
    layer.batch = batchNum;
    layer.h = height;
    layer.w = width;
    layer.c = channel;
    layer.outW = 1;
    layer.outH = 1;
    layer.outC = channel;
    layer.outputs = layer.outC;
    layer.inputs = height * width * channel;

    outputSize = layer.outputs * batchNum;
    layer.output =  calloc(outputSize, sizeof(float));
    layer.delta =   calloc(outputSize, sizeof(float));

#ifdef YN_GPU
    layer.outputGpu  = YnCudaMakeArray(layer.output, outputSize);
    layer.deltaGpu   = YnCudaMakeArray(layer.delta, outputSize);
#endif

    return layer;
}

void YnLayerAvgpoolResize(tYnLayer * layer,
        int32 width,
        int32 height)
{
    layer->w = width;
    layer->h = height;
    layer->inputs = height * width * layer->c;
}

void YnLayerAvgpoolForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int b,i,k;
    int out_index;
    int in_index;

    for (b = 0; b < layer.batch; b ++)
    {
        for (k = 0; k < layer.c; k ++)
        {
            out_index = k + b * layer.c;
            layer.output[out_index] = 0;

            for (i = 0; i < layer.h * layer.w; i ++)
            {
                in_index = i + layer.h * layer.w * (k + b * layer.c);
                layer.output[out_index] += netState.input[in_index];
            }

            layer.output[out_index] /= layer.h * layer.w;
        }
    }
}

void YnLayerAvgpoolBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int b,i,k;
    int out_index;
    int in_index;

    for (b = 0; b < layer.batch; b ++)
    {
        for (k = 0; k < layer.c; k ++)
        {
            out_index = k + b * layer.c;

            for (i = 0; i < layer.h * layer.w; i ++)
            {
                in_index = i + layer.h * layer.w * (k + b * layer.c);
                netState.delta[in_index] += layer.delta[out_index] / (layer.h * layer.w);
            }
        }
    }
}
