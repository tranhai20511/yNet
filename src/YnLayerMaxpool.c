//	File        :   YnLayerMaxpoolayer.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerMaxpool.h"
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
tYnLayer YnLayerMaxpoolMake(int batchNum,
        int height,
        int width,
        int channel,
        int size,
        int stride)
{
    fprintf(stderr, "Maxpool Layer: %d x %d x %d image, %d size, %d stride\n", height, width, channel, size, stride);

    int output_size;
    tYnLayer layer = {0};

    layer.type = cYnLayerMaxpool;
    layer.batch = batchNum;
    layer.h = height;
    layer.w = width;
    layer.c = channel;
    layer.outW = (width - 1) / stride + 1;
    layer.outH = (height - 1) / stride + 1;
    layer.outC = channel;
    layer.outputs = layer.outH * layer.outW * layer.outC;
    layer.inputs = height * width * channel;
    layer.size = size;
    layer.stride = stride;
    output_size = layer.outH * layer.outW * layer.outC * batchNum;
    layer.indexes = calloc(output_size, sizeof(int));
    layer.output =  calloc(output_size, sizeof(float));
    layer.delta =   calloc(output_size, sizeof(float));

#ifdef YN_GPU
    layer.indexesGpu = YnCudaMakeIntArray(output_size);
    layer.outputGpu  = YnCudaMakeArray(layer.output, output_size);
    layer.deltaGpu   = YnCudaMakeArray(layer.delta, output_size);
#endif

    return layer;
}

void YnLayerMaxpoolForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int b, i, j, k, m, n;
    int out_index;
    float max;
    int max_i;
    int cur_h;
    int cur_w;
    int index;
    int valid;
    float val;

    int w_offset = (- layer.size-1) / 2 + 1;
    int h_offset = (- layer.size-1) / 2 + 1;

    int h = (layer.h - 1) / layer.stride + 1;
    int w = (layer.w - 1) / layer.stride + 1;
    int c = layer.c;

    for (b = 0; b < layer.batch; b ++)
    {
        for (k = 0; k < c; k ++)
        {
            for (i = 0; i < h; i ++)
            {
                for (j = 0; j < w; j ++)
                {
                    out_index = j + w * (i + h * (k + c * b));
                    max = - FLT_MAX;
                    max_i = -1;

                    for (n = 0; n < layer.size; n ++)
                    {
                        for (m = 0; m < layer.size; m ++)
                        {
                            cur_h = h_offset + i * layer.stride + n;
                            cur_w = w_offset + j * layer.stride + m;
                            index = cur_w + layer.w * (cur_h + layer.h * (k + b * layer.c));

                            valid = ((cur_h >= 0) && (cur_h < layer.h) && (cur_w >= 0) && (cur_w < layer.w));
                            val = (valid != 0) ? netState.input[index] : (- FLT_MAX);

                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }

                    layer.output[out_index] = max;
                    layer.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void YnLayerMaxpoolBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    int index;
    int h = (layer.h - 1) / layer.stride + 1;
    int w = (layer.w - 1) / layer.stride + 1;
    int c = layer.c;

    for (i = 0; i < h * w * c * layer.batch; i ++)
    {
        index = layer.indexes[i];
        netState.delta[index] += layer.delta[i];
    }
}

void YnLayerMaxpoolResize(tYnLayer * layer,
        int width,
        int height)
{
    int output_size;
    int stride = layer->stride;
    layer->h = height;
    layer->w = width;
    layer->inputs = height * width * layer->c;

    layer->outW = (width - 1) / stride + 1;
    layer->outH = (height - 1) / stride + 1;
    layer->outputs = layer->outW * layer->outH * layer->c;
    output_size = layer->outputs * layer->batch;

    layer->indexes = realloc(layer->indexes, output_size * sizeof(int));
    layer->output = realloc(layer->output, output_size * sizeof(float));
    layer->delta = realloc(layer->delta, output_size * sizeof(float));

#ifdef YN_GPU

    YnCudaFreeArray((float *)layer->indexesGpu);
    YnCudaFreeArray(layer->outputGpu);
    YnCudaFreeArray(layer->deltaGpu);
    layer->indexesGpu = YnCudaMakeIntArray(output_size);
    layer->outputGpu  = YnCudaMakeArray(layer->output, output_size);
    layer->deltaGpu   = YnCudaMakeArray(layer->delta,  output_size);

#endif
}

tYnImage YnLayerMaxpoolImageGet(tYnLayer layer)
{
    int h = layer.outH;
    int w = layer.outW;
    int c = layer.c;

    return YnImageFloatToImage(w, h, c, layer.output);
}

tYnImage YnLayerMaxpoolGradientGet(tYnLayer layer)
{
    int h = layer.outH;
    int w = layer.outW;
    int c = layer.c;

    return YnImageFloatToImage(w, h, c, layer.delta);
}
