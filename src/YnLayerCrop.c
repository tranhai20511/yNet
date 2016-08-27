//	File        :   YnLayerCrop.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   15-08-2016
//	Author      :   haittt

#include "../include/YnLayerCrop.h"
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
tYnImage YnLayerCropMake(int32 batchNum,
        int32 height,
        int32 width,
        int32 channel,
        int32 cropHeight,
        int32 cropWidth,
        int32 flip,
        float angle,
        float saturation,
        float exposure)
{
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n",
            height, width, cropHeight, cropWidth, channel);

    tYnLayer layer = {0};

    layer.type = cYnLayerCrop;
    layer.batch = batchNum;
    layer.h = height;
    layer.w = width;
    layer.c = channel;
    layer.scale = (float)cropHeight / height;
    layer.flip = flip;
    layer.angle = angle;
    layer.saturation = saturation;
    layer.exposure = exposure;
    layer.outW = cropWidth;
    layer.outH = cropHeight;
    layer.outC = channel;
    layer.inputs =layer.w *layer.h *layer.c;
    layer.outputs =layer.outW *layer.outH *layer.outC;
    layer.output = calloc(layer.outputs * batchNum, sizeof(float));

#ifdef YN_GPU
    layer.outputGpu = YnCudaMakeArray(layer.output, layer.outputs * batchNum);
    layer.randGpu   = YnCudaMakeArray(0,layer.batch*8);
#endif

    return layer;
}

void YnLayerCropForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i, j, c, b, row, col;
    int index;
    int count = 0;
    int flip = (layer.flip && rand() % 2);
    int dh = rand() % (layer.h -layer.outH + 1);
    int dw = rand() % (layer.w -layer.outW + 1);
    float scale = 2;
    float trans = -1;

    if (layer.noadjust)
    {
        scale = 1;
        trans = 0;
    }

    if (!netState.train)
    {
        flip = 0;
        dh = (layer.h - layer.outH)/2;
        dw = (layer.w - layer.outW)/2;
    }

    for (b = 0; b < layer.batch; b ++)
    {
        for (c = 0; c < layer.c; c ++)
        {
            for (i = 0; i < layer.outH; i ++)
            {
                for (j = 0; j < layer.outW; j ++)
                {
                    if (flip)
                    {
                        col = layer.w - dw - j - 1;
                    }
                    else
                    {
                        col = j + dw;
                    }

                    row = i + dh;
                    index = col + layer.w * (row + layer.h * (c + layer.c * b));
                    layer.output[count++] = netState.input[index] * scale + trans;
                }
            }
        }
    }
}

tYnImage YnLayerCropImageGet(tYnLayer layer)
{
    int h = layer.outH;
    int w = layer.outW;
    int c = layer.outC;
    return YnImageFloatToImage(w, h, c, layer.output);
}

void YnLayerCropResize(tYnLayer layer,
        int32 width,
        int32 height)
{
   layer->w = width;
   layer->h = height;

   layer->outW = layer->scale * width;
   layer->outH = layer->scale * height;

   layer->inputs =layer->w * layer->h * layer->c;
   layer->outputs =layer->outH * layer->outW * layer->outC;

   layer->output = realloc(layer->output,layer->batch * layer->outputs * sizeof(float));

#ifdef YN_GPU
   YnCudaFreeArray(layer->outputGpu);
   layer->outputGpu = YnCudaMakeArray(layer->output, layer->outputs * layer->batch);
#endif

}
