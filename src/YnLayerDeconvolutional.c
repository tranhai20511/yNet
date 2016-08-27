//	File        :   YnLayerDeconvolutionalayer.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   21-08-2016
//	Author      :   haittt

#include "../include/YnLayerDeconvolutional.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
tYnLayer YnLayerDeconvolutionalMake(int batchNum,
        int height,
        int width,
        int channel,
        int num,
        int size,
        int stride,
        int pad,
        eYnActivationType activation)
{
    int i;
    float scale;

    int out_h;
    int out_w;
    tYnLayer layer = {0};
    layer.type = cYnLayerDeconvolutional;

    layer.h = height;
    layer.w = width;
    layer.c = channel;
    layer.n = num;
    layer.batch = batchNum;
    layer.stride = stride;
    layer.size = size;

    layer.filters = calloc(channel * num * size * size, sizeof(float));
    layer.filterUpdates = calloc(channel * num * size * size, sizeof(float));

    layer.biases = calloc(num, sizeof(float));
    layer.biasUpdates = calloc(num, sizeof(float));

    scale = 1. / sqrt(size * size * channel);

    for (i = 0; i < channel * num * size * size; i ++)
        layer.filters[i] = scale * rand_normal();

    for (i = 0; i < num; i ++)
    {
        layer.biases[i] = scale;
    }

    out_h = deconvolutional_out_height(layer);
    out_w = deconvolutional_out_width(layer);

    layer.outH = out_h;
    layer.outW = out_w;
    layer.outC = num;
    layer.outputs = layer.outW * layer.outH * layer.outC;
    layer.inputs = layer.w * layer.h * layer.c;

    layer.colImage = calloc(height * width * size * size * num, sizeof(float));
    layer.output = calloc(layer.batch*out_h * out_w * num, sizeof(float));
    layer.delta  = calloc(layer.batch*out_h * out_w * num, sizeof(float));

#ifdef YN_GPU
    layer.filtersGpu = YnCudaMakeArray(layer.filters, channel * num * size * size);
    layer.filterUpdatesGpu = YnCudaMakeArray(layer.filterUpdates, channel * num * size * size);

    layer.biasesGpu = YnCudaMakeArray(layer.biases, num);
    layer.biasUpdatesGpu = YnCudaMakeArray(layer.biasUpdates, num);

    layer.colImageGpu = YnCudaMakeArray(layer.colImage, height * width * size * size * num);
    layer.deltaGpu = YnCudaMakeArray(layer.delta, layer.batch * out_h * out_w * num);
    layer.outputGpu = YnCudaMakeArray(layer.output, layer.batch * out_h * out_w * num);
#endif

    layer.activation = activation;

    fprintf(stderr, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", height ,width ,channel ,num, out_h, out_w, num);

    return layer;
}

void YnLayerDeconvolutionalForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    float *a;
    float *b;
    float *c;
    int out_h = YnLayerDeconvolutionalOutHeightGet(layer);
    int out_w = YnLayerDeconvolutionalOutWidthGet(layer);
    int size = out_h * out_w;

    int m = layer.size * layer.size * layer.n;
    int n = layer.h * layer.w;
    int k = layer.c;

    YnBlasArrayFillValueSet(layer.output, layer.outputs * layer.batch, 1, 0);

    for (i = 0; i < layer.batch; i ++)
    {
        *a = layer.filters;
        *b = netState.input + i * layer.c * layer.h * layer.w;
        *c = layer.colImage;

        YnGemm(1,0,m,n,k,1,a,m,b,n,0,c,n);

        YnImageCol2Image(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.output + i * layer.n * size);
    }

    YnBlasArrayBiasAdd(layer.output, layer.biases, layer.batch, layer.n, size);
    YnActivationOutputArrayCal(layer.output, layer.batch * layer.n * size, layer.activation);
}

void YnLayerDeconvolutionalBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int m;
    int n;
    int k;
    float *a;
    float *b;
    float *c;
    float alpha = 1. / layer.batch;
    int out_h = YnLayerDeconvolutionalOutHeightGet(layer);
    int out_w = YnLayerDeconvolutionalOutWidthGet(layer);
    int size = out_h * out_w;
    int i;

    YnActivationGradientArrayCal(layer.output, size * layer.n * layer.batch, layer.activation, layer.delta);
    YnBlasArrayBackwardBias(layer.biasUpdates, layer.delta, layer.batch, layer.n, size);

    for (i = 0; i < layer.batch; i ++)
    {
        m = layer.c;
        n = layer.size*layer.size*layer.n;
        k = layer.h*layer.w;

        a = netState.input + i*m*n;
        b = layer.colImage;
        c = layer.filterUpdates;

        YnImageImage2Col(layer.delta + i * layer.n * size, layer.n, out_h, out_w,
                layer.size, layer.stride, 0, b);

        YnGemm(0, 1, m, n, k, alpha, a, k, b, k, 1, c, n);

        if (netState.delta)
        {
            m = layer.c;
            n = layer.h * layer.w;
            k = layer.size * layer.size * layer.n;

            a = layer.filters;
            b = layer.colImage;
            c = netState.delta + i * n * m;

            YnGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
    }
}

void YnLayerDeconvolutionalUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
{
    int size = layer.size * layer.size * layer.c * layer.n;

    YnBlasArrayAxpyValueSet(layer.biases, layer.biasUpdates, layer.n, 1, 1, learningRate);
    YnBlasArrayScaleValueSet(layer.filterUpdates, layer.n, 1, momentum);

    YnBlasArrayAxpyValueSet(layer.filterUpdates, layer.filters, size, 1, 1, - decay);
    YnBlasArrayAxpyValueSet(layer.filters, layer.filterUpdates, size, 1, 1, learningRate);
    YnBlasArrayScaleValueSet(layer.filterUpdates, size, 1, momentum);
}

void YnLayerDeconvolutionalResize(tYnLayer* layer,
        int width,
        int height)
{
    layer->h = height;
    layer->w = width;
    int out_h = YnLayerDeconvolutionalOutHeightGet(*layer);
    int out_w = YnLayerDeconvolutionalOutWidthGet(*layer);

    layer->colImage = realloc(layer->colImage, out_h * out_w * layer->size * layer->size * layer->c * sizeof(float));
    layer->output = realloc(layer->output, layer->batch * out_h * out_w * layer->n * sizeof(float));
    layer->delta  = realloc(layer->delta, layer->batch * out_h * out_w * layer->n * sizeof(float));

#ifdef YN_GPU
    YnCudaFreeArray(layer->colImageGpu);
    YnCudaFreeArray(layer->deltaGpu);
    YnCudaFreeArray(layer->outputGpu);

    layer->colImageGpu = YnCudaMakeArray(layer->colImage, out_h * out_w * layer->size * layer->size * layer->c);
    layer->deltaGpu = YnCudaMakeArray(layer->delta, layer->batch * out_h * out_w * layer->n);
    layer->outputGpu = YnCudaMakeArray(layer->output, layer->batch * out_h * out_w * layer->n);
#endif
}

tYnImage YnLayerDeconvolutionalImageGet(tYnLayer layer)
{
    int h, w, c;

    h = YnLayerDeconvolutionalOutHeightGet(layer);
    w = YnLayerDeconvolutionalOutWidthGet(layer);
    c = layer.n;

    return YnImageFloatToImage(w, h, c, layer.output);
}

tYnImage YnLayerDeconvolutionalGradientGet(tYnLayer layer)
{
    int h, w, c;

    h = YnLayerDeconvolutionalOutHeightGet(layer);
    w = YnLayerDeconvolutionalOutWidthGet(layer);
    c = layer.n;

    return YnImageFloatToImage(w, h, c, layer.delta);
}

int YnLayerDeconvolutionalOutHeightGet(tYnLayer layer)
{
    int h = layer.stride * (layer.h - 1) + layer.size;
    return h;
}

int YnLayerDeconvolutionalOutWidthGet(tYnLayer layer)
{
    int w = layer.stride * (layer.w - 1) + layer.size;
    return w;
}
