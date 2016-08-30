//	File        :   YnLayerConvolutional.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   02-08-2016
//	Author      :   haittt

#include "../include/YnLayerConvolutional.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
YN_STATIC
tYnImage * _YnLayerConvolutionalFiltersGet(tYnLayer layer)
YN_ALSWAY_INLINE;

/**************** Implement */
YN_STATIC
tYnImage * _YnLayerConvolutionalFiltersGet(tYnLayer layer)
{
    tYnImage *filters = calloc(layer.n, sizeof(tYnImage));
    int i;

    for (i = 0; i < layer.n; i ++)
    {
        filters[i] = YnImageCopy(YnLayerConvolutionalFilterGet(layer, i));
    }

    return filters;
}

tYnLayer YnLayerConvolutionalMake(int batchNum,
        int height,
        int width,
        int channel,
        int num,
        int size,
        int stride,
        int pad,
        eYnActivationType activation,
        int batchNormalize,
        int binary)
{
    int i;
    int outH;
    int outW;
    float scale;
    tYnLayer layer = {0};
    layer.type = cYnLayerConvolutional;

    layer.h = height;
    layer.w = width;
    layer.c = channel;
    layer.n = num;
    layer.binary = binary;
    layer.batch = batchNum;
    layer.stride = stride;
    layer.size = size;
    layer.pad = pad;
    layer.batchNormalize = batchNormalize;

    layer.filters = calloc(channel * num * size * size, sizeof(float));
    layer.filterUpdates = calloc(channel * num * size * size, sizeof(float));

    layer.biases = calloc(num, sizeof(float));
    layer.biasUpdates = calloc(num, sizeof(float));

    scale = sqrt(2./(size * size * channel));

    for (i = 0; i < channel * num * size * size; i ++)
        layer.filters[i] = scale*rand_uniform(-1, 1);

    outH = YnLayerConvolutionalOutHeightGet(layer);
    outW = YnLayerConvolutionalOutWidthGet(layer);
    layer.outH = outH;
    layer.outW = outW;
    layer.outC = num;
    layer.outputs = layer.outH * layer.outW * layer.outC;
    layer.inputs = layer.w * layer.h * layer.c;

    layer.colImage = calloc(outH * outW * size * size * channel, sizeof(float));
    layer.output = calloc(layer.batch * outH * outW * num, sizeof(float));
    layer.delta  = calloc(layer.batch * outH * outW * num, sizeof(float));

    if (binary)
    {
        layer.binaryFilters = calloc(channel * num * size * size, sizeof(float));
    }

    if (batchNormalize)
    {
        layer.scales = calloc(num, sizeof(float));
        layer.scaleUpdates = calloc(num, sizeof(float));
        for (i = 0; i < num; i ++)
        {
            layer.scales[i] = 1;
        }

        layer.mean = calloc(num, sizeof(float));
        layer.variance = calloc(num, sizeof(float));

        layer.rollingMean = calloc(num, sizeof(float));
        layer.rollingVariance = calloc(num, sizeof(float));
    }

#ifdef YN_GPU
    layer.filtersGpu = YnCudaMakeArray(layer.filters, channel * num * size * size);
    layer.filterUpdatesGpu = YnCudaMakeArray(layer.filterUpdates, channel * num * size * size);

    layer.biasesGpu = YnCudaMakeArray(layer.biases, num);
    layer.biasUpdatesGpu = YnCudaMakeArray(layer.biasUpdates, num);

    layer.scalesGpu = YnCudaMakeArray(layer.scales, num);
    layer.scaleUpdatesGpu = YnCudaMakeArray(layer.scaleUpdates, num);

    layer.colImageGpu = YnCudaMakeArray(layer.colImage, outH * outW * size * size * channel);
    layer.deltaGpu = YnCudaMakeArray(layer.delta, layer.batch* outH * outW * num);
    layer.outputGpu = YnCudaMakeArray(layer.output, layer.batch * outH * outW * num);

    if (binary)
    {
        layer.binaryFiltersGpu = YnCudaMakeArray(layer.filters, channel * num * size * size);
    }

    if (batchNormalize)
    {
        layer.meanGpu = YnCudaMakeArray(layer.mean, num);
        layer.varianceGpu = YnCudaMakeArray(layer.variance, num);

        layer.rollingMeanGpu = YnCudaMakeArray(layer.mean, num);
        layer.rollingVarianceGpu = YnCudaMakeArray(layer.variance, num);

        layer.meanDeltaGpu = YnCudaMakeArray(layer.mean, num);
        layer.varianceDeltaGpu = YnCudaMakeArray(layer.variance, num);

        layer.xGpu = YnCudaMakeArray(layer.output, layer.batch * outH * outW * num);
        layer.xNormGpu = YnCudaMakeArray(layer.output, layer.batch * outH * outW * num);
    }
#endif

    layer.activation = activation;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", height, width, channel, num, outH, outW, num);

    return layer;
}

void YnLayerConvolutionalForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int outH = YnLayerConvolutionalOutHeightGet(layer);
    int outW = YnLayerConvolutionalOutWidthGet(layer);
    int i;

    fill_cpu(layer.outputs*layer.batch, 0, layer.output, 1);

    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = outH*outW;

    float *a = layer.filters;
    float *b = layer.colImage;
    float *c = layer.output;

    for (i = 0; i < layer.batch; i ++)
    {
        YnImageImage2Col(netState.input, layer.c, layer.h, layer.w,
                layer.size, layer.stride, layer.pad, b);
        YnGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);

        c += n * m;
        netState.input += layer.c*layer.h*layer.w;
    }

    if (layer.batchNormalize)
    {
        if (netState.train)
        {
            YnBlasArrayMeanCal(layer.output, layer.batch, layer.n, layer.outH*layer.outW, layer.mean);
            YnBlasArrayVarianceCal(layer.output, layer.mean, layer.batch, layer.n, layer.outH*layer.outW, layer.variance);
            YnBlasArrayNormalizeCal(layer.output, layer.mean, layer.variance, layer.batch, layer.n, layer.outH * layer.outW);
        }
        else
        {
            YnBlasArrayNormalizeCal(layer.output, layer.rollingMean, layer.rollingVariance, layer.batch, layer.n, layer.outH*layer.outW);
        }

        YnBlasArrayBiasScale(layer.output, layer.scales, layer.batch, layer.n, outH*outW);
    }

    YnBlasArrayBiasAdd(layer.output, layer.biases, layer.batch, layer.n, outH*outW);

    YnActivationOutputArrayCal(layer.output, m * n * layer.batch, layer.activation);
}

void YnLayerConvolutionalBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    float *a;
    float *b;
    float *c;
    float *im;
    int m = layer.n;
    int n = layer.size * layer.size * layer.c;
    int k = YnLayerConvolutionalOutHeightGet(layer) * YnLayerConvolutionalOutWidthGet(layer);

    YnActivationGradientArrayCal(layer.output, m * k * layer.batch, layer.activation, layer.delta);
    YnBlasArrayBackwardBias(layer.biasUpdates, layer.delta, layer.batch, layer.n, k);

    for (i = 0; i < layer.batch; i ++)
    {
        a = layer.delta + i * m * k;
        b = layer.colImage;
        c = layer.filterUpdates;

        im = netState.input + i * layer.c * layer.h * layer.w;

        YnImageImage2Col(im, layer.c, layer.h, layer.w, layer.size, layer.stride, layer.pad, b);
        YnGemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

        if (netState.delta)
        {
            a = layer.filters;
            b = layer.delta + i*m*k;
            c = layer.colImage;

            YnGemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            YnImageCol2Image(layer.colImage, layer.c,  layer.h,  layer.w,  layer.size,
                    layer.stride, layer.pad, netState.delta + i * layer.c * layer.h * layer.w);
        }
    }
}

void YnLayerConvolutionalUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
{
    int size = layer.size * layer.size * layer.c * layer.n;

    YnBlasArrayAxpyValueSet(layer.biases, layer.biasUpdates, layer.n, 1, 1, learningRate / batch);
    YnBlasArrayScaleValueSet(layer.biasUpdates, layer.n, 1, momentum);

    YnBlasArrayAxpyValueSet(layer.filterUpdates, layer.filters, size, 1, 1, -decay * batch);
    YnBlasArrayAxpyValueSet(layer.filters, layer.filterUpdates, size, 1, 1, learningRate / batch);
    YnBlasArrayScaleValueSet(layer.filterUpdates, size, 1, momentum);
}

void YnLayerConvolutionalDenormalize(tYnLayer layer)
{
    float scale;
    int i, j;

    for (i = 0; i < layer.n; i ++)
    {
        scale = layer.scales[i] / sqrt(layer.rollingVariance[i] + .00001);

        for (j = 0; j < (layer.c * layer.size * layer.size); j ++)
        {
            layer.filters[(i * layer.c * layer.size * layer.size) + j] *= scale;
        }

        layer.biases[i] -= layer.rollingMean[i] * scale;
    }
}

void YnLayerConvolutionalResize(tYnLayer* layer,
        int width,
        int height)
{
    int outW;
    int outH;

    layer->w = width;
    layer->h = height;

    outW = YnLayerConvolutionalOutWidthGet(*layer);
    outH = YnLayerConvolutionalOutHeightGet(*layer);

    layer->outW = outW;
    layer->outH = outH;

    layer->outputs = layer->outH * layer->outW * layer->outC;
    layer->inputs = layer->w * layer->h * layer->c;

    layer->colImage = realloc(layer->colImage, outH * outW * layer->size * layer->size * layer->c * sizeof(float));
    layer->output = realloc(layer->output, layer->batch * outH * outW * layer->n * sizeof(float));
    layer->delta  = realloc(layer->delta, layer->batch * outH * outW * layer->n * sizeof(float));

#ifdef YN_GPU
    YnCudaFreeArray(layer->colImageGpu);
    YnCudaFreeArray(layer->deltaGpu);
    YnCudaFreeArray(layer->outputGpu);

    layer->colImageGpu = YnCudaMakeArray(layer->colImage, outH * outW * layer->size * layer->size * layer->c);
    layer->deltaGpu    = YnCudaMakeArray(layer->delta, layer->batch * outH * outW * layer->n);
    layer->outputGpu   = YnCudaMakeArray(layer->output, layer->batch * outH * outW * layer->n);
#endif
}

tYnImage * YnLayerConvolutionalVisualize(tYnLayer layer,
        char * window,
        tYnImage * filters)
{
    char buff[256];
    tYnImage delta;
    tYnImage dc;
    tYnImage *singleFilters = _YnLayerConvolutionalFiltersGet(layer);

    YnImageImagesShow(singleFilters, layer.n, window);

    delta = YnLayerConvolutionalImageGet(layer);
    dc = YnImageCollapseLayers(delta, 1);

    sprintf(buff, "%s: Output", window);

    YnImageFree(dc);

    return singleFilters;
}

tYnImage YnLayerConvolutionalImageGet(tYnLayer layer)
{
    int h, w, c;

    h = YnLayerConvolutionalOutHeightGet(layer);
    w = YnLayerConvolutionalOutWidthGet(layer);
    c = layer.n;

    return YnImageFloatToImage(w, h, c, layer.output);
}

tYnImage YnLayerConvolutionalGradientGet(tYnLayer layer)
{
    int h, w, c;

    h = YnLayerConvolutionalOutHeightGet(layer);
    w = YnLayerConvolutionalOutWidthGet(layer);
    c = layer.n;

    return YnImageFloatToImage(w,h,c,layer.delta);
}

tYnImage YnLayerConvolutionalFilterGet(tYnLayer layer,
        int i)
{
    int h = layer.size;
    int w = layer.size;
    int c = layer.c;

    return YnImageFloatToImage(w, h, c, layer.filters + i * h * w * c);
}

int YnLayerConvolutionalOutHeightGet(tYnLayer layer)
{
    int h = layer.h;

    if (!layer.pad)
        h -= layer.size;
    else
        h -= 1;

    return h / layer.stride + 1;
}

int YnLayerConvolutionalOutWidthGet(tYnLayer layer)
{
    int w = layer.w;

    if (!layer.pad)
        w -= layer.size;
    else
        w -= 1;

    return w / layer.stride + 1;
}

void YnLayerConvolutionalFiltersRescale(tYnLayer layer,
        float scale,
        float trans)
{
    int i;
    float sum;
    tYnImage im;

    for (i = 0; i < layer.n; i ++)
    {
        im = YnLayerConvolutionalFilterGet(layer, i);

        if (im.channel == 3)
        {
            YnImgaeScale(im, scale);
            sum = YnUtilArraySum(im.data, im.width * im.height * im.channel);
            layer.biases[i] += sum * trans;
        }
    }
}

int YnLayerConvolutionalFiltersRgbgr(tYnLayer layer)
{
    int i;
    tYnImage im;

    for (i = 0; i < layer.n; i ++)
    {
        im = YnLayerConvolutionalFilterGet(layer, i);

        if (im.channel == 3)
        {
            YnImageRgbgr(im);
        }
    }
}

