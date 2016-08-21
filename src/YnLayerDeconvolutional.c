//	File        :   YnLayerDeconvolutionalayer.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   21-08-2016
//	Author      :   haittt

#include "../include/YnLayerDeconvolutionalayer.h"

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
    tYnLayer layer = {0};
    layer.type = DECONVOLUTIONAL;

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
    for(i = 0; i < channel * num * size * size; i ++)
        layer.filters[i] = scale * rand_normal();

    for(i = 0; i < n; ++i){
        layer.biases[i] = scale;
    }
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);

    layer.out_h = out_h;
    layer.out_w = out_w;
    layer.out_c = n;
    layer.outputs = layer.out_w * layer.out_h * layer.out_c;
    layer.inputs = layer.w * layer.h * layer.c;

    layer.col_image = calloc(h*w*size*size*n, sizeof(float));
    layer.output = calloc(layer.batch*out_h * out_w * n, sizeof(float));
    layer.delta  = calloc(layer.batch*out_h * out_w * n, sizeof(float));

    #ifdef GPU
    layer.filters_gpu = cuda_make_array(layer.filters, c*n*size*size);
    layer.filter_updates_gpu = cuda_make_array(layer.filter_updates, c*n*size*size);

    layer.biases_gpu = cuda_make_array(layer.biases, n);
    layer.bias_updates_gpu = cuda_make_array(layer.bias_updates, n);

    layer.col_image_gpu = cuda_make_array(layer.col_image, h*w*size*size*n);
    layer.delta_gpu = cuda_make_array(layer.delta, layer.batch*out_h*out_w*n);
    layer.output_gpu = cuda_make_array(layer.output, layer.batch*out_h*out_w*n);
    #endif

    layer.activation = activation;

    fprintf(stderr, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

void YnLayerDeconvolutionalForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int outH = YnLayerDeconvolutionalOutHeightGet(layer);
    int outW = YnLayerDeconvolutionalOutWidthGet(layer);
    int i;

    fill_cpu(layer.outputs*layer.batch, 0, layer.output, 1);

    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = outH*outW;

    float *a = layer.filters;
    float *b = layer.colImage;
    float *c = layer.output;

    for(i = 0; i < layer.batch; i ++)
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

void YnLayerDeconvolutionalBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    float *a;
    float *b;
    float *c;
    float *im;
    int m = layer.n;
    int n = layer.size * layer.size * layer.c;
    int k = YnLayerDeconvolutionalOutHeightGet(layer) * YnLayerDeconvolutionalOutWidthGet(layer);

    YnActivationGradientArrayCal(layer.output, m * k * layer.batch, layer.activation, layer.delta);
    YnBlasArrayBackwardBias(layer.biasUpdates, layer.delta, layer.batch, layer.n, k);

    for(i = 0; i < layer.batch; i ++)
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

void YnLayerDeconvolutionalUpdate(tYnLayer layer,
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

void YnLayerDeconvolutionalDenormalize(tYnLayer layer)
{
    float scale;
    int i, j;

    for(i = 0; i < layer.n; i ++)
    {
        scale = layer.scales[i] / sqrt(layer.rollingVariance[i] + .00001);

        for(j = 0; j < (layer.c * layer.size * layer.size); j ++)
        {
            layer.filters[(i * layer.c * layer.size * layer.size) + j] *= scale;
        }

        layer.biases[i] -= layer.rollingMean[i] * scale;
    }
}

void YnLayerDeconvolutionalResize(tYnLayer* layer,
        int width,
        int height)
{
    int outW;
    int outH;

    layer->w = width;
    layer->h = height;

    outW = YnLayerDeconvolutionalOutWidthGet(*layer);
    outH = YnLayerDeconvolutionalOutHeightGet(*layer);

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

tYnImage * YnLayerDeconvolutionalVisualize(tYnLayer layer,
        char * window,
        tYnImage * filters)
{
    char buff[256];
    tYnImage delta;
    tYnImage dc;
    tYnImage *singleFilters = _YnLayerDeconvolutionalFiltersGet(layer);

    YnImageImagesShow(singleFilters, layer.n, window);

    delta = YnLayerDeconvolutionalImageGet(layer);
    dc = YnImageCollapseLayers(delta, 1);

    sprintf(buff, "%s: Output", window);

    YnImageFree(dc);

    return singleFilters;
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

    return YnImageFloatToImage(w,h,c,layer.delta);
}

tYnImage YnLayerDeconvolutionalFilterGet(tYnLayer layer,
        int i)
{
    int h = layer.size;
    int w = layer.size;
    int c = layer.c;

    return YnImageFloatToImage(w, h, c, layer.filters + i * h * w * c);
}

int YnLayerDeconvolutionalOutHeightGet(tYnLayer layer)
{
    int h = layer.h;

    if (!layer.pad)
        h -= layer.size;
    else
        h -= 1;

    return h / layer.stride + 1;
}

int YnLayerDeconvolutionalOutWidthGet(tYnLayer layer)
{
    int w = layer.w;

    if (!layer.pad)
        w -= layer.size;
    else
        w -= 1;

    return w / layer.stride + 1;
}

void YnLayerDeconvolutionalFiltersRescale(tYnLayer layer,
        float scale,
        float trans)
{
    int i;
    float sum;
    tYnImage im;

    for(i = 0; i < layer.n; i ++)
    {
        im = YnLayerDeconvolutionalFilterGet(layer, i);

        if (im.channel == 3)
        {
            YnImgaeScale(im, scale);
            sum = YnUtilArraySum(im.data, im.width * im.height * im.channel);
            layer.biases[i] += sum * trans;
        }
    }
}

int YnLayerDeconvolutionalFiltersRgbgr(tYnLayer layer)
{
    int i;
    tYnImage im;

    for(i = 0; i < layer.n; i ++)
    {
        im = YnLayerDeconvolutionalFilterGet(layer, i);

        if (im.channel == 3)
        {
            YnImageRgbgr(im);
        }
    }
}

