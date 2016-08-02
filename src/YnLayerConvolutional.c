//	File        :   YnLayerConvolutionalayer.c
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

/**************** Implement */
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

    for(i = 0; i < channel * num * size * size; i ++)
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
        for(i = 0; i < num; i ++)
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
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(layer.outputs*layer.batch, 0, layer.output, 1);

    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = out_h*out_w;

    float *a = layer.filters;
    float *b = layer.colImage;
    float *c = layer.output;

    for(i = 0; i < layer.batch; i ++)
    {
        YnImageImage2Col(netState.input, layer.c, layer.h, layer.w,
                layer.size, layer.stride, layer.pad, b);
        YnGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        c += n*m;
        state.input += layer.c*layer.h*layer.w;
    }

    if (layer.batch_normalize){
        if (state.train){
            mean_cpu(layer.output, layer.batch, layer.n, layer.out_h*layer.out_w, layer.mean);
            variance_cpu(layer.output, layer.mean, layer.batch, layer.n, layer.out_h*layer.out_w, layer.variance);
            normalize_cpu(layer.output, layer.mean, layer.variance, layer.batch, layer.n, layer.out_h*layer.out_w);
        } else {
            normalize_cpu(layer.output, layer.rolling_mean, layer.rolling_variance, layer.batch, layer.n, layer.out_h*layer.out_w);
        }
        scale_bias(layer.output, layer.scales, layer.batch, layer.n, out_h*out_w);
    }
    add_bias(layer.output, layer.biases, layer.batch, layer.n, out_h*out_w);

    activate_array(layer.output, m*n*layer.batch, layer.activation);
}

void YnLayerConvolutionalBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    int m = layer.n;
    int n = layer.size*layer.size*layer.c;
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array(layer.output, m*k*layer.batch, layer.activation, layer.delta);
    backward_bias(layer.bias_updates, layer.delta, layer.batch, layer.n, k);

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.delta + i*m*k;
        float *b = layer.col_image;
        float *c = layer.filter_updates;

        float *im = state.input+i*layer.c*layer.h*layer.w;

        im2col_cpu(im, layer.c, layer.h, layer.w,
                layer.size, layer.stride, layer.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if (state.delta){
            a = layer.filters;
            b = layer.delta + i*m*k;
            c = layer.col_image;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(layer.col_image, layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, layer.pad, state.delta+i*layer.c*layer.h*layer.w);
        }
    }
}

void YnLayerConvolutionalUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_cpu(layer.n, learning_rate/batch, layer.bias_updates, 1, layer.biases, 1);
    scal_cpu(layer.n, momentum, layer.bias_updates, 1);

    axpy_cpu(size, -decay*batch, layer.filters, 1, layer.filter_updates, 1);
    axpy_cpu(size, learning_rate/batch, layer.filter_updates, 1, layer.filters, 1);
    scal_cpu(size, momentum, layer.filter_updates, 1);
}

void YnLayerConvolutionalDenormalize(tYnLayer layer)
{
    int i, j;
    for(i = 0; i < layer.n; ++i){
        float scale = layer.scales[i]/sqrt(layer.rolling_variance[i] + .00001);
        for(j = 0; j < layer.c*layer.size*layer.size; ++j){
            layer.filters[i*layer.c*layer.size*layer.size + j] *= scale;
        }
        layer.biases[i] -= layer.rolling_mean[i] * scale;
    }
}

void YnLayerConvolutionalResize(tYnLayer layer,
        int width,
        int height)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->col_image = realloc(l->col_image,
            out_h*out_w*l->size*l->size*l->c*sizeof(float));
    l->output = realloc(l->output,
            l->batch*out_h * out_w * l->n*sizeof(float));
    l->delta  = realloc(l->delta,
            l->batch*out_h * out_w * l->n*sizeof(float));

#ifdef GPU
    cuda_free(l->col_imageGpu);
    cuda_free(l->deltaGpu);
    cuda_free(l->outputGpu);

    l->col_imageGpu = YnCudaMakeArray(l->col_image, out_h*out_w*l->size*l->size*l->c);
    l->deltaGpu =     YnCudaMakeArray(l->delta, l->batch*out_h*out_w*l->n);
    l->outputGpu =    YnCudaMakeArray(l->output, l->batch*out_h*out_w*l->n);
#endif
}

tYnImage * YnLayerConvolutionalVisualize(tYnLayer layer,
        char * window,
        tYnImage * filters)
{
    image *single_filters = get_filters(l);
    show_images(single_filters, layer.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_filters;
}

tYnImage YnLayerConvolutionalImageGet(tYnLayer layer)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = layer.n;
    return float_to_image(w,h,c,layer.output);
}

tYnImage YnLayerConvolutionalGradientGet(tYnLayer layer)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = layer.n;
    return float_to_image(w,h,c,layer.delta);
}

tYnImage YnLayerConvolutionalFilterGet(tYnLayer layer,
        int i)
{
    image *filters = calloc(layer.n, sizeof(image));
    int i;
    for(i = 0; i < layer.n; ++i){
        filters[i] = copy_image(get_convolutional_filter(l, i));
        //normalize_image(filters[i]);
    }
    return filters;
}

int YnLayerConvolutionalOutHeightGet(tYnLayer layer)
{
    int h = layer.h;
    if (!layer.pad) h -= layer.size;
    else h -= 1;
    return h/layer.stride + 1;
}

int YnLayerConvolutionalOutWidthGet(tYnLayer layer)
{
    int w = layer.w;
    if (!layer.pad) w -= layer.size;
    else w -= 1;
    return w/layer.stride + 1;
}

void YnLayerConvolutionalFiltersRescale(tYnLayer layer,
        float scale,
        float trans)
{
    int i;
    for(i = 0; i < layer.n; ++i){
        image im = get_convolutional_filter(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            layer.biases[i] += sum*trans;
        }
    }
}

int YnLayerConvolutionalFiltersRgbgr(tYnLayer layer)
{
    int i;
    for(i = 0; i < layer.n; ++i){
        image im = get_convolutional_filter(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

