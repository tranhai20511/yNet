//	File        :   YnLayerCropGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   15-08-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "../include/YnLayerCropGpu.h"
#include "../include/YnCudaGpu.h"
#include "../include/YnBlasGpu.h"
#include "../include/YnImageGpu.h"
}

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_GPU_DEVICE float _YnLayerCropGpuPixelGet(float *image,
        int w,
        int h,
        int x,
        int y,
        int c)
{
    if (x < 0 || x >= w || y < 0 || y >= h)
        return 0;

    return image[x + w * (y + c * h)];
}

YN_GPU_DEVICE float3 _YnLayerCropGpuRgbToHsv(float3 rgb)
{
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;

    float h, s, v;
    float max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    float min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    float delta = max - min;

    v = max;
    if (max == 0)
    {
        s = 0;
        h = -1;
    }
    else
    {
        s = delta/max;

        if (r == max)
        {
            h = (g - b) / delta;
        }
        else if (g == max)
        {
            h = 2 + (b - r) / delta;
        }
        else
        {
            h = 4 + (r - g) / delta;
        }

        if (h < 0)
            h += 6;
    }

    return make_float3(h, s, v);
}

YN_GPU_DEVICE float3 _YnLayerCropGpuHsvToRgb(float3 hsv)
{
    float h = hsv.x;
    float s = hsv.y;
    float v = hsv.z;
    int index;
    float r, g, b;
    float f, p, q, t;

    if (s == 0)
    {
        r = g = b = v;
    }
    else
    {
        index = (int) floorf(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));

        if (index == 0)
        {
            r = v;
            g = t;
            b = p;
        }
        else if (index == 1)
        {
            r = q;
            g = v;
            b = p;
        }
        else if (index == 2)
        {
            r = p;
            g = v;
            b = t;
        }
        else if (index == 3)
        {
            r = p;
            g = q;
            b = v;
        }
        else if (index == 4)
        {
            r = t;
            g = p;
            b = v;
        }
        else
        {
            r = v;
            g = p;
            b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);

    return make_float3(r, g, b);
}

YN_GPU_DEVICE float _YnLayerCropGpuBilinearInterpolate(float *image,
        int w,
        int h,
        float x,
        float y,
        int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * _YnLayerCropGpuPixelGet(image, w, h, ix,     iy,     c) +
                dy     * (1-dx) * _YnLayerCropGpuPixelGet(image, w, h, ix,     iy + 1, c) +
                (1-dy) *   dx   * _YnLayerCropGpuPixelGet(image, w, h, ix + 1, iy,     c) +
                dy     *   dx   * _YnLayerCropGpuPixelGet(image, w, h, ix + 1, iy+1,   c);

    return val;
}

YN_GPU_GLOBAL void  _YnLayerCropGpuLevelsImage(float *image,
        float *rand,
        int batch,
        int w,
        int h,
        int train,
        float saturation,
        float exposure,
        float translate,
        float scale,
        float shift)
{
    uint32 offset;
    int x, y;
    float rshift;
    float gshift;
    float bshift;
    float r0;
    float r1;
    float r2;
    float r3;
    float3 rgb;
    float3 hsv;
    float r;
    float g;
    float b;
    int size = batch * w * h;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size)
        return;

    x = id % w;
    id /= w;
    y = id % h;
    id /= h;
    rshift = rand[0];
    gshift = rand[1];
    bshift = rand[2];
    r0 = rand[8 * id + 0];
    r1 = rand[8 * id + 1];
    r2 = rand[8 * id + 2];
    r3 = rand[8 * id + 3];

    saturation = r0 * (saturation - 1) + 1;
    saturation = (r1 > .5) ? 1./saturation : saturation;
    exposure = r2 * (exposure - 1) + 1;
    exposure = (r3 > .5) ? 1. / exposure : exposure;

    offset = id * h * w * 3;
    image += offset;
    r = image[x + w * (y + h * 0)];
    g = image[x + w * (y + h * 1)];
    b = image[x + w * (y + h * 2)];

    rgb = make_float3(r,g,b);

    if (train)
    {
        hsv = _YnLayerCropGpuRgbToHsv(rgb);
        hsv.y *= saturation;
        hsv.z *= exposure;
        rgb = _YnLayerCropGpuHsvToRgb(hsv);
    }
    else
    {
        shift = 0;
    }

    image[x + w * (y + h * 0)] = rgb.x * scale + translate + (rshift - .5) * shift;
    image[x + w * (y + h * 1)] = rgb.y * scale + translate + (gshift - .5) * shift;
    image[x + w * (y + h * 2)] = rgb.z * scale + translate + (bshift - .5) * shift;
}

YN_GPU_GLOBAL void _YnLayerCropGpuForward(float *input,
        float *rand,
        int size,
        int c,
        int h,
        int w,
        int crop_height,
        int crop_width,
        int train,
        int flip,
        float angle,
        float *output)
{
    int i, j;
    int count;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size)
        return;

    float cx = w/2.;
    float cy = h/2.;

    int count = id;
    j = id % crop_width;
    id /= crop_width;
    i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    float r4 = rand[8*b + 4];
    float r5 = rand[8*b + 5];
    float r6 = rand[8*b + 6];
    float r7 = rand[8*b + 7];

    float dw = (w - crop_width)*r4;
    float dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5));
    angle = 2*angle*r7 - angle;
    if (!train){
        dw = (w - crop_width)/2.;
        dh = (h - crop_height)/2.;
        flip = 0;
        angle = 0;
    }

    input += w*h*c*b;

    float x = (flip) ? w - dw - j - 1 : j + dw;
    float y = i + dh;

    float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;
    float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;

    output[count] = _YnLayerCropBilineraInterpolate(input, w, h, rx, ry, k);
}

YN_EXTERN_C
void YnLayerCropGpuForward(tYnLayer * layer,
        tYnNetworkState netState)
{
    cuda_random(layer.rand_gpu, layer.batch*8);

    float radians = layer.angle*3.14159265/180.;

    float scale = 2;
    float translate = -1;
    if (layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;

    levels_image_kernel<<<YnCudaGridSize(size), YN_GPU_NUM_THREADS_IN_BLOCK>>>(state.input, layer.rand_gpu, layer.batch, layer.w, layer.h, state.train, layer.saturation, layer.exposure, translate, scale, layer.shift);
    check_error(cudaPeekAtLastError());

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    forward_crop_layer_kernel<<<YnCudaGridSize(size), YN_GPU_NUM_THREADS_IN_BLOCK>>>(state.input, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, state.train, layer.flip, radians, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}
