//	File        :   YnImage.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   04-07-2016
//	Author      :   haittt

extern "C" {
#include "../include/YnCuda.h"
#include "../include/YnImageGpu.h"
}

#ifdef YN_GPU

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_GPU_GLOBAL void YnImageGpuCol2ImageKernel(const int n,
        const float* data_col,
        const int height,
        const int width,
        const int ksize,
        const int pad,
        const int stride,
        const int height_col,
        const int width_col,
        float *data_im)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < n; index += blockDim.x * gridDim.x)
    {
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);

        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);

        // equivalent implementation
        int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);

        for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
        {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
            {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }

        data_im[index] += val;
    }
}

YN_EXTERN_C
void YnImageGpuCol2Image(float *data_col,
        int channels,
        int height,
        int width,
        int ksize,
        int stride,
        int pad,
        float *data_im)
{
    /*
    We are going to launch channels * height_col * width_col kernels, each
    kernel responsible for copying a single-channel grid.
    */

    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;

    YnImageGpuCol2ImageKernel<<<(num_kernels+BLOCK-1)/BLOCK,
        YN_GPU_NUM_THREADS_IN_BLOCK>>>(num_kernels, data_col, height, width, ksize, pad,
                    stride, height_col, width_col, data_im);
}

YN_GPU_GLOBAL void YnImageGpuImage2ColKernel(const int n,
        const float* data_im,
        const int height,
        const int width,
        const int ksize,
        const int pad,
        const int stride,
        const int height_col,
        const int width_col,
        float *data_col)
{
    int h;
    int w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < n; index += blockDim.x * gridDim.x)
    {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        const float* data_im_ptr = data_im;

        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;

        for (int i = 0; i < ksize; i ++)
        {
            for (int j = 0; j < ksize; j ++)
            {
                h = h_in + i;
                w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                        data_im_ptr[i * width + j] : 0;

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

YN_EXTERN_C
void YnImageGpuImage2Col(float *im,
        int channels,
        int height,
        int width,
        int ksize,
        int stride,
        int pad,
        float *data_col)
{
    /*
    We are going to launch channels * height_col * width_col kernels,
    each kernel responsible for copying a single-channel grid
    */

    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    YnImageGpuImage2ColKernel<<<(num_kernels+BLOCK-1)/BLOCK,
        YN_GPU_NUM_THREADS_IN_BLOCK>>>(num_kernels, im, height, width, ksize, pad,
                 stride, height_col, width_col, data_col);
}

#endif
