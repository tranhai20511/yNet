//	File        :   YnBlasGpu.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   28-06-2016
//	Author      :   haittt

#include "../include/YnBlasGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
YN_GPU_GLOBAL void _YnBlasNormalize(uint32 num,
        float * inArr,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial)
{
    int32 f = 0;
    int32 index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (index >= num)
        return;

    f = (index / spatial) % filters;

    inArr[index] = (inArr[index] - meanArr[f]) / (sqrt(varianceArr[f]) + .00001f);
}

YN_GPU_GLOBAL void _YnBlasNormalizeGradient(uint32 num,
        float * inArr,
        float * meanArr,
        float * varianceArr,
        float * meanGradientArr,
        float * varianceGradientArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * gradientArr)
{
    int32 f = 0;
    int32 index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (index >= num)
        return;

    f = (index / spatial) % filters;

    gradientArr[index] = gradientArr[index] * 1. / (sqrt(varianceArr[f]) + .00001f)
            + varianceGradientArr[f] * 2. * (inArr[index] - meanArr[f]) / (spatial * batch)
            + meanGradientArr[f] / (spatial * batch);
}

YN_GPU_GLOBAL void _YnBlasVarianceGradient(float * inArr,
        float * gradientArr,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceGradientArr)
{
    int32 j, k;
    int32 index;
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= filters)
        return;

    varianceGradientArr[i] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (k = 0; k < spatial; k ++)
        {
            index = j * filters * spatial + i * spatial + k;
            varianceGradientArr[i] += gradientArr[index] * (inArr[index] - meanArr[i]);
        }
    }

    varianceGradientArr[i] *= -.5 * pow(varianceArr[i] + .00001f, (float) (-3. / 2.));
}

YN_GPU_GLOBAL void _YnBlasAccumulate(float * inArr,
        uint32 num,
        int32 groups,
        float *sum)
{
    int32 k;
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= groups)
        return;

    sum[i] = 0;

    for (k = 0; k < num; k ++)
    {
        sum[i] += inArr[k * groups + i];
    }
}

YN_GPU_GLOBAL void _YnBlasFastMeanGradient(float * gradientArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanGradientArr)
{
    int32 i, j;
    const int32 threads = YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_BLOCK;
    int32 id = threadIdx.x;
    int32 filter = blockIdx.x;

    YN_GPU_SHARED_MEM float local[threads];
    local[id] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (i = 0; i < spatial; i += threads)
        {
            int32 index = j * spatial * filters + filter * spatial + i + id;
            local[id] += (i + id < spatial) ? gradientArr[index] : 0;
        }
    }

    if (id == 0)
    {
        meanGradientArr[filter] = 0;

        for (i = 0; i < threads; i ++)
        {
            meanGradientArr[filter] += local[i];
        }

        meanGradientArr[filter] *= (-1. / sqrt(varianceArr[filter] + .00001f));
    }
}

YN_GPU_GLOBAL void _YnBlasFastVarianceGradient(float * inArr,
        float * gradientArr,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceGradientArr)
{
    int32 i, j;
    const int32 threads = YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_BLOCK;
    int32 id = threadIdx.x;
    int32 filter = blockIdx.x;

    YN_GPU_SHARED_MEM float local[threads];
    local[id] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (i = 0; i < spatial; i += threads)
        {
            int32 index = j * spatial * filters + filter * spatial + i + id;

            local[id] += (i + id < spatial) ? gradientArr[index] * (inArr[index] - meanArr[filter]) : 0;
        }
    }

    if (id == 0)
    {
        varianceGradientArr[filter] = 0;

        for (i = 0; i < threads; i ++)
        {
            varianceGradientArr[filter] += local[i];
        }

        varianceGradientArr[filter] *= -.5 * pow(varianceArr[filter] + .00001f, (float) (-3. / 2.));
    }
}

YN_GPU_GLOBAL void _YnBlasMeanGradient(float * gradientArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanGradientArr)
{
    int32 j, k;
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= filters)
        return;

    meanGradientArr[i] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (k = 0; k < spatial; k ++)
        {
            int32 index = j * filters * spatial + i * spatial + k;
            meanGradientArr[i] += gradientArr[index];
        }
    }

    meanGradientArr[i] *= (-1. / sqrt(varianceArr[i] + .00001f));
}

YN_GPU_GLOBAL void _YnBlasMean(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
{
    int32 j, k;
    float scale = 1. / (batch * spatial);
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= filters)
        return;

    meanArr[i] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (k = 0; k < spatial; k ++)
        {
            int32 index = j * filters * spatial + i * spatial + k;
            meanArr[i] += inArr[index];
        }
    }

    meanArr[i] *= scale;
}

YN_GPU_GLOBAL void _YnBlasVariance(float * inArr,
        float * meanArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceArr)
{
    int32 j, k;
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    float scale = 1. / (batch * spatial);

    if (i >= filters)
        return;

    varianceArr[i] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (k = 0; k < spatial; k ++)
        {
            int32 index = j * filters * spatial + i * spatial + k;
            varianceArr[i] += pow((inArr[index] - meanArr[i]), 2);
        }
    }

    varianceArr[i] *= scale;
}

YN_GPU_GLOBAL void _YnBlasAxpy(uint32 num,
        float value,
        float * xArr,
        int32 offsetxArr,
        int32 incIdx,
        float * yArr,
        int32 offsetY,
        int32 incIdy)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[offsetY + i * incIdy] += value * xArr[offsetX + i * incIdx];
}

YN_GPU_GLOBAL void _YnBlasPow(uint32 num,
        float value,
        float * xArr,
        int32 incIdx,
        float * yArr,
        int32 incIdy)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[i * incIdy] = pow(xArr[i * incIdx], value);
}

YN_GPU_GLOBAL void _YnBlasConst(uint32 num,
        float value,
        float * xArr,
        int32 incIdx)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        xArr[i * incIdx] = value;
}

YN_GPU_GLOBAL void _YnBlasScale(uint32 num,
        float value,
        float * xArr,
        int32 incIdx)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        xArr[i * incIdx] *= value;
}

YN_GPU_GLOBAL void _YnBlasFill(uint32 num,
        float value,
        float * xArr,
        int32 incIdx)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        xArr[i * incIdx] = value;
}

YN_GPU_GLOBAL void _YnBlasMask(uint32 num,
        float * inArr,
        float maskNum,
        float *mask)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if ((i < num) && (mask[i] == maskNum))
        inArr[i] = maskNum;
}

YN_GPU_GLOBAL void _YnBlasCopy(uint32 num,
        float * xArr,
        int32 offsetxArr,
        int32 incIdx,
        float * yArr,
        int32 offsetY,
        int32 incIdy)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[i * incIdy + offsetY] = xArr[i * incIdx + offsetX];
}

YN_GPU_GLOBAL void _YnBlasMul(uint32 num,
        float * xArr,
        int32 incIdx,
        float * yArr,
        int32 incIdy)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[i * incIdy] *= xArr[i * incIdx];
}

YN_GPU_GLOBAL void _YnBlasFastMean(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
{
    int32 i, j;
    int32 index;
    const int32 threads = YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_BLOCK;
    int32 id = threadIdx.x;
    int32 filter = blockIdx.x;

    YN_GPU_SHARED_MEM float local[threads];
    local[id] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (i = 0; i < spatial; i += threads)
        {
            index = j * spatial * filters + filter * spatial + i + id;
            local[id] += (i + id < spatial) ? inArr[index] : 0;
        }
    }

    if (id == 0)
    {
        meanArr[filter] = 0;

        for (i = 0; i < threads; i ++)
        {
            meanArr[filter] += local[i];
        }

        meanArr[filter] /= spatial * batch;
    }
}

YN_GPU_GLOBAL void _YnBlasFastVariance(float * inArr,
        float * meanArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceArr)
{
    int32 i, j;
    const int32 threads = YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_BLOCK;
    int32 id = threadIdx.x;
    int32 filter = blockIdx.x;

    YN_GPU_SHARED_MEM float local[threads];
    local[id] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (i = 0; i < spatial; i += threads)
        {
            int32 index = j * spatial * filters + filter * spatial + i + id;

            local[id] +=
                    (i + id < spatial) ? pow((inArr[index] - meanArr[filter]), 2) : 0;
        }
    }

    if (id == 0)
    {
        varianceArr[filter] = 0;
        for (i = 0; i < threads; i ++)
        {
            varianceArr[filter] += local[i];
        }
        varianceArr[filter] /= spatial * batch;
    }
}

YN_GPU_GLOBAL void _YnBlasShortcut(int32 size,
        int32 minw,
        int32 minh,
        int32 minc,
        int32 stride,
        int32 sample,
        int32 batch,
        int32 widthAdd,
        int32 heightAdd,
        int32 channelAdd,
        float *addArr,
        int32 widthOut,
        int32 heightOut,
        int32 channelOut,
        float *outArr)
{
    int32 id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (id >= size)
        return;

    int32 i = id % minw;
    id /= minw;
    int32 j = id % minh;
    id /= minh;
    int32 k = id % minc;
    id /= minc;
    int32 b = id % batch;

    int32 out_index = i * sample + widthOut * (j * sample + heightOut * (k + channelOut * b));
    int32 add_index = i * stride + widthAdd * (j * stride + heightAdd * (k + channelAdd * b));

    outArr[out_index] += addArr[add_index];
}

YN_GPU_GLOBAL void _YnBlasSmoothL1(uint32 num,
        float *predArr,
        float *truthArr,
        float * gradientArr)
{
    int32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    float diff;

    if (i < num)
    {
        diff = truthArr[i] - predArr[i];

        if (abs(diff) > 1)
        {
            gradientArr[i] = diff;
        }
        else
        {
            gradientArr[i] = (diff > 0) ? 1 : -1;
        }
    }
}

/**************** Implement */
YN_EXTERN_C
void YnBlasGpuArrayConstValueSet(float * array,
        uint32 num,
        int32 incIdx,
        const float value)
{
    _YnBlasConst<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, value, array, incIdx);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
{
    _YnBlasMul<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, xArr, incIdx, yArr, incIdy);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 powVal)
{
    _YnBlasPow<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, powVal, xArr, incIdx, yArr, incIdy);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 mulVal)
{
    YnBlasGpuArrayAxpyValueSet(yArr, xArr, num, incIdy, 0, incIdx, 0, mulVal);
}

YN_EXTERN_C
void YnBlasGpuArrayAxpyOffsetValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 offsetY,
        int32 incIdx,
        int32 offsetX,
        int32 mulVal)
{
    _YnBlasAxpy<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, mulVal, xArr, offsetX, incIdx, yArr, offsetY, incIdy);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayScaleValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 scaleVal)
{
    _YnBlasScale<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, scaleVal, xArr, incIdx);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayFillValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 fillVal)
{
    _YnBlasFill<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, fillVal, xArr, incIdx);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
{
    YnBlasGpuArrayCopyOffsetValueSet(yArr, xArr, num, incIdy, 0, incIdx, 0);
}

YN_EXTERN_C
void YnBlasGpuArrayCopyOffsetValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 offsetY,
        int32 incIdx,
        int32 offsetX)
{
    _YnBlasCopy<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, xArr, offsetX, incIdx, yArr, offsetY, incIdy);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayMaskValueSet(float * xArr,
        uint32 num,
        float maskNum,
        float * maskArr)
{
    _YnBlasMask<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, xArr, maskNum, maskArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num)
{
    _YnBlasSmoothL1<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, preArr, truthArr, deltaArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuShortcut(int32 batch,
        int32 widthAdd,
        int32 heightAdd,
        int32 channelAdd,
        float * addArr,
        int32 widthOut,
        int32 heightOut,
        int32 channelOut,
        float * outArr)
{
    int minw = (widthAdd < widthOut) ? widthAdd : widthOut;
    int minh = (heightAdd < heightOut) ? heightAdd : heightOut;
    int minc = (channelAdd < channelOut) ? channelAdd : channelOut;

    int stride = widthAdd / widthOut;
    int sample = widthOut / widthAdd;

    assert(stride == heightAdd / heightOut);
    assert(sample == heightOut / heightAdd);

    if (stride < 1)
        stride = 1;
    if (sample < 1)
        sample = 1;

    int size = batch * minw * minh * minc;

    _YnBlasShortcut<<<cuda_gridsize(size), YN_GPU_NUM_THREADS_IN_BLOCK>>>(size,
            minw,
            minh,
            minc,
            stride,
            sample,
            batch,
            widthAdd,
            heightAdd,
            channelAdd,
            addArr,
            widthOut,
            heightOut,
            channelOut,
            outArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayMeanCal(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
{
    _YnBlasMean<<<cuda_gridsize(filters), YN_GPU_NUM_THREADS_IN_BLOCK>>>(inArr, batch, filters, spatial, meanArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayMeanGradientCal(float * gradientArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanGradientArr)
{
    _YnBlasMeanGradient<<<cuda_gridsize(filters), YN_GPU_NUM_THREADS_IN_BLOCK>>>(gradientArr, varianceArr, batch, filters, spatial, meanGradientArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayVarianceCal(float * arrayIn,
        float * meanArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceArr)
{
    _YnBlasVariance<<<cuda_gridsize(filters), YN_GPU_NUM_THREADS_IN_BLOCK>>>(arrayIn, meanArr, batch, filters, spatial, varianceArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayNormalizeCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial)
{
    uint32 num = batch * filters * spatial;

    _YnBlasNormalize<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, arrayIn, meanArr, varianceArr, batch, filters, spatial);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuArrayNormalizeGradientCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        float * meanGradientArr,
        float * varianceGradientArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * gradientArr)
{
    uint32 num = batch * filters * spatial;

    _YnBlasNormalizeGradient<<<cuda_gridsize(num), YN_GPU_NUM_THREADS_IN_BLOCK>>>(num, arrayIn, meanArr, varianceArr, meanGradientArr, varianceGradientArr, batch, filters, spatial, gradientArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuFastArrayMeanGradientCal(float * gradientArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanGradientArr)
{
    _YnBlasFastMeanGradient<<<filters, YN_GPU_NUM_THREADS_IN_BLOCK>>>(gradientArr, varianceArr, batch, filters, spatial, meanGradientArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuFastArrayVarianceGradientCal(float * arrayIn,
        float * gradientArr,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceGradientArr)
{
    _YnBlasFastVarianceGradient<<<filters, YN_GPU_NUM_THREADS_IN_BLOCK>>>(arrayIn, gradientArr, meanArr, varianceArr, batch, filters, spatial, varianceGradientArr);

    YnCudaCheckError(cudaPeekAtLastError());
}

YN_EXTERN_C
void YnBlasGpuFastArrayVarianceCal(float * arrayIn,
        float * meanArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceArr)
{
    _YnBlasFastVariance<<<filters, YN_GPU_NUM_THREADS_IN_BLOCK>>>(arrayIn, meanArr, batch, filters, spatial, varianceArr);

    YnCudaCheckError(cudaPeekAtLastError());
}


YN_EXTERN_C
void YnBlasGpuFastArrayMeanCal(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
{
    _YnBlasFastMean<<<filters, YN_GPU_NUM_THREADS_IN_BLOCK>>>(inArr, batch, filters, spatial, meanArr);

    YnCudaCheckError(cudaPeekAtLastError());
}
