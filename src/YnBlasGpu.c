//	File        :   YnBlasGpu.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   28-06-2016
//	Author      :   haittt

#include "../YnBlasGpu.h"

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
        uint32 batch,
        uint32 filters,
        uint32 spatial)
{
    uint32 f = 0;
    uint32 index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

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
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * gradientArr)
{
    uint32 f = 0;
    uint32 index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

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
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceGradientArr)
{
    uint32 j, k;
    uint32 index;
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

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
        uint32 groups,
        float *sum)
{
    uint32 k;
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

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
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanGradientArr)
{
    uint32 i, j;
    const uint32 threads = YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_BLOCK;
    uint32 id = threadIdx.x;
    uint32 filter = blockIdx.x;

    YN_GPU_SHARED_MEM float local[threads];
    local[id] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (i = 0; i < spatial; i += threads)
        {
            uint32 index = j * spatial * filters + filter * spatial + i + id;
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
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceGradientArr)
{
    uint32 i, j;
    const uint32 threads = YN_GPU_NUM_THREADS_IN_YN_GPU_NUM_THREADS_IN_BLOCK;
    uint32 id = threadIdx.x;
    uint32 filter = blockIdx.x;

    YN_GPU_SHARED_MEM float local[threads];
    local[id] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (i = 0; i < spatial; i += threads)
        {
            uint32 index = j * spatial * filters + filter * spatial + i + id;

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
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanGradientArr)
{
    uint32 j, k;
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= filters)
        return;

    meanGradientArr[i] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (k = 0; k < spatial; k ++)
        {
            uint32 index = j * filters * spatial + i * spatial + k;
            meanGradientArr[i] += gradientArr[index];
        }
    }

    meanGradientArr[i] *= (-1. / sqrt(varianceArr[i] + .00001f));
}

YN_GPU_GLOBAL void _YnBlasMean(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr)
{
    uint32 j, k;
    float scale = 1. / (batch * spatial);
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= filters)
        return;

    meanArr[i] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (k = 0; k < spatial; k ++)
        {
            uint32 index = j * filters * spatial + i * spatial + k;
            meanArr[i] += inArr[index];
        }
    }

    meanArr[i] *= scale;
}

YN_GPU_GLOBAL void _YnBlasVariance(float * inArr,
        float * meanArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr)
{
    uint32 j, k;
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    float scale = 1. / (batch * spatial);

    if (i >= filters)
        return;

    varianceArr[i] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (k = 0; k < spatial; k ++)
        {
            uint32 index = j * filters * spatial + i * spatial + k;
            varianceArr[i] += pow((inArr[index] - meanArr[i]), 2);
        }
    }

    varianceArr[i] *= scale;
}

YN_GPU_GLOBAL void _YnBlasAxpy(uint32 num,
        float value,
        float * xArr,
        uint32 offsetX,
        uint32 incIdx,
        float * yArr,
        uint32 offsetY,
        uint32 incIdy)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[offsetY + i * incIdy] += value * xArr[offsetX + i * incIdx];
}

YN_GPU_GLOBAL void _YnBlasPow(uint32 num,
        float value,
        float * xArr,
        uint32 incIdx,
        float * yArr,
        uint32 incIdy)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[i * incIdy] = pow(xArr[i * incIdx], value);
}

YN_GPU_GLOBAL void _YnBlasConst(uint32 num,
        float value,
        float * xArr,
        uint32 incIdx)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        xArr[i * incIdx] = value;
}

YN_GPU_GLOBAL void _YnBlasScale(uint32 num,
        float value,
        float * xArr,
        uint32 incIdx)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        xArr[i * incIdx] *= value;
}

YN_GPU_GLOBAL void _YnBlasFill(uint32 num,
        float value,
        float * xArr,
        uint32 incIdx)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        xArr[i * incIdx] = value;
}

YN_GPU_GLOBAL void _YnBlasMask(uint32 num,
        float * inArr,
        float maskNum,
        float *mask)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if ((i < num) && (mask[i] == maskNum))
        inArr[i] = maskNum;
}

YN_GPU_GLOBAL void _YnBlasCopy(uint32 num,
        float * xArr,
        uint32 offsetX,
        uint32 incIdx,
        float * yArr,
        uint32 offsetY,
        uint32 incIdy)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[i * incIdy + offsetY] = xArr[i * incIdx + offsetX];
}

YN_GPU_GLOBAL void _YnBlasMul(uint32 num,
        float * xArr,
        uint32 incIdx,
        float * yArr,
        uint32 incIdy)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < num)
        yArr[i * incIdy] *= xArr[i * incIdx];
}

YN_GPU_GLOBAL void _YnBlasFastMean(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr)
{
    uint32 i, j;
    uint32 index;
    const uint32 threads = YN_GPU_NUM_THREADS_IN_BLOCK;
    uint32 id = threadIdx.x;
    uint32 filter = blockIdx.x;

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
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr)
{
    uint32 i, j;
    const uint32 threads = YN_GPU_NUM_THREADS_IN_BLOCK;
    uint32 id = threadIdx.x;
    uint32 filter = blockIdx.x;

    YN_GPU_SHARED_MEM float local[threads];
    local[id] = 0;

    for (j = 0; j < batch; j ++)
    {
        for (i = 0; i < spatial; i += threads)
        {
            uint32 index = j * spatial * filters + filter * spatial + i + id;

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

YN_GPU_GLOBAL void _YnBlasShortcut(uint32 size,
        uint32 minw,
        uint32 minh,
        uint32 minc,
        uint32 stride,
        uint32 sample,
        uint32 batch,
        uint32 widthAdd,
        uint32 heightAdd,
        uint32 channelAdd,
        float *addArr,
        uint32 widthOut,
        uint32 heightOut,
        uint32 channelOut,
        float *outArr)
{
    uint32 id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (id >= size)
        return;

    uint32 i = id % minw;
    id /= minw;
    uint32 j = id % minh;
    id /= minh;
    uint32 k = id % minc;
    id /= minc;
    uint32 b = id % batch;

    uint32 out_index = i * sample + widthOut * (j * sample + heightOut * (k + channelOut * b));
    uint32 add_index = i * stride + widthAdd * (j * stride + heightAdd * (k + channelAdd * b));

    outArr[out_index] += addArr[add_index];
}

YN_GPU_GLOBAL void _YnBlasSmoothL1(uint32 num,
        float *predArr,
        float *truthArr,
        float * gradientArr)
{
    uint32 i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
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

void YnBlasGpuArrayConstValueSet(float * array,
        uint32 num,
        uint32 incIdx,
        const float value);

void YnBlasGpuArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

void YnBlasGpuArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 powVal);

void YnBlasGpuArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 mulVal);

void YnBlasGpuArrayScaleValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 scaleVal);

void YnBlasGpuArrayFillValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 fillVal);

void YnBlasGpuArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

void YnBlasGpuArrayDotValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);


void YnBlasGpuGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num);


void YnBlasGpuShortcut(uint32 batch,
        uint32 widthAdd,
        uint32 heightAdd,
        uint32 channelAdd,
        float * addArr,
        uint32 widthOut,
        uint32 heightOut,
        uint32 channelOut,
        float * outArr);


void YnBlasGpuArrayMeanCal(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr);


void YnBlasGpuArrayVarianceCal(float * arrayIn,
        float * meanArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr);


void YnBlasGpuArrayNormalizeCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial);


void YnBlasGpuFastArrayMeanGradientCal(float * gradientArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanGradientArr);

void YnBlasGpuFastArrayVarianceGradientCal(float * arrayIn,
        float * gradientArr,
        float * meanArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceGradientArr);


void YnBlasGpuFastArrayVarianceCal(float * arrayIn,
        float * meanArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr);


void YnBlasGpuFastArrayMeanCal(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr);
