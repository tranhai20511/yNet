//	File        :   YnBlas.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-06-2016
//	Author      :   haittt

#include "../include/YnBlas.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnBlasArrayConstValueSet(float * array,
        uint32 num,
        int32 incIdx,
        const float value)
{
    int32 i;

    for (i = 0; i < num; i ++)
        array[i * incIdx] = value;
}

void YnBlasArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
{
    int32 i;

    for (i = 0; i < num; i ++)
        yArr[i * incIdy] *= xArr[i * incIdx];
}

void YnBlasArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 powVal)
{
    int32 i;

    for (i = 0; i < num; i ++)
        yArr[i * incIdy] = pow(xArr[i * incIdx], powVal);
}

void YnBlasArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 mulVal)
{
    int32 i;

    for (i = 0; i < num; i ++)
        yArr[i * incIdy] += mulVal * xArr[i * incIdx];
}

void YnBlasArrayScaleValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 scaleVal)
{
    int32 i;

    for (i = 0; i < num; i ++)
        xArr[i * incIdx] *= scaleVal;
}

void YnBlasArrayFillValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 fillVal)
{
    int32 i;

    for (i = 0; i < num; i ++)
        xArr[i * incIdx] = fillVal;
}

void YnBlasArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
{
    int32 i;

    for (i = 0; i < num; i ++)
        yArr[i * incIdy] = xArr[i * incIdx];
}

float YnBlasArrayDotValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
{
    int32 i;
    float dot = 0;

    for (i = 0; i < num; i ++)
        dot += xArr[i * incIdx] * yArr[i * incIdy];

    return dot;
}

/*
 * Smooth gradient array
 */
void YnBlasGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num)
{
    int32 i;
    float diff = 0;

    for (i = 0; i < num; i++)
    {
        diff = truthArr[i] - preArr[i];

        if (fabs(diff) > 1)
            deltaArr[i] = diff;
        else
            deltaArr[i] = (diff > 0) ? 1 : -1;
    }
}

/*
 * Shortcut
 */
void YnBlasShortcut(int32 batch,
        int32 widthAdd,
        int32 heightAdd,
        int32 channelAdd,
        float * addArr,
        int32 widthOut,
        int32 heightOut,
        int32 channelOut,
        float * outArr)
{
    int32 i, j, k, b;
    int32 outIndex = 0;
    int32 addIndex = 0;
    int32 stride = widthAdd / widthOut;
    int32 sample = widthOut / widthAdd;

    int32 minw = (widthAdd < widthOut) ? widthAdd : widthOut;
    int32 minh = (heightAdd < heightOut) ? heightAdd : heightOut;
    int32 minc = (channelAdd < channelOut) ? channelAdd : channelOut;

    assert(stride == heightAdd / heightOut);
    assert(sample == heightOut / heightAdd);

    if (stride < 1)
        stride = 1;

    if (sample < 1)
        sample = 1;

    for (b = 0; b < batch; ++b)
    {
        for (k = 0; k < minc; ++k)
        {
            for (j = 0; j < minh; ++j)
            {
                for (i = 0; i < minw; ++i)
                {
                    outIndex = i * sample + widthOut * (j * sample + heightOut * (k + channelOut * b));
                    addIndex = i * stride  + widthAdd * (j * stride  + heightAdd * (k + channelAdd * b));
                    outArr[outIndex] += outArr[addIndex];
                }
            }
        }
    }
}

/*
 * Calculate mean array
 */
void YnBlasArrayMeanCal(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
{
    float scale = 1. / (batch * spatial);
    int32 i, j, k;
    int32 index;

    for (i = 0; i < filters; ++i)
    {
        meanArr[i] = 0;
        for (j = 0; j < batch; ++j)
        {
            for (k = 0; k < spatial; ++k)
            {
                index = (j * filters +   i) * spatial + k;
                meanArr[i] += inArr[index];
            }
        }
        meanArr[i] *= scale;
    }
}

/*
 * Calculate variance array
 */
void YnBlasArrayVarianceCal(float * inArr,
        float * meanArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceArr)
{
    float scale = 1. / (batch * spatial);
    int32 i, j, k;
    int32 index;

    for (i = 0; i < filters; ++i)
    {
        varianceArr[i] = 0;
        for (j = 0; j < batch; ++j)
        {
            for (k = 0; k < spatial; ++k)
            {
                int32 index = j * filters * spatial + i * spatial + k;
                varianceArr[i] += pow((inArr[index] - meanArr[i]), 2);
            }
        }
        varianceArr[i] *= scale;
    }
}

/*
 * Calculate normalize array
 */
void YnBlasArrayNormalizeCal(float * inArr,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial)
{
    int32 b, f, i;

    for (b = 0; b < batch; ++b)
    {
        for (f = 0; f < filters; ++f)
        {
            for (i = 0; i < spatial; ++i)
            {
                int32 index = b * filters * spatial + f * spatial + i;
                inArr[index] = (inArr[index] - meanArr[f]) / (sqrt(varianceArr[f]) + .00001f);
            }
        }
    }
}

void YnBlasArrayBiasAdd(float * output,
        float * biases,
        int batch,
        int num,
        int size)
{
    int i, j, b;

    for(b = 0; b < batch; b ++)
    {
        for(i = 0; i < num; i ++)
        {
            for(j = 0; j < size; j ++)
            {
                output[(b * num + i) * size + j] += biases[i];
            }
        }
    }
}

void YnBlasArrayBackwardBias(float * biasUpdates,
        float * gradient,
        int batch,
        int num,
        int size)
{
    int i, b;

    for(b = 0; b < batch; b ++)
    {
        for(i = 0; i < num; i ++)
        {
            biasUpdates[i] += YnUtilArraySum(gradient + size * (i + b * num), size);
        }
    }
}

void YnBlasArrayBiasScale(float * output,
        float * scales,
        int batch,
        int num,
        int size)
{
    int i, j, b;

    for(b = 0; b < batch; b ++)
    {
        for(i = 0; i < num; i ++)
        {
            for(j = 0; j < size; j ++)
            {
                output[(b * num + i) * size + j] *= scales[i];
            }
        }
    }
}

void YnBlasArrayBackwardScale(float * xNorm,
        float * delta,
        int batch,
        int num,
        int size,
        float *scaleUpdates)
{
    int i, b, f;
    int index;
    float sum;

    for(f = 0; f < num; f ++)
    {
        sum = 0;
        for(b = 0; b < batch; b ++)
        {
            for(i = 0; i < size; i ++)
            {
                index = i + size * (f + num * b);
                sum += delta[index] * xNorm[index];
            }
        }

        scaleUpdates[f] += sum;
    }
}

void YnBlasArrayMeanGradient(float * gradient,
        float * variance,
        int batch,
        int filters,
        int spatial,
        float *meanGradient)
{
    int i ,j ,k;
    int index;

    for(i = 0; i < filters; i ++)
    {
        variance[i] = 0;

        for (j = 0; j < batch; j ++)
        {
            for (k = 0; k < spatial; k ++)
            {
                index = j * filters * spatial + i * spatial + k;
                meanGradient[i] += gradient[index];
            }
        }

        meanGradient[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}

void YnBlasArrayVarianceGradient(float * arrayIn,
        float * delta,
        float * mean,
        float * variance,
        int batch,
        int filters,
        int spatial,
        float *varianceDelta)
{
    int index;
    int i ,j ,k;

    for(i = 0; i < filters; i ++)
    {
        varianceDelta[i] = 0;
        for(j = 0; j < batch; j ++)
        {
            for(k = 0; k < spatial; k ++)
            {
                index = j * filters * spatial + i * spatial + k;
                varianceDelta[i] += delta[index] * (arrayIn[index] - mean[i]);
            }
        }

        varianceDelta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

void YnBlasArrayNormalizeGradient(float * arrayIn,
        float * mean,
        float * variance,
        float * mean_delta,
        float * varianceGradient,
        int batch,
        int filters,
        int spatial,
        float *delta)
{
    int index;
    int f, j, k;

    for(j = 0; j < batch; j ++)
    {
        for(f = 0; f < filters; f ++)
        {
            for(k = 0; k < spatial; k ++)
            {
                index = j * filters * spatial + f * spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + varianceGradient[f] * 2. * (arrayIn[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}
