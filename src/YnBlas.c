//	File        :   YnBlas.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   28-06-2016
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
