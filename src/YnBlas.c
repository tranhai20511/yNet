//	File        :   YnBlas.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   28-06-2016
//	Author      :   haittt

#include "../YnBlas.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

YN_FINAL void YnBlasArrayConstValueSet(float * array,
        uint32 num,
        uint32 incIdx,
        const float value)
{
    uint32 i;

    for(i = 0; i < num; i ++)
        array[i * incIdx] = value;
}

YN_FINAL void YnBlasArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx)
{
    uint32 i;

    for(i = 0; i < num; i ++)
        yArr[i * incIdy] *= xArr[i * incIdx];
}

YN_FINAL void YnBlasArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 powVal)
{
    uint32 i;

    for(i = 0; i < num; i ++)
        yArr[i * incIdy] = pow(xArr[i * incIdx], powVal);
}

YN_FINAL void YnBlasArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 mulVal)
{
    uint32 i;

    for(i = 0; i < num; i ++)
        yArr[i * incIdy] += mulVal * xArr[i * incIdx];
}

YN_FINAL void YnBlasArrayScaleValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 scaleVal)
{
    uint32 i;

    for(i = 0; i < num; i ++)
        xArr[i * incIdx] *= scaleVal;
}

YN_FINAL void YnBlasArrayFillValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 fillVal)
{
    uint32 i;

    for(i = 0; i < num; i ++)
        xArr[i * incIdx] = fillVal;
}

YN_FINAL void YnBlasArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx)
{
    uint32 i;

    for(i = 0; i < num; i ++)
        yArr[i * incIdy] = xArr[i * incIdx];
}

/*
 * Smooth gradient array
 */
YN_FINAL void YnBlasGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num)
{
    uint32 i;
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
YN_FINAL void YnBlasShortcut(uint32 batch,
        uint32 widthAdd,
        uint32 heightAdd,
        uint32 channelAdd,
        float * addArr,
        uint32 widthOut,
        uint32 heightOut,
        uint32 channelOut,
        float * outArr)
{
    uint32 i, j, k, b;
    uint32 outIndex = 0;
    uint32 addIndex = 0;
    uint32 stride = widthAdd / widthOut;
    uint32 sample = widthOut / widthAdd;

    uint32 minw = (widthAdd < widthOut) ? widthAdd : widthOut;
    uint32 minh = (heightAdd < heightOut) ? heightAdd : heightOut;
    uint32 minc = (channelAdd < channelOut) ? channelAdd : channelOut;

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
YN_FINAL void YnBlasArrayMeanCal(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr)
{
    float scale = 1. / (batch * spatial);
    uint32 i, j, k;
    uint32 index;

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
YN_FINAL void YnBlasArrayVarianceCal(float * inArr,
        float * meanArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr)
{
    float scale = 1. / (batch * spatial);
    uint32 i, j, k;
    uint32 index;

    for (i = 0; i < filters; ++i)
    {
        varianceArr[i] = 0;
        for (j = 0; j < batch; ++j)
        {
            for (k = 0; k < spatial; ++k)
            {
                uint32 index = j * filters * spatial + i * spatial + k;
                varianceArr[i] += pow((inArr[index] - meanArr[i]), 2);
            }
        }
        varianceArr[i] *= scale;
    }
}

/*
 * Calculate normalize array
 */
YN_FINAL void YnBlasArrayNormalizeCal(float * inArr,
        float * meanArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial)
{
    uint32 b, f, i;

    for (b = 0; b < batch; ++b)
    {
        for (f = 0; f < filters; ++f)
        {
            for (i = 0; i < spatial; ++i)
            {
                uint32 index = b * filters * spatial + f * spatial + i;
                inArr[index] = (inArr[index] - meanArr[f]) / (sqrt(varianceArr[f]) + .00001f);
            }
        }
    }
}
