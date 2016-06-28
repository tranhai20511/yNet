#ifndef YNBLASGPU_H
#define YNBLASGPU_H

#include "../YnBlas.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

/*
 * Set value for array elements
 */
YN_FINAL void YnBlasGpuArrayConstValueSet(float * array,
        uint32 num,
        uint32 incIdx,
        const float value);

YN_FINAL void YnBlasGpuArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

YN_FINAL void YnBlasGpuArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 powVal);

YN_FINAL void YnBlasGpuArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 mulVal);

YN_FINAL void YnBlasGpuArrayScaleValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 scaleVal);

YN_FINAL void YnBlasGpuArrayFillValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 fillVal);

YN_FINAL void YnBlasGpuArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

YN_FINAL void YnBlasGpuArrayDotValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

/*
 * Smooth gradient array
 */
YN_FINAL void YnBlasGpuGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num);

/*
 * Shortcut
 */
YN_FINAL void YnBlasGpuShortcut(uint32 batch,
        uint32 widthAdd,
        uint32 heightAdd,
        uint32 channelAdd,
        float * addArr,
        uint32 widthOut,
        uint32 heightOut,
        uint32 channelOut,
        float * outArr);

/*
 * Calculate mean array
 */
YN_FINAL void YnBlasGpuArrayMeanCal(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr);

/*
 * Calculate variance array
 */
YN_FINAL void YnBlasGpuArrayVarianceCal(float * arrayIn,
        float * meanArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr);

/*
 * Calculate normalize array
 */
YN_FINAL void YnBlasGpuArrayNormalizeCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial);

/*
 * Fast calculate mean gradient array
 */
YN_FINAL void YnBlasGpuFastArrayMeanGradientCal(float * gradientArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanGradientArr);
/*
 * Fast calculate variance gradient array
 */
YN_FINAL void YnBlasGpuFastArrayVarianceGradientCal(float * arrayIn,
        float * gradientArr,
        float * meanArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceGradientArr);

/*
 * Fast calculate variance array
 */
YN_FINAL void YnBlasGpuFastArrayVarianceCal(float * arrayIn,
        float * meanArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr);

/*
 * Fast calculate mean array
 */
YN_FINAL void YnBlasGpuFastArrayMeanCal(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr);

#ifdef __cplusplus
}
#endif
