#ifndef YNBLAS_H
#define YNBLAS_H

#include "../YnStd.h"

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
YN_FINAL void YnBlasArrayConstValueSet(float * array,
        uint32 num,
        uint32 incIdx,
        const float value);

YN_FINAL void YnBlasArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

YN_FINAL void YnBlasArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 powVal);

YN_FINAL void YnBlasArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx,
        uint32 mulVal);

YN_FINAL void YnBlasArrayScaleValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 scaleVal);

YN_FINAL void YnBlasArrayFillValueSet(float * xArr,
        uint32 num,
        uint32 incIdx,
        uint32 fillVal);

YN_FINAL void YnBlasArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

YN_FINAL void YnBlasArrayDotValueSet(float * yArr,
        float * xArr,
        uint32 num,
        uint32 incIdy,
        uint32 incIdx);

/*
 * Smooth gradient array
 */
YN_FINAL void YnBlasGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num);

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
        float * outArr);

/*
 * Calculate mean array
 */
YN_FINAL void YnBlasArrayMeanCal(float * inArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * meanArr);

/*
 * Calculate variance array
 */
YN_FINAL void YnBlasArrayVarianceCal(float * arrayIn,
        float * meanArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial,
        float * varianceArr);

/*
 * Calculate normalize array
 */
YN_FINAL void YnBlasArrayNormalizeCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        uint32 batch,
        uint32 filters,
        uint32 spatial);

#ifdef __cplusplus
}
#endif
