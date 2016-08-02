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
YN_FINAL
void YnBlasArrayConstValueSet(float * array,
        uint32 num,
        int32 incIdx,
        const float value)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 powVal)
YN_ALSWAY_INLINE;

YN_FINAL void YnBlasArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 mulVal)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayScaleValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 scaleVal)
YN_ALSWAY_INLINE;

YN_FINAL void YnBlasArrayFillValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 fillVal)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
YN_ALSWAY_INLINE;

YN_FINAL
float YnBlasArrayDotValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
YN_ALSWAY_INLINE;


/*
 * Smooth gradient
 */
YN_FINAL
void YnBlasGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num)
YN_ALSWAY_INLINE;

/*
 * Shortcut
 */
YN_FINAL
void YnBlasShortcut(int32 batch,
        int32 widthAdd,
        int32 heightAdd,
        int32 channelAdd,
        float * addArr,
        int32 widthOut,
        int32 heightOut,
        int32 channelOut,
        float * outArr)
YN_ALSWAY_INLINE;

/*
 * Calculate mean
 */
YN_FINAL
void YnBlasArrayMeanCal(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
YN_ALSWAY_INLINE;

/*
 * Calculate variance
 */
YN_FINAL
void YnBlasArrayVarianceCal(float * arrayIn,
        float * meanArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceArr)
YN_ALSWAY_INLINE;

/*
 * Calculate normalize
 */
YN_FINAL
void YnBlasArrayNormalizeCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayBiasAdd(float *output,
        float *biases,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayBackwardBias(float *biasUpdates,
        float *gradient,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayBiasScale(float * output,
        float * scales,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayBackwardScale(float * xNorm,
        float *delta,
        int batch,
        int num,
        int size,
        float *scaleUpdates)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayMeanGradient(float * gradient,
        float * variance,
        int batch,
        int filters,
        int spatial,
        float *meanGradient)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayVarianceGradient(float * arrayIn,
        float * delta,
        float * mean,
        float * variance,
        int batch,
        int filters,
        int spatial,
        float *varianceDelta)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasArrayNormalizeGradient(float * arrayIn,
        float * mean,
        float * variance,
        float * mean_delta,
        float * varianceGradient,
        int batch,
        int filters,
        int spatial,
        float *delta)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNBLAS_H */
