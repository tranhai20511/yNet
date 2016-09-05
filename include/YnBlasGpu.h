#ifndef YNBLASGPU_H
#define YNBLASGPU_H

#include "YnBlas.h"

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
void YnBlasGpuArrayConstValueSet(float * array,
        uint32 num,
        int32 incIdx,
        const float value)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayMultipleValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayPowValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 powVal)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayAxpyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx,
        int32 mulVal)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayAxpyOffsetValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 offsetY,
        int32 incIdx,
        int32 offsetX,
        int32 mulVal)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayScaleValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 scaleVal)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayFillValueSet(float * xArr,
        uint32 num,
        int32 incIdx,
        int32 fillVal)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayCopyValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 incIdx)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayCopyOffsetValueSet(float * yArr,
        float * xArr,
        uint32 num,
        int32 incIdy,
        int32 offsetY,
        int32 incIdx,
        int32 offsetX)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayMaskValueSet(float * xArr,
        uint32 num,
        float maskNum,
        float * maskArr)
YN_ALSWAY_INLINE;

/*
 * Smooth gradient
 */
YN_FINAL
void YnBlasGpuGradientSmoothL1(float * preArr,
        float * truthArr,
        float * deltaArr,
        uint32 num)
YN_ALSWAY_INLINE;

/*
 * Shortcut
 */
YN_FINAL
void YnBlasGpuShortcut(int32 batch,
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
void YnBlasGpuArrayMeanCal(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayMeanGradientCal(float * gradientArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanGradientArr)
YN_ALSWAY_INLINE;

/*
 * Calculate variance
 */
YN_FINAL
void YnBlasGpuArrayVarianceCal(float * arrayIn,
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
void YnBlasGpuArrayNormalizeCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuArrayNormalizeGradientCal(float * arrayIn,
        float * meanArr,
        float * varianceArr,
        float * meanGradientArr,
        float * varianceGradientArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * gradientArr)
YN_ALSWAY_INLINE;

/*
 * Fast calculate mean gradient array
 */
YN_FINAL
void YnBlasGpuFastArrayMeanGradientCal(float * gradientArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanGradientArr)
YN_ALSWAY_INLINE;
/*
 * Fast calculate variance gradient array
 */
YN_FINAL
void YnBlasGpuFastArrayVarianceGradientCal(float * arrayIn,
        float * gradientArr,
        float * meanArr,
        float * varianceArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceGradientArr)
YN_ALSWAY_INLINE;

/*
 * Fast calculate variance array
 */
YN_FINAL
void YnBlasGpuFastArrayVarianceCal(float * arrayIn,
        float * meanArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * varianceArr)
YN_ALSWAY_INLINE;

/*
 * Fast calculate mean array
 */
YN_FINAL
void YnBlasGpuFastArrayMeanCal(float * inArr,
        int32 batch,
        int32 filters,
        int32 spatial,
        float * meanArr)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuBiasScale(float *output,
        float *biases,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuBackwardScale(float *x_norm,
        float *delta,
        int batch,
        int num,
        int size,
        float *scale_updates)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuBiasAdd(float *output,
        float *biases,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

YN_FINAL
void YnBlasGpuBiasBackward(float *bias_updates,
        float *delta,
        int batch,
        int num,
        int size)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNBLASGPU_H */
