#ifndef YNACTIVATIONGPU_H
#define YNACTIVATIONGPU_H

#include "../YnActivation.h"


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

#ifdef GPU

/*
 *  GPU: Calculation activation output value
 */
YN_GPU_DEVICE
float YnActivationGpuOutputCal(const float inVal ,
        const eYnActivationType actType)
YN_ALSWAY_INLINE;

/*
 *  GPU: Calculation gradient value
 */
YN_GPU_DEVICE
float YnActivationGpuGradientCal(const float inVal ,
        const eYnActivationType actType)
YN_ALSWAY_INLINE;

/*
 *  GPU: Calculation activation output value for array
 */
YN_GPU_GLOBAL
void YnActivationGpuOutputArrayCal(float * array,
        uint32 num,
        eYnActivationType actType)
YN_ALSWAY_INLINE;

YN_FINAL
eYnRetCode YnActivationCallGpuOutputArrayCal(float * array,
        const uint32 num,
        const eYnActivationType actType)
YN_ALSWAY_INLINE;

/*
 *  GPU: Calculation gradient value for array
 */
YN_GPU_GLOBAL
void YnActivationGpuGradientArrayCal(float * array,
        uint32 num,
        eYnActivationType actType,
        float * gradientArray)
YN_ALSWAY_INLINE;

YN_FINAL
eYnRetCode YnActivationCallGpuGradientArrayCal(const float * array,
        const uint32 num,
        const eYnActivationType actType,
        float * gradientArray)
YN_ALSWAY_INLINE;

#endif


#ifdef __cplusplus
}
#endif
