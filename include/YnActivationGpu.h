#ifndef YNACTIVATIONGPU_H
#define YNACTIVATIONGPU_H

#include "YnActivation.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef YN_GPU

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
/*
 *  GPU: Calculation activation output value for array
 */
YN_FINAL
eYnRetCode YnActivationGpuOutputArrayCal(float * array,
        uint32 num,
        eYnActivationType actType)
YN_ALSWAY_INLINE;

/*
 *  GPU: Calculation gradient value for array
 */
YN_FINAL
void YnActivationGpuGradientArrayCal(float * array,
        uint32 num,
        eYnActivationType actType,
        float * gradientArray)
YN_ALSWAY_INLINE;

#endif

#ifdef __cplusplus
}
#endif

#endif /* YNACTIVATIONGPU_H */
