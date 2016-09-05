#ifndef YNACTIVATION_H
#define YNACTIVATION_H

#include "YnStd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

#define mYnActivationLinear(inVal)   (inVal)

#define mYnGradientLinear(inVal)     (1)

#define mYnActivationRelu(inVal) \
    ((inVal) * ((inVal) > 0))

#define mYnGradientRelu(inVal) \
    ((inVal) > 0)

#define mYnActivationElu(inVal) \
    ((inVal) * ((inVal) >= 0) + (exp((inVal)) - 1) * ((inVal) < 0))

#define mYnGradientElu(inVal) \
    (((inVal) >= 0) + ((inVal) + 1) * ((inVal) < 0))

#define mYnActivationRamp(inVal) \
    ((inVal) * ((inVal) > 0) + 0.1 * (inVal))

#define mYnGradientRamp(inVal) \
    (((inVal) > 0) + 0.1)

#define mYnActivationLeaky(inVal) \
    ((inVal) * ((inVal) > 0) + 0.1 * (inVal) *((inVal) <= 0))

#define mYnGradientLeaky(inVal) \
    (((inVal) > 0) + 0.1 * ((inVal) <= 0))

#define mYnActivationTanh(inVal) \
    ((exp(2 * (inVal)) - 1) / (exp(2 * (inVal)) + 1))

#define mYnGradientTanh(inVal) \
    (1 - (inVal) * (inVal))

#define mYnActivationPlse(inVal) \
    (((inVal) < -4) ? (.01 * ((inVal) + 4)) :\
    (((inVal) > 4) ? (.01 * ((inVal) - 4) + 1) : (.125 * (inVal) + .5)))

#define mYnGradientPlse(inVal) \
        (((inVal) < 0 || (inVal) > 1) ? .01 : .125)

#define mYnActivationLogistic(inVal) \
    (1. / (1. + exp(-(inVal))))

#define mYnGradientLogistic(inVal) \
    ((inVal) * (1 - (inVal)))

#define mYnActivationLoggy(inVal) \
    (2. / (1. + exp(-(inVal))) - 1)

#define mYnGradientLoggy(inVal) \
    (2 * (1 - (((inVal) + 1.) / 2.)) * (((inVal) + 1.) / 2.))

/**************** Enum */
/*
 *	Activation function types
 */
typedef enum eYnActivationType {
	cYnActivationLogistic,
	cYnActivationRelu,
	cYnActivationLinear,
	cYnActivationRamp,
	cYnActivationTanh,
	cYnActivationPlse,
	cYnActivationLeaky,
	cYnActivationElu,
	cYnActivationLoggy,
}eYnActivationType;


/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

/*
 *	Get Activation function types from string
 */
YN_FINAL
eYnRetCode YnActivationTypeFromStringGet(char * string,
        eYnActivationType* type)
YN_ALSWAY_INLINE;

/*
 *	Get Activation function string types
 */
YN_FINAL
char * YnActivationTypeStringGet(eYnActivationType type)
YN_ALSWAY_INLINE;

/*
 *  Calculation activation output value
 */
YN_FINAL
eYnRetCode YnActivationOutputCal(const float inVal ,
        const eYnActivationType actType,
        float * output)
YN_ALSWAY_INLINE;

/*
 *  Calculation gradient value
 */
YN_FINAL
eYnRetCode YnActivationGradientCal(const float inVal ,
        const eYnActivationType actType,
        float * gradient)
YN_ALSWAY_INLINE;

/*
 *  Calculation activation output value for array
 */
YN_FINAL
eYnRetCode YnActivationOutputArrayCal(float * array,
        const uint32 num,
        const eYnActivationType actType)
YN_ALSWAY_INLINE;

/*
 *  Calculation gradient value for array
 */
YN_FINAL
eYnRetCode YnActivationGradientArrayCal(const float * array,
        const uint32 num,
        const eYnActivationType actType,
        float * gradientArray)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNACTIVATION_H */
