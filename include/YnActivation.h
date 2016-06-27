#ifndef YNACTIVATIONS_H
#define YNACTIVATIONS_H

#include "../YnStd.h"


#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**/
#define mYnActivationLinear (x)   (x)

#define mYnGradientLinear (x)     (1)

#define mYnActivationRelu (x)\
    ((x) * ((x) > 0))

#define mYnGradientRelu (x)\
    ((x) > 0)

#define mYnActivationElu (x)\
    ((x) * ((x) >= 0) + (exp((x)) - 1) * ((x) < 0))

#define mYnGradientElu (x)\
    (((x) >= 0) + ((x) + 1) * ((x) < 0))

#define mYnActivationRamp (x)\
    ((x) * ((x) > 0) + 0.1 * (x))

#define mYnGradientRamp (x)\
    (((x) > 0) + 0.1)

#define mYnActivationLeaky (x)\
    ((x) * ((x) > 0) + 0.1 * (x) *((x) <= 0))

#define mYnGradientLeaky (x)\
    (((x) > 0) + 0.1 * ((x) <= 0))

#define mYnActivationTanh (x)\
    ((exp(2 * (x)) - 1) / (exp(2 * (x)) + 1))

#define mYnGradientTanh (x)\
    (1 - (x) * (x))

#define mYnActivationPlse (x)\
    (((x) < -4) ? (.01 * ((x) + 4)) :\
    (((x) > 4) ? (.01 * ((x) - 4) + 1) : (.125 * (x) + .5)))

#define mYnGradientPlse (x)\
        (((x) < 0 || (x) > 1) ? .01 : .125)

#define mYnActivationLogistic (x)\
    (1. / (1. + exp(-(x))))

#define mYnGradientLogistic (x)\
    ((x) * (1 - (x)))

#define mYnActivationLoggy (float x)\
    (2. / (1. + exp(-(x))) - 1)

#define mYnGradientLoggy (x)\
    (2 * (1 - (((x) + 1.) / 2.)) * (((x) + 1.) / 2.))

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
YN_FINAL eYnActivationType YnActivationTypeFromStringGet(char * string);

/*
 *	Get Activation function string types
 */
YN_FINAL char * YnActivationTypeStringGet(eYnActivationType type);


#ifdef __cplusplus
}
#endif
