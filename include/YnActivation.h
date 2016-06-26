#ifndef YNACTIVATIONS_H
#define YNACTIVATIONS_H

#include "../YnStd.h"


#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/*
 *	Activation function types
 */
typedef enum eYnActivationType {
	cYnActivationLogistic,
	cYnActivationRelu,
	cYnActivationRelie,
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
