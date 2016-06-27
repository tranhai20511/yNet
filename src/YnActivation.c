//	File        :   YnActivation.c
//	Brief       :   Implement activation function methods.
//	DD-MM_YYYY  :   26-06-2016
//	Author      :   haittt

#include "../YnActivation.h"


/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

eYnReturnCode YnActivationTypeFromStringGet(char * string, eYnActivationType* type)
{
    if (!string || !type)
        return eYnReturnNull;

    mYnRetEqual(strcmp(string, "logistic"), 0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "loggy"),    0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "relu"),     0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "elu"),      0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "plse"),     0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "linear"),   0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "ramp"),     0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "leaky"),    0, cYnActivationLogistic);
    mYnRetEqual(strcmp(string, "tanh"),     0, cYnActivationLogistic);

    return eYnReturnInvalidParam;
}

YN_FINAL char * YnActivationTypeStringGet(eYnActivationType type)
{
    switch (type)
    {
        case cYnActivationLogistic:
            return "logistic";
        case cYnActivationLoggy:
            return "loggy";
        case cYnActivationRelu:
            return "relu";
        case cYnActivationElu:
            return "elu";
        case cYnActivationRamp:
            return "ramp";
        case cYnActivationLinear:
            return "linear";
        case cYnActivationTanh:
            return "tanh";
        case cYnActivationPlse:
            return "plse";
        case cYnActivationLeaky:
            return "leaky";
        default:
            return "unknown";
            break;
    }

    return "unknown";
}
