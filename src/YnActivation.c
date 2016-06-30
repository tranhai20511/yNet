//	File        :   YnActivation.c
//	Brief       :   Implement methods.
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

eYnRetCode YnActivationTypeFromStringGet(char * string,
        eYnActivationType* type)
{
    if (!string || !type)
        return eYnRetNull;

    mYnRetEqualPointer(strcmp(string, "logistic"), 0, cYnActivationLogistic);
    mYnRetEqualPointer(strcmp(string, "loggy"),    0, cYnActivationLoggy);
    mYnRetEqualPointer(strcmp(string, "relu"),     0, cYnActivationRelu);
    mYnRetEqualPointer(strcmp(string, "elu"),      0, cYnActivationElu);
    mYnRetEqualPointer(strcmp(string, "plse"),     0, cYnActivationPlse);
    mYnRetEqualPointer(strcmp(string, "linear"),   0, cYnActivationLinear);
    mYnRetEqualPointer(strcmp(string, "ramp"),     0, cYnActivationRamp);
    mYnRetEqualPointer(strcmp(string, "leaky"),    0, cYnActivationLeaky);
    mYnRetEqualPointer(strcmp(string, "tanh"),     0, cYnActivationTanh);

    return eYnRetOk;
}

char * YnActivationTypeStringGet(eYnActivationType type)
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

eYnRetCode YnActivationOutputCal(const float inVal ,
        const eYnActivationType actType,
        float * output)
{
    if (!output)
        return eYnRetNull;

    *output = 0;

    switch (actType)
    {
        case cYnActivationLogistic:
            *output = mYnActivationLogistic(inVal);
            break;
        case cYnActivationLoggy:
            *output = mYnActivationLoggy(inVal);
            break;
        case cYnActivationRelu:
            *output = mYnActivationRelu(inVal);
            break;
        case cYnActivationElu:
            *output = mYnActivationElu(inVal);
            break;
        case cYnActivationRamp:
            *output = mYnActivationRamp(inVal);
            break;
        case cYnActivationLinear:
            *output = mYnActivationLinear(inVal);
            break;
        case cYnActivationTanh:
            *output = mYnActivationTanh(inVal);
            break;
        case cYnActivationPlse:
            *output = mYnActivationPlse(inVal);
            break;
        case cYnActivationLeaky:
            *output = mYnActivationLeaky(inVal);
            break;
        default:
            return eYnRetInvalidParam;
            break;
    }

    return eYnRetOk;
}

eYnRetCode YnActivationGradientCal(const float inVal ,
        const eYnActivationType actType,
        float * gradient)
{
    if (!gradient)
        return eYnRetNull;

    *gradient = 0;

    switch (actType)
    {
        case cYnActivationLogistic:
            *gradient = mYnGradientLogistic(inVal);
            break;
        case cYnActivationLoggy:
            *gradient = mYnGradientLoggy(inVal);
            break;
        case cYnActivationRelu:
            *gradient = mYnGradientRelu(inVal);
            break;
        case cYnActivationElu:
            *gradient = mYnGradientElu(inVal);
            break;
        case cYnActivationRamp:
            *gradient = mYnGradientRamp(inVal);
            break;
        case cYnActivationLinear:
            *gradient = mYnGradientLinear(inVal);
            break;
        case cYnActivationTanh:
            *gradient = mYnGradientTanh(inVal);
            break;
        case cYnActivationPlse:
            *gradient = mYnGradientPlse(inVal);
            break;
        case cYnActivationLeaky:
            *gradient = mYnGradientLeaky(inVal);
            break;
        default:
            return eYnRetInvalidParam;
            break;
    }

    return eYnRetOk;
}

eYnRetCode YnActivationOutputArrayCal(float * array,
        const uint32 num,
        const eYnActivationType actType)
{
    if (!array)
        return eYnRetNull;

    int32 idx = 0;

    for(idx = 0; idx < num; idx ++)
    {
        YnActivationGradientCal(array[idx], actType, &(array[idx]));
    }

    return eYnRetOk;
}


eYnRetCode YnActivationGradientArrayCal(const float * array,
        const uint32 num,
        const eYnActivationType actType,
        float * gradientArray)
{
    if (!array)
        return eYnRetNull;

    int32 idx = 0;

    for(idx = 0; idx < num; idx ++)
    {
        YnActivationGradientCal(array[idx], actType, &(gradientArray[idx]));
    }

    return eYnRetOk;
}
