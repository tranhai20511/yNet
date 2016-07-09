//	File        :   YnActivationGpu.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   27-06-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "../YnActivationGpu.h"


/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
/*
 *  GPU: Calculation activation output value
 */
YN_GPU_DEVICE float _YnActivationLinear(float inVal)
{
    return mYnActivationLinear(inVal);
}

YN_GPU_DEVICE float _YnActivationLogistic(float inVal)
{
    return mYnActivationLogistic(inVal);
}

YN_GPU_DEVICE float _YnActivationLoggy(float inVal)
{
    return mYnActivationLoggy(inVal);
}

YN_GPU_DEVICE float _YnActivationRelu(float inVal)
{
    return mYnActivationRelu(inVal);
}

YN_GPU_DEVICE float _YnActivationElu(float inVal)
{
    return mYnActivationElu(inVal);
}

YN_GPU_DEVICE float _YnActivationRamp(float inVal)
{
    return mYnActivationRamp(inVal);
}

YN_GPU_DEVICE float _YnActivationLeaky(float inVal)
{
    return mYnActivationLeaky(inVal);
}

YN_GPU_DEVICE float _YnActivationTanh(float inVal)
{
    return mYnActivationTanh(inVal);
}

YN_GPU_DEVICE float _YnActivationPlse(float inVal)
{
    return mYnActivationPlse(inVal);
}

/*
 *  GPU: Calculation gradient value
 */
YN_GPU_DEVICE float _YnGradientLinear(float inVal)
{
    return mYnGradientLinear(inVal);
}

YN_GPU_DEVICE float _YnGradientLogistic(float inVal)
{
    return mYnGradientLogistic(inVal);
}

YN_GPU_DEVICE float _YnGradientLoggy(float inVal)
{
    return mYnGradientLoggy(inVal);
}

YN_GPU_DEVICE float _YnGradientRelu(float inVal)
{
    return mYnGradientRelu(inVal);
}

YN_GPU_DEVICE float _YnGradientElu(float inVal)
{
    return mYnGradientElu(inVal);
}

YN_GPU_DEVICE float _YnGradientRamp(float inVal)
{
    return mYnGradientRamp(inVal);
}

YN_GPU_DEVICE float _YnGradientLeaky(float inVal)
{
    return mYnGradientLeaky(inVal);
}

YN_GPU_DEVICE float _YnGradientTanh(float inVal)
{
    return mYnGradientTanh(inVal);
}

YN_GPU_DEVICE float _YnGradientPlse(float inVal)
{
    return mYnGradientPlse(inVal);
}

/**************** Implement */

YN_GPU_DEVICE float YnActivationGpuOutputCal(const float inVal ,
        const eYnActivationType actType)
{
    switch (actType)
    {
        case cYnActivationLogistic:
            return _YnActivationLogistic(inVal);
            break;
        case cYnActivationLoggy:
            return _YnActivationLoggy(inVal);
            break;
        case cYnActivationRelu:
            return _YnActivationRelu(inVal);
            break;
        case cYnActivationElu:
            return _YnActivationElu(inVal);
            break;
        case cYnActivationRamp:
            return _YnActivationRamp(inVal);
            break;
        case cYnActivationLinear:
            return _YnActivationLinear(inVal);
            break;
        case cYnActivationTanh:
            return _YnActivationTanh(inVal);
            break;
        case cYnActivationPlse:
            return _YnActivationPlse(inVal);
            break;
        case cYnActivationLeaky:
            return _YnActivationLeaky(inVal);
            break;
        default:
            return eYnRetInvalidParam;
            break;
    }

    return eYnRetOk;
}

YN_GPU_DEVICE float YnActivationGpuGradientCal(const float inVal ,
        const eYnActivationType actType)
{
    switch (actType)
    {
        case cYnActivationLogistic:
            return _YnGradientLogistic(inVal);
            break;
        case cYnActivationLoggy:
            return _YnGradientLoggy(inVal);
            break;
        case cYnActivationRelu:
            return _YnGradientRelu(inVal);
            break;
        case cYnActivationElu:
            return _YnGradientElu(inVal);
            break;
        case cYnActivationRamp:
            return _YnGradientRamp(inVal);
            break;
        case cYnActivationLinear:
            return _YnGradientLinear(inVal);
            break;
        case cYnActivationTanh:
            return _YnGradientTanh(inVal);
            break;
        case cYnActivationPlse:
            return _YnGradientPlse(inVal);
            break;
        case cYnActivationLeaky:
            return _YnGradientLeaky(inVal);
            break;
        default:
            return eYnRetInvalidParam;
            break;
    }

    return eYnRetOk;
}

YN_GPU_GLOBAL void YnActivationGpuOutputArrayCal(float * array,
        uint32 num,
        eYnActivationType actType)
{
    int32 idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (idx < num)
        array[idx] = YnActivationGpuOutputCal(array[idx], actType);
}

eYnRetCode YnActivationCallGpuOutputArrayCal(float * array,
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

YN_GPU_GLOBAL void YnActivationGpuGradientArrayCal(float * array,
        uint32 num,
        eYnActivationType actType,
        float * gradientArray)
{
    int32 idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (idx)
        gradientArray[idx] *= YnActivationGpuGradientCal(array[idx], actType);
}

eYnRetCode YnActivationCallGpuGradientArrayCal(const float * array,
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
