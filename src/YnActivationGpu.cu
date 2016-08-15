//	File        :   YnActivationGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   27-06-2016
//	Author      :   haittt

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "../include/YnActivationGpu.h"
}

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
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
YN_GPU_DEVICE float YnActivationGpuKernelCal(const float inVal ,
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

YN_GPU_GLOBAL void YnActivationGpuKernelOutputArrayCal(float * array,
        uint32 num,
        eYnActivationType actType)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < num)
        array[i] = YnActivationGpuKernelCal(array[i], actType);
}

YN_EXTERN_C
void YnActivationGpuOutputArrayCal(float * array,
        uint32 num,
        eYnActivationType actType)
{
    YnActivationGpuKernelOutputArrayCal<<<cuda_gridsize(n), BLOCK>>>(array, num, actType);
    YnCudaCheckError(cudaPeekAtLastError());
}

YN_GPU_GLOBAL void YnActivationGpuKernelGradientArrayCal(float * array,
        uint32 num,
        eYnActivationType actType,
        float * gradientArray)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < num)
        gradientArray[i] *= YnActivationGpuGradientCal(array[i], actType);
}

YN_EXTERN_C
void YnActivationGpuGradientArrayCal(float * array,
        uint32 num,
        eYnActivationType actType,
        float * gradientArray)
{
    YnActivationGpuKernelGradientArrayCal<<<cuda_gridsize(n), BLOCK>>>(array, num, actType, gradientArray);
    YnCudaCheckError(cudaPeekAtLastError());
}
