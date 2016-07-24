//	File        :   YnBlas.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   28-06-2016
//	Author      :   haittt

#include "assert.h"
#include <stdlib.h>
#include <time.h>

#include "../YnCudaGpu.h"
#include "../YnUtil.h"
#include "../YnBlas.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnCudaCheckError(cudaError_t errorStatus)
{
    char buffer[YN_CHAR_BUFF] = {0};
    cudaError_t status = cudaGetLastError();

    if (errorStatus != cudaSuccess)
    {
        const char *s = cudaGetErrorString(errorStatus);

        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, YN_CHAR_BUFF, "CUDA Error: %s", s);
        YnUtilError(buffer);
    }

    memset(buffer, 0, YN_CHAR_BUFF * sizeof(char))
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(error);

        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, YN_CHAR_BUFF, "CUDA Error Prev: %s", s);
        YnUtilError(buffer);
    }
}

dim3 YnCudaGridSize(uint32 num)
{
    dim3 d = {0};
    int32 k = (num - 1) / YN_GPU_NUM_THREADS_IN_BLOCK + 1;
    int32 x = k;
    int32 y = 1;

    if (x >= YN_GPU_MAX_NUM_THREAD_X)
    {
        x = ceil( sqrt(k) );
        y = (num - 1) / (x * YN_GPU_NUM_THREADS_IN_BLOCK) + 1;
    }

    d = { x, y, 1 };

    return d;
}

cublasHandle_t YnCudaBlasHandle()
{
    static bool init = false;
    static cublasHandle_t handle;

    if (!init)
    {
        cublasCreate(&handle);
        init = true;
    }

    return handle;
}

float * YnCudaMakeArray(float * cpuArr,
        uint32 num)
{
    float * gpuArr = NULL;

    uint32 size = sizeof(float) * num;

    cudaError_t status = cudaMalloc((void **) &gpuArr, size);
    YnCudaCheckError(status);

    if (cpuArr)
    {
        status = cudaMemcpy(gpuArr, cpuArr, size, cudaMemcpyHostToDevice);
        YnCudaCheckError(status);
    }

    if (!gpuArr)
        YnUtilError("Cuda malloc failed\n");

    return gpuArr;
}

float * YnCudaMakeIntArray(uint32 num)
{
    int * gpuArr = NULL;

    uint32 size = sizeof(int) * num;

    cudaError_t status = cudaMalloc((void **) &gpuArr, size);

    YnCudaCheckError(status);

    return gpuArr;
}

float * YnCudaMakeRamdomArray(float * gpuArr,
        uint32 num)
{
    static curandGenerator_t gen;
    static bool init = false;

    if (!init)
    {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(0));
        init = true;
    }

    curandGenerateUniform(gen, gpuArr, num);
    YnCudaCheckError(cudaPeekAtLastError());
}

float * YnCudaCompareArray(float * gpuArr,
        float * cpuArr,
        uint32 num,
        char * errorS)
{
    float *tmp = calloc(num, sizeof(float));
    float errAcc = 0;

    YnCudaArrayPullFrom(gpuArr, tmp, num);
    YnBlasArrayAxpyValueSet(tmp, cpuArr, num, 1, 1, -1);

    errAcc = YnBlasArrayDotValueSet(tmp, tmp, num, 1, 1);

    printf("Error %s: %f\n", errorS, sqrt(errAcc / num));
    YnUtilFree(tmp);

    return errAcc;
}

void YnCudaFreeArray(float * gpuArr)
{
    cudaError_t status = cudaFree(gpuArr);

    YnCudaCheckError(status);
}

void YnCudaArrayPushToGpu(float * gpuArr,
        float * cpuArr,
        uint32 num)
{
    uint32 size = sizeof(float) * num;

    cudaError_t status = cudaMemcpy(gpuArr, cpuArr, size, cudaMemcpyHostToDevice);

    YnCudaCheckError(status);
}

void YnCudaArrayPullFromGpu(float * gpuArr,
        float * cpuArr,
        uint32 num)
{
    uint32 size = sizeof(float) * num;

    cudaError_t status = cudaMemcpy(cpuArr, gpuArr, size, cudaMemcpyDeviceToHost);

    check_YnUtilError(status);
}
