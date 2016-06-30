#ifndef YNCUDAGPU_H
#define YNCUDAGPU_H

#include "../YnStd.h"

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */
#define YN_GPU

/* Only excuted & called by device GPU */
#define YN_GPU_DEVICE   __device__

/* Excuted by GPU & called by CPU */
#define YN_GPU_GLOBAL       __global__

/* GPU share memory */
#define YN_GPU_SHARED_MEM   __shared__

#define YN_GPU_NUM_THREADS_IN_BLOCK     (512)
#define YN_GPU_MAX_NUM_THREAD_X         (65536)

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */
extern int gpuIndex;

/**************** Local Implement */

/**************** Implement */

/*
 *  Check cuda errors
 */
YN_FINAL void YnCudaCheckError(cudaError_t error);

/*
 *  Convert num to 3 dim
 */
YN_FINAL dim3 YnCudaGridSize(uint32 num);

/*
 *  Get cuda blas handle
 */
YN_FINAL cublasHandle_t YnCudaBlasHandle();

/*
 *  Get cuda make array
 */
YN_FINAL float * YnCudaMakeArray(float * cpuArr,
        uint32 num);

/*
 *  Get cuda make array
 */
YN_FINAL float * YnCudaMakeIntArray(uint32 num);

/*
 *  Generate cuda random array
 */
YN_FINAL float * YnCudaMakeRamdomArray(float * gpuArr,
        uint32 num);

/*
 *  Compare cuda array with cpu array
 */
YN_FINAL float * YnCudaCompareArray(float * gpuArr,
        float * cpuArr,
        uint32 num,
        char * error);

/*
 *  Free cuda array
 */
YN_FINAL void YnCudaFreeArray(float * gpuArr);

/*
 *  Push cuda array
 */
YN_FINAL void YnCudaArrayPushToGpu(float * gpuArr,
        float * cpuArr,
        uint32 num);

/*
 *  Pull cuda array
 */
YN_FINAL void YnCudaArrayPullFromGpu(float * gpuArr,
        float * cpuArr,
        uint32 num);


#ifdef __cplusplus
}
#endif
