#ifndef YNCUDAGPU_H
#define YNCUDAGPU_H

#include "../YnCuda.h"

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */
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

/**************** Local Implement */

/**************** Implement */
/*
 *  Check cuda errors
 */
YN_FINAL
void YnCudaCheckError(cudaError_t error)
YN_ALSWAY_INLINE;

/*
 *  Convert num to 3 dim
 */
YN_FINAL
dim3 YnCudaGridSize(uint32 num)
YN_ALSWAY_INLINE;

/*
 *  Get cuda blas handle
 */
YN_FINAL
cublasHandle_t YnCudaBlasHandle()
YN_ALSWAY_INLINE;

/*
 *  Get cuda make array
 */
YN_FINAL
float * YnCudaMakeArray(float * cpuArr,
        uint32 num)
YN_ALSWAY_INLINE;

/*
 *  Get cuda make array
 */
YN_FINAL
float * YnCudaMakeIntArray(uint32 num)
YN_ALSWAY_INLINE;

/*
 *  Generate cuda random array
 */
YN_FINAL float * YnCudaMakeRamdomArray(float * gpuArr,
        uint32 num)
YN_ALSWAY_INLINE;

/*
 *  Compare cuda array with cpu array
 */
YN_FINAL
float * YnCudaCompareArray(float * gpuArr,
        float * cpuArr,
        uint32 num,
        char * error)
YN_ALSWAY_INLINE;

/*
 *  Free cuda array
 */
YN_FINAL
void YnCudaFreeArray(float * gpuArr)
YN_ALSWAY_INLINE;

/*
 *  Push cuda array
 */
YN_FINAL
void YnCudaArrayPushToGpu(float * gpuArr,
        float * cpuArr,
        uint32 num)
YN_ALSWAY_INLINE;

/*
 *  Pull cuda array
 */
YN_FINAL
void YnCudaArrayPullFromGpu(float * gpuArr,
        float * cpuArr,
        uint32 num)
YN_ALSWAY_INLINE;

YN_FINAL
void YnCudaRandomArray(float * gpuArr,
        uint32 num)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNCUDAGPU_H */
