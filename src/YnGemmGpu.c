//	File        :   YnGemmGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   31-06-2016
//	Author      :   haittt

#include "../include/YnGemmGpu.h"
#include "../include/YnCudaGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_STATIC
void YnGemmOnGpu(int TA,
        int TB,
        int M,
        int N,
        int K,
        float ALPHA,
        float *A_gpu,
        int lda,
        float *B_gpu,
        int ldb,
        float BETA,
        float *C_gpu,
        int ldc)
{
    cublasHandle_t handle = YnCudaBlasHandle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                        (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);

    YnCudaCheckError(status);
}

void YnGemmGpu(int TA,
        int TB,
        int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float BETA,
        float *C,
        int ldc)
{
    float * A_gpu = YnCudaMakeArray(A, (TA ? (lda * K) : (lda * M)));
    float * B_gpu = YnCudaMakeArray(B, (TB ? (ldb * N) : (ldb * K)));
    float * C_gpu = YnCudaMakeArray(C, ldc * M);

    YnGemmOnGpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    YnCudaArrayPullFromGpu(C_gpu, C, ldc*M);
    YnCudaFreeArray(A_gpu);
    YnCudaFreeArray(B_gpu);
    YnCudaFreeArray(C_gpu);
}
