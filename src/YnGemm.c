//	File        :   YnGemm.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   31-06-2016
//	Author      :   haittt

#include "../include/YnGemm.h"
#include "../include/YnUtil.h"
#include "../include/YnCuda.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
YN_STATIC
void YnGemmNn(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
YN_ALSWAY_INLINE;

YN_STATIC
void YnGemmNt(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
YN_ALSWAY_INLINE;

YN_STATIC
void YnGemmTn(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
YN_ALSWAY_INLINE;

YN_STATIC
void YnGemmTt(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
YN_ALSWAY_INLINE;

/**************** Implement */
YN_STATIC
void YnGemmNn(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
{
    int i, j, k;
    register float A_PART;

    for (i = 0; i < M; i ++)
    {
        for (k = 0; k < K; k ++)
        {
            A_PART = ALPHA*A[i * lda + k];

            for (j = 0; j < N; j ++)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

YN_STATIC
void YnGemmNt(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
{
    int i, j, k;
    register float sum;

    for (i = 0; i < M; i ++)
    {
        for (j = 0; j < N; j ++)
        {
            sum = 0;
            for (k = 0; k < K; k ++)
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }

            C[i * ldc + j] += sum;
        }
    }
}

YN_STATIC
void YnGemmTn(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
{
    int i, j, k;
    register float A_PART;

    for (i = 0; i < M; i ++)
     {
        for (k = 0; k < K; k ++)
        {
            A_PART = ALPHA * A[k * lda + i];

            for (j = 0; j < N; j ++)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

YN_STATIC
void YnGemmTt(int M,
        int N,
        int K,
        float ALPHA,
        float *A,
        int lda,
        float *B,
        int ldb,
        float *C,
        int ldc)
{
    int i, j, k;
    register float sum;

    for (i = 0; i < M; i ++)
    {
        for (j = 0; j < N; j ++)
        {
            sum = 0;

            for (k = 0; k < K; k ++)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }

            C[i * ldc + j] += sum;
        }
    }
}

float * YnGemmRandomMatrix(int rows,
        int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));

    for (i = 0; i < rows*cols; i ++)
    {
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void YnGemmTimeRandomMatrix(int TA,
        int TB,
        int m,
        int k,
        int n)
{
    float *a;
    float *b;
    float *c;
    int lda;
    int ldb;
    int i;
    clock_t start, end;

    if (!TA)
        a = YnGemmRandomMatrix(m, k);
    else
        a = YnGemmRandomMatrix(k, m);

    lda = (!TA) ? k : m;
    if (!TB)
        b = YnGemmRandomMatrix(k, n);
    else
        b = YnGemmRandomMatrix(n, k);

    ldb = (!TB) ? n : k;
    c = YnGemmRandomMatrix(m,n);


    start = clock();
    for (i = 0; i<10; i ++)
    {
        YnGemm(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
    }

    end = clock();

    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    YnUtilFree(a);
    YnUtilFree(b);
    YnUtilFree(c);
}

void YnGemm(int TA,
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
    int i, j;

    for (i = 0; i < M; i ++)
    {
        for (j = 0; j < N; j ++)
        {
            C[i * ldc + j] *= BETA;
        }
    }

    if (!TA && !TB)
        YnGemmNn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        YnGemmTn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)
        YnGemmNt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        YnGemmTt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}
