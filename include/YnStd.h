#ifndef YNSTD_H
#define YNSTD_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../YnCuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

#define GPU

#define YN_VIRTUAL
#define YN_FINAL
#define YN_GPU

/* Only excuted & called by device GPU */
#define YN_GPU_DEVICE   __device__

/* Excuted by GPU & called by CPU */
#define YN_GPU_GLOBAL       __global__

/* GPU share memory */
#define YN_GPU_SHARED_MEM   __shared__

#define YN_STATIC           static
#define YN_STATIC_INLINE    static inline


#define YN_GPU_NUM_THREADS_IN_BLOCK  (512)

/**************** Macro */
#define mYnRetEqualPointer(_val, _exVal, _retVal)  if ((_val) == (_exVal)) (*(type)) = (_retVal)

/**************** Typedef */

typedef unsigned char       byte;
typedef unsigned char       uint8;
typedef signed short int    int16;
typedef unsigned short int  uint16;
typedef signed int          int32;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;
typedef long long           int64;
typedef unsigned char       uchar;


/**************** Enum */

typedef enum eYnRetCode {
    eYnRetOk,
    eYnRetGenericErr,
    eYnRetOor,
    eYnRetNull,
    eYnRetInvalidParam,
}eYnRetCode;

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */



#ifdef __cplusplus
}
#endif
