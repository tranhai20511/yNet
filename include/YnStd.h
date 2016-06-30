#ifndef YNSTD_H
#define YNSTD_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "../YnCuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */
#define GPU

#define YN_VIRTUAL
#define YN_FINAL

#define YN_STATIC           static
#define YN_STATIC_INLINE    static inline
#define YN_EXTERN_C         extern "C"


#define YN_CHAR_BUFF        (1024)

/**************** Macro */
#define mYnRetEqualPointer(_val, _exVal, _retVal)  if ((_val) == (_exVal)) (*(type)) = (_retVal)
/*#define mYnErrorCheck(_ret)*/

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
