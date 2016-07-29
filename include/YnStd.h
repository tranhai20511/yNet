#ifndef YNSTD_H
#define YNSTD_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */
#define YN_GPU
#define YN_OPENCV

#define YN_VIRTUAL
#define YN_FINAL

#define YN_INLINE           inline
#define YN_STATIC           static
#define YN_STATIC_INLINE    static inline
#define YN_EXTERN_C         extern "C"
#define YN_ALSWAY_INLINE    __attribute__((always_inline))


#define YN_CHAR_BUFF        (1024)
#define YN_CUS_NUM          (-1234)

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

/**************** Macro */
#define NULL (void *)0
#define mYnRetEqualPointer(_val, _exVal, _retVal)  if ((_val) == (_exVal)) (*(type)) = (_retVal)
#define mYnNullRetNull(_val)  if ((_val) == NULL) return NULL
#define mYnNullRetRet(_val, _ret)  if ((_val) == NULL) return _ret
#define mYnNullRet(_val, _ret)  if ((_val) == NULL) return
/*#define mYnErrorCheck(_ret)*/

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

#ifdef __cplusplus
}
#endif

#endif /* YNSTD_H */
