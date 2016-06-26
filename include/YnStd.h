#ifndef YNSTD_H
#define YNSTD_H

#include "cuda.h"
#include "math.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */
#define YN_VIRTUAL
#define YN_FINAL
#define YN_GPU

#define YN_GPU_DEVICE       __device__
#define YN_GPU_GLOBAL       __global__

#define YN_STATIC           static
#define YN_STATIC_INLINE    static inline

/**************** Macro */
#define mYnRetEqual(_val, _exVal, _retVal)  if ((_val) == (_exVal)) return (_retVal)

/**************** Enum */

typedef enum eYnReturnCode {
    eYnReturnOk,
    eYnReturnGenericErr,
    eYnReturnOor,
    eYnReturnNull,
    eYnReturnInvalidParam,
}eYnReturnCode;

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */



#ifdef __cplusplus
}
#endif
