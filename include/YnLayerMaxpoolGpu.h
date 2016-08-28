#ifndef YNLAYERMAXPOOLGPU_H
#define YNLAYERMAXPOOLGPU_H

#include "../YnLayerMaxpool.h"
#include "../YnCudaGpu.h"
#include "../YnBlasGpu.h"
#include "../YnImageGpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
void YnLayerMaxpoolGpuForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerMaxpoolGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERMAXPOOLGPU_H */
