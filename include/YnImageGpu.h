#ifndef YNIMAGEGPU_H
#define YNIMAGEGPU_H

#include "../YnImage.h"

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
void YnImageGpuImage2Col(float *im,
        int channels,
        int height,
        int width,
        int ksize,
        int stride,
        int pad,
        float *data_col)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageGpuCol2Image(float *data_col,
        int channels,
        int height,
        int width,
        int ksize,
        int stride,
        int pad,
        float *data_im)
YN_ALSWAY_INLINE;



#ifdef __cplusplus
}
#endif

#endif /* YNIMAGEGPU_H */
