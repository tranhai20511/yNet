#ifndef YNLAYERCONVOLUTIONAL_H
#define YNLAYERCONVOLUTIONAL_H

#include "YnLayer.h"
#include "YnNetwork.h"
#include "YnActivation.h"
#include "YnLayer.h"
#include "YnImage.h"

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
tYnLayer YnLayerConvolutionalMake(int batchNum,
        int height,
        int width,
        int channel,
        int num,
        int size,
        int stride,
        int pad,
        eYnActivationType activation,
        int batchNormalize,
        int binary)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalUpdate(tYnLayer layer,
        int32 batch,
        float learningRate,
        float momentum,
        float decay)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalDenormalize(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalResize(tYnLayer* layer,
        int width,
        int height)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage * YnLayerConvolutionalVisualize(tYnLayer layer,
        char * window,
        tYnImage * filters)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerConvolutionalImageGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerConvolutionalGradientGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerConvolutionalFilterGet(tYnLayer layer,
        int i)
YN_ALSWAY_INLINE;

YN_FINAL
int YnLayerConvolutionalOutHeightGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
int YnLayerConvolutionalOutWidthGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalFiltersRescale(tYnLayer layer,
        float scale,
        float trans)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerConvolutionalFiltersRgbgr(tYnLayer layer)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERCONVOLUTIONAL_H */
