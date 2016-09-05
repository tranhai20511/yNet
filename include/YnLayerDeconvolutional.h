#ifndef YNLAYERDECONVOLUTIONAL_H
#define YNLAYERDECONVOLUTIONAL_H

#include "YnLayer.h"
#include "YnNetwork.h"
#include "YnActivation.h"
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
tYnLayer YnLayerDeconvolutionalMake(int batchNum,
        int height,
        int width,
        int channel,
        int num,
        int size,
        int stride,
        eYnActivationType activation)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalForward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalBackward(tYnLayer layer,
        tYnNetworkState netState)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalUpdate(tYnLayer layer,
        float learningRate,
        float momentum,
        float decay)
YN_ALSWAY_INLINE;

YN_FINAL
void YnLayerDeconvolutionalResize(tYnLayer* layer,
        int width,
        int height)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerDeconvolutionalImageGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnLayerDeconvolutionalGradientGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
int YnLayerDeconvolutionalOutHeightGet(tYnLayer layer)
YN_ALSWAY_INLINE;

YN_FINAL
int YnLayerDeconvolutionalOutWidthGet(tYnLayer layer)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYERDECONVOLUTIONAL_H */
