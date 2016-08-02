#ifndef YNIMAGE_H
#define YNIMAGE_H

#include "../YnBBox.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */
typedef struct tYnImage{
    int height;
    int width;
    int channel;
    float * data;
} tYnImage;

/**************** Local variables */


/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

/*
 * Get color value
 */
YN_FINAL
float YnImageColorGet(uint32 channel,
        int x,
        int max)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageDrawLabel(tYnImage image,
        int x, int max)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageDrawBox(tYnImage a,
        int x1,
        int y1,
        int x2,
        int y2,
        float r,
        float g,
        float b)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageDrawBoxWidth(tYnImage a,
        int x1,
        int y1,
        int x2,
        int y2,
        int w,
        float r,
        float g,
        float b)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageDrawBbox(tYnImage a,
        tYnBBox bbox,
        int w,
        float r,
        float g,
        float b)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageDrawDetections(tYnImage im,
        int num,
        float thresh,
        tYnBBox *boxes,
        float **probs,
        char **names,
        tYnImage *labels,
        int classes)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageFlip(tYnImage a)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageDistance(tYnImage a,
        tYnImage b)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageEmbed(tYnImage source,
        tYnImage dest,
        int dx,
        int dy)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageCollapseLayers(tYnImage source,
        int border)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageConstrain(tYnImage im)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageNormalize(tYnImage p)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageCopy(tYnImage p)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageRgbgr(tYnImage im)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageShow(tYnImage p,
        const char *name)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageSave(tYnImage im,
        const char *name)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageShowLayers(tYnImage p,
        char *name)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageShowCollapsed(tYnImage p,
        char *name)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageMakeEmpty(int w,
        int h,
        int c)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageMake(int w,
        int h,
        int c)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageMakeRandom(int w,
        int h,
        int c)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageFloatToImage(int w,
        int h,
        int c,
        float *data)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageRotate(tYnImage im,
        float rad)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageTranslate(tYnImage m,
        float s)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImgaeScale(tYnImage m,
        float s)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageCrop(tYnImage im,
        int dx,
        int dy,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
float YnImageThreeWayMax(float a,
        float b,
        float c)
YN_ALSWAY_INLINE;

YN_FINAL
float YnImageThreeWayMin(float a,
        float b,
        float c)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageRgbToHsv(tYnImage im)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageHsvToRgb(tYnImage im)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageGrayscale(tYnImage im)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageThreshold(tYnImage im,
        float thresh)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageBlend(tYnImage fore,
        tYnImage back,
        float alpha)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageScaleChannel(tYnImage im,
        int c,
        float v)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageSaturate(tYnImage im,
        float sat)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageExposure(tYnImage im,
        float sat)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageSaturateExposure(tYnImage im,
        float sat,
        float exposure)
YN_ALSWAY_INLINE;

YN_FINAL
float YnImageBilinearInterpolate(tYnImage im,
        float x,
        float y,
        int c)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageResize(tYnImage im,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageTestResize(char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageLoadStb(char *filename,
        int channels)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageLoad(char *filename,
        int w,
        int h,
        int c)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageLoadColor(char *filename,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageGetLayer(tYnImage m,
        int l)
YN_ALSWAY_INLINE;

YN_FINAL
float YnImageGetPixel(tYnImage m,
        int x,
        int y,
        int c)
YN_ALSWAY_INLINE;

YN_FINAL
float YnImageGetPixelExtend(tYnImage m,
        int x,
        int y,
        int c)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageSetPixel(tYnImage m,
        int x,
        int y,
        int c,
        float val)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageAddPixel(tYnImage m,
        int x,
        int y,
        int c,
        float val)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImagePrint(tYnImage m)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageCollapseVert(tYnImage *ims,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageCollapseHorz(tYnImage *ims,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageShow(tYnImage *ims,
        int n,
        char *window)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageFree(tYnImage m)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageImage2Col(float* data_col,
         int channels,
         int height,
         int width,
         int ksize,
         int stride,
         int pad,
         float* data_im)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageCol2Image(float* data_col,
         int channels,
         int height,
         int width,
         int ksize,
         int stride,
         int pad,
         float* data_im)
YN_ALSWAY_INLINE;

/* OPENCV Image */
#ifdef YN_OPENCV

YN_FINAL
void YnImageCvShow(tYnImage p,
        const char *name)
YN_ALSWAY_INLINE;

YN_FINAL
void YnImageSaveImage(tYnImage p,
        char *name)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageCvIplToimage(IplImage* src)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnImageCvLoad(char *filename,
        int channels)
YN_ALSWAY_INLINE;

#endif

#ifdef __cplusplus
}
#endif

#endif /* YNIMAGE_H */
