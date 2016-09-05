//	File        :   YnImageCv.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   24-07-2016
//	Author      :   haittt

#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

#include "../include/YnImage.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */
uint32 windows = 0;

/**************** Local Implement */

/**************** Implement */
void YnImageCvShow(tYnImage p,
        const char *name)
{
    char buff[256];
    int x, y, k;
    tYnImage copy = YnImageCopy(p);
    IplImage * disp;
    int step;

    sprintf(buff, "%s", name);

    YnImageConstrain(copy);
    if (p.channel == 3)
        YnImageRgbgr(copy);

    disp = cvCreateImage(cvSize(p.width, p.height), IPL_DEPTH_8U, p.channel);
    step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL);

    windows ++;
    for (y = 0; y < p.height; y ++)
    {
        for (x = 0; x < p.width; x ++)
        {
            for (k = 0; k < p.channel; k ++)
            {
                disp->imageData[(y * step) + (x * p.channel) + k] = (unsigned char)(YnImagePixelGet(copy, x, y, k) * 255);
            }
        }
    }

    YnImageFree(copy);
    cvShowImage(buff, disp);
    cvReleaseImage(&disp);
}

void YnImageSaveImage(tYnImage p,
        char *name)
{
    IplImage *disp;
    tYnImage copy = YnImageCopy(p);
    int x, y ,k;
    int step;
    char buff[256];

    sprintf(buff, "%s.jpg", name);

    YnImageRgbgr(copy);
    disp = cvCreateImage(cvSize(p.width, p.height), IPL_DEPTH_8U, p.channel);

    step = disp->widthStep;
    for (y = 0; y < p.height; y ++)
    {
        for (x = 0; x < p.width; x ++)
        {
            for (k= 0; k < p.channel; k ++)
            {
                disp->imageData[y * step + x * p.channel + k] = (unsigned char)(YnImagePixelGet(copy, x, y, k) * 255);
            }
        }
    }

    cvSaveImage(buff, disp,0);
    cvReleaseImage(&disp);

    YnImageFree(copy);
}

tYnImage YnImageFromStreamGet(CvCapture *cap)
{
    tYnImage im;
    IplImage* src = cvQueryFrame(cap);

    if (!src)
        return YnImageMakeEmpty(0, 0, 0);

    im = YnImageCvIplToimage(src);
    YnImageRgbgr(im);

    return im;
}

tYnImage YnImageLoadColor(char *filename,
        int w,
        int h)
{
    return YnImageLoad(filename, w, h, 3);
}

tYnImage YnImageLoad(char *filename,
        int w,
        int h,
        int c)
{
    tYnImage resized;

#ifdef YN_OPENCV
    tYnImage out = YnImageCvLoad(filename, c);
#else
    tYnImage out = YnImageLoadStb(filename, c);
#endif

    if ((h && w) && ((h != out.height) || (w != out.width)))
    {
        resized = YnImageResize(out, w, h);
        YnImageFree(out);
        out = resized;
    }

    return out;
}

tYnImage YnImageCvLoad(char *filename,
        int channels)
{
    IplImage* src = 0;
    tYnImage out;
    int flag = -1;

    if (channels == 0)
        flag = -1;
    else if (channels == 1)
        flag = 0;
    else if (channels == 3)
        flag = 1;
    else
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);

    if ((src = cvLoadImage(filename, flag)) == 0)
    {
        printf("Cannot load image \"%s\"\n", filename);
        exit(0);
    }

    out = (tYnImage)YnImageCvIplToimage(src);

    cvReleaseImage(&src);
    YnImageRgbgr(out);

    return out;
}

tYnImage YnImageCvIplToimage(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k, count = 0;
    tYnImage out = YnImageMake(w, h, c);

    for (k = 0; k < c; k ++)
    {
        for (i = 0; i < h; i ++)
        {
            for (j = 0; j < w; j ++)
            {
                out.data[count++] = data[(i * step) + (j * c) + k] / 255.;
            }
        }
    }

    return out;
}
