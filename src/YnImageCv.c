//	File        :   YnImageCv.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   24-07-2016
//	Author      :   haittt

#include "stb_image.h"
#include "stb_image_write.h"
#include "../YnImage.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"


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
    int x,y,k;
    tYnImage copy = YnImageCopy(p);
    IplImage *disp;
    int step;

    sprintf(buff, "%s", name);

    YnImageConstrain(copy);
    if(p.c == 3)
        YnImageRgbgr(copy);

    *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL);

    ++windows;
    for(y = 0; y < p.h; y ++)
    {
        for(x = 0; x < p.w; x ++)
        {
            for(k = 0; k < p.c; k ++)
            {
                disp->imageData[(y * step) + (x * p.c) + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
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
    int x,y,k;
    int step;
    char buff[256];

    sprintf(buff, "%s.jpg", name);

    YnImageRgbgr(copy);
    disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);

    step = disp->widthStep;
    for(y = 0; y < p.h; y ++)
    {
        for(x = 0; x < p.w; x ++)
        {
            for(k= 0; k < p.c; k ++)
            {
                disp->imageData[y * step + x * p.c + k] = (unsigned char)(YnImageGet(copy, x, y, k) * 255);
            }
        }
    }

    cvSaveImage(buff, disp,0);
    cvReleaseImage(&disp);

    YnImageFree(copy);
}

tYnImage YnImageCvIplToimage(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k, count=0;
    tYnImage out = YnImageMake(w, h, c);

    for(k= 0; k < c; k ++)
    {
        for(i = 0; i < h; i ++)
        {
            for(j = 0; j < w; j ++)
            {
                out.data[count++] = data[(i * step) + (j * c) + k] / 255.;
            }
        }
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

    out = YnImageIplToImage(src);

    cvReleaseImage(&src);
    YnImageRgbgr(out);

    return out;
}

#endif
