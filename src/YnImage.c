//	File        :   YnImage.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   04-07-2016
//	Author      :   haittt

#include "../YnCuda.h"
#include "../YnImage.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#endif

#include "stb_image.h"
#include "stb_image_write.h"

/**************** Define */
#define class_test_car      (6)
#define class_test_person   (14)
#define class_test_bike     (1)
#define class_test_motor    (13)
#define class_test_bus      (5)
#define box_num             (7*7*2)

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */
float colors[6][3] = { {1,0,1},{0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

/**************** Local Implement */

/**************** Implement */

float YnImageGetColor(int c,
        int x,
        int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    float r = 0;

    ratio -= i;
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c];

    return r;
}

void YnImageDrawLabel(tYnImage a,
        int r,
        int c,
        tYnImage label,
        const float *rgb)
{
	int i, j, k;
    float ratio = (float) label.w / label.h;
    int h = label.h;
    int w = ratio * h;
    float val;
    tYnImage rl = YnImageResize(label, w, h);

    if (r - h >= 0)
        r = r - h;

    for(j = 0; j < h && j + r < a.h; ++j)
    {
        for(i = 0; i < w && i + c < a.w; ++i)
        {
            for(k = 0; k < label.c; ++k)
            {
                val =  YnImageGetPixel(rl, i, j, k);
                YnImageSetPixel(a, i + c, j + r, k, rgb[k] * val);
            }
        }
    }

    YnImageFree(rl);
}


void YnImageDrawBox(tYnImage a,
        int x1,
        int y1,
        int x2,
        int y2,
        float r,
        float g,
        float b)
{
    int i;

    if (x1 < 0)
        x1 = 0;
    if (x1 >= a.w)
        x1 = a.w-1;
    if (x2 < 0)
        x2 = 0;
    if (x2 >= a.w)
        x2 = a.w-1;

    if (y1 < 0)
        y1 = 0;
    if (y1 >= a.h)
        y1 = a.h-1;
    if (y2 < 0)
        y2 = 0;
    if (y2 >= a.h)
        y2 = a.h-1;

    for(i = x1; i <= x2; i ++)
    {
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }

    for(i = y1; i <= y2; i ++)
    {
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void YnImageDrawBoxWidth(tYnImage a,
        int x1,
        int y1,
        int x2,
        int y2,
        int w,
        float r,
        float g,
        float b)
{
    int i;

    for(i = 0; i < w; i ++)
    {
        YnImageDrawBox(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

void YnImageDrawBbox(tYnImage a,
        tYnBBox bbox,
        int w,
        float r,
        float g,
        float b)
{
    int i;
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    for(i = 0; i < w; i ++)
    {
        YnImageDrawBox(a, left + i, top + i, right - i, bot - i, r, g, b);
    }
}

void YnImageDrawDetections(tYnImage im,
        int num,
        float thresh,
        tYnBBox *boxes,
        float **probs,
        char **names,
        tYnImage *labels,
        int classes)
{
    int i;
    int width;
    int offset;
    float red;
    float green;
    float blue;
    float rgb[3];
    tYnBBox b;
    int left;
    int right;
    int top;
    int bot;

    for(i = 0; i < num; i ++)
    {
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];

        if (prob > thresh)
        {
            width = pow(prob, 1./2.)*10+1;
            printf("%s: %.2f\n", names[class], prob);
            offset = class*17 % classes;
            red = get_color(0,offset,classes);
            green = get_color(1,offset,classes);
            blue = get_color(2,offset,classes);

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            b = boxes[i];

            left  = (b.x-b.w/2.)*im.w;
            right = (b.x+b.w/2.)*im.w;
            top   = (b.y-b.h/2.)*im.h;
            bot   = (b.y+b.h/2.)*im.h;

            if (left < 0)
                left = 0;
            if (right > im.w - 1)
                right = im.w - 1;
            if (top < 0)
                top = 0;
            if (bot > im.h-1)
                bot = im.h-1;

            YnImageDrawBoxWidth(im, left, top, right, bot, width,
                    red, green, blue);

            if (labels)
                YnImgeDrawLabel(im, top + width, left, labels[class], rgb);
        }
    }
}

void YnImageFlip(tYnImage a)
{
    float swap;
    int flip;
    int index;
    int i,j,k;

    for(k = 0; k < a.c; k ++)
    {
        for(i = 0; i < a.h; i ++)
        {
            for(j = 0; j < a.w/2; j ++)
            {
                index = j + a.w*(i + a.h*(k));
                flip = (a.w - j - 1) + a.w*(i + a.h*(k));
                swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

tYnImage YnImageDistance(tYnImage a,
        tYnImage b)
{
    int i,j;
    tYnImage dist = YnImageMake(a.w, a.h, 1);

    for(i = 0; i < a.c; i ++)
    {
        for(j = 0; j < a.h * a.w; j ++)
        {
            dist.data[j] += pow(a.data[i * a.h * a.w + j] -
                    b.data[i * a.h * a.w + j],2);
        }
    }

    for(j = 0; j < a.h * a.w; j ++)
    {
        dist.data[j] = sqrt(dist.data[j]);
    }

    return dist;
}

void YnImageEmbed(tYnImage source,
        tYnImage dest,
        int dx,
        int dy)
{
    float val;
    int x,y,k;

    for(k = 0; k < source.c; k ++)
    {
        for(y = 0; y < source.h; y ++)
        {
            for(x = 0; x < source.w; x ++)
            {
                val = YnImageGetPixel(source, x,y,k);
                YnOmgeSetPixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

tYnImage YnImageCollapseLayers(tYnImage source,
        int border)
{
    int i;
    int h_offset;
    int h = source.h;

    h = (h + border) * source.c - border;
    tYnImage dest = YnImgeMake(source.w, h, 1);

    for(i = 0; i < source.c; i ++)
    {
        tYnImage layer = YnImageGetLayer(source, i);
        h_offset = i * (source.h + border);
        YnImageEmbed(layer, dest, 0, h_offset);
        YnImageFree(layer);
    }

    return dest;
}

void YnImageConstrain(tYnImage im)
{
    int i;

    for(i = 0; i < im.w * im.h * im.c; i ++)
    {
        if (im.data[i] < 0)
            im.data[i] = 0;

        if (im.data[i] > 1)
            im.data[i] = 1;
    }
}

void YnImageNormalize(tYnImage p)
{
    float *min = calloc(p.c, sizeof(float));
    float *max = calloc(p.c, sizeof(float));
    int i,j;
    float v;

    for(i = 0; i < p.c; i ++)
        min[i] = max[i] = p.data[i * p.h * p.w];

    for(j = 0; j < p.c; j ++)
    {
        for(i = 0; i < p.h * p.w; i ++)
        {
            v = p.data[i + (j * p.h * p.w)];

            if (v < min[j])
                min[j] = v;

            if (v > max[j])
                max[j] = v;
        }
    }

    for(i = 0; i < p.c; i ++)
    {
        if ((max[i] - min[i]) < .000000001)
        {
            min[i] = 0;
            max[i] = 1;
        }
    }

    for(j = 0; j < p.c; j ++)
    {
        for(i = 0; i < p.w * p.h; i ++)
        {
            p.data[i + j * p.h * p.w] =
                    (p.data[i + j * p.h * p.w] - min[j])/(max[j] - min[j]);
        }
    }

    YnUtilFree(min);
    YnUtilFree(max);
}

tYnImage YnImageCopy(tYnImage p)
{
    tYnImage copy = p;

    copy.data = calloc(p.h * p.w * p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));

    return copy;
}

void YnImageRgbgr(tYnImage im)
{
    int i;
    float swap;

    for(i = 0; i < im.w * im.h; i ++)
    {
        swap = im.data[i];
        im.data[i] = im.data[i + im.w * im.h * 2];
        im.data[i + im.w * im.h * 2] = swap;
    }
}

void YnImageShow(tYnImage p,
        const char *name)
{
#ifdef OPENCV
    YnImageCvShow(p, name);
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    YnImageSave(p, name);
#endif
}

void YnImageSave(tYnImage im,
        const char *name)
{
    char buff[256];
    int i,k;
    int success;
    unsigned char *data = calloc(im.w * im.h * im.c, sizeof(char));

    sprintf(buff, "%s.png", name);

    for(k = 0; k < im.c; k ++)
    {
        for(i = 0; i < im.w * im.h; i ++)
        {
            data[i*im.c+k] =
                    (unsigned char)(255 * im.data[i + k * im.w * im.h]);
        }
    }

    success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w * im.c);
    YnUtilFree(data);

    if (!success)
        fprintf(stderr, "Failed to write image %s\n", buff);
}

void YnImageShowLayers(tYnImage p,
        char *name)
{
    int i;
    char buff[256];

    for(i = 0; i < p.c; i ++)
    {
        sprintf(buff, "%s - Layer %d", name, i);
        tYnImage layer = YnImageGetLayer(p, i);
        YnImageShow(layer, buff);
        YnImageFree(layer);
    }
}

void YnImageShowCollapsed(tYnImage p,
        char *name)
{
    tYnImage c = YnImageCollapseLayers(p, 1);
    YnImageShow(c, name);
    YnImageFree(c);
}

tYnImage YnImageMakeEmpty(int w,
        int h,
        int c)
{
    tYnImage out;

    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;

    return out;
}

tYnImage YnImageMake(int w,
        int h,
        int c)
{
    tYnImage out = YnImageMakeEmpty(w, h, c);

    out.data = calloc(h * w * c , sizeof(float));

    return out;
}

tYnImage YnImageMakeRandom(int w,
        int h,
        int c)
{
    tYnImage out = YnImageMakeEmpty(w, h, c);
    int i;

    out.data = calloc(h*w*c, sizeof(float));

    for(i = 0; i < w * h * c; i ++)
    {
        out.data[i] = (YnUtilRandomNormalNum() * .75) + .25;
    }
    return out;
}

tYnImage YnImageFloatToImage(int w,
        int h,
        int c,
        float *data)
{
    tYnImage out = YnImageMakeEmpty(w, h, c);
    out.data = data;
    return out;
}

tYnImage YnImageRotate(tYnImage im,
        float rad)
{
    int x, y, c;
    float rx;
    float ry;
    float val;
    float cx = im.w/2.;
    float cy = im.h/2.;
    tYnImage rot = YnImageMake(im.w, im.h, im.c);

    for(c = 0; c < im.c; c ++)
    {
        for(y = 0; y < im.h; y ++)
        {
            for(x = 0; x < im.w; x ++)
            {
                rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
                ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
                val = bilinear_interpolate(im, rx, ry, c);
                YnImageSetPixel(rot, x, y, c, val);
            }
        }
    }

    return rot;
}

void YnImageTranslate(tYnImage m,
        float s)
{
    int i;
    for(i = 0; i < m.h * m.w * m.c; i ++)
        m.data[i] += s;
}

void YnImgaeScale(tYnImage m,
        float s)
{
    int i;
    for(i = 0; i < m.h * m.w * m.c; i ++)
        m.data[i] *= s;
}

tYnImage YnImageCrop(tYnImage im,
        int dx,
        int dy,
        int w,
        int h)
{
    int i, j, k;
    int r,c;
    float val;
    tYnImage cropped = YnImageMake(w, h, im.c);

    for(k = 0; k < im.c; k ++)
    {
        for(j = 0; j < h; j ++)
        {
            for(i = 0; i < w; i ++)
            {
                r = j + dy;
                c = i + dx;
                val = 0;

                if ((r >= 0) && (r < im.h) && (c >= 0) && (c < im.w))
                {
                    val = YnImageGetPixel(im, c, r, k);
                }

                YnIMageSetPixel(cropped, i, j, k, val);
            }
        }
    }

    return cropped;
}

float YnImageThreeWayMax(float a,
        float b,
        float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float YnImageThreeWayMin(float a,
        float b,
        float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

/* ref darket & http://www.cs.rit.edu/~ncs/color/t_convert.html*/
void YnImageRgbToHsv(tYnImage im)
{
    int i, j;
    float r, g, b;
    float h, s, v;
    float max, min, delta;

    assert(im.c == 3);

    for(j = 0; j < im.h; j ++)
    {
        for(i = 0; i < im.w; i ++)
        {
            r = YnImageGetPixel(im, i , j, 0);
            g = YnImageGetPixel(im, i , j, 1);
            b = YnImageGetPixel(im, i , j, 2);
            max = YnImageThreeWayMax(r,g,b);
            min = YnImageThreeWayMin(r,g,b);
            delta = max - min;
            v = max;

            if (max == 0)
            {
                s = 0;
                h = -1;
            }
            else
            {
                s = delta/max;
                if (r == max)
                {
                    h = (g - b) / delta;
                }
                else if (g == max)
                {
                    h = 2 + (b - r) / delta;
                }
                else
                {
                    h = 4 + (r - g) / delta;
                }

                if (h < 0)
                    h += 6;
            }
            YnImageSetPixel(im, i, j, 0, h);
            YnImageThreeWay(im, i, j, 1, s);
            YnImageThreeWay(im, i, j, 2, v);
        }
    }
}

void YnImageHsvToRgb(tYnImage im)
{
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    int index;

    assert(im.c == 3);

    for(j = 0; j < im.h; j ++)
    {
        for(i = 0; i < im.w; i ++)
        {
            h = YnImageGetPixel(im, i , j, 0);
            s = YnImageGetPixel(im, i , j, 1);
            v = YnImageGetPixel(im, i , j, 2);

            if (s == 0)
            {
                r = g = b = v;
            }
            else
            {
                index = floor(h);
                f = h - index;
                p = v * (1 - s);
                q = v * (1 - (s * f));
                t = v * (1 - (s * (1-f)));

                if (index == 0)
                {
                    r = v;
                    g = t;
                    b = p;
                }
                else if (index == 1)
                {
                    r = q;
                    g = v;
                    b = p;
                }
                else if (index == 2)
                {
                    r = p;
                    g = v;
                    b = t;
                }
                else if (index == 3)
                {
                    r = p;
                    g = q;
                    b = v;
                }
                else if (index == 4)
                {
                    r = t;
                    g = p;
                    b = v;
                }
                else
                {
                    r = v;
                    g = p;
                    b = q;
                }
            }

            YnImageSetPixel(im, i, j, 0, r);
            YnImageSetPixel(im, i, j, 1, g);
            YnImageSetPixel(im, i, j, 2, b);
        }
    }
}

tYnImage YnImageGrayscale(tYnImage im)
{
    int i, j, k;
    float scale[] = {0.587, 0.299, 0.114};
    tYnImage gray = YnImageMake(im.w, im.h, 1);

    assert(im.c == 3);

    for(k = 0; k < im.c; k ++)
    {
        for(j = 0; j < im.h; j ++)
        {
            for(i = 0; i < im.w; i ++)
            {
                gray.data[i + (im.w * j)] += scale[k] * YnImageGetPixel(im, i, j, k);
            }
        }
    }

    return gray;
}

tYnImage YnImageThreshold(tYnImage im,
        float thresh)
{
    int i;
    tYnImage t = YnImageMake(im.w, im.h, im.c);

    for(i = 0; i < im.w * im.h * im.c; i ++)
    {
        t.data[i] = (im.data[i] > thresh) ? 1 : 0;
    }

    return t;
}

tYnImage YnImageBlend(tYnImage fore,
        tYnImage back,
        float alpha)
{
    float val;
    int i, j, k;
    tYnImage blend = YnImageMake(fore.w, fore.h, fore.c);

    assert((fore.w == back.w) && (fore.h == back.h) && (fore.c == back.c));

    for(k = 0; k < fore.c; k ++)
    {
        for(j = 0; j < fore.h; j ++)
        {
            for(i = 0; i < fore.w; i ++)
            {
                val = alpha * YnImageGetPixel(fore, i, j, k) +
                    (1 - alpha)* get_pixel(back, i, j, k);

                YnImageSetPixel(blend, i, j, k, val);
            }
        }
    }

    return blend;
}

void YnImageScaleChannel(tYnImage im,
        int c,
        float v)
{
    float pix;
    int i, j;

    for(j = 0; j < im.h; j ++)
    {
        for(i = 0; i < im.w; i ++)
        {
            pix = YnImageGetPixel(im, i, j, c);
            pix = pix*v;
            YnIMageSetPixel(im, i, j, c, pix);
        }
    }
}

void YnImageSaturate(tYnImage im,
        float sat)
{
    YnImageRgbToHsv(im);
    YnImageSaleChannel(im, 1, sat);
    YnImageHsvToRgb(im);
    YnImageConstrain(im);
}

void YnImageExposure(tYnImage im,
        float sat)
{
    YnImageRgbToHsv(im);
    YnImageScaleChannel(im, 2, sat);
    YnImageHsvToRgb(im);
    YnIMageConstrain(im);
}

void YnImageSaturateExposure(tYnImage im,
        float sat,
        float exposure)
{
    YnImageRgbToHsv(im);
    YnImageScaleChannel(im, 1, sat);
    YnImageScaleChannel(im, 2, exposure);
    YnImageHsvToRgb(im);
    YnImageConstrain(im);
}

float YnImageBilinearInterpolate(tYnImage im,
        float x,
        float y,
        int c)
{
    float val;
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);
    float dx = x - ix;
    float dy = y - iy;

    val = (1-dy)    * (1-dx) * YnImageGetPixelExtend(im, ix,    iy,     c) +
            dy      * (1-dx) * YnImageGetPixelExtend(im, ix,    iy+1,   c) +
            (1-dy)  *   dx   * YnImageGetPixelExtend(im, ix+1,  iy,     c) +
            dy      *   dx   * YnImageGetPixelExtend(im, ix+1,  iy+1,   c);

    return val;
}

tYnImage YnImageResize(tYmImage im,
        int w,
        int h)
{
    float val;
    float sx;
    int ix;
    float dx;
    float sy;
    int iy;
    float dy;
    tYnImage resized = YnImageMake(w, h, im.c);
    tYnImage part = YnImageMake(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);

    for(k = 0; k < im.c; k ++)
    {
        for(r = 0; r < im.h; r ++)
        {
            for(c = 0; c < w; c ++)
            {
                val = 0;
                if (c == w-1 || im.w == 1)
                {
                    val = get_pixel(im, im.w-1, r, k);
                }
                else
                {
                    sx = c * w_scale;
                    ix = (int) sx;
                    dx = sx - ix;
                    val = (1 - dx) * YnImageGetPixel(im, ix, r, k) +
                            dx * YnImageGetPixel(im, ix + 1, r, k);
                }

                YnImageSetPixel(part, c, r, k, val);
            }
        }
    }

    for(k = 0; k < im.c; k ++)
    {
        for(r = 0; r < h; r ++)
        {
            sy = r * h_scale;
            iy = (int) sy;
            dy = sy - iy;

            for(c = 0; c < w; c ++)
            {
                val = (1-dy) * YnImageGetPixel(part, c, iy, k);
                YnImageSetPixel(resized, c, r, k, val);
            }

            if ((r == h-1) || (im.h == 1))
                continue;

            for(c = 0; c < w; c ++)
            {
                val = dy * YnImageGetPixel(part, c, iy+1, k);
                YnImageAddPixel(resized, c, r, k, val);
            }
        }
    }

    YnImageFree(part);
    return resized;
}

void YnImageTestResize(char *filename)
{
    tYnImage im = YnImageLoad(filename, 0,0, 3);
    float mag = YnUtilArrayMag(im.data, im.w*im.h*im.c);
    tYnImage gray = YnImageGrayscale(im);

    printf("L2 Norm: %f\n", mag);

    tYnImage sat2 = copy_image(im);
    saturate_image(sat2, 2);

    tYnImage sat5 = copy_image(im);
    saturate_image(sat5, .5);

    tYnImage exp2 = copy_image(im);
    exposure_image(exp2, 2);

    tYnImage exp5 = copy_image(im);
    exposure_image(exp5, .5);

    #ifdef YN_GPU
    tYnImage r = YnImageResize(im, im.w, im.h);
    tYnImage black = YnImageMake(im.w * 2 + 3, im.h * 2 + 3, 9);
    tYnImage black2 = YnImageMake(im.w, im.h, 3);

    float *r_gpu = YnCudaMakeArray(r.data, r.w * r.h * r.c);
    float *black_gpu = YnCudaMakeArray(black.data, black.w * black.h * black.c);
    float *black2_gpu = YnCudaMakeArray(black2.data, black2.w * black2.h * black2.c);
    YnBlasGpuShortcut(3, r.w, r.h, 1, r_gpu, black.w, black.h, 3, black_gpu);

    YnBlasGpuShortcut(3, black.w, black.h, 3, black_gpu, black2.w, black2.h, 1, black2_gpu);
    YnCudaArrayPullFromGpu(black_gpu, black.data, black.w*black.h*black.c);
    YnCudaArrayPullFromGpu(black2_gpu, black2.data, black2.w*black2.h*black2.c);
    YnImageShowLayers(black, "Black");
    YnImageShow(black2, "Recreate");
    #endif

    YnImageShow(im, "Original");
    YnImageShow(gray, "Gray");
    YnImageShow(sat2, "Saturation-2");
    YnImageShow(sat5, "Saturation-.5");
    YnImageShow(exp2, "Exposure-2");
    YnImageShow(exp5, "Exposure-.5");

#ifdef OPENCV
    cvWaitKey(0);
#endif

}

tYnImage YnImageLoadStb(char *filename,
        int channels)
{
    int i,j,k;
    int w, h, c;
    int dst_index, src_index;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    tYnImage im;

    if (!data)
    {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }

    if (channels)
        c = channels;

    im = YnImageMake(w, h, c);
    for(k = 0; k < c; k ++)
    {
        for(j = 0; j < h; j ++)
        {
            for(i = 0; i < w; i ++)
            {
                dst_index = i + w*j + w*h*k;
                src_index = k + c*i + c*w*j;

                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    YnUtilFree(data);

    return im;
}

tYnImage YnImageLoad(char *filename,
        int w,
        int h,
        int c)
{
    tYnImage resized;

#ifdef OPENCV
    tYnImage out = YnImageCvLoad(filename, c);
#else
    tYnImage out = YnImageLoadStb(filename, c);
#endif

    if ((h && w) && ((h != out.h) || (w != out.w)))
    {
        resized = YnImageResize(out, w, h);
        YnImageFree(out);
        out = resized;
    }

    return out;
}

tYnImage YnImageLoadColor(char *filename,
        int w,
        int h)
{
    return YnImageLoad(filename, w, h, 3);
}

tYnImage YnImageGetLayer(tYnImage m,
        int l)
{
    tYnImage out = YnImageMake(m.w, m.h, 1);
    int i;

    for(i = 0; i < (m.h * m.w); i ++)
    {
        out.data[i] = m.data[i + l * m.h * m.w];
    }

    return out;
}

float YnImageGetPixel(tYnImage m,
        int x,
        int y,
        int c)
{
    assert((x < m.w) && (y < m.h) && (c < m.c));
    return m.data[(c * m.h * m.w) + (y * m.w) + x];
}

float YnImageGetPixelExtend(tYnImage m,
        int x,
        int y,
        int c)
{
    if ((x < 0) || (x >= m.w) || (y < 0) || (y >= m.h) || (c < 0) || (c >= m.c))
        return 0;

    return YnImageGetPixel(m, x, y, c);
}

void YnImageSetPixel(tYnImage m,
        int x,
        int y,
        int c,
        float val)
{
    assert((x < m.w) && (y < m.h) && (c < m.c));
    m.data[c * m.h * m.w + y * m.w + x] = val;
}

void YnImageAddPixel(tYnImage m,
        int x,
        int y,
        int c,
        float val)
{
    assert((x < m.w) && (y < m.h) && (c < m.c));
    m.data[(c * m.h * m.w) + (y * m.w) + x] += val;
}

void YnImagePrint(tYnImage m)
{
    int i, j, k;

    for(i =0 ; i < m.c; i ++)
    {
        for(j =0 ; j < m.h; j ++)
        {
            for(k = 0; k < m.w; k ++)
            {
                printf("%.2lf, ", m.data[i * m.h * m.w + j * m.w + k]);
                if (k > 30)
                    break;
            }

            printf("\n");
            if (j > 30)
                break;
        }

        printf("\n");
    }

    printf("\n");
}

tYnImage YnImageCollapseVert(tYnImage *ims,
        int n)
{
    int i,j;
    int color = 1;
    int border = 1;
    int h,w,c;
    int h_offset;
    int w_offset;
    image layer;

    w = ims[0].w;
    h = (ims[0].h + border) * n - border;
    c = ims[0].c;

    if ((c != 3) || !color)
    {
        w = (w + border) * c - border;
        c = 1;
    }

    tYnImage filters = YnImageMake(w, h, c);

    for(i = 0; i < n; i ++)
    {
        h_offset = i * (ims[0].h + border);
        tYnImage copy = YnImageCopy(ims[i]);

        if ((c == 3) && color)
        {
            YnImageEmbed(copy, filters, 0, h_offset);
        }
        else
        {
            for(j = 0; j < copy.c; j ++)
            {
                w_offset = j * (ims[0].w + border);
                layer = YnImageGetLayer(copy, j);
                YnImageEmbed(layer, filters, w_offset, h_offset);
                YnImageFree(layer);
            }
        }
        YnImageFree(copy);
    }
    return filters;
}

tYnImage YnImageCollapseHorz(tYnImage *ims,
        int n)
{
    int i,j;
    tYnImage filters;
    int color = 1;
    int border = 1;
    int h,w,c;
    int w_offset;
    int h_offset;
    tYnImage copy;
    tYnImage layer;
    int size = ims[0].h;

    h = size;
    w = (ims[0].w + border) * n - border;
    c = ims[0].c;

    if (c != 3 || !color)
    {
        h = (h+border)*c - border;
        c = 1;
    }

    filters = YnImageMake(w, h, c);

    for(i = 0; i < n; i ++)
    {
        w_offset = i * (size+border);
        copy = YnImageCopy(ims[i]);

        if ((c == 3) && color)
        {
            YnImageEmbed(copy, filters, w_offset, 0);
        }
        else
        {
            for(j = 0; j < copy.c; j ++)
            {
                h_offset = j*(size + border);
                layer = get_image_layer(copy, j);
                YnImageEmbed(layer, filters, w_offset, h_offset);
                YnImageFree(layer);
            }
        }
        YnImageFree(copy);
    }

    return filters;
}

void YnImageShow(tYnImage *ims,
        int n,
        char *window)
{
    tYnImage m = YnImageCollapseVert(ims, n);
    image sized;

    YnImageNormalize(m);
    sized = YnImageResize(m, m.w, m.h);
    YnImageSave(sized, window);
    YnImageShow(sized, window);
    YnImageFree(sized);
    YnImageFree(m);
}

void YnImageFree(tYnImage m)
{
    YnUtilFree(m.data);
}
