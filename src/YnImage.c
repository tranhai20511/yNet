//	File        :   YnImage.channel
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   04-07-2016
//	Author      :   haittt

#include "../YnCuda.h"
#include "../YnImage.h"

#ifdef YN_OPENCV
#include "opencv2/highgui/highgui_c.height"
#include "opencv2/imgproc/imgproc_c.height"
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
YN_STATIC
float _YnImage2ColPixelGet(float *im,
        int height,
        int width,
        int channels,
        int row,
        int col,
        int channel,
        int pad)
YN_ALSWAY_INLINE;

YN_STATIC
void _YnImage2ColPixelAdd(float *im,
        int height,
        int width,
        int channels,
        int row,
        int col,
        int channel,
        int pad,
        float val)
YN_ALSWAY_INLINE;

/**************** Implement */
YN_STATIC
float _YnImage2ColPixelGet(float *im,
        int height,
        int width,
        int channels,
        int row,
        int col,
        int channel,
        int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return 0;

    return im[col + width*(row + height*channel)];
}

YN_STATIC
void _YnImage2ColPixelAdd(float *im,
        int height,
        int width,
        int channels,
        int row,
        int col,
        int channel,
        int pad,
        float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return;

    im[col + width * (row + height * channel)] += val;
}

float YnImageColorGet(int c,
        int x,
        int max)
{
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    float r = 0;

    ratio -= i;
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];

    return r;
}

void YnImageDrawLabel(tYnImage a,
        int r,
        int c,
        tYnImage label,
        const float *rgb)
{
	int i, j, k;
    float ratio = (float) label.width / label.height;
    int h = label.height;
    int w = ratio * h;
    float val;
    tYnImage rl = YnImageResize(label, w, h);

    if (r - h >= 0)
        r = r - h;

    for (j = 0; j < h && j + r < a.height; ++j)
    {
        for (i = 0; i < w && i + c < a.width; i ++)
        {
            for (k = 0; k < label.channel; ++k)
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
    if (x1 >= a.width)
        x1 = a.width-1;
    if (x2 < 0)
        x2 = 0;
    if (x2 >= a.width)
        x2 = a.width-1;

    if (y1 < 0)
        y1 = 0;
    if (y1 >= a.height)
        y1 = a.height-1;
    if (y2 < 0)
        y2 = 0;
    if (y2 >= a.height)
        y2 = a.height-1;

    for (i = x1; i <= x2; i ++)
    {
        a.data[i + y1*a.width + 0*a.width*a.height] = r;
        a.data[i + y2*a.width + 0*a.width*a.height] = r;

        a.data[i + y1*a.width + 1*a.width*a.height] = g;
        a.data[i + y2*a.width + 1*a.width*a.height] = g;

        a.data[i + y1*a.width + 2*a.width*a.height] = b;
        a.data[i + y2*a.width + 2*a.width*a.height] = b;
    }

    for (i = y1; i <= y2; i ++)
    {
        a.data[x1 + i*a.width + 0*a.width*a.height] = r;
        a.data[x2 + i*a.width + 0*a.width*a.height] = r;

        a.data[x1 + i*a.width + 1*a.width*a.height] = g;
        a.data[x2 + i*a.width + 1*a.width*a.height] = g;

        a.data[x1 + i*a.width + 2*a.width*a.height] = b;
        a.data[x2 + i*a.width + 2*a.width*a.height] = b;
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

    for (i = 0; i < w; i ++)
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
    int left  = (bbox.x-bbox.width/2)*a.width;
    int right = (bbox.x+bbox.width/2)*a.width;
    int top   = (bbox.y-bbox.height/2)*a.height;
    int bot   = (bbox.y+bbox.height/2)*a.height;

    for (i = 0; i < w; i ++)
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

    for (i = 0; i < num; i ++)
    {
        int class = YnUtilArrayMaxIndex(probs[i], classes);
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

            left  = (b.x-b.width/2.)*im.width;
            right = (b.x+b.width/2.)*im.width;
            top   = (b.y-b.height/2.)*im.height;
            bot   = (b.y+b.height/2.)*im.height;

            if (left < 0)
                left = 0;
            if (right > im.width - 1)
                right = im.width - 1;
            if (top < 0)
                top = 0;
            if (bot > im.height-1)
                bot = im.height-1;

            YnImageDrawBoxWidth(im, left, top, right, bot, width,
                    red, green, blue);

            if (labels)
                YnImgeDrawLabel(im, top + width, left, labels[class], rgb);
        }
    }
}

void YnImageDrawDetections1(tYnImage im,
        int num,
        float thresh,
        tYnBBox *boxes,
        float **probs,
        char **names,
        tYnImage *labels,
        int classes,
        tYnBBoxSend * boxSend,
        unsigned char *numBox)
{
    int countBox = 0;
    int i;
    int sameClass = 0;
    int class;
    float prob;
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

    for (i = 0; i < num; i ++)
    {
        class = YnUtilArrayMaxIndex(probs[i], classes);

        if ((class != class_test_car)
            && (class != class_test_person)
            && (class != class_test_bike)
            && (class != class_test_motor)
            && (class != class_test_bus))
            continue;

        prob = probs[i][class];
        if(prob > thresh)
        {
            width = pow(prob, 1./2.) * 10 + 1;
            printf("%s: %.2f\n", names[class], prob);
            offset = class * 17 % classes;

            red = YnImageColorGet(0, offset, classes);
            green = YnImageColorGet(1, offset, classes);
            blue = YnImageColorGet(2, offset, classes);
            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            b = boxes[i];

            left  = (b.x - b.width) * im.width;
            right = (b.x + b.width) * im.width;
            top   = (b.y - b.height) * im.height;
            bot   = (b.y + b.height) * im.height;

            if(left < 0)
                left = 0;
            if(right > im.width - 1)
                right = im.width - 1;
            if(top < 0)
                top = 0;
            if(bot > im.height - 1)
                bot = im.height - 1;

            YnImageDrawBoxWidth(im, left, top, right, bot, width, red, green, blue);

            boxSend[countBox].x = b.x;
            boxSend[countBox].y = b.y;
            boxSend[countBox].height = b.height;
            boxSend[countBox].width = b.width;
            boxSend[countBox].classId = class;
            countBox ++;

            if (labels)
                YnImgeDrawLabel(im, top + width, left, labels[class], rgb);
        }
    }

    *numBox = countBox;
}
void YnImageFlip(tYnImage a)
{
    float swap;
    int flip;
    int index;
    int i,j,k;

    for (k = 0; k < a.channel; k ++)
    {
        for (i = 0; i < a.height; i ++)
        {
            for (j = 0; j < a.width/2; j ++)
            {
                index = j + a.width * (i + a.height*(k));
                flip = (a.width - j - 1) + a.width*(i + a.height*(k));

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
    tYnImage dist = YnImageMake(a.width, a.height, 1);

    for (i = 0; i < a.channel; i ++)
    {
        for (j = 0; j < a.height * a.width; j ++)
        {
            dist.data[j] += pow(a.data[i * a.height * a.width + j] -
                    b.data[i * a.height * a.width + j],2);
        }
    }

    for (j = 0; j < a.height * a.width; j ++)
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

    for (k = 0; k < source.channel; k ++)
    {
        for (y = 0; y < source.height; y ++)
        {
            for (x = 0; x < source.width; x ++)
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
    int h = source.height;

    h = (h + border) * source.channel - border;
    tYnImage dest = YnImgeMake(source.width, h, 1);

    for (i = 0; i < source.channel; i ++)
    {
        tYnImage layer = YnImageGetLayer(source, i);
        h_offset = i * (source.height + border);
        YnImageEmbed(layer, dest, 0, h_offset);
        YnImageFree(layer);
    }

    return dest;
}

void YnImageConstrain(tYnImage im)
{
    int i;

    for (i = 0; i < im.width * im.height * im.channel; i ++)
    {
        if (im.data[i] < 0)
            im.data[i] = 0;

        if (im.data[i] > 1)
            im.data[i] = 1;
    }
}

void YnImageNormalize(tYnImage p)
{
    float *min = calloc(p.channel, sizeof(float));
    float *max = calloc(p.channel, sizeof(float));
    int i,j;
    float v;

    for (i = 0; i < p.channel; i ++)
        min[i] = max[i] = p.data[i * p.height * p.width];

    for (j = 0; j < p.channel; j ++)
    {
        for (i = 0; i < p.height * p.width; i ++)
        {
            v = p.data[i + (j * p.height * p.width)];

            if (v < min[j])
                min[j] = v;

            if (v > max[j])
                max[j] = v;
        }
    }

    for (i = 0; i < p.channel; i ++)
    {
        if ((max[i] - min[i]) < .000000001)
        {
            min[i] = 0;
            max[i] = 1;
        }
    }

    for (j = 0; j < p.channel; j ++)
    {
        for (i = 0; i < p.width * p.height; i ++)
        {
            p.data[i + j * p.height * p.width] =
                    (p.data[i + j * p.height * p.width] - min[j])/(max[j] - min[j]);
        }
    }

    YnUtilFree(min);
    YnUtilFree(max);
}

tYnImage YnImageCopy(tYnImage p)
{
    tYnImage copy = p;

    copy.data = calloc(p.height * p.width * p.channel, sizeof(float));
    memcpy(copy.data, p.data, p.height * p.width * p.channel * sizeof(float));

    return copy;
}

void YnImageRgbgr(tYnImage im)
{
    int i;
    float swap;

    for (i = 0; i < im.width * im.height; i ++)
    {
        swap = im.data[i];
        im.data[i] = im.data[i + im.width * im.height * 2];
        im.data[i + im.width * im.height * 2] = swap;
    }
}

void YnImageShow(tYnImage p,
        const char *name)
{
#ifdef YN_OPENCV
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
    unsigned char *data = calloc(im.width * im.height * im.channel, sizeof(char));

    sprintf(buff, "%s.png", name);

    for (k = 0; k < im.channel; k ++)
    {
        for (i = 0; i < im.width * im.height; i ++)
        {
            data[i*im.channel+k] =
                    (unsigned char)(255 * im.data[i + k * im.width * im.height]);
        }
    }

    success = stbi_write_png(buff, im.width, im.height, im.channel, data, im.width * im.channel);
    YnUtilFree(data);

    if (!success)
        fprintf(stderr, "Failed to write image %s\n", buff);
}

void YnImageShowLayers(tYnImage p,
        char *name)
{
    int i;
    char buff[256];

    for (i = 0; i < p.channel; i ++)
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
    out.height = h;
    out.width = w;
    out.channel = c;

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

    for (i = 0; i < w * h * c; i ++)
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
    float cx = im.width/2.;
    float cy = im.height/2.;
    tYnImage rot = YnImageMake(im.width, im.height, im.channel);

    for (c = 0; c < im.channel; c ++)
    {
        for (y = 0; y < im.height; y ++)
        {
            for (x = 0; x < im.width; x ++)
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
    for (i = 0; i < m.height * m.width * m.channel; i ++)
        m.data[i] += s;
}

void YnImgaeScale(tYnImage m,
        float s)
{
    int i;
    for (i = 0; i < m.height * m.width * m.channel; i ++)
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
    tYnImage cropped = YnImageMake(w, h, im.channel);

    for (k = 0; k < im.channel; k ++)
    {
        for (j = 0; j < h; j ++)
        {
            for (i = 0; i < w; i ++)
            {
                r = j + dy;
                c = i + dx;
                val = 0;

                if ((r >= 0) && (r < im.height) && (c >= 0) && (c < im.width))
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
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c);
}

float YnImageThreeWayMin(float a,
        float b,
        float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

/* ref darket & http://www.channels.rit.edu/~ncs/color/t_convert.heighttml*/
void YnImageRgbToHsv(tYnImage im)
{
    int i, j;
    float r, g, b;
    float h, s, v;
    float max, min, delta;

    assert(im.channel == 3);

    for (j = 0; j < im.height; j ++)
    {
        for (i = 0; i < im.width; i ++)
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

    assert(im.channel == 3);

    for (j = 0; j < im.height; j ++)
    {
        for (i = 0; i < im.width; i ++)
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
    tYnImage gray = YnImageMake(im.width, im.height, 1);

    assert(im.channel == 3);

    for (k = 0; k < im.channel; k ++)
    {
        for (j = 0; j < im.height; j ++)
        {
            for (i = 0; i < im.width; i ++)
            {
                gray.data[i + (im.width * j)] += scale[k] * YnImageGetPixel(im, i, j, k);
            }
        }
    }

    return gray;
}

tYnImage YnImageThreshold(tYnImage im,
        float thresh)
{
    int i;
    tYnImage t = YnImageMake(im.width, im.height, im.channel);

    for (i = 0; i < im.width * im.height * im.channel; i ++)
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
    tYnImage blend = YnImageMake(fore.width, fore.height, fore.channel);

    assert((fore.width == back.width) && (fore.height == back.height) && (fore.channel == back.channel));

    for (k = 0; k < fore.channel; k ++)
    {
        for (j = 0; j < fore.height; j ++)
        {
            for (i = 0; i < fore.width; i ++)
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

    for (j = 0; j < im.height; j ++)
    {
        for (i = 0; i < im.width; i ++)
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

tYnImage YnImageResize(tYnImage im,
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
    tYnImage resized = YnImageMake(w, h, im.channel);
    tYnImage part = YnImageMake(w, im.height, im.channel);
    int r, c, k;
    float w_scale = (float)(im.width - 1) / (w - 1);
    float h_scale = (float)(im.height - 1) / (h - 1);

    for (k = 0; k < im.channel; k ++)
    {
        for (r = 0; r < im.height; r ++)
        {
            for (c = 0; c < w; c ++)
            {
                val = 0;
                if (c == w-1 || im.width == 1)
                {
                    val = get_pixel(im, im.width-1, r, k);
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

    for (k = 0; k < im.channel; k ++)
    {
        for (r = 0; r < h; r ++)
        {
            sy = r * h_scale;
            iy = (int) sy;
            dy = sy - iy;

            for (c = 0; c < w; c ++)
            {
                val = (1-dy) * YnImageGetPixel(part, c, iy, k);
                YnImageSetPixel(resized, c, r, k, val);
            }

            if ((r == h-1) || (im.height == 1))
                continue;

            for (c = 0; c < w; c ++)
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
    float mag = YnUtilArrayMag(im.data, im.width*im.height*im.channel);
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
    tYnImage r = YnImageResize(im, im.width, im.height);
    tYnImage black = YnImageMake(im.width * 2 + 3, im.height * 2 + 3, 9);
    tYnImage black2 = YnImageMake(im.width, im.height, 3);

    float *r_gpu = YnCudaMakeArray(r.data, r.width * r.height * r.channel);
    float *black_gpu = YnCudaMakeArray(black.data, black.width * black.height * black.channel);
    float *black2_gpu = YnCudaMakeArray(black2.data, black2.width * black2.height * black2.channel);
    YnBlasGpuShortcut(3, r.width, r.height, 1, r_gpu, black.width, black.height, 3, black_gpu);

    YnBlasGpuShortcut(3, black.width, black.height, 3, black_gpu, black2.width, black2.height, 1, black2_gpu);
    YnCudaArrayPullFromGpu(black_gpu, black.data, black.width*black.height*black.channel);
    YnCudaArrayPullFromGpu(black2_gpu, black2.data, black2.width*black2.height*black2.channel);
    YnImageShowLayers(black, "Black");
    YnImageShow(black2, "Recreate");
    #endif

    YnImageShow(im, "Original");
    YnImageShow(gray, "Gray");
    YnImageShow(sat2, "Saturation-2");
    YnImageShow(sat5, "Saturation-.5");
    YnImageShow(exp2, "Exposure-2");
    YnImageShow(exp5, "Exposure-.5");

#ifdef YN_OPENCV
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
    for (k = 0; k < c; k ++)
    {
        for (j = 0; j < h; j ++)
        {
            for (i = 0; i < w; i ++)
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

tYnImage YnImageLoadColor(char *filename,
        int w,
        int h)
{
    return YnImageLoad(filename, w, h, 3);
}

tYnImage YnImageGetLayer(tYnImage m,
        int l)
{
    tYnImage out = YnImageMake(m.width, m.height, 1);
    int i;

    for (i = 0; i < (m.height * m.width); i ++)
    {
        out.data[i] = m.data[i + l * m.height * m.width];
    }

    return out;
}

float YnImageGetPixel(tYnImage m,
        int x,
        int y,
        int c)
{
    assert((x < m.width) && (y < m.height) && (c < m.channel));
    return m.data[(c * m.height * m.width) + (y * m.width) + x];
}

float YnImageGetPixelExtend(tYnImage m,
        int x,
        int y,
        int c)
{
    if ((x < 0) || (x >= m.width) || (y < 0) || (y >= m.height) || (c < 0) || (c >= m.channel))
        return 0;

    return YnImageGetPixel(m, x, y, c);
}

void YnImageSetPixel(tYnImage m,
        int x,
        int y,
        int c,
        float val)
{
    assert((x < m.width) && (y < m.height) && (c < m.channel));
    m.data[c * m.height * m.width + y * m.width + x] = val;
}

void YnImageAddPixel(tYnImage m,
        int x,
        int y,
        int c,
        float val)
{
    assert((x < m.width) && (y < m.height) && (c < m.channel));
    m.data[(c * m.height * m.width) + (y * m.width) + x] += val;
}

void YnImagePrint(tYnImage m)
{
    int i, j, k;

    for (i =0 ; i < m.channel; i ++)
    {
        for (j =0 ; j < m.height; j ++)
        {
            for (k = 0; k < m.width; k ++)
            {
                printf("%.2lf, ", m.data[i * m.height * m.width + j * m.width + k]);
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
    tYnImage layer;

    w = ims[0].width;
    h = (ims[0].height + border) * n - border;
    c = ims[0].channel;

    if ((c != 3) || !color)
    {
        w = (w + border) * c - border;
        c = 1;
    }

    tYnImage filters = YnImageMake(w, h, c);

    for (i = 0; i < n; i ++)
    {
        h_offset = i * (ims[0].height + border);
        tYnImage copy = YnImageCopy(ims[i]);

        if ((c == 3) && color)
        {
            YnImageEmbed(copy, filters, 0, h_offset);
        }
        else
        {
            for (j = 0; j < copy.channel; j ++)
            {
                w_offset = j * (ims[0].width + border);
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
    int size = ims[0].height;

    h = size;
    w = (ims[0].width + border) * n - border;
    c = ims[0].channel;

    if (c != 3 || !color)
    {
        h = (h+border)*c - border;
        c = 1;
    }

    filters = YnImageMake(w, h, c);

    for (i = 0; i < n; i ++)
    {
        w_offset = i * (size+border);
        copy = YnImageCopy(ims[i]);

        if ((c == 3) && color)
        {
            YnImageEmbed(copy, filters, w_offset, 0);
        }
        else
        {
            for (j = 0; j < copy.channel; j ++)
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

void YnImageImagesShow(tYnImage *ims,
        int n,
        char *window)
{
    tYnImage m = YnImageCollapseVert(ims, n);
    tYnImage sized;

    YnImageNormalize(m);
    sized = YnImageResize(m, m.width, m.height);
    YnImageSave(sized, window);
    YnImageShow(sized, window);
    YnImageFree(sized);
    YnImageFree(m);
}

void YnImageFree(tYnImage m)
{
    YnUtilFree(m.data);
}

void YnImageImage2Col(float* data_im,
        int channels,
        int height,
        int width,
        int ksize,
        int stride,
        int pad,
        float* data_col)
{
    int c,h,w;
    int channels_col;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int w_offset;
    int h_offset;
    int c_im;
    int im_row;
    int im_col;
    int col_index;

    if (pad)
    {
        height_col = 1 + (height - 1) / stride;
        width_col = 1 + (width - 1) / stride;
        pad = ksize/2;
    }

    channels_col = channels * ksize * ksize;

    for (c = 0; c < channels_col; c ++)
    {
        w_offset = c % ksize;
        h_offset = (c / ksize) % ksize;
        c_im = c / ksize / ksize;

        for (h = 0; h < height_col; h ++)
        {
            for (w = 0; w < width_col; w ++)
            {
                im_row = h_offset + h * stride;
                im_col = w_offset + w * stride;
                col_index = (c * height_col + h) * width_col + w;

                data_col[col_index] = _YnImage2ColPixelGet(data_im, height, width, channels,
                                                           im_row, im_col, c_im, pad);
            }
        }
    }
}

void YnImageCol2Image(float* data_col,
         int channels,
         int height,
         int width,
         int ksize,
         int stride,
         int pad,
         float* data_im)
{
    int c,h,w;
    int channels_col;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int w_offset;
    int h_offset;
    int c_im;
    int im_row;
    int im_col;
    int col_index;
    double val;


    if (pad)
    {
        height_col = 1 + (height - 1) / stride;
        width_col = 1 + (width - 1) / stride;
        pad = ksize/2;
    }

    channels_col = channels * ksize * ksize;

    for (c = 0; c < channels_col; c ++)
    {
        w_offset = c % ksize;
        h_offset = (c / ksize) % ksize;
        c_im = c / ksize / ksize;

        for (h = 0; h < height_col; h ++)
        {
            for (w = 0; w < width_col; w ++)
            {
                im_row = h_offset + h * stride;
                im_col = w_offset + w * stride;
                col_index = (c * height_col + h) * width_col + w;
                val = data_col[col_index];

                _YnImage2ColPixelAdd(data_im, height, width, channels, im_row, im_col, c_im, pad, val);
            }
        }
    }
}
