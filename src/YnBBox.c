//	File        :   YnBBox.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   03-07-2016
//	Author      :   haittt

#include "../include/YnUtil.h"
#include "../include/YnBBox.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

typedef struct tYnBBoxSortable{
    int index;
    int class;

    float **probs;
} tYnBBoxSortable;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
YN_STATIC_INLINE
tYnBBoxD _YnBBoxD(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
float _YnBBoxOverlap(float x1,
        float width1,
        float x2,
        float width2)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
float _YnBBoxIntersection(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
float _YnBBoxUnion(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnBBoxD _YnBBoxIntersectionD(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnBBoxD _YnBBoxUnionD(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int _YnBBoxNmsComparator(const void *box1,
        const void *box2)
YN_ALSWAY_INLINE;

/**************** Implement */

YN_STATIC_INLINE
tYnBBoxD _YnBBoxD(tYnBBox box1,
        tYnBBox box2)
{
    tYnBBoxD d = {0};
    float l1 = 0;
    float l2 = 0;
    float r1 = 0;
    float r2 = 0;
    float t1 = 0;
    float t2 = 0;
    float b1 = 0;
    float b2 = 0;

    l1 = box1.x - box1.width / 2;
    l2 = box2.x - box2.width / 2;
    if (l1 > l2)
    {
        d.dx -= 1;
        d.dwidth += .5;
    }

    r1 = box1.x + box1.width / 2;
    r2 = box2.x + box2.width / 2;
    if (r1 < r2)
    {
        d.dx += 1;
        d.dwidth += .5;
    }

    if (l1 > r2)
    {
        d.dx = -1;
        d.dwidth = 0;
    }
    if (r1 < l2)
    {
        d.dx = 1;
        d.dwidth = 0;
    }

    t1 = box1.y - box1.height / 2;
    t2 = box2.y - box2.height / 2;
    if (t1 > t2)
    {
        d.dy -= 1;
        d.dheight += .5;
    }

    b1 = box1.y + box1.height / 2;
    b2 = box2.y + box2.height / 2;
    if (b1 < b2)
    {
        d.dy += 1;
        d.dheight += .5;
    }

    if (t1 > b2)
    {
        d.dy = -1;
        d.dheight = 0;
    }
    if (b1 < t2)
    {
        d.dy = 1;
        d.dheight = 0;
    }
    return d;
}

YN_STATIC_INLINE
float _YnBBoxOverlap(float x1,
        float width1,
        float x2,
        float width2)
{
    float l1 = x1 - width1 / 2;
    float l2 = x2 - width2 / 2;

    float left = l1 > l2 ? l1 : l2;

    float r1 = x1 + width1 / 2;
    float r2 = x2 + width2 / 2;

    float right = r1 < r2 ? r1 : r2;

    return right - left;
}

YN_STATIC_INLINE
float _YnBBoxIntersection(tYnBBox box1,
        tYnBBox box2)
{
    float w = _YnBBoxOverlap(box1.x, box1.width, box2.x, box2.width);
    float h = _YnBBoxOverlap(box1.y, box1.height, box2.y, box2.height);

    if (w < 0 || h < 0)
        return 0;

    return (w * h);
}

YN_STATIC_INLINE
float _YnBBoxUnion(tYnBBox box1,
        tYnBBox box2)
{
    float i = _YnBBoxIntersection(box1, box2);
    float u = box1.width * box1.height + box2.width * box2.height - i;

    return u;
}

YN_STATIC_INLINE
tYnBBoxD _YnBBoxIntersectionD(tYnBBox box1,
        tYnBBox box2)
{
    tYnBBoxD di = {0};
    float w = _YnBBoxOverlap(box1.x, box1.width, box2.x, box2.width);
    float h = _YnBBoxOverlap(box1.y, box1.height, box2.y, box2.height);
    tYnBBoxD dover = _YnBBoxD(box1, box2);


    di.dwidth = dover.dwidth * h;
    di.dx = dover.dx * h;
    di.dheight = dover.dheight * w;
    di.dy = dover.dy * w;

    return di;
}

YN_STATIC_INLINE
tYnBBoxD _YnBBoxUnionD(tYnBBox box1,
        tYnBBox box2)
{
    tYnBBoxD dunion = {0};

    tYnBBoxD dIntersec = _YnBBoxIntersectionD(box1, box2);
    dunion.dwidth = box1.height - dIntersec.dwidth;
    dunion.dheight = box1.width - dIntersec.dheight;
    dunion.dx = - dIntersec.dx;
    dunion.dy = - dIntersec.dy;

    return dunion;
}

YN_STATIC_INLINE
int _YnBBoxNmsComparator(const void * pBox1,
        const void * pBox2)
{
    float diff = 0;
    tYnBBoxSortable box1 = *(tYnBBoxSortable *) pBox1;
    tYnBBoxSortable box2 = *(tYnBBoxSortable *) pBox2;

    diff = box1.probs[box1.index][box2.class] - box2.probs[box2.index][box2.class];

    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;

    return 0;
}

tYnBBox YnBBoxFromArray(float * array)
{
    tYnBBox retBox = {0};

    retBox.x = array[0];
    retBox.y = array[1];
    retBox.width = array[2];
    retBox.height = array[3];

    return retBox;
}

float YnBBoxIou(tYnBBox box1,
        tYnBBox box2)
{
    return (_YnBBoxIntersection(box1, box2) / _YnBBoxUnion(box1, box2));
}

float YnBBoxRmse(tYnBBox box1,
        tYnBBox box2)
{
    return sqrt(pow(box1.x - box2.x, 2) +
                pow(box1.y - box2.y, 2) +
                pow(box1.width - box2.width, 2) +
                pow(box1.height - box2.height, 2));
}

tYnBBoxD YnBBoxIouD(tYnBBox box1,
        tYnBBox box2)
{
    float u = _YnBBoxUnion(box1, box2);
    float i = _YnBBoxIntersection(box1, box2);
    tYnBBoxD di = _YnBBoxIntersectionD(box1, box2);
    tYnBBoxD du = _YnBBoxUnionD(box1, box2);
    tYnBBoxD dd = { 0, 0, 0, 0 };

    /* Fixme */
    if (i <= 0 || 1)
    {
        dd.dx = box2.x - box1.x;
        dd.dy = box2.y - box1.y;
        dd.dwidth= box2.width- box1.width;
        dd.dheight = box2.height - box1.height;
        return dd;
    }

    dd.dx = 2 * pow((1 - (i / u)), 1) * (di.dx * u - du.dx * i) / (u * u);
    dd.dy = 2 * pow((1 - (i / u)), 1) * (di.dy * u - du.dy * i) / (u * u);
    dd.dwidth= 2 * pow((1 - (i / u)), 1) * (di.dwidth* u - du.dwidth* i) / (u * u);
    dd.dheight = 2 * pow((1 - (i / u)), 1) * (di.dheight * u - du.dheight * i) / (u * u);

    return dd;
}

/* Scan all boxes */
void YnBBoxNms(tYnBBox * boxes,
        float ** probs,
        uint32 totalBox,
        uint32 classes,
        float thresh)
{
    uint32 i, j, k;
    tYnBBoxSortable *sortBoxes = calloc(totalBox, sizeof(tYnBBoxSortable));
    tYnBBox a, b;
    uint32 indexInBoxesData = 0;

    for (i = 0; i < totalBox; i ++)
    {
        sortBoxes[i].index = i;
        sortBoxes[i].class = 0;
        sortBoxes[i].probs = probs;
    }

    for (k = 0; k < classes; k ++)
    {
        for (i = 0; i < totalBox; i ++)
        {
            sortBoxes[i].class = k;
        }

        /* Sort all boxes by prob value */
        qsort(sortBoxes, totalBox, sizeof(tYnBBoxSortable), _YnBBoxNmsComparator);

        for (i = 0; i < totalBox; i ++)
        {
            if (probs[sortBoxes[i].index][k] == 0)
                continue;

            indexInBoxesData = sortBoxes[i].index;
            a = boxes[indexInBoxesData];

            for (j = i + 1; j < totalBox; j ++)
            {
                b = boxes[sortBoxes[j].index];

                if (YnBBoxIou(a, b) > thresh)
                {
                    probs[sortBoxes[j].index][k] = 0;
                }
            }
        }
    }

    YnUtilFree(sortBoxes);
}

void YnBBoxNmsSort(tYnBBox * boxes,
        float ** probs,
        uint32 totalBox,
        uint32 classes,
        float thresh)
{
    int i, j, k;
    int any = 0;

    for (i = 0; i < totalBox; i ++)
    {
        any = 0;

        for (k = 0; k < classes; k ++)
        {
            any = any || (probs[i][k] > 0);
        }

        if (!any)
        {
            continue;
        }

        for (j = i + 1; j < totalBox; j ++)
        {
            if (YnBBoxIou(boxes[i], boxes[j]) > thresh)
            {
                for (k = 0; k < classes; ++k)
                {
                    if (probs[i][k] < probs[j][k])
                        probs[i][k] = 0;
                    else
                        probs[j][k] = 0;
                }
            }
        }
    }
}

tYnBBox YnBBoxDecode(tYnBBox box,
        tYnBBox anchor)
{
    tYnBBox encode = {0};

    encode.x = (box.x - anchor.x) / anchor.width;
    encode.y = (box.y - anchor.y) / anchor.height;
    encode.width = log2(box.width / anchor.width);
    encode.height = log2(box.height / anchor.height);

    return encode;
}

tYnBBox YnBBoxEncode(tYnBBox box,
		tYnBBox anchor)
{
    tYnBBox decode = {0};

    decode.x = box.x * anchor.width + anchor.x;
    decode.y = box.y * anchor.height + anchor.y;
    decode.width = pow(2., box.width) * anchor.width;
    decode.height = pow(2., box.height) * anchor.height;

    return decode;
}
