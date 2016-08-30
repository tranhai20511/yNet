#ifndef YNBBOX_H
#define YNBBOX_H

#include "../YnStd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */
typedef struct tYnBBox{
    float x, y;
    float width;
    float height;
} tYnBBox;

typedef struct tYnBBoxSend{
    float x, y;
    float width;
    float height;

    int classId;
} tYnBBoxSend;

typedef struct tYnBBoxD{
    float dx, dy;
    float dwidth, dheight;
} tYnBBoxD;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

/*
 * Convert array to box
 */
YN_FINAL
tYnBBox YnBBoxFromArray(float * array)
YN_ALSWAY_INLINE;

/*
 * Box Intersection over union
 */
YN_FINAL
float YnBBoxIou(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

/*
 * Box rmse
 */
YN_FINAL
float YnBBoxRmse(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

/*
 * Box IOU D
 */
YN_FINAL
tYnBBoxD YnBBoxIouD(tYnBBox box1,
        tYnBBox box2)
YN_ALSWAY_INLINE;

/*
 * Box NMS
 */
YN_FINAL
void YnBBoxNms(tYnBBox * boxes,
        float ** probs,
        uint32 totalBox,
        uint32 classes,
        float thresh
        )
YN_ALSWAY_INLINE;

/*
 * Box NMS sort
 */
YN_FINAL
void YnBBoxNmsSort(tYnBBox * boxes,
        float ** probs,
        uint32 totalBox,
        uint32 classes,
        float thresh
        )
YN_ALSWAY_INLINE;

/*
 * Box decode
 */
YN_FINAL
tYnBBox YnBBoxDecode(tYnBBox * box,
        tYnBBox * anchor)
YN_ALSWAY_INLINE;

/*
 * Box encode
 */
YN_FINAL
tYnBBox YnBBoxEncode(tYnBBox * box,
        tYnBBox * anchor)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNBBOX_H */
