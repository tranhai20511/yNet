#ifndef YNIMAGE_H
#define YNIMAGE_H

#include "../YnStd.h"

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

/*
 * Get color value
 */
YN_FINAL
void YnImageDrawLabel(tYnImage image,
        int x, int max)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif
