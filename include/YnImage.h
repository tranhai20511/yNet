#ifndef YNIMAGE_H
#define YNIMAGE_H

#include "../YnStd.h"

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
 * Set value for array elements
 */
YN_FINAL

#ifdef __cplusplus
}
#endif
