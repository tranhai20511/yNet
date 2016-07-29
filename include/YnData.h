#ifndef YNDATA_H
#define YNDATA_H

#include "../YnMatrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Typedef */
typedef struct YnData{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
} YnData;

/**************** Macro */

/**************** Enum */
typedef enum eYnDataType{
    cYnDataClassification,
    cYnDataDetection,
    cYnDataCaptcha,
    cYnDataRegion,
    cYnDataImage,
    cYnDataCompare,
    cYnDataWriting,
    cYnDataSwag
} eYnDataType;

/**************** Struct */
typedef struct tYnData{
    int w;
    int h;
    tYnMatrix x;
    tYnMatrix y;
    int shallow;
} tYnData;

typedef struct tYnDataLoadArgs{
    char ** paths;
    char * path;
    int n;
    int m;
    char ** labels;
    int h;
    int w;
    int outW;
    int outH;
    int nh;
    int nw;
    int numBoxes;
    int classes;
    int background;
    float jitter;
    tYnData * d;
    tYnImage * im;
    tYnImage * resized;
    eYnDataType type;
} tYnDataLoadArgs;

typedef struct tYnDataBoxLabel{
    int id;
    float x;
    float y;
    float w;
    float h;
    float left;
    float right;
    float top;
    float bottom;
} tYnDataBoxLabel;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
uint32 YndataSeedGet(void)
YN_ALSWAY_INLINE;

YN_FINAL
void YndataSeedSet(uint32 seed)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNDATA_H */
