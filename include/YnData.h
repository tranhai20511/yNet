#ifndef YNDATA_H
#define YNDATA_H

#include "YnMatrix.h"
#include "YnImage.h"
#include "YnList.h"
#include "YnUtil.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Typedef */

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
int YnDataSeedGet(void)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataSeedSet(int seed)
YN_ALSWAY_INLINE;

YN_FINAL
tYnList * YnDataPathsGet(char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
char ** YnDataRandomPathsGet(char ** paths,
        int n,
        int m)
YN_ALSWAY_INLINE;

YN_FINAL
char ** YnDataFindReplacePaths(char **paths,
        int n,
        char *find,
        char *replace)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnDataLoadImagePathsGray(char **paths,
        int n,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnDataLoadImagePaths(char **paths,
        int n,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnDataBoxLabel * YnDataReadBoxes(char *filename,
        int *n)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataRandomizeBoxes(tYnDataBoxLabel *boxes,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataCorrectBoxes(tYnDataBoxLabel *boxes,
        int n,
        float dx,
        float dy,
        float sx,
        float sy,
        int flip)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataFillTruthSwag(char *path,
        float *truth,
        int classes,
        int flip,
        float dx,
        float dy,
        float sx,
        float sy)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataFillTruthRegion(char *path,
        float *truth,
        int classes,
        int numBoxes,
        int flip,
        float dx,
        float dy,
        float sx,
        float sy)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataFillTruthDetection(char *path,
        float *truth,
        int classes,
        int numBoxes,
        int flip,
        int background,
        float dx,
        float dy,
        float sx,
        float sy)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataPrintLetters(float *pred,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataFillTruthCaptcha(char *path,
        int n,
        float *truth)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadCaptcha(char **paths,
        int n,
        int m,
        int k,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadCaptchaEncode(char **paths,
        int n,
        int m,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataFillTruth(char *path,
        char **labels,
        int k,
        float *truth)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnDataLoadLabelsPaths(char **paths,
        int n,
        char **labels,
        int k)
YN_ALSWAY_INLINE;

YN_FINAL
char ** YnDataLabelsGet(char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataFree(tYnData data)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadRegion(int n,
        char **paths,
        int m,
        int w,
        int h,
        int size,
        int classes,
        float jitter)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadCompare(int n,
        char **paths,
        int m,
        int classes,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadCompare(int n,
        char **paths,
        int m,
        int classes,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadSwag(char **paths,
        int n,
        int classes,
        float jitter)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadDetection(int n,
        char **paths,
        int m,
        int classes,
        int w,
        int h,
        int numBoxes,
        int background)
YN_ALSWAY_INLINE;

YN_FINAL
void * YnDataLoadThread(void *ptr)
YN_ALSWAY_INLINE;

YN_FINAL
pthread_t YnDataLoadInThread(tYnDataLoadArgs args)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoadWriting(char **paths,
        int n,
        int m,
        int w,
        int h,
        int out_w,
        int out_h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataLoad(char **paths,
        int n,
        int m,
        char **labels,
        int k,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnDataConcatMatrix(tYnMatrix m1,
        tYnMatrix m2)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData YnDataConcat(tYnData d1,
        tYnData d2)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataRandomBatchGet(tYnData d,
        int n,
        float * X,
        float * y)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataNextBatchGet(tYnData d,
        int n,
        int offset,
        float *X,
        float *y)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataRandomize(tYnData d)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataScaleRows(tYnData d,
        float s)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataTranslateRows(tYnData d,
        float s)
YN_ALSWAY_INLINE;

YN_FINAL
void YnDataNormalizeRows(tYnData d)
YN_ALSWAY_INLINE;

YN_FINAL
tYnData * YnDataSplit(tYnData d,
        int part,
        int total)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNDATA_H */
