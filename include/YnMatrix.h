#ifndef YNMATRIX_H
#define YNMATRIX_H

#include "YnStd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */
typedef struct tYnMatrix{
    uint32 rows;
    uint32 cols;
    float ** vals;
} tYnMatrix;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
tYnMatrix YnMatrixMake(uint32 rows,
        uint32 cols)
YN_ALSWAY_INLINE;

YN_FINAL
void YnMatrixFree(tYnMatrix m)
YN_ALSWAY_INLINE;

YN_FINAL
void YnMatrixPrint(tYnMatrix m)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnMatrixCsvToMatrix(char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnMatrixHoldOut(tYnMatrix * matrix,
        int num)
YN_ALSWAY_INLINE;

YN_FINAL
float YnMatrixTopAccuracy(tYnMatrix truth,
        tYnMatrix guess,
        int k)
YN_ALSWAY_INLINE;

YN_FINAL
void YnMatrixAdd(tYnMatrix from,
        tYnMatrix to)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnMatrixPopColumn(tYnMatrix * matrix,
        int c)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNMATRIX_H */
