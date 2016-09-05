#ifndef YNVOC_H
#define YNVOC_H

#include "YnNetwork.h"

#ifdef YN_GPU
#include "YnNetworkGpu.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Typedef */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
void YnVocTrain(char *cfgfile,
        char *weightfile)
YN_ALSWAY_INLINE;

YN_FINAL
void YnVocConvertDetections(float *predictions,
        int classes,
        int num,
        int square,
        int side,
        int w,
        int h,
        float thresh,
        float **probs,
        tYnBBox *boxes,
        int only_objectness)
YN_ALSWAY_INLINE;

YN_FINAL
void YnVocTest(char *cfgfile,
        char *weightfile,
        char *filename,
        float thresh)
YN_ALSWAY_INLINE;

YN_FINAL
void YnVocDemo(char *cfgfile,
        char *weightfile,
        float thresh,
        int cam_index,
        const char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnVocRun(int argc,
        char **argv)
YN_ALSWAY_INLINE;

#ifdef YN_OPENCV

YN_FINAL
void YnVocCpuDemo(char *cfgfile,
        char *weightfile,
        float thresh,
        int cam_index,
		const char *filename)
YN_ALSWAY_INLINE;

#endif

#ifdef YN_GPU

YN_FINAL
void YnVocGpuDemo(char *cfgfile,
        char *weightfile,
        float thresh,
        int cam_index,
        char *filename)
YN_ALSWAY_INLINE;

#endif

#ifdef __cplusplus
}
#endif

#endif /* YNVOC_H */
