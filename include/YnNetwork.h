#ifndef YNNETWORK_H
#define YNNETWORK_H

#include "../YnLayer.h"
#include "../YnData.h"
#include "../YnMatrix.h"
#include "../YnParser.h"
#include "../YnUtil.h"
#include "../YnBlas.h"
#include "../YnLayerCrop.h"
#include "../YnLayerConnected.h"
#include "../YnLayerConvolutional.h"
#include "../YnLayerActivation.h"
#include "../YnLayerAvgpool.h"
#include "../YnLayerDeconvolutional.h"
#include "../YnLayerDetection.h"
#include "../YnLayerMaxpool.h"
#include "../YnLayerCost.h"
#include "../YnLayerSoftmax.h"
#include "../YnLayerDropout.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Typedef */

/**************** Macro */

/**************** Enum */
typedef enum eYnNetworkLearnRatePolicy{
    eYnNetworkLearnRateConstant,
    eYnNetworkLearnRateStep,
    eYnNetworkLearnRateExp,
    eYnNetworkLearnRatePoly,
    eYnNetworkLearnRateSteps,
    eYnNetworkLearnRateSig,
} eYnNetworkLearnRatePolicy;

/**************** Struct */
typedef struct tYnNetwork{
    int n;
    int batch;
    int * seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    tYnLayer * layers;
    int outputs;
    float * output;
    eYnNetworkLearnRatePolicy policy;

    float learningRate;
    float gamma;
    float scale;
    float power;
    int timeSteps;
    int step;
    int max_batches;
    float * scales;
    int   * steps;
    int num_steps;

    int inputs;
    int h;
    int w;
    int c;

    float ** inputGpu;
    float ** truthGpu;
}tYnNetwork;

typedef struct tYnNetworkState {
    float *truth;
    float *input;
    float *delta;
    int train;
    int index;
    tYnNetwork net;
} tYnNetworkState;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
int YnNetworkGetCurrentBatch(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkResetMomentum(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkCurrentRateget(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
char *YnNetworkLayerStringGet(eYnLayerType type)
YN_ALSWAY_INLINE;

YN_FINAL
tYnNetwork YnNetworkMake(int n)
YN_ALSWAY_INLINE;

YN_FINAL
tYnNetwork YnNetworkMake(int n)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkForward(tYnNetwork net,
        tYnNetworkState state)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkUpdate(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnNetworkOutputGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkCostGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
int YnNnetworkPredictedClassNetworkGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
int YnNnetworkPredictedClassNetworkGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkBackward(tYnNetwork net,
        tYnNetworkState state)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkTrainDatum(tYnNetwork net,
        float * x,
        float * y)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkTrainSgd(tYnNetwork net,
        tYnData d,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkTrain(tYnNetwork net,
        tYnData d)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkTrainBatch(tYnNetwork net,
        tYnData d,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkBatchSet(tYnNetwork *net,
        int b)
YN_ALSWAY_INLINE;

YN_FINAL
int YnNetworkResize(tYnNetwork * net,
        int w,
        int h)
YN_ALSWAY_INLINE;

YN_FINAL
int YnNetworkOutputSizeGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
int YnNetworkInputSizeGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
tYnLayer YnNetworkDetectionLayerGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnNetworkImageLayerGet(tYnNetwork net,
        int i)
YN_ALSWAY_INLINE;

YN_FINAL
tYnImage YnNetworkImageGet(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkVisualize(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkTopPredictions(tYnNetwork net,
        int k,
        int *index)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnNetworkPredict(tYnNetwork net,
        float *input)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnNetworkPredictDataMulti(tYnNetwork net,
        tYnData test,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
tYnMatrix YnNetworkPredictData(tYnNetwork net,
        tYnData test)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkPrint(tYnNetwork net)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkCompare(tYnNetwork n1,
        tYnNetwork n2,
        tYnData test)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkAccuracy(tYnNetwork net,
        tYnData d)
YN_ALSWAY_INLINE;

YN_FINAL
float * YnnNetworkAccuracies(tYnNetwork net,
        tYnData d,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
float YnNetworkAccuracyMulti(tYnNetwork net,
        tYnData d,
        int n)
YN_ALSWAY_INLINE;

YN_FINAL
void YnNetworkFree(tYnNetwork net)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNNETWORK_H */
