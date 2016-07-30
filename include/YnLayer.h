#ifndef YNLAYER_H
#define YNLAYER_H

#include "../YnStd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */
typedef enum eYnLayerType{
    cYnLayerConvolutional,
    cYnLayerDeconvolutional,
    cYnLayerConnected,
    cYnLayerMaxpool,
    cYnLayerSoftmax,
    cYnLayerDetection,
    cYnLayerDropout,
    cYnLayerCrop,
    cYnLayerRoute,
    cYnLayerCost,
    cYnLayerNormalization,
    cYnLayerAvgpool,
    cYnLayerLocal,
    cYnLayerShortcut,
    cYnLayerActive,
    cYnLayerRnn
} eYnLayerType;

typedef enum eYnLayerCostType{
    cYnLayerCostSse,
    cYnLayerCostMasked,
    cYnLayerCostSmooth,
}eYnLayerCostType;

/**************** Struct */
typedef struct tYnLayer{
    eYnLayerType type;
    eYnActivationType activation;
    eYnLayerCostType costType;

    int n;
    int batchNormalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int truths;
    int h,w,c;
    int outH, outW, outC;
    int groups;
    int size;
    int side;
    int stride;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int steps;
    int hidden;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;

    float alpha;
    float beta;
    float kappa;

    float coordScale;
    float objectScale;
    float noobjectScale;
    float classScale;

    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    int * indexes;
    float * rand;
    float * cost;
    float * filters;
    float * filterUpdates;
    float * state;

    float * binaryFilters;

    float * biases;
    float * biasUpdates;

    float * scales;
    float * scaleUpdates;

    float * weights;
    float * weightUpdates;

    float * colImage;
    int   * inputLayers;
    int   * inputSizes;
    float * delta;
    float * output;
    float * squared;
    float * norms;

    float * spatialMean;
    float * mean;
    float * variance;

    float * meanDelta;
    float * varianceDelta;

    float * rollingMean;
    float * rollingVriance;

    float * x;
    float * xNorm;

    tYnLayer * inputLayer;
    tYnLayer * selfLayer;
    tYnLayer * outputLayer;

    int * indexesGpu;
    float * stateGpu;
    float * filtersGpu;
    float * filterUpdatesGpu;

    float * binaryFiltersGpu;
    float * meanFiltersGpu;

    float * spatialMeanGpu;
    float * spatialVarianceGpu;

    float * meanGpu;
    float * varianceGpu;

    float * rollingMeanGpu;
    float * rollingVarianceGpu;

    float * spatialMean_deltaGpu;
    float * spatialVariance_deltaGpu;

    float * varianceDeltaGpu;
    float * meanDeltaGpu;

    float * colImageGpu;

    float * xGpu;
    float * xNormGpu;
    float * weightsGpu;
    float * weightUpdatesGpu;

    float * biasesGpu;
    float * biasUpdatesGpu;

    float * scalesGpu;
    float * scaleUpdatesGpu;

    float * outputGpu;
    float * deltaGpu;
    float * randGpu;
    float * squaredGpu;
    float * normsGpu;
}tYnLayer;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
void YnLayerFree(tYnLayer layer)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNLAYER_H */
