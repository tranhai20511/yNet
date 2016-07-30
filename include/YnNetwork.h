#ifndef YNNETWORK_H
#define YNNETWORK_H

#include "../YnLayer.h"

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

#ifdef __cplusplus
}
#endif

#endif /* YNNETWORK_H */
