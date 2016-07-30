//	File        :   YnNetwork.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   26-07-2016
//	Author      :   haittt

#include "../include/YnNetwork.h"
#include "../include/YnLayerCrop.h"
#include "../include/YnLayerConnected.h"
#include "../include/YnLayerRnn.h"
#include "../include/YnLayerLocal.h"
#include "../include/YnLayerConvolutional.h"
#include "../include/YnLayerActivation.h"
#include "../include/YnLayerDeconvolutional.h"
#include "../include/YnLayerDetection.h"
#include "../include/YnLayerNormalization.h"
#include "../include/YnLayerMaxpool.h"
#include "../include/YnLayerAvgpool.h"
#include "../include/YnLayerCost.h"
#include "../include/YnLayerSoftmax.h"
#include "../include/YnLayerDropout.h"
#include "../include/YnLayerRoute.h"
#include "../include/YnLayerShortcut.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
int YnNetworkGetCurrentBatch(tYnNetwork net)
{
    return ((*net.seen) / (net.batch * net.subdivisions));
}

void YnNetworkResetMomentum(tYnNetwork net)
{
    if (net.momentum == 0)
        return;

    net.learningRate = 0;
    net.momentum = 0;
    net.decay = 0;

#ifdef YN_GPU
    if (YnCudaGpuIndexGet() >= 0)
        YnNetworkUpdateGpu(net);
#endif
}

float YnNetworkCurrentRateget(tYnNetwork net)
{
    int batchNum = YnNetworkCurrentBatchGet(net);
    int i;
    float rate;

    switch (net.policy)
    {
        case eYnNetworkLearnRateConstant:
            return net.learningRate;
        case eYnNetworkLearnRateStep:
            return net.learningRate * pow(net.scale, batchNum / net.step);
        case eYnNetworkLearnRateSteps:
            rate = net.learningRate;

            for (i = 0; i < net.num_steps; i ++)
            {
                if (net.steps[i] > batchNum)
                    return rate;

                rate *= net.scales[i];

                if (net.steps[i] > batchNum - 1)
                    YnNwtworkResetMomentum(net);
            }

            return rate;
        case eYnNetworkLearnRateExp:
            return net.learningRate * pow(net.gamma, batchNum);
        case eYnNetworkLearnRatePoly:
            return net.learningRate * pow(1 - (float)batchNum / net.max_batches, net.power);
        case eYnNetworkLearnRateSig:
            return net.learningRate * (1. / (1. + exp(net.gamma * (batchNum - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learningRate;
    }
}

char *YnNetworkLayerStringGet(eYnLayerType type)
{
    switch(type)
    {
        case cYnLayerConvolutional:
            return "convolutional";
        case cYnLayerActive:
            return "activation";
        case cYnLayerLocal:
            return "local";
        case cYnLayerDeconvolutional:
            return "deconvolutional";
        case cYnLayerConnected:
            return "connected";
        case cYnLayerRnn:
            return "rnn";
        case cYnLayerMaxpool:
            return "maxpool";
        case cYnLayerAvgpool:
            return "avgpool";
        case cYnLayerSoftmax:
            return "softmax";
        case cYnLayerDetection:
            return "detection";
        case cYnLayerDropout:
            return "dropout";
        case cYnLayerCrop:
            return "crop";
        case cYnLayerCost:
            return "cost";
        case cYnLayerRoute:
            return "route";
        case cYnLayerShortcut:
            return "shortcut";
        case cYnLayerNormalization:
            return "normalization";
        default:
            break;
    }

    return "none";
}

tYnNetwork YnNetworkMake(int n)
{
    tYnNetwork net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(tYnLayer));
    net.seen = calloc(1, sizeof(int));

    #ifdef YN_GPU
    net.inputGpu = calloc(1, sizeof(float *));
    net.truthGpu = calloc(1, sizeof(float *));
    #endif

    return net;
}

void YnNetworkForward(tYnNetwork net,
        tYnNetworkState state)
{
    int i;
    tYnLayer layer;

    for (i = 0; i < net.n; i ++)
    {
        state.index = i;
        layer = net.layers[i];

        if (layer.delta)
        {
            YnBlasArrayScaleValueSet(layer.delta, layer.outputs * layer.batch, 1, 0);
        }

        switch(layer.type)
        {
            case cYnLayerConvolutional:
                YnLayerConvolutionalForward(layer, state);
                break;
            case cYnLayerActive:
                YnLayerActiveForward(layer, state);
                break;
            case cYnLayerLocal:
                YnLayerLocalForward(layer, state);
                break;
            case cYnLayerDeconvolutional:
                YnLayerDeconvolutionalForward(layer, state);
                break;
            case cYnLayerConnected:
                YnLayerConnectedForward(layer, state);
                break;
            case cYnLayerRnn:
                YnLayerRnnForward(layer, state);
                break;
            case cYnLayerMaxpool:
                YnLayerMaxpoolForward(layer, state);
                break;
            case cYnLayerAvgpool:
                YnLayerAvgpoolForward(layer, state);
                break;
            case cYnLayerSoftmax:
                YnLayerSoftmaxForward(layer, state);
                break;
            case cYnLayerDetection:
                YnLayerDetectionForward(layer, state);
                break;
            case cYnLayerDropout:
                YnLayerDropoutForward(layer, state);
                break;
            case cYnLayerCrop:
                YnLayerCropForward(layer, state);
                break;
            case cYnLayerCost:
                YnLayerCostForward(layer, state);
                break;
            case cYnLayerRoute:
                YnLayerRouteForward(layer, state);
                break;
            case cYnLayerShortcut:
                YnLayerShortcutForward(layer, state);
                break;
            case cYnLayerNormalization:
                YnLayerNormalizationForward(layer, state);
                break;
            default:
                break;
        }

        state.input = layer.output;
    }
}

void YnNetworkUpdate(tYnNetwork net)
{
    int i;
    tYnLayer layer;
    int updateBatch = net.batch * net.subdivisions;
    float rate = get_current_rate(net);

    for (i = 0; i < net.n; i ++)
    {
        layer = net.layers[i];
        if (layer.type == cYnLayerConvolutional)
        {
            YnLayerConvolutionalUpdate(layer, updateBatch, rate, net.momentum, net.decay);
        }
        else if (layer.type == cYnLayerDeconvolutional)
        {
            YnLayerDeconvolutionalUpdate(layer, rate, net.momentum, net.decay);
        }
        else if (layer.type == cYnLayerConnected)
        {
            YnLayerConnectedUpdate(layer, updateBatch, rate, net.momentum, net.decay);
        }
        else if (layer.type == cYnLayerRnn)
        {
            YnLayerRnnUpdate(layer, updateBatch, rate, net.momentum, net.decay);
        }
        else if (layer.type == cYnLayerLocal)
        {
            YnLayerLocalUpdate(layer, updateBatch, rate, net.momentum, net.decay);
        }
    }
}

float * YnNetworkOutputGet(tYnNetwork net)
{
    int i;

    for (i = (net.n - 1); i > 0; i --)
        if (net.layers[i].type != cYnLayerCost)
            break;

    return net.layers[i].output;
}

float YnNetworkCostGet(tYnNetwork net)
{
    int i;
    float sum = 0;
    int count = 0;

    for (i = 0; i < net.n; i ++)
    {
        if (net.layers[i].type == cYnLayerCost)
        {
            sum += net.layers[i].output[0];
            ++count;
        }

        if (net.layers[i].type == cYnLayerDetection)
        {
            sum += net.layers[i].cost[0];
            ++count;
        }
    }

    return sum / count;
}

int YnNnetworkPredictedClassNetworkGet(tYnNetwork net)
{
    float *out = YnNetworkOutputGet(net);
    int k = YnNetworkOutputSizeGet(net);
    return YnUtilArrayMaxIndex(out, k);
}

void backward_network(tYnNetwork net, tYnNetworkState state)
{
    int i;
    tYnLayer layer;
    tYnLayer prev;
    float * original_input = state.input;
    float * original_delta = state.delta;

    for (i = net.n-1; i >= 0; i --)
    {
        state.index = i;

        if (i == 0)
        {
            state.input = original_input;
            state.delta = original_delta;
        }
        else
        {
            prev = net.layers[i - 1];
            state.input = prev.output;
            state.delta = prev.delta;
        }

        layer = net.layers[i];

        switch(layer.type)
        {
            case cYnLayerConvolutional:
                YnLayerConvolutionalBackward(layer, state);
                break;
            case cYnLayerActive:
                YnLayerActiveBackward(layer, state);
                break;
            case cYnLayerLocal:
                YnLayerLocalBackward(layer, state);
                break;
            case cYnLayerDeconvolutional:
                YnLayerDeconvolutionalBackward(layer, state);
                break;
            case cYnLayerConnected:
                YnLayerConnectedBackward(layer, state);
                break;
            case cYnLayerRnn:
                YnLayerRnnBackward(layer, state);
                break;
            case cYnLayerMaxpool:
                YnLayerMaxpoolBackward(layer, state);
                break;
            case cYnLayerAvgpool:
                YnLayerAvgpoolBackward(layer, state);
                break;
            case cYnLayerSoftmax:
                YnLayerSoftmaxBackward(layer, state);
                break;
            case cYnLayerDetection:
                YnLayerDetectionBackward(layer, state);
                break;
            case cYnLayerDropout:
                YnLayerDropoutBackward(layer, state);
                break;
            case cYnLayerCrop:
                YnLayerCropBackward(layer, state);
                break;
            case cYnLayerCost:
                YnLayerCostBackward(layer, state);
                break;
            case cYnLayerRoute:
                YnLayerRouteBackward(layer, state);
                break;
            case cYnLayerShortcut:
                YnLayerShortcutBackward(layer, state);
                break;
            case cYnLayerNormalization:
                YnLayerNormalizationBackward(layer, state);
                break;
            default:
                break;
        }
    }
}

float YnNetworkTrainDatum(tYnNetwork net,
        float * x,
        float * y)
{
    tYnNetworkState state;
    float error;

    *net.seen += net.batch;

#ifdef YN_GPU
    if (YnCudaGpuIndexGet() >= 0)
        return YnNwteorkGpuTrainDatum(net, x, y);
#endif

    state.index = 0;
    state.net = net;
    state.input = x;
    state.delta = 0;
    state.truth = y;
    state.train = 1;

    YnNetworkForward(net, state);
    YnNetworkBackward(net, state);
    error = YnNetworkCostGet(net);

    if (((*net.seen) / net.batch) % net.subdivisions == 0)
        YnNetworkUpdate(net);

    return error;
}

float YnNetworkTrainSgd(tYnNetwork net,
        tYnData d,
        int n)
{
    int i;
    float err;
    float sum = 0;
    int batch = net.batch;
    float *X = calloc(batch * d.x.cols, sizeof(float));
    float *y = calloc(batch * d.y.cols, sizeof(float));

    for (i = 0; i < n; i ++)
    {
        YnDataRandomBatchGet(d, batch, X, y);
        err = YnNetworkTrainDatum(net, X, y);
        sum += err;
    }

    YnUtilFree(X);
    YnUtilFree(y);

    return (float)sum / (n * batch);
}

float YnNetworkTrain(tYnNetwork net,
        tYnData d)
{
    int i;
    float err;
    float sum = 0;
    int batch = net.batch;
    int n = d.x.rows / batch;
    float *X = calloc(batch * d.x.cols, sizeof(float));
    float *y = calloc(batch * d.y.cols, sizeof(float));

    for (i = 0; i < n; i ++)
    {
        YnNetworkNextBatchGet(d, batch, i * batch, X, y);
        err = YnNetworkTrainDatum(net, X, y);
        sum += err;
    }

    YnUtilFree(X);
    YnUtilFree(y);

    return (float)sum/(n * batch);
}

float YnNetworkTrainBatch(tYnNetwork net,
        tYnData d,
        int n)
{
    int i,j;
    float sum = 0;
    int batch = 2;
    tYnNetworkState state;

    state.index = 0;
    state.net = net;
    state.train = 1;
    state.delta = 0;

    for (i = 0; i < n; i ++)
    {
        for (j = 0; j < batch; j ++)
        {
            int index = rand() % d.x.rows;
            state.input = d.x.vals[index];
            state.truth = d.y.vals[index];
            YnNetworkForward(net, state);
            YnNetworkBackward(net, state);

            sum += YnNetworkCostGet(net);
        }

        YnNetworkUpdate(net);
    }

    return (float)sum/(n * batch);
}

void YnNetworkBatchSet(tYnNetwork *net,
        int b)
{
    int i;
    net->batch = b;

    for (i = 0; i < net->n; i ++)
    {
        net->layers[i].batch = b;
    }
}

int YnNetworkResize(tYnNetwork * net,
        int w,
        int h)
{
    int i;
    int inputs = 0;
    tYnLayer layer;

    net->w = w;
    net->h = h;

    for (i = 0; i < net->n; i ++)
    {
        layer = net->layers[i];

        if (layer.type == cYnLayerConvolutional)
        {
            YnLayerConvolutionalResize(&layer, w, h);
        }
        else if (layer.type == cYnLayerCrop)
        {
            YnLayerCropResize(&layer, w, h);
        }
        else if (layer.type == cYnLayerCrop)
        {
            YnLayerMaxpoolResize(&layer, w, h);
        }
        else if (layer.type == cYnLayerAvgpool)
        {
            YnLayerAvgpoolResize(&layer, w, h);
        }
        else if (layer.type == cYnLayerNormalization)
        {
            YnLayerNormalizationResize(&layer, w, h);
        }
        else if (layer.type == cYnLayerCost)
        {
            YnLayerCostResize(&layer, inputs);
        }
        else
        {
            error("Cannot resize this type of layer");
        }

        inputs = layer.outputs;
        net->layers[i] = layer;

        w = layer.outW;
        h = layer.outH;

        if (layer.type == cYnLayerAvgpool)
            break;
    }
    return 0;
}

int YnNetworkOutputSizeGet(tYnNetwork net)
{
    int i;

    for (i = net.n-1; i > 0; i --)
        if (net.layers[i].type != cYnLayerCost)
            break;

    return net.layers[i].outputs;
}

int YnNetworkInputSizeGet(tYnNetwork net)
{
    return net.layers[0].inputs;
}

tYnLayerDetection get_network_detection_layer(network net)
{
    int i;
    for (i = 0; i < net.n; i ++){
        if (net.layers[i].type == DETECTION){
            return net.layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    detection_layer l = {0};
    return l;
}

image get_network_image_layer(network net, int i)
{
    layer l = net.layers[i];
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network net)
{
    int i;
    for (i = net.n-1; i >= 0; i --){
        image m = get_network_image_layer(net, i);
        if (m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for (i = 0; i < net.n; i ++){
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    }
}

void top_predictions(network net, int k, int *index)
{
    int size = get_network_output_size(net);
    float *out = get_network_output(net);
    top_k(out, size, k, index);
}


float *network_predict(network net, float *input)
{
#ifdef YN_GPU
    if (gpu_index >= 0)  return network_predict_gpu(net, input);
#endif

    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}

matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for (i = 0; i < test.X.rows; i += net.batch){
        for (b = 0; b < net.batch; ++b){
            if (i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for (m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for (b = 0; b < net.batch; ++b){
                if (i+b == test.X.rows) break;
                for (j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for (i = 0; i < test.X.rows; i += net.batch){
        for (b = 0; b < net.batch; ++b){
            if (i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for (b = 0; b < net.batch; ++b){
            if (i+b == test.X.rows) break;
            for (j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;
}

void print_network(network net)
{
    int i,j;
    for (i = 0; i < net.n; i ++){
        layer l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if (n > 100) n = 100;
        for (j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if (n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for (i = 0; i < g1.rows; i ++){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if (p1 == truth){
            if (p2 == truth) ++d;
            else ++c;
        }else{
            if (p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den);
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}


float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; i ++){
        free_layer(net.layers[i]);
    }
    free(net.layers);
    #ifdef YN_GPU
    if (*net.input_gpu) cuda_free(*net.input_gpu);
    if (*net.truth_gpu) cuda_free(*net.truth_gpu);
    if (net.input_gpu) free(net.input_gpu);
    if (net.truth_gpu) free(net.truth_gpu);
    #endif
}
