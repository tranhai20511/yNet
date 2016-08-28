//	File        :   YnNetwork.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   26-07-2016
//	Author      :   haittt

#include "../include/YnNetwork.h"

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
                YnLayerActivationForward(layer, state);
                break;
            case cYnLayerLocal:
                /*YnLayerLocalForward(layer, state);*/
                break;
            case cYnLayerDeconvolutional:
                YnLayerDeconvolutionalForward(layer, state);
                break;
            case cYnLayerConnected:
                YnLayerConnectedForward(layer, state);
                break;
            case cYnLayerRnn:
                /*YnLayerRnnForward(layer, state);*/
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
                /*YnLayerRouteForward(layer, state);*/
                break;
            case cYnLayerShortcut:
                /*YnLayerShortcutForward(layer, state);*/
                break;
            case cYnLayerNormalization:
                /*YnLayerNormalizationForward(layer, state);*/
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
    float rate = YnNetworkCurrentRateget(net);

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
            /*YnLayerRnnUpdate(layer, updateBatch, rate, net.momentum, net.decay);*/
        }
        else if (layer.type == cYnLayerLocal)
        {
            /*YnLayerLocalUpdate(layer, updateBatch, rate, net.momentum, net.decay);*/
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

void YnNetworkBackward(tYnNetwork net,
        tYnNetworkState state)
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
                YnLayerActivationBackward(layer, state);
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
                /*YnLayerRnnBackward(layer, state);*/
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
            case cYnLayerCost:
                YnLayerCostBackward(layer, state);
                break;
            case cYnLayerRoute:
                /*YnLayerRouteBackward(layer, state);*/
                break;
            case cYnLayerShortcut:
                /*YnLayerShortcutBackward(layer, state);*/
                break;
            case cYnLayerNormalization:
                /*YnLayerNormalizationBackward(layer, state);*/
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
        return YnNetworkGpuTrainDatum(net, x, y);
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
            /*YnLayerNormalizationResize(&layer, w, h);*/
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

tYnLayer YnNetworkDetectionLayerGet(tYnNetwork net)
{
    int i;

    for (i = 0; i < net.n; i ++)
    {
        if (net.layers[i].type == cYnLayerDetection)
        {
            return net.layers[i];
        }
    }

    fprintf(stderr, "Detection layer not found!!\n");
    tYnLayer layer = {0};

    return layer;
}

tYnImage YnNetworkImageLayerGet(tYnNetwork net,
        int i)
{
    tYnLayer layer = net.layers[i];

    if (layer.outW && layer.outH && layer.outC)
    {
        return YnImageFloatToImage(layer.outW, layer.outH, layer.outC, layer.output);
    }

    tYnImage def = {0};

    return def;
}

tYnImage YnNetworkImageGet(tYnNetwork net)
{
    int i;
    tYnImage m;

    for (i = net.n - 1; i >= 0; i --)
    {
        m = YnNetworkImageLayerGet(net, i);
        if (m.height != 0)
            return m;
    }

    tYnImage def = {0};

    return def;
}

void YnNetworkVisualize(tYnNetwork net)
{
    tYnImage *prev = 0;

    int i;
    char buff[256];
    tYnLayer layer;

    for (i = 0; i < net.n; i ++)
    {
        sprintf(buff, "Layer %d", i);
        layer = net.layers[i];

        if (layer.type == cYnLayerConvolutional)
        {
            prev = YnLayerConvolutionalVisualize(layer, buff, prev);
        }
    }
}

void YnNetworkTopPredictions(tYnNetwork net,
        int k,
        int *index)
{
    int size = YnNetworkOutputSizeGet(net);
    float *out = YnNetworkOutputGet(net);

    YnUtilTop(out, size, k, index);
}

float * YnNetworkPredict(tYnNetwork net,
        float *input)
{

#ifdef YN_GPU
    if (YnCudaGpuIndexGet() >= 0)  return YnNetworkGpuPredict(net, input);
#endif

    float *out;
    tYnNetworkState state;

    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    YnNetworkForward(net, state);
    out = get_network_output(net);

    return out;
}

tYnMatrix YnNetworkPredictDataMulti(tYnNetwork net,
        tYnData test,
        int n)
{
    tYnMatrix pred;
    float *X;
    float *out;
    int i, j, b, m;
    int k = get_network_output_size(net);

    pred = YnMatrixMake(test.x.rows, k);
    X = calloc(net.batch*test.x.rows, sizeof(float));

    for (i = 0; i < test.x.rows; i += net.batch)
    {
        for (b = 0; b < net.batch; b ++)
        {
            if (i + b == test.x.rows)
                break;

            memcpy(X + b * test.x.cols, test.x.vals[i + b], test.x.cols * sizeof(float));
        }

        for (m = 0; m < n; m ++)
        {
            out = YnNetworkPredict(net, X);

            for (b = 0; b < net.batch; b ++)
            {
                if (i + b == test.x.rows)
                    break;

                for (j = 0; j < k; j ++)
                {
                    pred.vals[i + b][j] += out[j + b * k] / n;
                }
            }
        }
    }

    YnUtilFree(X);

    return pred;
}

tYnMatrix YnNetworkPredictData(tYnNetwork net,
        tYnData test)
{
    tYnMatrix pred;
    int i, j, b;
    float *X;
    float *out;
    int k = YnNetworkOutputSizeGet(net);

    pred = YnMatrixMake(test.x.rows, k);
    X = calloc(net.batch * test.x.cols, sizeof(float));

    for (i = 0; i < test.x.rows; i += net.batch)
    {
        for (b = 0; b < net.batch; b ++)
        {
            if (i + b == test.x.rows)
                break;

            memcpy(X + b * test.x.cols, test.x.vals[i+b], test.x.cols * sizeof(float));
        }

        out = YnNetworkPredict(net, X);

        for (b = 0; b < net.batch; b ++)
        {
            if (i + b == test.x.rows)
                break;

            for (j = 0; j < k; j ++)
            {
                pred.vals[i + b][j] = out[j + b * k];
            }
        }
    }

    YnUtilFree(X);

    return pred;
}

void YnNetworkPrint(tYnNetwork net)
{
    int i,j;
    tYnLayer layer;
    float *output;
    float mean;
    float vari;
    int n;

    for (i = 0; i < net.n; i ++)
    {
        layer = net.layers[i];
        output = layer.output;
        n = layer.outputs;
        mean = YnUtilArrayMean(output, n);
        vari = YnUtilArrayVariance(output, n);

        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n", i, mean, vari);

        if (n > 100)
            n = 100;

        for (j = 0; j < n; j ++)
            fprintf(stderr, "%f, ", output[j]);

        if (n == 100)
            fprintf(stderr,".....\n");

        fprintf(stderr, "\n");
    }
}

void YnNetworkCompare(tYnNetwork n1,
        tYnNetwork n2,
        tYnData test)
{
    int i;
    int a,b,c,d;
    int truth;
    int p1;
    int p2;
    float num;
    float den;

    tYnMatrix g1 = YnNetworkPredictData(n1, test);
    tYnMatrix g2 = YnNetworkPredictData(n2, test);

    a = b = c = d = 0;

    for (i = 0; i < g1.rows; i ++)
    {
        truth = YnUtilArrayMaxIndex(test.y.vals[i], test.y.cols);
        p1 = YnUtilArrayMaxIndex(g1.vals[i], g1.cols);
        p2 = YnUtilArrayMaxIndex(g2.vals[i], g2.cols);

        if (p1 == truth)
        {
            if (p2 == truth)
                d ++;
            else
                c ++;

        }
        else
        {
            if (p2 == truth)
                b ++;
            else
                a ++;
        }
    }

    printf("%5d %5d\n%5d %5d\n", a, b, c, d);

    num = pow((abs(b - c) - 1.), 2.);
    den = b + c;

    printf("%f\n", num/den);
}

float YnNetworkAccuracy(tYnNetwork net,
        tYnData d)
{
    tYnMatrix guess = YnNetworkPredictData(net, d);
    float acc = YnMatrixTopAccuracy(d.y, guess, 1);
    YnMatrixFree(guess);
    return acc;
}

float * YnnNetworkAccuracies(tYnNetwork net,
        tYnData d,
        int n)
{
    static float acc[2];
    tYnMatrix guess = YnNetworkPredictData(net, d);

    acc[0] = YnMatrixTopAccuracy(d.y, guess, 1);
    acc[1] = YnMatrixTopAccuracy(d.y, guess, n);

    YnMatrixFree(guess);

    return acc;
}

float YnNetworkAccuracyMulti(tYnNetwork net,
        tYnData d,
        int n)
{
    tYnMatrix guess = YnNetworkPredictDataMulti(net, d, n);
    float acc = YnMatrixTopAccuracy(d.y, guess,1);
    YnMatrixFree(guess);
    return acc;
}

void YnNetworkFree(tYnNetwork net)
{
    int i;

    for (i = 0; i < net.n; i ++)
    {
        YnLayerFree(net.layers[i]);
    }

    YnUtilFree(net.layers);

#ifdef YN_GPU
    if (*net.inputGpu) YnCudaFreeArray(*net.inputGpu);
    if (*net.truthGpu) YnCudaFreeArray(*net.truthGpu);
    if (net.inputGpu) YnUtilFree(net.inputGpu);
    if (net.truthGpu) YnUtilFree(net.truthGpu);
#endif
}
