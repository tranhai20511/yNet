//	File        :   YnLayerSoftmaxayer.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerSoftmax.h"
#include "../include/YnUtilayer.h"
#include "../include/YnCuda.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
tYnLayer YnLayerSoftmaxMake(int batchNum,
        int inputs,
        int groups)
{
    assert(inputs % groups == 0);
    fprintf(stderr, "Softmax Layer: %d inputs\n", inputs);

    tYnLayer layer = {0};

    layer.type = cYnLayerSoftmax;
    layer.batch = batchNum;
    layer.groups = groups;
    layer.inputs = inputs;
    layer.outputs = inputs;
    layer.output = calloc(inputs * batchNum, sizeof(float));
    layer.delta = calloc(inputs * batchNum, sizeof(float));

#ifdef YN_GPU
    layer.outputGpu = YnCudaMakeArray(layer.output, inputs * batchNum);
    layer.deltaGpu = YnCudaMakeArray(layer.delta, inputs * batchNum);
#endif

    return layer;
}

void YnLayerSoftmaxArray(float * input,
        int num,
        float temp,
        float * output)
{
    int i;
    float sum = 0;
    float largest = - FLT_MAX;

    for (i = 0; i < num; i ++)
    {
        if (input[i] > largest)
            largest = input[i];
    }

    for (i = 0; i < num; i ++)
    {
        sum += exp((input[i] / temp) - (largest / temp));
    }

    if(sum)
        sum = largest / temp + log(sum);
    else
        sum = largest - 100;

    for (i = 0; i < num; i ++)
    {
        output[i] = exp(input[i] / temp-sum);
    }
}

void YnLayerSoftmaxForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int b;
    int inputs = layer.inputs / layer.groups;
    int batch = layer.batch * layer.groups;

    for (b = 0; b < batch; b ++)
    {
        YnLayerSoftmaxArray(netState.input + b * inputs, inputs, layer.temperature, layer.output + b * inputs);
    }
}

void YnLayerSoftmaxBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;

    for (i = 0; i < layer.inputs * layer.batch; i ++)
    {
        netState.delta[i] += layer.delta[i];
    }
}
