//	File        :   YnLayerDetectionayerGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerDetectionGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnLayerDetectionGpuForward(tYnLayer layer,
        tYnNetworkState netState)
{
    float *in_cpu;
    float *truth_cpu;
    int num_truth;
    tYnNetworkState cpu_netState = {0};

    if (!netState.train)
    {
        YnBlasGpuArrayCopyValueSet(layer.outputGpu, netState.input, layer.batch * layer.inputs, 1, 1);
        return;
    }

    in_cpu = calloc(layer.batch * layer.inputs, sizeof(float));
    truth_cpu = 0;

    if (netState.truth)
    {
        num_truth = layer.batch * layer.side * layer.side * (1 + layer.coords+layer.classes);
        truth_cpu = calloc(num_truth, sizeof(float));
        YnCudaArrayPullFromGpu(netState.truth, truth_cpu, num_truth);
    }

    YnCudaArrayPullFromGpu(netState.input, in_cpu, layer.batch * layer.inputs);
    cpu_netState.train = netState.train;
    cpu_netState.truth = truth_cpu;
    cpu_netState.input = in_cpu;

    YnLayerDetectionForward(layer, cpu_netState);
    YnCudaArrayPushToGpu(layer.outputGpu, layer.output, layer.batch * layer.outputs);
    YnCudaArrayPushToGpu(layer.deltaGpu, layer.delta, layer.batch * layer.inputs);
    YnUtilFree(cpu_netState.input);

    if (cpu_netState.truth)
        YnUtilFree(cpu_netState.truth);
}

void YnLayerDetectionGpuBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    YnBlasGpuArrayAxpyValueSet(netState.delta, layer.deltaGpu, layer.batch * layer.inputs, 1, 1, 1);
}
