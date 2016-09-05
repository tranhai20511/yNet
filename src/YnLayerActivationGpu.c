//	File        :   YnLayerActivationGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   30-07-2016
//	Author      :   haittt

#include "../include/YnLayerActivationGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnLayerActivationGpuForward(tYnLayer * layer,
        tYnNetworkState netState)
{
    YnBlasGpuArrayCopyValueSet(layer.output, netState.input, layer.outputs * layer.batch, 1, 1);
    YnActivationGpuOutputArrayCal(layer.output, layer.outputs * layer.batch, layer.activation);
}

void YnLayerActivationGpuBackward(tYnLayer * layer,
        tYnNetworkState netState)
{
    YnActivationGpuGradientArrayCal(layer.output, layer.outputs * layer.batch, layer.activation, layer.delta);
    YnBlasGpuArrayCopyValueSet(netState.delta, layer.delta, layer.outputs * layer.batch, 1, 1);
}
