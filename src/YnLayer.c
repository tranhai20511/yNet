//	File        :   YnLayer.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   26-07-2016
//	Author      :   haittt

#include "../include/YnLayer.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_STATIC
void YnLayerDropoutFreeRand(tYnLayer layer)
{
	YnUtilFree(layer.rand);

#ifdef YN_GPU
    if (layer.randGpu)
        YnCudaFreeArray(layer.randGpu);
#endif
}

YN_STATIC void YnLayerGpuFree(tYnLayer layer)
{
#ifdef YN_GPU
    YnCudaFreeArray((float *)layer.indexesGpu);
    YnCudaFreeArray(layer.filtersGpu);
    YnCudaFreeArray(layer.filterUpdatesGpu);
    YnCudaFreeArray(layer.colImageGpu);
    YnCudaFreeArray(layer.weightsGpu);
    YnCudaFreeArray(layer.biasesGpu);
    YnCudaFreeArray(layer.weightUpdatesGpu);
    YnCudaFreeArray(layer.biasUpdatesGpu);
    YnCudaFreeArray(layer.outputGpu);
    YnCudaFreeArray(layer.deltaGpu);
    YnCudaFreeArray(layer.randGpu);
    YnCudaFreeArray(layer.squaredGpu);
    YnCudaFreeArray(layer.normsGpu);
#endif
}

void YnLayerFree(tYnLayer layer)
{
    if (layer.type == cYnLayerDropout)
    {
        YnLayerDropoutFreeRand(layer);
        return;
    }

    YnUtilFree(layer.indexes);
    YnUtilFree(layer.rand);
    YnUtilFree(layer.cost);
    YnUtilFree(layer.filters);
    YnUtilFree(layer.filterUpdates);
    YnUtilFree(layer.biases);
    YnUtilFree(layer.biasUpdates);
    YnUtilFree(layer.weights);
    YnUtilFree(layer.weightUpdates);
    YnUtilFree(layer.colImage);
    YnUtilFree(layer.inputLayers);
    YnUtilFree(layer.inputSizes);
    YnUtilFree(layer.delta);
    YnUtilFree(layer.output);
    YnUtilFree(layer.squared);
    YnUtilFree(layer.norms);

    YnLayerGpuFree(layer);
}
