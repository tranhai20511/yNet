//	File        :   YnLayerDropout.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   28-08-2016
//	Author      :   haittt

#include "../include/YnLayerDropout.h"
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
tYnLayer YnLayerDropoutMake(int batchNum,
        int inputs,
        float probability)
{
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
    tYnLayer layer = {0};

    layer.type = DROPOUT;
    layer.probability = probability;
    layer.inputs = inputs;
    layer.outputs = inputs;
    layer.batch = batch;
    layer.rand = calloc(inputs*batch, sizeof(float));
    layer.scale = 1./(1.-probability);
    #ifdef GPU
    layer.rand_gpu = cuda_make_array(layer.rand, inputs*batch);
    #endif
    return l;
}

void YnLayerDropoutForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    if (!state.train) return;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        float r = rand_uniform(0, 1);
        layer.rand[i] = r;
        if(r < layer.probability) state.input[i] = 0;
        else state.input[i] *= layer.scale;
    }
}

void YnLayerDropoutBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i;
    if(!state.delta) return;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        float r = layer.rand[i];
        if(r < layer.probability) state.delta[i] = 0;
        else state.delta[i] *= layer.scale;
    }
}

void YnLayerDropoutResize(tYnLayer layer,
        int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}
