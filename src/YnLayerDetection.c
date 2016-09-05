//	File        :   YnLayerDetectionayer.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   27-08-2016
//	Author      :   haittt

#include "../include/YnLayerDetection.h"
#include "../include/YnLayerSoftmax.h"
#include "../include/YnActivation.h"
#include "../include/YnBlas.h"
#include "../include/YnBBox.h"
#include "../include/YnGpu.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
tYnLayer YnLayerDetectionMake(int batchNum,
        int inputs,
        int num,
        int side,
        int classes,
        int coords,
        int rescore)
{
    tYnLayer layer = {0};
    layer.type = cYnLayerDetection;

    layer.n = num;
    layer.batch = batchNum;
    layer.inputs = inputs;
    layer.classes = classes;
    layer.coords = coords;
    layer.rescore = rescore;
    layer.side = side;

    assert(side * side * ((1 + layer.coords) * layer.n + layer.classes) == inputs);

    layer.cost = calloc(1, sizeof(float));
    layer.outputs = layer.inputs;
    layer.truths = layer.side*layer.side*(1+layer.coords + layer.classes);
    layer.output = calloc(batchNum * layer.outputs, sizeof(float));
    layer.delta = calloc(batchNum * layer.outputs, sizeof(float));

#ifdef YN_GPU
    layer.outputGpu = cuda_make_array(layer.output, batchNum * layer.outputs);
    layer.deltaGpu = cuda_make_array(layer.delta, batchNum * layer.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return layer;
}

void YnLayerDetectionForward(tYnLayer layer,
        tYnNetworkState netState)
{
    int i,j;
    int index;
    int offset;
    int b;
    float avg_iou;
    float avg_cat;
    float avg_allcat;
    float avg_obj;
    float avg_anyobj;
    int count;
    int size;
    int truth_index;
    int is_obj;
    int p_index;
    int best_index;
    float best_iou;
    float best_rmse;
    int class_index;
    int locations = layer.side * layer.side;
    int box_index;
    int tbox_index;
    tYnBBox truth, out;
    float iou;
    float rmse;

    memcpy(layer.output, netState.input, layer.outputs * layer.batch * sizeof(float));

    if (layer.softmax)
    {
        for (b = 0; b < layer.batch; b ++)
        {
            index = b * layer.inputs;

            for (i = 0; i < locations; i ++)
            {
                offset = i * layer.classes;
                YnLayerSoftmaxArray(layer.output + index + offset, layer.classes, 1,
                        layer.output + index + offset);
            }

            offset = locations * layer.classes;
            YnActivationOutputArrayCal(layer.output + index + offset, locations*layer.n*(1+layer.coords), cYnActivationLogistic);
        }
    }

    if (netState.train)
    {
        avg_iou = 0;
        avg_cat = 0;
        avg_allcat = 0;
        avg_obj = 0;
        avg_anyobj = 0;
        count = 0;
        *(layer.cost) = 0;

        size = layer.inputs * layer.batch;
        memset(layer.delta, 0, size * sizeof(float));

        for (b = 0; b < layer.batch; b ++)
        {
            index = b * layer.inputs;

            for (i = 0; i < locations; i ++)
            {
                truth_index = (b*locations + i)*(1+layer.coords+layer.classes);
                is_obj = netState.truth[truth_index];

                for (j = 0; j < layer.n; j ++)
                {
                    p_index = index + locations * layer.classes + i * layer.n + j;
                    layer.delta[p_index] = layer.noobjectScale * (0 - layer.output[p_index]);
                    *(layer.cost) += layer.noobjectScale * pow(layer.output[p_index], 2);
                    avg_anyobj += layer.output[p_index];
                }

                best_index = -1;
                best_iou = 0;
                best_rmse = 20;

                if (!is_obj)
                {
                    continue;
                }

                class_index = index + i * layer.classes;

                for (j = 0; j < layer.classes; j ++)
                {
                    layer.delta[class_index + j] = layer.classScale * (netState.truth[truth_index + 1 + j] - layer.output[class_index + j]);
                    *(layer.cost) += layer.classScale * pow(netState.truth[truth_index + 1 + j] - layer.output[class_index + j], 2);

                    if (netState.truth[truth_index + 1 + j])
                        avg_cat += layer.output[class_index + j];

                    avg_allcat += layer.output[class_index + j];
                }

                truth = YnBBoxFromArray(netState.truth + truth_index + 1 + layer.classes);
                truth.x /= layer.side;
                truth.y /= layer.side;

                for (j = 0; j < layer.n; j ++)
                {
                    box_index = index + locations * (layer.classes + layer.n) + (i * layer.n + j) * layer.coords;
                    out = YnBBoxFromArray(layer.output + box_index);

                    out.x /= layer.side;
                    out.y /= layer.side;

                    if (layer.sqrt)
                    {
                        out.width = out.width * out.width;
                        out.height = out.height * out.height;
                    }

                    iou  = YnBBoxIou(out, truth);
                    rmse = YnBBoxRmse(out, truth);

                    if (best_iou > 0 || iou > 0)
                    {
                        if (iou > best_iou)
                        {
                            best_iou = iou;
                            best_index = j;
                        }
                    }
                    else
                    {
                        if (rmse < best_rmse)
                        {
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if (layer.forced)
                {
                    if (truth.width * truth.height < .1)
                    {
                        best_index = 1;
                    }
                    else
                    {
                        best_index = 0;
                    }
                }

                box_index = index + locations*(layer.classes + layer.n) + (i*layer.n + best_index) * layer.coords;
                tbox_index = truth_index + 1 + layer.classes;

                out = YnBBoxFromArray(layer.output + box_index);
                out.x /= layer.side;
                out.y /= layer.side;

                if (layer.sqrt)
                {
                    out.width = out.width * out.width;
                    out.height = out.height * out.height;
                }
                iou = YnBBoxIou(out, truth);

                p_index = index + locations * layer.classes + i * layer.n + best_index;
                *(layer.cost) -= layer.noobjectScale * pow(layer.output[p_index], 2);
                *(layer.cost) += layer.objectScale * pow(1-layer.output[p_index], 2);
                avg_obj += layer.output[p_index];
                layer.delta[p_index] = layer.objectScale * (1.-layer.output[p_index]);

                if (layer.rescore)
                {
                    layer.delta[p_index] = layer.objectScale * (iou - layer.output[p_index]);
                }

                layer.delta[box_index+0] = layer.coordScale*(netState.truth[tbox_index + 0] - layer.output[box_index + 0]);
                layer.delta[box_index+1] = layer.coordScale*(netState.truth[tbox_index + 1] - layer.output[box_index + 1]);
                layer.delta[box_index+2] = layer.coordScale*(netState.truth[tbox_index + 2] - layer.output[box_index + 2]);
                layer.delta[box_index+3] = layer.coordScale*(netState.truth[tbox_index + 3] - layer.output[box_index + 3]);

                if (layer.sqrt)
                {
                    layer.delta[box_index+2] = layer.coordScale*(sqrt(netState.truth[tbox_index + 2]) - layer.output[box_index + 2]);
                    layer.delta[box_index+3] = layer.coordScale*(sqrt(netState.truth[tbox_index + 3]) - layer.output[box_index + 3]);
                }

                *(layer.cost) += pow(1-iou, 2);
                avg_iou += iou;

                ++count;
            }

            if (layer.softmax)
            {
                YnActivationGradientArrayCal(layer.output + index + locations * layer.classes, locations * layer.n * (1 + layer.coords),
                        cYnActivationLogistic, layer.delta + index + locations * layer.classes);
            }
        }

        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n",
                avg_iou / count,
                avg_cat / count,
                avg_allcat / (count * layer.classes),
                avg_obj / count,
                avg_anyobj / (layer.batch * locations * layer.n), count);
    }
}

void YnLayerDetectionBackward(tYnLayer layer,
        tYnNetworkState netState)
{
    YnBlasArrayAxpyValueSet(netState.delta, layer.delta, layer.batch * layer.inputs, 1, 1, 1);
}
