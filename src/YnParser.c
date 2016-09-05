//	File        :   YnParser.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   24-07-2016
//	Author      :   haittt

#include "../include/YnList.h"
#include "../include/YnOptionList.h"
#include "../include/YnUtil.h"
#include "../include/YnNetwork.h"
#include "../include/YnActivation.h"
#include "../include/YnLayerCrop.h"
#include "../include/YnLayerCost.h"
#include "../include/YnLayerConvolutional.h"
#include "../include/YnLayerActivation.h"
#include "../include/YnLayerDeconvolutional.h"
#include "../include/YnLayerConnected.h"
#include "../include/YnLayerMaxpool.h"
#include "../include/YnLayerSoftmax.h"
#include "../include/YnLayerDropout.h"
#include "../include/YnLayerDetection.h"
#include "../include/YnLayerAvgpool.h"

#include "../include/YnParser.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
YN_STATIC_INLINE
int YnParserIsNetwork(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsConvolutional(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsActivation(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsLocal(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsDeconvolutional(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsConnected(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsRnn(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsMaxpool(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsAvgpool(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsDropout(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsSoftmax(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsNormalization(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsCrop(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsShortcut(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsCost(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsDetection(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
int YnParserIsRoute(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnList * YnParserReadCfg(char *filename)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
void YnParserFreeSection(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
eYnNetworkLearnRatePolicy YnParserPolicyGet(char *s)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
void YnParserTransposeMatrix(float *a,
        int rows,
        int cols)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
void YnParserData(char *data,
        float *a,
        int n)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserDeconvolutional(tYnList *options,
        tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserConvolutional(tYnList *options,
        tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
void YnParserConnectedWeightsSave(tYnLayer layer,
                                  FILE *fp)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserConnected(tYnList *options,
                           tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserSoftmax(tYnList *options,
                         tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserDetection(tYnList *options,
                           tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserCost(tYnList *options,
                      tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserCrop(tYnList *options,
                       tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserMaxpool(tYnList *options,
                         tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserAvgpool(tYnList *options,
                         tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserDropout(tYnList *options,
                         tYnParserSizeParams params)
YN_ALSWAY_INLINE;

YN_STATIC_INLINE
tYnLayer YnParserActivation(tYnList *options,
                            tYnParserSizeParams params)
YN_ALSWAY_INLINE;

/**************** Implement */
YN_STATIC_INLINE
int YnParserIsShortcut(tYnParserSection *s)
{
    return (strcmp(s->type, "[shortcut]") == 0);
}

YN_STATIC_INLINE
int YnParserIsCrop(tYnParserSection *s)
{
    return (strcmp(s->type, "[crop]") == 0);
}

YN_STATIC_INLINE
int YnParserIsCost(tYnParserSection *s)
{
    return (strcmp(s->type, "[cost]") == 0);
}

YN_STATIC_INLINE
int YnParserIsDetection(tYnParserSection *s)
{
    return (strcmp(s->type, "[detection]") == 0);
}

YN_STATIC_INLINE
int YnParserIsLocal(tYnParserSection *s)
{
    return (strcmp(s->type, "[local]") == 0);
}

YN_STATIC_INLINE
int YnParserIsDeconvolutional(tYnParserSection *s)
{
    return ((strcmp(s->type, "[deconv]") == 0) ||
            (strcmp(s->type, "[deconvolutional]") == 0));
}

YN_STATIC_INLINE
int YnParserIsConvolutional(tYnParserSection *s)
{
    return ((strcmp(s->type, "[conv]") == 0) ||
            (strcmp(s->type, "[convolutional]") == 0));
}

YN_STATIC_INLINE
int YnParserIsActivation(tYnParserSection *s)
{
    return (strcmp(s->type, "[activation]") == 0);
}

YN_STATIC_INLINE
int YnParserIsNetwork(tYnParserSection *s)
{
    return ((strcmp(s->type, "[net]") == 0) ||
            (strcmp(s->type, "[network]") == 0));
}

YN_STATIC_INLINE
int YnParserIsRnn(tYnParserSection *s)
{
    return (strcmp(s->type, "[rnn]") == 0);
}

YN_STATIC_INLINE
int YnParserIsConnected(tYnParserSection *s)
{
    return ((strcmp(s->type, "[conn]") == 0) ||
            (strcmp(s->type, "[connected]") == 0));
}

YN_STATIC_INLINE
int YnParserIsMaxpool(tYnParserSection *s)
{
    return ((strcmp(s->type, "[max]") == 0) ||
            (strcmp(s->type, "[maxpool]") == 0));
}

YN_STATIC_INLINE
int YnParserIsAvgpool(tYnParserSection *s)
{
    return ((strcmp(s->type, "[avg]") == 0) ||
            (strcmp(s->type, "[avgpool]") == 0));
}

YN_STATIC_INLINE
int YnParserIsDropout(tYnParserSection *s)
{
    return (strcmp(s->type, "[dropout]") == 0);
}

YN_STATIC_INLINE
int YnParserIsNormalization(tYnParserSection *s)
{
    return ((strcmp(s->type, "[lrn]") == 0) ||
            (strcmp(s->type, "[normalization]") == 0));
}

YN_STATIC_INLINE
int YnParserIsSoftmax(tYnParserSection *s)
{
    return ((strcmp(s->type, "[soft]") == 0) ||
            (strcmp(s->type, "[softmax]") == 0));
}

YN_STATIC_INLINE
int YnParserIsRoute(tYnParserSection *s)
{
    return (strcmp(s->type, "[route]") == 0);
}

YN_STATIC_INLINE
tYnList * YnParserReadCfg(char *filename)
{
    char *line;
    int nu = 0;
    tYnList *sections;
    tYnParserSection *current = 0;
    FILE *file = fopen(filename, "r");

    if (file == 0)
        YnUtilErrorOpenFile(filename);

    sections = YnListMake(NULL);

    while ((line = YnUtilFileGetLine(file)) != 0)
    {
        ++ nu;
        YnUtilStripString(line);

        switch(line[0])
        {
            case '[':
                current = malloc(sizeof(tYnParserSection));
                YnListInsert(sections, current);
                current->options = YnListMake(NULL);
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                YnUtilFree(line);
                break;
            default:
                if (!YnOptionRead(line, current->options))
                {
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    YnUtilFree(line);
                }
                break;
        }
    }

    fclose(file);

    return sections;
}

YN_STATIC_INLINE
void YnParserFreeSection(tYnParserSection *s)
{
    YnUtilFree(s->type);
    tYnListNode *n = s->options->front;

    while (n)
    {
        tYnOptionKeyVal *pair = (tYnOptionKeyVal *)n->val;
        YnUtilFree(pair->key);
        YnUtilFree(pair);
        tYnListNode *next = n->next;
        YnUtilFree(n);
        n = next;
    }

    YnUtilFree(s->options);
    YnUtilFree(s);
}

YN_STATIC_INLINE
eYnNetworkLearnRatePolicy YnParserPolicyGet(char *s)
{
    if (strcmp(s, "poly") == 0)       return cYnNetworkLearnRatePoly;
    if (strcmp(s, "constant") == 0)   return cYnNetworkLearnRateConstant;
    if (strcmp(s, "step") == 0)       return cYnNetworkLearnRateStep;
    if (strcmp(s, "exp") == 0)        return cYnNetworkLearnRateExp;
    if (strcmp(s, "sigmoid") == 0)    return cYnNetworkLearnRateSig;
    if (strcmp(s, "steps") == 0)      return cYnNetworkLearnRateSteps;

    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);

    return cYnNetworkLearnRateConstant;
}

YN_STATIC_INLINE
void YnParserTransposeMatrix(float *a,
        int rows,
        int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;

    for (x = 0; x < rows; x ++)
    {
        for (y = 0; y < cols; y ++)
        {
            transpose[(y * rows) + x] = a[(x * cols) + y];
        }
    }

    memcpy(a, transpose, rows * cols * sizeof(float));

    YnUtilFree(transpose);
}

YN_STATIC_INLINE
void YnParserData(char *data,
        float *a,
        int n)
{
    int i;
    int done;
    char *curr;
    char *next;
    if (!data)
        return;

    curr = data;
    next = data;
    done = 0;

    for (i = 0; (i < n) && (!done); i ++)
    {
        while ((*++next !='\0') && (*next != ','));

        if (*next == '\0')
            done = 1;

        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

YN_STATIC_INLINE
tYnLayer YnParserDeconvolutional(tYnList *options,
        tYnParserSizeParams params)
{
    tYnLayer layer = {0};
    eYnActivationType activation;
    int batch, h, w, c;
    char *weights;
    char *biases;

    int n = YnOptionFindInt(options, "filters",1);
    int size = YnOptionFindInt(options, "size",1);
    int stride = YnOptionFindInt(options, "stride",1);
    char *activation_s = YnOptionFindStr(options, "activation", "logistic");

    if (YnActivationTypeFromStringGet(activation_s, &activation) != eYnRetOk)
    {
        YnUtilError("Get deconvolutional type failed");
        return layer;
    }

    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;

    if (!(h && w && c))
        YnUtilError("Layer before deconvolutional layer must output image");

    layer = YnLayerDeconvolutionalMake(batch, h, w, c, n, size, stride, activation);

    weights = YnOptionFindStr(options, "weights", 0);
    biases = YnOptionFindStr(options, "biases", 0);
    YnParserData(weights, layer.filters, c * n * size * size);
    YnParserData(biases, layer.biases, n);

#ifdef YN_GPU
    if (weights || biases)
        YnLayerDeconvolutionalGpuPush(layer);
#endif

    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserConvolutional(tYnList *options,
        tYnParserSizeParams params)
{
    tYnLayer layer = {0};
    eYnActivationType activation;
    int batch, h, w, c;
    int batch_normalize;
    int binary;
    char *weights;
    char *biases;

    int n = YnOptionFindInt(options, "filters",1);
    int size = YnOptionFindInt(options, "size",1);
    int stride = YnOptionFindInt(options, "stride",1);
    int pad = YnOptionFindInt(options, "pad", 0);
    char *activation_s = YnOptionFindStr(options, "activation", "logistic");

    if (YnActivationTypeFromStringGet(activation_s, &activation) != eYnRetOk)
    {
        YnUtilError("Get convolutional type failed");
        return layer;
    }

    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        YnUtilError("Layer before convolutional layer must output image");

    batch_normalize = YnOptionFindIntQuiet(options, "batch_normalize", 0);
    binary = YnOptionFindIntQuiet(options, "binary", 0);

    layer = YnLayerConvolutionalMake(batch, h, w, c, n, size, stride, pad, activation, batch_normalize, binary);
    layer.flipped = YnOptionFindIntQuiet(options, "flipped", 0);

    weights = YnOptionFindStr(options, "weights", 0);
    biases = YnOptionFindStr(options, "biases", 0);
    YnParserData(weights, layer.filters, c * n * size * size);
    YnParserData(biases, layer.biases, n);

#ifdef YN_GPU
    if (weights || biases)
        YnLayerConvolutionalGpuPush(layer);
#endif

    return layer;
}

YN_STATIC_INLINE
void YnParserConnectedWeightsSave(tYnLayer layer,
                                  FILE *fp)
{
#ifdef YN_GPU
    if (YnCudaGpuIndexGet() >= 0)
    {
        YnLayerConnectedGpuPull(layer);
    }
#endif
    fwrite(layer.biases, sizeof(float), layer.outputs, fp);
    fwrite(layer.weights, sizeof(float), layer.outputs * layer.inputs, fp);

    if (layer.batchNormalize)
    {
        fwrite(layer.scales, sizeof(float), layer.outputs, fp);
        fwrite(layer.rollingMean, sizeof(float), layer.outputs, fp);
        fwrite(layer.rollingVariance, sizeof(float), layer.outputs, fp);
    }
}

YN_STATIC_INLINE
tYnLayer YnParserConnected(tYnList *options,
                           tYnParserSizeParams params)
{
    tYnLayer layer= {0};
    eYnActivationType activation;
    char *weights;
    char *biases;

    int output = YnOptionFindInt(options, "output",1);
    char *activation_s = YnOptionFindStr(options, "activation", "logistic");
    int batch_normalize = YnOptionFindIntQuiet(options, "batch_normalize", 0);

    if (YnActivationTypeFromStringGet(activation_s, &activation) != eYnRetOk)
    {
        YnUtilError("Get connected type failed");
        return layer;
    }

    layer = YnLayerConnectedMake(params.batch, params.inputs, output, activation, batch_normalize);

    weights = YnOptionFindStr(options, "weights", 0);
    biases = YnOptionFindStr(options, "biases", 0);
    YnParserData(biases, layer.biases, output);
    YnParserData(weights, layer.weights, params.inputs * output);

#ifdef YN_GPU
    if (weights || biases)
        YnLayerConnectedGpuPush(layer);
#endif

    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserSoftmax(tYnList *options,
                         tYnParserSizeParams params)
{
    int groups = YnOptionFindIntQuiet(options, "groups",1);
    tYnLayer layer = YnLayerSoftmaxMake(params.batch, params.inputs, groups);
    layer.temperature = YnOptionFindFloatQuiet(options, "temperature", 1);
    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserDetection(tYnList *options,
                           tYnParserSizeParams params)
{
    int coords = YnOptionFindInt(options, "coords", 1);
    int classes = YnOptionFindInt(options, "classes", 1);
    int rescore = YnOptionFindInt(options, "rescore", 0);
    int num = YnOptionFindInt(options, "num", 1);
    int side = YnOptionFindInt(options, "side", 7);
    tYnLayer layer = YnLayerDetectionMake(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = YnOptionFindInt(options, "softmax", 0);
    layer.sqrt = YnOptionFindInt(options, "sqrt", 0);

    layer.coordScale = YnOptionFindFloat(options, "coord_scale", 1);
    layer.forced = YnOptionFindInt(options, "forced", 0);
    layer.objectScale = YnOptionFindFloat(options, "object_scale", 1);
    layer.noobjectScale = YnOptionFindFloat(options, "noobject_scale", 1);
    layer.classScale = YnOptionFindFloat(options, "class_scale", 1);
    layer.jitter = YnOptionFindFloat(options, "jitter", .2);

    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserCost(tYnList *options,
                      tYnParserSizeParams params)
{
    char *type_s = YnOptionFindStr(options, "type", "sse");
    eYnLayerCostType type = YnLayerCostStringToType(type_s);
    float scale = YnOptionFindFloatQuiet(options, "scale",1);
    tYnLayer layer = YnLayerCostMake(params.batch, params.inputs, type, scale);
    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserCrop(tYnList *options,
                       tYnParserSizeParams params)
{
    tYnLayer layer;
    int batch, h, w, c;
    int noadjust;
    int crop_height = YnOptionFindInt(options, "crop_height",1);
    int crop_width = YnOptionFindInt(options, "crop_width",1);
    int flip = YnOptionFindInt(options, "flip",0);
    float angle = YnOptionFindFloat(options, "angle",0);
    float saturation = YnOptionFindFloat(options, "saturation",1);
    float exposure = YnOptionFindFloat(options, "exposure",1);

    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;

    if (!(h && w && c))
        YnUtilError("Layer before crop layer must output image");

    noadjust = YnOptionFindIntQuiet(options, "noadjust",0);

    layer = YnLayerCropMake(batch, h, w, c, crop_height,crop_width,flip, angle, saturation, exposure);
    layer.shift = YnOptionFindFloat(options, "shift", 0);
    layer.noadjust = noadjust;

    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserMaxpool(tYnList *options,
                         tYnParserSizeParams params)
{
    tYnLayer layer;
    int batch, h, w, c;
    int stride = YnOptionFindInt(options, "stride",1);
    int size = YnOptionFindInt(options, "size",stride);

    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;

    if (!(h && w && c))
        YnUtilError("Layer before maxpool layer must output image");

    layer = YnLayerMaxpoolMake(batch, h, w, c, size, stride);

    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserAvgpool(tYnList *options,
                         tYnParserSizeParams params)
{
    tYnLayer layer;
    int batch, w, h, c;

    w = params.w;
    h = params.h;
    c = params.c;
    batch = params.batch;

    if (!(h && w && c))
        YnUtilError("Layer before avgpool layer must output image.");

    layer = YnLayerAvgpoolMake(batch, w, h, c);

    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserDropout(tYnList *options,
                         tYnParserSizeParams params)
{
    float probability = YnOptionFindFloat(options, "probability", .5);
    tYnLayer layer = YnLayerDropoutMake(params.batch, params.inputs, probability);
    layer.outW = params.w;
    layer.outH = params.h;
    layer.outC = params.c;
    return layer;
}

YN_STATIC_INLINE
tYnLayer YnParserActivation(tYnList *options,
                            tYnParserSizeParams params)
{
    tYnLayer layer = {0};
    char *activation_s = YnOptionFindStr(options, "activation", "linear");
    eYnActivationType activation;

    if (YnActivationTypeFromStringGet(activation_s, &activation) != eYnRetOk)
    {
        YnUtilError("Get activation type failed");
        return layer;
    }

    layer = YnLayerActivationMake(params.batch, params.inputs, activation);

    layer.outH = params.h;
    layer.outW = params.w;
    layer.outC = params.c;
    layer.h = params.h;
    layer.w = params.w;
    layer.c = params.c;

    return layer;
}

void YnParserNetOptions(tYnList *options,
        tYnNetwork *net)
{
    char *policy_s;
    char *l;
    char *p;
    int len;
    int n;
    int i;
    int *steps;
    float *scales;
    int step;
    float scale;

    net->batch = YnOptionFindInt(options, "batch",1);
    net->learningRate = YnOptionFindFloat(options, "learning_rate", .001);
    net->momentum = YnOptionFindFloat(options, "momentum", .9);
    net->decay = YnOptionFindFloat(options, "decay", .0001);
    int subdivs = YnOptionFindInt(options, "subdivisions",1);
    net->timeSteps = YnOptionFindIntQuiet(options, "time_steps",1);
    net->batch /= subdivs;
    net->batch *= net->timeSteps;
    net->subdivisions = subdivs;

    net->h = YnOptionFindIntQuiet(options, "height",0);
    net->w = YnOptionFindIntQuiet(options, "width",0);
    net->c = YnOptionFindIntQuiet(options, "channels",0);
    net->inputs = YnOptionFindIntQuiet(options, "inputs", net->h * net->w * net->c);

    if (!net->inputs && !(net->h && net->w && net->c))
        YnUtilError("No input parameters supplied");

    policy_s = YnOptionFindStr(options, "policy", "constant");

    net->policy = YnParserPolicyGet(policy_s);
    if (net->policy == cYnNetworkLearnRateStep)
    {
        net->step = YnOptionFindInt(options, "step", 1);
        net->scale = YnOptionFindFloat(options, "scale", 1);
    }
    else if (net->policy == cYnNetworkLearnRateSteps)
    {
        l = YnOptionFind(options, "steps");
        p = YnOptionFind(options, "scales");

        if (!l || !p)
            YnUtilError("STEPS policy must have steps and scales in cfg file");

        len = strlen(l);
        n = 1;

        for(i = 0; i < len; i ++)
            if (l[i] == ',')
                n ++;

        steps = calloc(n, sizeof(int));
        scales = calloc(n, sizeof(float));

        for (i = 0; i < n; i ++)
        {
            step = atoi(l);
            scale = atof(p);
            l = strchr(l, ',') + 1;
            p = strchr(p, ',') + 1;
            steps[i] = step;
            scales[i] = scale;
        }

        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;

    }
    else if (net->policy == cYnNetworkLearnRateExp)
    {
        net->gamma = YnOptionFindFloat(options, "gamma", 1);
    }
    else if (net->policy == cYnNetworkLearnRateSig)
    {
        net->gamma = YnOptionFindFloat(options, "gamma", 1);
        net->step = YnOptionFindInt(options, "step", 1);
    }
    else if (net->policy == cYnNetworkLearnRatePoly)
    {
        net->power = YnOptionFindFloat(options, "power", 1);
    }

    net->max_batches = YnOptionFindInt(options, "max_batches", 0);
}

tYnNetwork YnParserNetworkCfg(char *filename)
{
    tYnLayer layer;
    tYnNetwork net;
    tYnParserSizeParams params;
    tYnParserSection *s;
    tYnList *options;
    int count;

    tYnList *sections = YnParserReadCfg(filename);
    tYnListNode *n = sections->front;

    if (!n)
        YnUtilError("Config file has no sections");

    net = YnNetworkMake(sections->size - 1);

    s = (tYnParserSection *)n->val;
    options = s->options;

    if (!YnParserIsNetwork(s))
        YnUtilError("First section must be [net] or [network]");

    YnParserNetOptions(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    params.batch = net.batch;
    params.timeSteps = net.timeSteps;

    n = n->next;
    count = 0;
    YnParserFreeSection(s);

    while (n)
    {
        params.index = count;
        fprintf(stderr, "%d: ", count);
        s = (tYnParserSection *)n->val;
        options = s->options;
        memset(&layer, 0, sizeof(tYnLayer));

        if (YnParserIsConvolutional(s))
            layer = YnParserConvolutional(options, params);
        /*else if (YnParserIsLocal(s))
            layer = YnParserLocal(options, params);*/
        else if (YnParserIsActivation(s))
            layer = YnParserActivation(options, params);
        else if (YnParserIsDeconvolutional(s))
            layer = YnParserDeconvolutional(options, params);
        /*else if (YnParserIsRnn(s))
            layer = YnParserRnn(options, params);*/
        else if (YnParserIsConnected(s))
            layer = YnParserConnected(options, params);
        else if (YnParserIsCrop(s))
            layer = YnParserCrop(options, params);
        else if (YnParserIsCost(s))
            layer = YnParserCost(options, params);
        else if (YnParserIsDetection(s))
            layer = YnParserDetection(options, params);
        else if (YnParserIsSoftmax(s))
            layer = YnParserSoftmax(options, params);
        /*else if (YnParserIsNormalization(s))
            layer = YnParserNormalization(options, params);*/
        else if (YnParserIsMaxpool(s))
            layer = YnParserMaxpool(options, params);
        else if (YnParserIsAvgpool(s))
            layer = YnParserAvgpool(options, params);
        /*else if (YnParserIsRoute(s))
            layer = YnParserRoute(options, params, net);*/
        /*else if (YnParserIsShortcut(s))
            layer = YnParserShortcut(options, params, net);*/
        else if (YnParserIsDropout(s))
        {
            layer = YnParserDropout(options, params);
            layer.output = net.layers[count - 1].output;
            layer.delta = net.layers[count - 1].delta;

#ifdef YN_GPU
            layer.outputGpu = net.layers[count - 1].outputGpu;
            layer.deltaGpu = net.layers[count - 1].deltaGpu;
#endif
        }
        else
        {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        layer.dontload = YnOptionFindIntQuiet(options, "dontload", 0);
        layer.dontloadscales = YnOptionFindIntQuiet(options, "dontloadscales", 0);
        YnOptionUnused(options);
        net.layers[count] = layer;
        YnParserFreeSection(s);
        n = n->next;
        count ++;

        if (n)
        {
            params.h = layer.outH;
            params.w = layer.outW;
            params.c = layer.outC;
            params.inputs = layer.outputs;
        }
    }

    YnListFree(sections);
    net.outputs = YnNetworkOutputSizeGet(net);
    net.output = YnNetworkOutputGet(net);
    return net;
}

void YnParserWeightsUptoSave(tYnNetwork net,
                             char *filename,
                             int cutoff)
{
    fprintf(stderr, "Saving weights to %s\n", filename);

    int i;
    int num;
    tYnLayer layer;
    int major = 0;
    int minor = 1;
    int revision = 0;
    FILE *fp = fopen(filename, "w");

    if (!fp)
        YnUtilErrorOpenFile(filename);

    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net.seen, sizeof(int), 1, fp);

    for(i = 0; i < net.n && i < cutoff; i ++)
    {
        layer = net.layers[i];
        if (layer.type == cYnLayerConvolutional)
        {
#ifdef YN_GPU
            if (YnCudaGpuIndexGet() >= 0)
                YnLayerConvolutionalGpuPull(layer);
#endif
            num = layer.n * layer.c * layer.size * layer.size;
            fwrite(layer.biases, sizeof(float), layer.n, fp);

            if (layer.batchNormalize)
            {
                fwrite(layer.scales, sizeof(float), layer.n, fp);
                fwrite(layer.rollingMean, sizeof(float), layer.n, fp);
                fwrite(layer.rollingVariance, sizeof(float), layer.n, fp);
            }

            fwrite(layer.filters, sizeof(float), num, fp);

        }
        if (layer.type == cYnLayerConnected)
        {
            YnParserConnectedWeightsSave(layer, fp);
        }
        if (layer.type == cYnLayerRnn)
        {
            /*
            YnParserConnectedWeightsSave(*(layer.inputLayer), fp);
            YnParserConnectedWeightsSave(*(layer.selfLayer), fp);
            YnParserConnectedWeightsSave(*(layer.outputLayer), fp);
            */
        }
        if (layer.type == cYnLayerLocal)
        {
            /*
            #ifdef YN_GPU
            if (YnCudaGpuIndexGet() >= 0)
                YnLayerLocalGpuPull(layer);
            #endif

            int locations = layer.outW * layer.outH;
            int size = layer.size * layer.size * layer.c * layer.n * locations;
            fwrite(layer.biases, sizeof(float), layer.outputs, fp);
            fwrite(layer.filters, sizeof(float), size, fp);
             */
        }
    }

    fclose(fp);
}

void YnParserWeightsSave(tYnNetwork net,
                         char *filename)
{
    YnParserWeightsUptoSave(net, filename, net.n);
}


void YnPareserWeightsDoubleSave(tYnNetwork net,
                                char *filename)
{
    fprintf(stderr, "Saving doubled weights to %s\n", filename);
    int i, j, k;
    float zero;
    int index;
    tYnLayer layer;
    FILE *fp = fopen(filename, "w");

    if (!fp)
        YnUtilErrorOpenFile(filename);

    fwrite(&net.learningRate, sizeof(float), 1, fp);
    fwrite(&net.momentum, sizeof(float), 1, fp);
    fwrite(&net.decay, sizeof(float), 1, fp);
    fwrite(net.seen, sizeof(int), 1, fp);

    for(i = 0; i < net.n; i ++)
    {
        layer = net.layers[i];
        if (layer.type == cYnLayerConvolutional)
        {

#ifdef YN_GPU
            if (YnCudaGpuIndexGet() >= 0)
                YnLayerConvolutionalGpuPull(layer);
#endif
            zero = 0;
            fwrite(layer.biases, sizeof(float), layer.n, fp);
            fwrite(layer.biases, sizeof(float), layer.n, fp);

            for (j = 0; j < layer.n; j ++)
            {
                index = j * layer.c * layer.size * layer.size;
                fwrite(layer.filters + index, sizeof(float), layer.c * layer.size * layer.size, fp);

                for (k = 0; k < layer.c * layer.size * layer.size; k ++)
                    fwrite(&zero, sizeof(float), 1, fp);
            }
            for (j = 0; j < layer.n; j ++)
            {
                index = j * layer.c * layer.size * layer.size;
                for (k = 0; k < layer.c * layer.size * layer.size; k ++)
                    fwrite(&zero, sizeof(float), 1, fp);

                fwrite(layer.filters + index, sizeof(float), layer.c * layer.size * layer.size, fp);
            }
        }
    }

    fclose(fp);
}

void YnParserConnectedWeightsLoad(tYnLayer layer,
                                   FILE *fp,
                                   int transpose)
{
    fread(layer.biases, sizeof(float), layer.outputs, fp);
    fread(layer.weights, sizeof(float), layer.outputs * layer.inputs, fp);

    if (transpose)
    {
        YnParserTransposeMatrix(layer.weights, layer.inputs, layer.outputs);
    }

    if (layer.batchNormalize && (!layer.dontloadscales))
    {
        fread(layer.scales, sizeof(float), layer.outputs, fp);
        fread(layer.rollingMean, sizeof(float), layer.outputs, fp);
        fread(layer.rollingVariance, sizeof(float), layer.outputs, fp);
    }

#ifdef YN_GPU
    if (YnCudaGpuIndexGet() >= 0)
        YnLayerConnectedGpuPush(layer);
#endif

}

void YnParserWeightsUptoLoad(tYnNetwork *net,
                             char *filename,
                             int cutoff)
{
    tYnLayer layer;
    int major;
    int minor;
    int revision;
    int transpose;
    int i;
    int num;
    /*int locations;*/
    /*int size;*/

    FILE *fp = fopen(filename, "rb");
    if (!fp)
        YnUtilErrorOpenFile(filename);

    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);

    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    fread(net->seen, sizeof(int), 1, fp);
    transpose = (major > 1000) || (minor > 1000);

    for(i = 0; i < net->n && i < cutoff; i ++)
    {
        layer = net->layers[i];

        if (layer.dontload)
            continue;

        if (layer.type == cYnLayerConvolutional)
        {
            num = layer.n * layer.c * layer.size * layer.size;
            fread(layer.biases, sizeof(float), layer.n, fp);

            if (layer.batchNormalize && (!layer.dontloadscales))
            {
                fread(layer.scales, sizeof(float), layer.n, fp);
                fread(layer.rollingMean, sizeof(float), layer.n, fp);
                fread(layer.rollingVariance, sizeof(float), layer.n, fp);
            }

            fread(layer.filters, sizeof(float), num, fp);
            if (layer.flipped)
            {
                YnParserTransposeMatrix(layer.filters, layer.c * layer.size * layer.size, layer.n);
            }
#ifdef YN_GPU
            if (YnCudaGpuIndexGet() >= 0)
                YnLayerConvolutionalGpuPush(layer);
#endif
        }

        if (layer.type == cYnLayerDeconvolutional)
        {
            num = layer.n * layer.c * layer.size * layer.size;
            fread(layer.biases, sizeof(float), layer.n, fp);
            fread(layer.filters, sizeof(float), num, fp);

#ifdef YN_GPU
            if (YnCudaGpuIndexGet() >= 0)
                YnLayerConvolutionalGpuPush(layer);
#endif
        }
        if (layer.type == cYnLayerConnected)
        {
        	YnParserConnectedWeightsLoad(layer, fp, transpose);
        }
        if (layer.type == cYnLayerRnn)
        {
            /*
            YnParserConnectedWeightsLoad(*(layer.inputLayer), fp, transpose);
            YnParserConnectedWeightsLoad(*(layer.selfLayer), fp, transpose);
            YnParserConnectedWeightsLoad(*(layer.outputLayer), fp, transpose);
            */
        }
        if (layer.type == cYnLayerLocal)
        {
            /*
            locations = layer.outW * layer.outH;
            size = layer.size * layer.size * layer.c * layer.n * locations;
            fread(layer.biases, sizeof(float), layer.outputs, fp);
            fread(layer.filters, sizeof(float), size, fp);

            #ifdef YN_GPU
            if (YnCudaGpuIndexGet() >= 0)
                YnLayerLocalGpuPush(layer);
            #endif
             */

        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void YnParserWeightsLoad(tYnNetwork *net,
                         char *filename)
{
    YnParserWeightsUptoLoad(net, filename, net->n);
}

