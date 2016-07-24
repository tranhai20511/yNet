//	File        :   YnParser.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   24-07-2016
//	Author      :   haittt

#include "../YnList.h"
#include "../YnOptionList.h"
#include "../YnUtil.h"
#include "../YnActivations.h"
#include "../YnLayerCrop.h"
#include "../YnLayerCost.h"
#include "../YnLayerConvolutional.h"
#include "../YnLayerActivation.h"
#include "../YnLayerNormalization.h"
#include "../YnLayerDeconvolutional.h"
#include "../YnLayerConnected.h"
#include "../YnLayerRnn.h"
#include "../YnLayerMaxpool.h"
#include "../YnLayerSoftmax.h"
#include "../YnLayerDropout.h"
#include "../YnLayerDetection.h"
#include "../YnLayerAvgpool.h"
#include "../YnLayerLocal.h"
#include "../YnLayerRoute.h"
#include "../YnLayerShortcut.h"

#include "../YnParser.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */
YN_STATIC
int YnParserIsNetwork(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsConvolutional(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsActivation(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsLocal(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsDeconvolutional(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsConnected(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsRnn(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsMaxpool(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsAvgpool(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsDropout(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsSoftmax(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsNormalization(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsCrop(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsShortcut(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsCost(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsDetection(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
int YnParserIsRoute(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
tYnList * YnParserReadConfig(char *filename)
YN_ALSWAY_INLINE;

YN_STATIC
tYnList * YnParserFreeSection(tYnParserSection *s)
YN_ALSWAY_INLINE;

YN_STATIC
tYnNetworkLearnRatePolicy * YnParserGetPolicy(char *s)
YN_ALSWAY_INLINE;

YN_STATIC
void YnParserTransposeMatrix(float *a,
        int rows,
        int cols)
YN_ALSWAY_INLINE;

/**************** Implement */
YN_STATIC
int YnParserIsShortcut(tYnParserSection *s)
{
    return (strcmp(s->type, "[shortcut]") == 0);
}

YN_STATIC
int YnParserIsCrop(tYnParserSection *s)
{
    return (strcmp(s->type, "[crop]") == 0);
}

YN_STATIC
int YnParserIsCost(tYnParserSection *s)
{
    return (strcmp(s->type, "[cost]") == 0);
}

YN_STATIC
int YnParserIsDetection(tYnParserSection *s)
{
    return (strcmp(s->type, "[detection]") == 0);
}

YN_STATIC
int YnParserIsLocal(tYnParserSection *s)
{
    return (strcmp(s->type, "[local]") == 0);
}

YN_STATIC
int YnParserIsDeconvolutional(tYnParserSection *s)
{
    return ((strcmp(s->type, "[deconv]") == 0) ||
            (strcmp(s->type, "[deconvolutional]") == 0));
}

YN_STATIC
int YnParserIsConvolutional(tYnParserSection *s)
{
    return ((strcmp(s->type, "[conv]") == 0) ||
            (strcmp(s->type, "[convolutional]") == 0));
}

YN_STATIC
int YnParserIsActivation(tYnParserSection *s)
{
    return (strcmp(s->type, "[activation]") == 0);
}

YN_STATIC
int YnParserIsNetwork(tYnParserSection *s)
{
    return ((strcmp(s->type, "[net]") == 0) ||
            (strcmp(s->type, "[network]") == 0));
}

YN_STATIC
int YnParserIsRnn(tYnParserSection *s)
{
    return (strcmp(s->type, "[rnn]") == 0);
}

YN_STATIC
int YnParserIsConnected(tYnParserSection *s)
{
    return ((strcmp(s->type, "[conn]") == 0) ||
            (strcmp(s->type, "[connected]") == 0));
}

YN_STATIC
int YnParserIsMaxpool(tYnParserSection *s)
{
    return ((strcmp(s->type, "[max]") == 0) ||
            (strcmp(s->type, "[maxpool]") == 0));
}

YN_STATIC
int YnParserIsAvgpool(tYnParserSection *s)
{
    return ((strcmp(s->type, "[avg]") == 0) ||
            (strcmp(s->type, "[avgpool]") == 0));
}

YN_STATIC
int YnParserIsDropout(tYnParserSection *s)
{
    return (strcmp(s->type, "[dropout]") == 0);
}

YN_STATIC
int YnParserIsNormalization(tYnParserSection *s)
{
    return ((strcmp(s->type, "[lrn]") == 0) ||
            (strcmp(s->type, "[normalization]") == 0));
}

YN_STATIC
int YnParserIsSoftmax(tYnParserSection *s)
{
    return ((strcmp(s->type, "[soft]") == 0) ||
            (strcmp(s->type, "[softmax]") == 0));
}

YN_STATIC
int YnParserIsRoute(tYnParserSection *s)
{
    return (strcmp(s->type, "[route]") == 0);
}

YN_STATIC
tYnList * YnParserReadCfg(char *filename)
{
    char *line;
    int nu = 0;
    tYnList *sections;
    tYnParserSection *current = 0;
    FILE *file = fopen(filename, "r");

    if (file == 0)
        file_error(filename);

    sections = YnListNew();

    while((line = fgetl(file)) != 0)
    {
        ++ nu;
        YnUtilStripString(line);

        switch(line[0])
        {
            case '[':
                current = malloc(sizeof(tYnParserSection));
                YnListInsert(sections, current);
                current->options = YnListNew();
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

YN_STATIC
tYnList * YnParserFreeSection(tYnParserSection *s)
{
    YnUtilFree(s->type);
    tYnListNode *n = s->options->front;

    while(n)
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

YN_STATIC
eYnNetworkLearnRatePolicy * YnParserGetPolicy(char *s)
{
    if (strcmp(s, "poly") == 0)       return cYnNetworkPoly;
    if (strcmp(s, "constant") == 0)   return cYnNetworkConstant;
    if (strcmp(s, "step") == 0)       return cYnNetworkStep;
    if (strcmp(s, "exp") == 0)        return cYnNetworkExp;
    if (strcmp(s, "sigmoid") == 0)    return cYnNetworkSig;
    if (strcmp(s, "steps") == 0)      return cYnNetworkSteps;

    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);

    return cYnNetworkConstant;
}

YN_STATIC
void YnParserTransposeMatrix(float *a,
        int rows,
        int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;

    for(x = 0; x < rows; x ++)
    {
        for(y = 0; y < cols; y ++)
        {
            transpose[(y * rows) + x] = a[(x * cols) + y];
        }
    }

    memcpy(a, transpose, rows * cols * sizeof(float));

    YnUtilFree(transpose);
}

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
        while((*++next !='\0') && (*next != ','));

        if (*next == '\0')
            done = 1;

        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}
