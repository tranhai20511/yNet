#ifndef YNPARSER_H
#define YNPARSER_H

#include "YnNetwork.h"
#include "YnOptionList.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */
typedef struct tYnParserSection{
    char *type;
    tYnList *options;
}tYnParserSection;

typedef struct tYnParserSizeParams{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int timeSteps;
} tYnParserSizeParams;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
tYnNetwork YnParserNetworkCfg(char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserNetOptions(tYnList *options,
        tYnNetwork *net)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserSaveNetwork(tYnNetwork net,
        char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserWeightsUptoSave(tYnNetwork net,
         char *filename,
         int cutoff)
YN_ALSWAY_INLINE;

YN_FINAL
void YnPareserWeightsDoubleSave(tYnNetwork net,
        char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserWeightsSave(tYnNetwork net,
         char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserWeightsLoad(tYnNetwork *net,
         char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserWeightsUptoLoad(tYnNetwork *net,
         char *filename,
         int cutoff)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserConnectedWeightsLoad(tYnLayer layer,
       FILE *fp,
       int transpose)
YN_ALSWAY_INLINE;

#endif /* YNPARSER_H */

