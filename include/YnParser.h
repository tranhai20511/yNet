#ifndef YNPARSER_H
#define YNPARSER_H

#include "../YnOptionList.h"

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
    int time_steps;
} tYnParserSizeParams;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
tYnNetwork YnParserNetworkConfig(char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserSaveNetwork(tYnNetwork net,
        char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserSaveWeights(tYnNetwork net,
        char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserSaveWeightsUpto(tYnNetwork net,
        char *filename,
        int cutoff)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserSaveWeightsDouble(tYnNetwork net,
        char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserLoadWeights(tYnNetwork *net,
        char *filename)
YN_ALSWAY_INLINE;

YN_FINAL
void YnParserLoadWeightsUpto(tYnNetwork *net,
        char *filename,
        int cutoff)
YN_ALSWAY_INLINE;

#endif /* YNPARSER_H */
