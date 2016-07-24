#ifndef YNOPTIONLIST_H
#define YNOPTIONLIST_H

#include "../YnList.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */
typedef struct tYnOptionKeyVal{
    char *key;
    char *val;
    int used;
} tYnOptionKeyVal;

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
YN_FINAL
int YnOptionRead(char *s,
        tYnList *options)
YN_ALSWAY_INLINE;

YN_FINAL
void YnOptionInsert(tYnList *l,
        char *key,
        char *val)
YN_ALSWAY_INLINE;

YN_FINAL
void YnOptionUnused(tYnList *l)
YN_ALSWAY_INLINE;

YN_FINAL
char * YnOptionFind(tYnList *l,
        char *key)
YN_ALSWAY_INLINE;

YN_FINAL
char * YnOptionFindStr(tYnList *l,
        char *key,
        char *def)
YN_ALSWAY_INLINE;

YN_FINAL
int YnOptionFindInt(tYnList *l,
        char *key,
        int def)
YN_ALSWAY_INLINE;

YN_FINAL
int YnOptionFindIntQuiet(tYnList *l,
        char *key,
        int def)
YN_ALSWAY_INLINE;

YN_FINAL
float YnOptionFindFloatQuiet(tYnList *l,
        char *key,
        float def)
YN_ALSWAY_INLINE;

YN_FINAL
float YnOptionFindFloat(tYnList *l,
        char *key,
        float def)
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNOPTIONLIST_H */
