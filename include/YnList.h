#ifndef YNLIST_H
#define YNLIST_H

#include "../YnStd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Typedef */
typedef struct tYnListNode{
    void * val;
    tYnListNode * next;
    tYnListNode * prev;
} tYnListNode;

typedef struct tYnList{
    uint32 size;
    tYnListNode * front;
    tYnListNode * back;

    void * freeFunc;
} tYnList;

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */


/*
 * Create new list
 */
YN_FINAL
tYnList * YnListNew(void * freeFunc)
YN_ALSWAY_INLINE;

/*
 * Pop from list
 */
YN_FINAL
void * YnListPop(tYnList * list)
YN_ALSWAY_INLINE;

/*
 * Insert to list
 */
YN_FINAL
void * YnListInsert(tYnList * list, void * pVal)
YN_ALSWAY_INLINE;

/*
 * Free from node
 */
YN_FINAL
void YnListFreeFromNode(tYnList * list, tYnListNode * node)
YN_ALSWAY_INLINE;

/*
 *  Free all member value
 */
YN_FINAL
void * YnListFreeAllValue(tYnList * list)
YN_ALSWAY_INLINE;

/*
 *  Convert list to array
 */
YN_FINAL
void ** YnListToArr(tYnList * list)
YN_ALSWAY_INLINE;


#ifdef __cplusplus
}
#endif

#endif /* YNLIST_H */
