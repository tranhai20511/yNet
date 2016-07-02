//	File        :   YnList.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   01-07-2016
//	Author      :   haittt

#include "../YnList.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

tYnList * YnListNew(void * freeFunc)
{
    tYnList * list = malloc(sizeof(tYnList));
    mYnNullRetNull(list);

    memset(list, 0, sizeof(tYnList));
    list->freeFunc = freeFunc;

    return list;
}

void * YnListPop(tYnList * list)
{
    tYnListNode * popNode = NULL;
    void * popVal = NULL;

    mYnNullRetNull(list);
    mYnNullRetNull(list->back);

    popNode = list->back;
    popVal = popNode->val;

    list->back = popNode->prev;

    if (list->back)
        list->back->next = NULL;

    list->size --;

    YnUtilFree(popNode);

    return popVal;
}

void * YnListInsert(tYnList * list, void * pVal)
{
    tYnListNode * newNode = malloc(sizeof(tYnListNode));
    mYnNullRetNull(newNode);

    newNode->val = pVal;
    newNode->next = NULL;

    if (!list->back)
    {
        list->front = newNode;
        newNode->prev = NULL;
    }
    else
    {
        list->back->next = newNode;
        newNode->prev = list->back;
    }

    list->back = newNode;
    list->size ++;

    return pVal;
}

void YnListFreeFromNode(tYnList * list, tYnListNode * node)
{
    tYnListNode * nextNode = NULL;

    while (node)
    {
        nextNode = node->next;

        if (list->freeFunc)
            list->freeFunc(node->val);

        YnUtilFree(node);

        node = nextNode;
    }
}

void YnListFree(tYnList * list)
{
    YnListFreeFromNode(list->front);
    YnUtilFree(list);
}

void YnListFreeAllValue(tYnList * list)
{
    tYnListNode * node = list->front;

    while (node)
    {
        if (list->freeFunc)
            list->freeFunc(node->val);

        node = node->next;
    }
}

void ** YnListToArr(tYnList * list)
{
    tYnListNode * node = NULL;
    int count = 0;

    void ** array = calloc(list->size, sizeof(void*));
    mYnNullRetNull(array);

    node = list->front;
    while (node)
    {
        array[count++] = node->val;
        node = node->next;
    }

    return array;
}

