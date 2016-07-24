//	File        :   YnOptionList.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   04-07-2016
//	Author      :   haittt

#include "../YnOptionList.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
int YnOptionRead(char *s,
        tYnList *options)
{
    uint32 i;
    uint32 len = strlen(s);
    char *val = 0;
    char *key;

    for (i = 0; i < len; i ++)
    {
        if (s[i] == '=')
        {
            s[i] = '\0';
            val = s + i + 1;
            break;
        }
    }

    if (i == (len - 1))
        return 0;

    key = s;
    YnOptionInsert(options, key, val);

    return 1;
}

void YnOptionInsert(tYnList *l,
        char *key,
        char *val)
{
    tYnOptionKeyVal *p = malloc(sizeof(tYnOptionKeyVal));

    p->key = key;
    p->val = val;
    p->used = 0;

    YnListInsert(l, p);
}

void YnOptionUnused(tYnList *l)
{
    tYnListNode *n = l->front;

    while(n)
    {
        tYnOptionKeyVal *p = (tYnOptionKeyVal *)n->val;

        if (!p->used)
        {
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }

        n = n->next;
    }
}

char * YnOptionFind(tYnList *l,
        char *key)
{
    tYnListNode *n = l->front;

    while(n)
    {
        tYnOptionKeyVal *p = (tYnOptionKeyVal *)n->val;

        if (strcmp(p->key, key) == 0)
        {
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }

    return 0;
}

char * YnOptionFindStr(tYnList *l,
        char *key,
        char *def)
{
    char *v = YnOptionFind(l, key);

    if (v)
        return v;

    if (def)
        fprintf(stderr, "%s: Using default '%s'\n", key, def);

    return def;
}

int YnOptionFindInt(tYnList *l,
        char *key,
        int def)
{
    char *v = YnOptionFind(l, key);

    if (v)
        return atoi(v);

    fprintf(stderr, "%s: Using default '%d'\n", key, def);

    return def;
}

int YnOptionFindIntQuiet(tYnList *l,
        char *key,
        int def)
{
    char *v = YnOptionFind(l, key);

    if (v)
        return atoi(v);

    return def;
}

float YnOptionFindFloatQuiet(tYnList *l,
        char *key,
        float def)
{
    char *v = YnOptionFind(l, key);

    if (v)
        return atof(v);

    return def;
}

float YnOptionFindFloat(tYnList *l,
        char *key,
        float def)
{
    char *v = YnOptionFind(l, key);

    if (v)
        return atof(v);

    fprintf(stderr, "%s: Using default '%lf'\n", key, def);

    return def;
}
