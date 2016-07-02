//  File        :   YnUtil.c
//  Brief       :   Implement yNet standard functions.
//  DD-MM_YYYY  :   26-06-2016
//  Author      :   haittt

#include "../YnUtil.h"


/**************** Define */
#define PI2     (6.2831853071795864769252866)

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

void YnUtilFree (void * mem)
{

    if (mem)
    {
        free(mem);
        mem = NULL;
    }
}
