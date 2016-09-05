//	File        :   YnGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   04-09-2016
//	Author      :   haittt

#include "../include/YnGpu.h"
#include "../include/YnUtil.h"
#include "../include/YnBlas.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */
static int gpuIndex = 0;

/**************** Local Implement */

/**************** Implement */
int YnCudaGpuIndexGet(void)
{
    return gpuIndex;
}

void YnCudaGpuIndexSet(int index)
{
    gpuIndex = index;
}
