//	File        :   YnNet.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   29-08-2016
//	Author      :   haittt

#include "../include/YnVoc.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        fprintf(stderr, "not enough param\n", argv[0]);
        return 0;
    }

    if  (YnUtilFindCharArg(argc, argv, "-nogpu"))
        YnCudaGpuIndexSet(-1);
    else
        YnCudaGpuIndexSet(YnUtilFindIntArg(argc, argv, "-i", 0));

#ifndef YN_GPU
    YnCudaGpuIndexSet(-1);
#else
    if(YnCudaGpuIndexGet() >= 0)
    {
        cudaError_t status = cudaSetDevice(YnCudaGpuIndexGet());
        YnCudaCheckError(status);
    }
#endif

    if(strcmp(argv[1], "voc") == 0)
        YnVocRun(argc, argv);
    else
        fprintf(stderr, "option isn't supported: %s\n", argv[1]);

    return 0;
}
