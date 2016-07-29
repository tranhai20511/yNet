#ifndef YNUTIL_H
#define YNUTIL_H

#include "../YnList.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */

/*
 * Free mem
 */
YN_FINAL
void YnUtilFree (void * mem)
YN_ALSWAY_INLINE;

/*
 * Free mem
 */
YN_FINAL
void YnUtilFreeArrPtrs (void ** mem,
        uint32 num)
YN_ALSWAY_INLINE;

/*
 * Set value a argument to 0
 */
YN_FINAL
void YnUtilDelArg (uint32 argc,
        char ** argv,
        uint32 index)
YN_ALSWAY_INLINE;

/*
 * Find a argument
 */
YN_FINAL
bool YnUtilFindDelArg (uint32 argc,
        char ** argv,
        char * argFind)
YN_ALSWAY_INLINE;

/*
 * Find a argument int value
 * %return: next int argument, delete other arguments
 */
YN_FINAL
int YnUtilFindIntArg (uint32 argc,
        char ** argv,
        char * argFind)
YN_ALSWAY_INLINE;

/*
 * Find a argument float value
 * %return: next float argument, delete other arguments
 */
YN_FINAL
float YnUtilFindFloatArg(uint32 argc,
        char ** argv,
        char * argFind)
YN_ALSWAY_INLINE;

/*
 * Find a argument char value
 * %return: next char argument, delete other arguments
 */
YN_FINAL
char * YnUtilFindCharArg(uint32 argc,
        char ** argv,
        char * argFind)
YN_ALSWAY_INLINE;

/*
 * Find base argument in config path
 * %return: name of config model
 */
YN_FINAL
char * YnUtilFindBaseConfig(char * configPath)
YN_ALSWAY_INLINE;

/*
 * Convert char to int num
 */
YN_FINAL
int32 YnUtilNumCharToInt(char c)
YN_ALSWAY_INLINE;

/*
 * Convert num int to char
 */
YN_FINAL
char YnUtilIntToNumChar(char c)
YN_ALSWAY_INLINE;


/*
 * Replace char in string
 * %return: new string with replaced char
 */
YN_FINAL
char * YnUtilReplaceChar(char * str,
        char * from,
        char * to)
YN_ALSWAY_INLINE;

/*
 * Get second from clock
 */
YN_FINAL
float YnUtilSecFromClock(clock_t clock)
YN_ALSWAY_INLINE;

/*
 * Get top
 */
YN_FINAL
void YnUtilTop(float * array,
        uint32 numOrigin,
        uint32 numIndex,
        int * indexArr)
YN_ALSWAY_INLINE;

/*
 * Error exit
 */
YN_FINAL
void YnUtilError(const char * s)
YN_ALSWAY_INLINE;

/*
 * Error malloc
 */
YN_FINAL
void YnUtilErrorMalloc()
YN_ALSWAY_INLINE;

/*
 * Error open file
 */
YN_FINAL
void YnUtilErrorOpenFile(const char * s)
YN_ALSWAY_INLINE;

/*
 * Split string
 */
YN_FINAL
tYnList * YnUtilSplitString(char * str,
        char delim)
YN_ALSWAY_INLINE;

/*
 * Strip special characters in string
 */
YN_FINAL
void YnUtilStripString(char * str)
YN_ALSWAY_INLINE;

/*
 * Strip special characters in string
 */
YN_FINAL
void YnUtilStripStringSpec(char * str,
        char specChar)
YN_ALSWAY_INLINE;

/*
 * File get line
 */
YN_FINAL
char * YnUtilFileGetLine(FILE * file)
YN_ALSWAY_INLINE;

/*
 * Read file
 */
YN_FINAL
void YnUtilFileRead(int fd,
        char * buffer,
        uint32 size);
YN_ALSWAY_INLINE;

/*
 * Write file
 */
YN_FINAL
void YnUtilFileWrite(int fd,
        char * buffer,
        uint32 size);
YN_ALSWAY_INLINE;

/*
 * str copy
 */
YN_FINAL
char * YnUtilStringCopy(char * str);
YN_ALSWAY_INLINE;

/*
 * Count fields in line
 */
YN_FINAL
uint32 YnUtilLineFieldCount(char * line);
YN_ALSWAY_INLINE;

/*
 * Parse fields in line
 */
YN_FINAL
float * YnUtilLineFieldParse(char * line,
        uint32 numField);
YN_ALSWAY_INLINE;

/*
 * Constrain value
 */
YN_FINAL
float YnUtilConstrain(float min,
        float max,
        float val);
YN_ALSWAY_INLINE;

/*
 * Sum array
 */
YN_FINAL
float YnUtilArraySum(char * array,
        uint32 numField);
YN_ALSWAY_INLINE;

/*
 * Mean array
 */
YN_FINAL
float YnUtilArrayMean(char * array,
        uint32 numField);
YN_ALSWAY_INLINE;

/*
 * Variance array
 */
YN_FINAL
float YnUtilArrayVariance(float * array,
        uint32 num);
YN_ALSWAY_INLINE;

/*
 * Mean square error array
 */
YN_FINAL
float YnUtilArrayMse(float * arrayErr,
        uint32 num);
YN_ALSWAY_INLINE;

/*
 * Normalize array
 */
YN_FINAL
void YnUtilArrayNormalize(float * array,
        uint32 num);
YN_ALSWAY_INLINE;

/*
 * Mean square array
 */
YN_FINAL
float YnUtilArrayMag(float * arrayErr,
        uint32 num);
YN_ALSWAY_INLINE;

/*
 * Scale array
 */
YN_FINAL
void YnUtilArrayScale(float * array,
        uint32 num,
        float scale);
YN_ALSWAY_INLINE;

/*
 * Translate array
 */
YN_FINAL
void YnUtilArrayTranslate(float * array,
        int n,
        float s)
YN_ALSWAY_INLINE;

/*
 * Find index of max value element in array
 */
YN_FINAL
float YnUtilArrayMaxIndex(float * array,
        uint32 num);
YN_ALSWAY_INLINE;

/*
 * Get random normal number
 */
YN_FINAL
float YnUtilRandomNormalNum();
YN_ALSWAY_INLINE;

/*
 * Get random uniform number
 */
YN_FINAL
float YnUtilRandomUniformNum(float min, float max);
YN_ALSWAY_INLINE;

#ifdef __cplusplus
}
#endif

#endif /* YNUTIL_H */
