//	File        :   YnUtil.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   03-07-2016
//	Author      :   haittt

#include <time.h>
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

void YnUtilFreeArrPtrs (void ** mem,
        uint32 num)
{
    uint32 idx = 0;

    mYnNullRet(mem);

    for (idx = 0; idx < num; idx ++)
    {
        YnUtilFree(mem[idx]);
    }

    YnUtilFree(mem);
}

void YnUtilDelArg (uint32 argc,
        char ** argv,
        uint32 index)
{
    uint32 i;

    for (i = index; i < (argc - 1); i ++)
    {
        argv[i] = argv[i + 1];
    }

    argv[i] = 0;
}

bool YnUtilFindDelArg (uint32 argc,
        char ** argv,
        char * argFind)
{
    uint32 i;

    for (i = 0; i < argc; i ++)
    {
        if (!argv[i])
            continue;

        if (strcmp(argv[i], argFind) == 0)
        {
            YnUtilDelArg(argc, argv, i);
            return true;
        }
    }

    return false;
}

int YnUtilFindIntArg (uint32 argc,
        char ** argv,
        char * argFind)
{
    uint32 i;
    int ret = 0;

    for (i = 0; i < argc - 1; i ++)
    {
        if (!argv[i])
            continue;

        if (strcmp(argv[i], argFind) == 0)
        {
            ret = atoi(argv[i + 1]);

            YnUtilDelArg(argc, argv, i);
            YnUtilDelArg(argc, argv, i);

            break;
        }
    }

    return ret;
}

float YnUtilFindFloatArg(uint32 argc,
        char ** argv,
        char * argFind)
{
    uint32 i;
    float ret = 0;

    for (i = 0; i < argc - 1; i ++)
    {
        if (!argv[i])
            continue;

        if (strcmp(argv[i], argFind) == 0)
        {
            ret = atof(argv[i + 1]);

            YnUtilDelArg(argc, argv, i);
            YnUtilDelArg(argc, argv, i);

            break;
        }
    }

    return ret;
}

char * YnUtilFindCharArg(uint32 argc,
        char ** argv,
        char * argFind)
{
    uint32 i;
    char * ret = 0;

    for (i = 0; i < argc - 1; i ++)
    {
        if (!argv[i])
            continue;

        if (strcmp(argv[i], argFind) == 0)
        {
            ret = argv[i + 1];

            YnUtilDelArg(argc, argv, i);
            YnUtilDelArg(argc, argv, i);

            break;
        }
    }

    return ret;
}

char * YnUtilFindBaseConfig(char * configPath)
{
    char * c = configPath;
    char * next = NULL;

    while ((next = strchr(c, '/')))
    {
        c = next + 1;
    }

    c = YnUtilStringCopy(c);
    next = strchr(c, '.');

    if (next)
        *next = 0;

    return c;
}

int32 YnUtilNumCharToInt(char c)
{
    return (c < 58) ? (c - 48) : (c - 87);
}

char YnUtilIntToNumChar(char c)
{
    if (c == 36)
        return '.';

    return (c < 10) ? (c + 48) : (c + 87);
}

char * YnUtilReplaceChar(char * str,
        char * from,
        char * to)
{
    static char buffer[YN_CHAR_BUFF];
    char * p = NULL;

    if (!(p = strstr(str, from)))
        return str;

    strncpy(buffer, str, p - str);

    buffer[p-str] = '\0';

    sprintf(buffer+(p-str), "%s%s", to, p+strlen(from));

    return buffer;
}

float YnUtilSecFromClock(clock_t clock)
{
    return (float)clock / CLOCKS_PER_SEC;
}

void YnUtilTop(float * array,
        uint32 numOrigin,
        uint32 numIndex,
        int * indexArr)
{
    uint32 i, j;

    for (j = 0; j < numIndex; j ++)
        indexArr[j] = -1;

    for (i = 0; i < numOrigin; i ++)
    {
        int curr = i;
        for (j = 0; j < numIndex; j ++)
        {
            if ((indexArr[j] < 0) || array[curr] > array[indexArr[j]])
            {
                int swap = curr;
                curr = indexArr[j];
                indexArr[j] = swap;
            }
        }
    }
}

void YnUtilError(const char *s)
{
    perror(s)
            ;
    exit(0);
}

void YnUtilErrorMalloc()
{
    fprintf(stderr, "Malloc error\n");

    exit(-1);
}

void YnUtilErrorOpenFile(const char * s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);

    exit(0);
}

tYnList * YnUtilSplitString(char * str,
        char delim)
{
    uint32 i;
    uint32 len = strlen(str);

    tYnList * list = YnListNew(NULL);

    YnListInsert(list, str);

    for (i = 0; i < len; i ++)
    {
        if (str[i] == delim)
        {
            str[i] = '\0';
            YnListInsert(list, &(str[i + 1]));
        }
    }

    return list;
}

void YnUtilStripString(char * str)
{
    uint32 i;
    uint32 len = strlen(str);
    uint32 offset = 0;
    char c;

    for (i = 0; i < len; ++i)
    {
        c = str[i];
        if (c == ' ' || c == '\t' || c == '\n')
            offset ++;
        else
            str[i - offset] = c;
    }

    str[len - offset] = '\0';
}

void YnUtilStripStringSpec(char * str,
        char specChar)
{
    uint32 i;
    uint32 len = strlen(str);
    uint32 offset = 0;
    char c;

    for (i = 0; i < len; ++i)
    {
        c = str[i];
        if (c == specChar)
            offset ++;
        else
            str[i - offset] = c;
    }

    str[len - offset] = '\0';
}

char * YnUtilFileGetLine(FILE * file)
{
    int32 curr = 0;
    uint32 size = YN_CHAR_BUFF;
    char *line = NULL;
    int32 readsize = 0;

    line = malloc(size * sizeof(char));
    mYnNullRetNull(line);

    if(feof(file))
        return NULL;

    if (!fgets(line, size, file))
    {
        YnUtilFree(line);
        return NULL;
    }

    curr = strlen(line);
    while ((line[curr - 1] != '\n') && !feof(file))
    {
        if (curr == size - 1)
        {
            size *= 2;
            line = realloc(line, size * sizeof(char));
            if (!line)
            {
                printf("%ld\n", size);
                YnUtilErrorMalloc();
            }
        }

        readsize = size - curr;
        if (readsize > INT_MAX)
            readsize = INT_MAX - 1;

        fgets(&line[curr], readsize, file);
        curr = strlen(line);
    }

    if (line[curr - 1] == '\n')
        line[curr - 1] = '\0';

    return line;
}

void YnUtilFileRead(int fd,
        char * buffer,
        uint32 size)
{
    uint32 num = 0;
    int32 next = 0;

    while (num < size)
    {
        next = read(fd, buffer + num, size - num);

        if (next <= 0)
            YnUtilError("read failed");

        num += next;
    }
}

void YnUtilFileWrite(int fd,
        char * buffer,
        uint32 size)
{
    uint32 num = 0;
    int32 next = 0;

    while(num < size)
    {
        next = write(fd, buffer + num, size - num);

        if (next <= 0)
            error("write failed");

        num += next;
    }
}

char * YnUtilStringCopy(char * str)
{
    char *copy = NULL;

    copy = malloc(strlen(str) + 1);

    strncpy(copy, str, strlen(str) + 1);

    return copy;
}

uint32 YnUtilLineFieldCount(char * line)
{
    uint32 count = 0;
    uint32 done = 0;
    char * c = NULL;

    for (c = line; !done; c ++)
    {
        done = (*c == '\0');

        if (*c == ',' || done)
            ++count;
    }

    return count;
}

float * YnUtilLineFieldParse(char * line,
        uint32 numField)
{
    float * field = NULL;
    char *c, *p, *end;
    int count = 0;
    int done = 0;

    field = calloc(numField, sizeof(float));

    for (c = line, p = line; !done; ++c)
    {
        done = (*c == '\0');

        if (*c == ',' || done)
        {
            *c = '\0';
            field[count] = strtod(p, &end);

            if (p == c)
                field[count] = nan("");

            if (end != c && (end != c - 1 || *end != '\r'))
                field[count] = nan(""); //DOS file formats!

            p = c + 1;

            ++count;
        }
    }
    return field;
}

float YnUtilConstrain(float min,
        float max,
        float val)
{
    if (val < min)
        return min;

    if (val > max)
        return max;

    return val;
}


float YnUtilArraySum(char * array,
        uint32 numField)
{
    uint32 i = 0;
    float sum = 0;

    for (i = 0; i < numField; i ++)
        sum += array[i];

    return sum;
}

float YnUtilArrayMean(char * array,
        uint32 numField)
{
    return YnUtilArraySum(array, numField)/numField;
}


float YnUtilArrayVariance(float * array,
        uint32 num)
{
    uint i = 0;
    float sum = 0;
    float mean = 0;
    float variance = 0;

    mean = YnUtilArrayMean(array, num);

    for (i = 0; i < num; i ++)
    {
        sum += (array[i] - mean) * (array[i] - mean);
    }

    variance = sum / num;

    return variance;
}

float YnUtilArrayMse(float * arrayErr,
        uint32 num)
{
    uint i = 0;
    float sum = 0;

    for (i = 0; i < num; ++i)
        sum += arrayErr[i] * arrayErr[i];

    return sqrt(sum / num);
}

void YnUtilArrayNormalize(float * array,
        uint32 num)
{
    uint32 i = 0;
    float mu = 0;
    float sigma = 0;

    mu = YnUtilArrayMean(array, num);
    sigma = sqrt(YnUtilArrayVariance(array, num));

    for (i = 0; i < num; i ++)
    {
        array[i] = (array[i] - mu) / sigma;
    }

    mu = YnUtilArrayMean(array, num);
    sigma = sqrt(YnUtilArrayVariance(array, num));
}

float YnUtilArrayMag(float * arrayErr,
        uint32 num)
{
    uint32 i = 0;
    float sum = 0;

    for (i = 0; i < num; ++i)
    {
        sum += arrayErr[i] * arrayErr[i];
    }

    return sqrt(sum);
}

void YnUtilArrayScale(float * array,
        uint32 num,
        float scale)
{
    uint32 i = 0;

    for (i = 0; i < num; ++i)
    {
        array[i] *= scale;
    }
}

float YnUtilArrayMaxIndex(float * array,
        uint32 num)
{
    int i, max_i = 0;
    float max = array[0];

    if (num <= 0)
        return -1;

    for (i = 1; i < num; ++i)
    {
        if (array[i] > max)
        {
            max = array[i];
            max_i = i;
        }
    }

    return max_i;
}

float YnUtilRandomNormalNum()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    if (haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if (rand1 < 1e-100)
        rand1 = 1e-100;

    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * PI2;

    return sqrt(rand1) * cos(rand2);
}

float YnUtilRandomUniformNum(float min, float max)
{
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}
