//	File        :   YnData.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   27-07-2016
//	Author      :   haittt

#include "../include/YnData.h"

/**************** Define */
#define NUMCHARS 37

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */
static uint32 seed;

/**************** Local Implement */

/**************** Implement */
int YndataSeedGet(void)
{
    return seed;
}

void YndataSeedSet(int seedVal)
{
    seed = seedVal;
}

tYnList * YnDataPathsGet(char *filename)
{
    char *path;
    tYnList * lines;
    FILE *file = fopen(filename, "r");

    if (!file)
        YnUtilErrorOpenFile(filename);

    lines = YnListMake();
    while((path = fgetl(file)))
    {
        YnLisyInsert(lines, path);
    }

    fclose(file);
    return lines;
}

char ** YnDataRandomPathsGet(char ** paths,
        int n,
        int m)
{
    int index;
    char **random_paths = calloc(n, sizeof(char *));
    int i;

    for (i = 0; i < n; i ++)
    {
        index = rand_r(&seed)%m;
        random_paths[i] = paths[index];

        if (i == 0)
            printf("%s\n", paths[index]);
    }

    return random_paths;
}

char ** YnDataFindReplacePaths(char **paths,
        int n,
        char *find,
        char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;

    for (i = 0; i < n; i ++)
    {
        char *replaced = YnUtilReplaceChar(paths[i], find, replace);
        replace_paths[i] = YnUtilStringCopy(replaced);
    }
    return replace_paths;
}

tYnMatrix YnDataLoadImagePathsGray(char **paths,
        int n,
        int w,
        int h)
{
    int i;
    tYnMatrix X;
    tYnImage image;
    tYnImage gray;

    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for (i = 0; i < n; i ++)
    {
        image = YnImageLoad(paths[i], w, h, 3);
        gray = YnImageGrayscale(image);
        YnImageFree(image);
        image = gray;

        X.vals[i] = image.data;
        X.cols = image.height * image.width * image.channel;
    }
    return X;
}

tYnMatrix YnDataLoadImagePaths(char **paths,
        int n,
        int w,
        int h)
{
    int i;
    tYnMatrix X;
    tYnImage image;

    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float *));
    X.cols = 0;

    for (i = 0; i < n; i ++)
    {
        image = YnImageLoadColor(paths[i], w, h);
        X.vals[i] = image.data;
        X.cols = image.height * image.width * image.channel;
    }

    return X;
}

tYnDataBoxLabel * YnDataReadBoxes(char *filename,
        int *n)
{
    float x, y, h, w;
    int id;
    int count = 0;
    tYnDataBoxLabel * boxes = calloc(1, sizeof(tYnDataBoxLabel));
    FILE *file = fopen(filename, "r");

    if (!file)
        YnUtilErrorOpenFile(filename);

    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
    {
        boxes = realloc(boxes, (count + 1) * sizeof(tYnDataBoxLabel));
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;

        ++count;
    }

    fclose(file);
    *n = count;

    return boxes;
}

void YnDataRandomizeBoxes(tYnDataBoxLabel *boxes,
        int n)
{
    int i;
    int index;

    for (i = 0; i < n; i ++)
    {
        tYnDataBoxLabel swap = boxes[i];
        index = rand_r(&seed)%n;
        boxes[i] = boxes[index];
        boxes[index] = swap;
    }
}

void YnDataCorrectBoxes(tYnDataBoxLabel *boxes,
        int n,
        float dx,
        float dy,
        float sx,
        float sy,
        int flip)
{
    int i;
    float swap;

    for (i = 0; i < n; i ++)
    {
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if (flip)
        {
            swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void YnDataFillTruthSwag(char *path,
        float *truth,
        int classes,
        int flip,
        float dx,
        float dy,
        float sx,
        float sy)
{
    float x,y,w,h;
    int id;
    int i;
    int index;
    tYnDataBoxLabel * boxes;
    int count = 0;
    char *labelpath = YnUtilReplaceChar(path, "images", "labels");

    labelpath = YnUtilReplaceChar(labelpath, "JPEGImages", "labels");
    labelpath = YnUtilReplaceChar(labelpath, ".jpg", ".txt");
    labelpath = YnUtilReplaceChar(labelpath, ".JPG", ".txt");
    labelpath = YnUtilReplaceChar(labelpath, ".JPEG", ".txt");

    boxes = YnDataReadBoxes(labelpath, &count);
    YnDataRandomizeBoxes(boxes, count);
    YnDataCorrectBoxes(boxes, count, dx, dy, sx, sy, flip);

    for (i = 0; i < count && i < 30; i ++)
    {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0)
            continue;

        index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes)
            truth[index+id] = 1;
    }

    YnDataFree(boxes);
}

void YnDataFillTruthRegion(char *path,
        float *truth,
        int classes,
        int numBoxes,
        int flip,
        float dx,
        float dy,
        float sx,
        float sy)
{
    float x,y,w,h;
    int id;
    int i;
    int col;
    int row;
    int index;
    int count = 0;
    tYnDataBoxLabel * boxes;
    char *labelpath = YnUtilReplaceChar(path, "images", "labels");
    labelpath = YnUtilReplaceChar(labelpath, "JPEGImages", "labels");
    labelpath = YnUtilReplaceChar(labelpath, ".jpg", ".txt");
    labelpath = YnUtilReplaceChar(labelpath, ".JPG", ".txt");
    labelpath = YnUtilReplaceChar(labelpath, ".JPEG", ".txt");

    boxes = YnDataReadBoxes(labelpath, &count);
    YnDataRandomizeBoxes(boxes, count);
    YnDataCorrectBoxes(boxes, count, dx, dy, sx, sy, flip);

    for (i = 0; i < count; i ++)
    {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if ((w < .01) || (h < .01))
            continue;

        col = (int)(x * numBoxes);
        row = (int)(y * numBoxes);

        x = x * numBoxes - col;
        y = y * numBoxes - row;

        index = (col + (row * numBoxes)) * (5 + classes);
        if (truth[index])
            continue;

        truth[index ++] = 1;
        if (id < classes)
            truth[index+id] = 1;
        index += classes;

        truth[index ++] = x;
        truth[index ++] = y;
        truth[index ++] = w;
        truth[index ++] = h;
    }

    YnDataFree(boxes);
}

void YnDataFillTruthDetection(char *path,
        float *truth,
        int classes,
        int numBoxes,
        int flip,
        int background,
        float dx,
        float dy,
        float sx,
        float sy)
{
    tYnDataBoxLabel *boxes;
    int count = 0;
    float x,y,w,h;
    float left, top, right, bot;
    int id;
    int i;
    float swap;
    int col;
    int row;
    int index;
    char *labelpath = YnUtilReplaceChar(path, "JPEGImages", "labels");
    labelpath = YnUtilReplaceChar(labelpath, ".jpg", ".txt");
    labelpath = YnUtilReplaceChar(labelpath, ".JPEG", ".txt");


    boxes = YnDataReadBoxes(labelpath, &count);
    YnDataRandomizeBoxes(boxes, count);

    if (background)
    {
        for (i = 0; i < (numBoxes * numBoxes * (4 + classes + background)); i += (4 + classes + background))
        {
            truth[i] = 1;
        }
    }

    for (i = 0; i < count; i ++)
    {
        left  = boxes[i].left  * sx - dx;
        right = boxes[i].right * sx - dx;
        top   = boxes[i].top   * sy - dy;
        bot   = boxes[i].bottom* sy - dy;
        id = boxes[i].id;

        if (flip)
        {
            swap = left;
            left = 1. - right;
            right = 1. - swap;
        }

        left =  YnUtilConstrain(0, 1, left);
        right = YnUtilConstrain(0, 1, right);
        top =   YnUtilConstrain(0, 1, top);
        bot =   YnUtilConstrain(0, 1, bot);

        x = (left + right)/2;
        y = (top + bot)/2;
        w = (right - left);
        h = (bot - top);

        if (x <= 0 || x >= 1 || y <= 0 || y >= 1)
            continue;

        col = (int)(x*numBoxes);
        row = (int)(y*numBoxes);

        x = (x * numBoxes) - col;
        y = (y * numBoxes) - row;

        w = YnUtilConstrain(0, 1, w);
        h = YnUtilConstrain(0, 1, h);

        if (w < .01 || h < .01)
            continue;

        w = pow(w, 1./2.);
        h = pow(h, 1./2.);

        index = (col + row * numBoxes) * (4 + classes + background);
        if (truth[index + classes + background + 2])
            continue;

        if (background)
            truth[index++] = 0;
        truth[index + id] = 1;
        index += classes;

        truth[index ++] = x;
        truth[index ++] = y;
        truth[index ++] = w;
        truth[index ++] = h;
    }

    YnUtilFree(boxes);
}

void YnDataPrintLetters(float *pred,
        int n)
{
    int i;
    int index;

    for (i = 0; i < n; i ++)
    {
        index = YnUtilArrayMaxIndex(pred + (i * NUMCHARS), NUMCHARS);
        printf("%c", YnUtilIntToNumChar(index));
    }

    printf("\n");
}

void YnDataFillTruthCaptcha(char *path,
        int n,
        float *truth)
{
    int i;
    int index;
    char *begin = strrchr(path, '/');
    begin ++;

    for (i = 0; (i < strlen(begin)) && (i < n) && (begin[i] != '.'); i ++)
    {
        index = YnUtilNumCharToInt(begin[i]);
        if (index > 35)
            printf("Bad %c\n", begin[i]);

        truth[(i * NUMCHARS) + index] = 1;
    }

    for (;i < n; i ++)
    {
        truth[(i * NUMCHARS) + NUMCHARS - 1] = 1;
    }
}

tYnData YnDataLoadCaptcha(char **paths,
        int n,
        int m,
        int k,
        int w,
        int h)
{
    tYnData d;
    int i;

    if (m)
        paths = YnDataRandomPaths(paths, n, m);

    d.shallow = 0;
    d.x = YnDataLoadImagePaths(paths, n, w, h);
    d.y = YnMatrixMake(n, k*NUMCHARS);

    for (i = 0; i < n; i ++)
    {
        YnDataFillTruthCaptcha(paths[i], k, d.y.vals[i]);
    }

    if (m)
        YnUtilFree(paths);

    return d;
}

tYnData YnDataLoadCaptchaEncode(char **paths,
        int n,
        int m,
        int w,
        int h)
{
    tYnData d;

    if (m)
        paths = YnDataRandomPaths(paths, n, m);

    d.shallow = 0;
    d.x = YnDataLoadImagePaths(paths, n, w, h);
    d.x.cols = 17100;
    d.y = d.x;

    if (m)
        YnUtilFree(paths);

    return d;
}

void YnDataFillTruth(char *path,
        char **labels,
        int k,
        float *truth)
{
    int i;
    int count = 0;

    memset(truth, 0, k * sizeof(float));

    for (i = 0; i < k; i ++)
    {
        if (strstr(path, labels[i]))
        {
            truth[i] = 1;
            ++count;
        }
    }

    if (count != 1)
        printf("Too many or too few labels: %d, %s\n", count, path);
}

tYnMatrix YnDataLoadLabelsPaths(char **paths,
        int n,
        char **labels,
        int k)
{
    int i;
    tYnMatrix y = YnMatrixMake(n, k);

    for (i = 0; i < n && labels; i ++)
    {
        YnDataFillTruth(paths[i], labels, k, y.vals[i]);
    }

    return y;
}

char ** YnDataLabelsGet(char *filename)
{
    tYnList *plist = YnDataPathsGet(filename);
    char **labels = (char **)YnListToArr(plist);

    YnListFree(plist);

    return labels;
}

void YnDataFree(tYnData data)
{
    if (!data.shallow)
    {
        YnMatrixFree(data.x);
        YnMatrixFree(data.y);
    }
    else
    {
        YnUtilFree(data.x.vals);
        YnUtilFree(data.y.vals);
    }
}

tYnData YnDaraLoadRegion(int n,
        char **paths,
        int m,
        int w,
        int h,
        int size,
        int classes,
        float jitter)
{
    char **random_paths = YnDataRandomPathsGet(paths, n, m);
    int i;
    int k;
    int oh;
    int ow;

    int dw;
    int dh;

    int pleft;
    int pright;
    int ptop;
    int pbot;

    int swidth;
    int sheight;

    float sx;
    float sy;
    float dx;
    float dy;

    int flip;
    tYnData d;
    tYnImage orig;
    tYnImage cropped;
    tYnImage sized;

    d.shallow = 0;
    d.x.rows = n;
    d.x.vals = calloc(d.x.rows, sizeof(float*));
    d.x.cols = h * w * 3;

    k = size * size * (5 + classes);
    d.y = YnMatrixMake(n, k);

    for (i = 0; i < n; i ++)
    {
        orig = YnImageLoadColor(random_paths[i], 0, 0);

        oh = orig.height;
        ow = orig.width;

        dw = (ow * jitter);
        dh = (oh * jitter);

        pleft  = YnUtilRandUniform(-dw, dw);
        pright = YnUtilRandUniform(-dw, dw);
        ptop   = YnUtilRandUniform(-dh, dh);
        pbot   = YnUtilRandUniform(-dh, dh);

        swidth =  ow - pleft - pright;
        sheight = oh - ptop - pbot;

        sx = (float)swidth  / ow;
        sy = (float)sheight / oh;

        flip = rand_r(&seed) % 2;
        cropped = YnImageCrop(orig, pleft, ptop, swidth, sheight);

        dx = ((float)pleft / ow)/sx;
        dy = ((float)ptop / oh)/sy;

        sized = YnImageResize(cropped, w, h);
        if (flip)
            YnImageFlip(sized);

        d.x.vals[i] = sized.data;
        YnDataFillTruthRegion(random_paths[i], d.y.vals[i],
                classes, size, flip, dx, dy, 1./sx, 1./sy);
        YnImageFree(orig);
        YnImageFree(cropped);
    }

    YnUtilFree(random_paths);

    return d;
}

tYnData YnDataLoadCompare(int n,
        char **paths,
        int m,
        int classes,
        int w,
        int h)
{
    int i, j, k;
    int id;
    float iou;
    tYnData d;
    tYnImage im1;
    tYnImage im2;
    char *imlabel1;
    char *imlabel2;
    FILE *fp1;
    FILE *fp2;

    if (m)
        paths = YnDataRandomPathsGet(paths, 2 * n, m);

    d.shallow = 0;
    d.x.rows = n;
    d.x.vals = calloc(d.x.rows, sizeof(float*));
    d.x.cols = h * w * 6;

    k = 2 * classes;
    d.y = YnMatrixMake(n, k);

    for (i = 0; i < n; i ++)
    {
        im1 = YnImageLoadColor(paths[i * 2],   w, h);
        im2 = YnImageLoadColor(paths[i * 2 + 1], w, h);

        d.x.vals[i] = calloc(d.x.cols, sizeof(float));
        memcpy(d.x.vals[i],         im1.data, h * w * 3 * sizeof(float));
        memcpy(d.x.vals[i] + h * w * 3, im2.data, h * w * 3 * sizeof(float));

        imlabel1 = YnUtilReplaceChar(paths[i*2],   "imgs", "labels");
        imlabel1 = YnUtilReplaceChar(imlabel1, "jpg", "txt");
        fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2)
        {
            if (d.y.vals[i][2 * id] < iou)
                d.y.vals[i][2 * id] = iou;
        }

        imlabel2 = YnUtilReplaceChar(paths[i * 2 + 1], "imgs", "labels");
        imlabel2 = YnUtilReplaceChar(imlabel2, "jpg", "txt");
        fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2)
        {
            if (d.y.vals[i][2 * id + 1] < iou) d.y.vals[i][2 * id + 1] = iou;
        }

        for (j = 0; j < classes; j ++)
        {
            if (d.y.vals[i][2 * j] > .5 &&  d.y.vals[i][2 * j + 1] < .5)
            {
                d.y.vals[i][2 * j] = 1;
                d.y.vals[i][2 * j + 1] = 0;
            }
            else if (d.y.vals[i][2 * j] < .5 &&  d.y.vals[i][2 * j + 1] > .5)
            {
                d.y.vals[i][2 * j] = 0;
                d.y.vals[i][2 * j + 1] = 1;
            }
            else
            {
                d.y.vals[i][2 * j] = YN_CUS_NUM;
                d.y.vals[i][2 * j + 1] = YN_CUS_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        YnImageFree(im1);
        YnImageFree(im2);
    }

    if (m)
        YnUtilFree(paths);

    return d;
}

tYnData YnDataLoadSwag(char **paths,
        int n,
        int classes,
        float jitter)
{
    int h;
    int w;
    int k;
    int dw;
    int dh;
    int pleft;
    int pright;
    int ptop;
    int pbot;
    int swidth;
    int sheight;
    float sx;
    float sy;
    int flip;
    tYnData d;
    int index = rand_r(&seed)%n;
    char *random_path = paths[index];
    tYnImage cropped;
    tYnImage orig = YnImageLoadColor(random_path, 0, 0);

    h = orig.height;
    w = orig.width;

    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.x.rows = 1;
    d.x.vals = calloc(d.x.rows, sizeof(float*));
    d.x.cols = h * w * 3;

    k = (4 + classes) * 30;
    d.y = YnMatrixMake(1, k);

    dw = w * jitter;
    dh = h * jitter;

    pleft  = rand_uniform(-dw, dw);
    pright = rand_uniform(-dw, dw);
    ptop   = rand_uniform(-dh, dh);
    pbot   = rand_uniform(-dh, dh);

    swidth =  w - pleft - pright;
    sheight = h - ptop - pbot;

    sx = (float)swidth  / w;
    sy = (float)sheight / h;

    flip = rand_r(&seed)%2;
    cropped = YnImageCrop(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    tYnImage sized = YnImageResize(cropped, w, h);
    if (flip)
        YnImageFlip(sized);

    d.x.vals[0] = sized.data;
    YnDataFillTruthSwag(random_path, d.y.vals[0],
            classes, flip, dx, dy, 1./sx, 1./sy);

    YnImageFree(orig);
    YnImageFree(cropped);

    return d;
}

tYnData YnDataLoadDetection(int n,
        char **paths,
        int m,
        int classes,
        int w,
        int h,
        int numBoxes,
        int background)
{
    char **random_paths = YnDataRandomPathsGet(paths, n, m);
    int i;
    tYnData d;
    tYnImage orig;
    tYnImage cropped;
    tYnImage sized;
    int k;
    int oh;
    int ow;
    int dw;
    int dh;
    int pleft;
    int pright;
    int ptop;
    int pbot;
    int swidth;
    int sheight;
    int flip;
    float sx;
    float sy;
    float dx;
    float dy;

    d.shallow = 0;
    d.x.rows = n;
    d.x.vals = calloc(d.x.rows, sizeof(float *));
    d.x.cols = h * w * 3;

    k = numBoxes*numBoxes*(4+classes+background);
    d.y = YnMatrixMake(n, k);

    for (i = 0; i < n; i ++)
    {
        orig = YnImageLoadColor(random_paths[i], 0, 0);

        oh = orig.height;
        ow = orig.width;
        dw = ow/10;
        dh = oh/10;
        pleft  = YnUtilRandomUniformNum(-dw, dw);
        pright = YnUtilRandomUniformNum(-dw, dw);
        ptop   = YnUtilRandomUniformNum(-dh, dh);
        pbot   = YnUtilRandomUniformNum(-dh, dh);

        swidth =  ow - pleft - pright;
        sheight = oh - ptop - pbot;

        sx = (float)swidth  / ow;
        sy = (float)sheight / oh;

        flip = rand_r(&seed)%2;
        cropped = YnImageCrop(orig, pleft, ptop, swidth, sheight);

        dx = ((float)pleft/ow)/sx;
        dy = ((float)ptop /oh)/sy;

        sized = YnIMageResize(cropped, w, h);
        if (flip)
            YnImageFlip(sized);

        d.x.vals[i] = sized.data;

        fill_truth_detection(random_paths[i], d.y.vals[i], classes, numBoxes, flip, background, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

void * YnDataLoadThread(void *ptr)
{

#ifdef YN_GPU
    cudaError_t status = cudaSetDevice(gpu_index);
    YnCudaCheckError(status);
#endif

    tYnDataLoadArgs a = *(struct load_args*)ptr;
    switch (a.type)
    {
    case cYnDataClassification:
        *a.d = YnDataLoad(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
        break;
    case cYnDataDetection:
        *a.d = YnDataLoadDetection(a.n, a.paths, a.m, a.classes, a.w, a.h, a.numBoxes, a.background);
        break;
    case cYnDataRegion:
        *a.d = YnDataLoadRegion(a.n, a.paths, a.m, a.w, a.h, a.numBoxes, a.classes, a.jitter);
        break;
    case cYnDataImage:
        *(a.im) = YnImageLoadColor(a.path, 0, 0);
        *(a.resized) = YnImageResize(*(a.im), a.w, a.h);
        break;
    case cYnDataCompare:
        *a.d = YnImageLoadCompare(a.n, a.paths, a.m, a.classes, a.w, a.h);
        break;
    case cYnDataWriting:
        *a.d = YnImageLoadWriting(a.paths, a.n, a.m, a.w, a.h, a.outW, a.outH);
        break;
    case cYnDataSwag:
        *a.d = YnImageLoadSwag(a.paths, a.n, a.classes, a.jitter);
        break;
    default:
        break;
    }

    YnUtilFree(ptr);
    return 0;
}

pthread_t YnDataLoadInThread(tYnDataLoadArgs args)
{
    pthread_t thread;
    struct tYnDataLoadArgs *ptr = calloc(1, sizeof(struct tYnDataLoadArgs));

    *ptr = args;
    if (pthread_create(&thread, 0, YnDataLoadThread, ptr))
        error("Thread creation failed");

    return thread;
}

tYnData YnDAtaLoadWriting(char **paths,
        int n,
        int m,
        int w,
        int h,
        int out_w,
        int out_h)
{
    int i;
    char **replace_paths = YnUtilReplaceChar(paths, n, ".png", "-label.png");
    tYnData d;

    if (m)
        paths = YnDAteRandomPathsGet(paths, n, m);

    d.shallow = 0;
    d.x = YnDataLoadImagePaths(paths, n, w, h);
    d.y = YnDataLoadImagePathsGray(replace_paths, n, out_w, out_h);

    if (m)
        free(paths);

    for (i = 0; i < n; i ++)
        YnUtilFree(replace_paths[i]);

    YnUtilFree(replace_paths);
    return d;
}

tYnData YnDataLoad(char **paths,
        int n,
        int m,
        char **labels,
        int k,
        int w,
        int h)
{
    tYnData d;

    if (m)
        paths = YnDAteRandomPathsGet(paths, n, m);

    d.shallow = 0;
    d.x = YnDataLoadImagePaths(paths, n, w, h);
    d.y = YnDataLoadLabelsPaths(paths, n, labels, k);

    if (m)
        YnUtilFree(paths);

    return d;
}

tYnMatrix YnDataConcatMatrix(tYnMatrix m1,
        tYnMatrix m2)
{
    int i, count = 0;
    tYnMatrix m;

    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float *));

    for (i = 0; i < m1.rows; i ++)
    {
        m.vals[count++] = m1.vals[i];
    }

    for (i = 0; i < m2.rows; i ++)
    {
        m.vals[count++] = m2.vals[i];
    }

    return m;
}

tYnData YnDataConcat(tYnData d1,
        tYnData d2)
{
    tYnData d;

    d.shallow = 1;
    d.x = YnDataConcatMatrix(d1.x, d2.x);
    d.y = YnDataConcatMatrix(d1.y, d2.y);

    return d;
}

tYnData YnDataLoadCategoricalCsv(char *filename,
        int target,
        int k)
{
    tYnData d;
    tYnMatrix X;
    tYnMatrix y;
    float *truth_1d;
    float **truth;

    d.shallow = 0;
    X = YnDataCsvToMatrix(filename);
    truth_1d = YnDataPopColumn(&X, target);
    truth = YnDataOneHotEncode(truth_1d, X.rows, k);

    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.x = X;
    d.y = y;

    YnUtilFree(truth_1d);

    return d;
}

tYnData YnDataLoadCifar10_data(char *filename)
{
    int class;
    long i,j;
    tYnData d;
    tYnMatrix X;
    tYnMatrix y;
    unsigned char bytes[3073];
    FILE *fp = fopen(filename, "rb");

    X = YnMatrixMake(10000, 3072);
    y = YnMatrixMake(10000, 10);
    d.shallow = 0;
    d.x = X;
    d.y = y;

    if (!fp)
        file_error(filename);

    for (i = 0; i < 10000; i ++)

        fread(bytes, 1, 3073, fp);
        class = bytes[0];
        y.vals[i][class] = 1;

        for (j = 0; j < X.cols; j ++)
        {
            X.vals[i][j] = (double)bytes[j + 1];
        }
    }

    YnDataTranslateRows(d, -128);
    YnDataScaleRows(d, 1./128);

    fclose(fp);
    return d;
}

void YnDataRandomBatchGet(tYnData d,
        int n,
        float * X,
        float * y)
{
    int j;
    int index;

    for (j = 0; j < n; ++j)
    {
        index = rand_r(&seed) % d.x.rows;
        memcpy(X + j * d.x.cols, d.x.vals[index], d.x.cols * sizeof(float));
        memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float));
    }
}

void YnDataNextBatchGet(tYnData d,
        int n,
        int offset,
        float *X,
        float *y)
{
    int j;
    int index;

    for (j = 0; j < n; j ++)
    {
        index = offset + j;
        memcpy(X + j * d.x.cols, d.x.vals[index], d.x.cols*sizeof(float));
        memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}


tYnData YnDataLoadAllCifar10()
{
    int i, j, b;
    int class;
    tYnData d;
    tYnMatrix X;
    tYnMatrix y;
    char buff[256];
    unsigned char bytes[3073];
    FILE *fp;

    X = YnMatrixMake(50000, 3072);
    y = YnMatrixMake(50000, 10);

    d.shallow = 0;
    d.x = X;
    d.y = y;

    for (b = 0; b < 5; b ++)
    {
        sprintf(buff, "data/cifar10/data_batch_%d.bin", b + 1);
        fp = fopen(buff, "rb");

        if (!fp)
            file_error(buff);

        for (i = 0; i < 10000; i ++)
        {
            fread(bytes, 1, 3073, fp);
            class = bytes[0];
            y.vals[i + b * 10000][class] = 1;

            for (j = 0; j < X.cols; j ++)
            {
                X.vals[i + b * 10000][j] = (double)bytes[j + 1];
            }
        }

        fclose(fp);
    }

    YnDataTranslateRows(d, -128);
    YnDataScaleRows(d, 1./128);

    return d;
}

void YnDataRandomize(tYnData d)
{
    int index;
    int i;
    float *swap;

    for (i = d.x.rows - 1; i > 0; i --)
    {
        index = rand_r(&seed) % i;

        swap = d.x.vals[index];
        d.x.vals[index] = d.x.vals[i];
        d.x.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void YnDataScaleRows(tYnData d,
        float s)
{
    int i;

    for (i = 0; i < d.x.rows; i ++)
    {
        YnUtilArrayScale(d.x.vals[i], d.x.cols, s);
    }
}

void YnDataTranslateRows(tYnData d, \
        float s)
{
    int i;

    for (i = 0; i < d.x.rows; i ++)
    {
        YnUtilArrayTranslate(d.x.vals[i], d.x.cols, s);
    }
}

void YnDataNormalizeRows(tYnData d)
{
    int i;

    for (i = 0; i < d.x.rows; i ++)
    {
        YnUtilArrayNormalize(d.x.vals[i], d.x.cols);
    }
}

tYnData * YnDataSplit(tYnData d,
        int part,
        int total)
{
    int i;
    tYnData train;
    tYnData test;
    tYnData *split = calloc(2, sizeof(tYnData));
    int start = part * d.x.rows / total;
    int end = (part + 1) * d.x.rows / total;

    train.shallow = test.shallow = 1;
    test.x.rows = test.y.rows = end - start;
    train.x.rows = train.y.rows = d.x.rows - (end - start);
    train.x.cols = test.x.cols = d.x.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.x.vals = calloc(train.x.rows, sizeof(float *));
    test.x.vals = calloc(test.x.rows, sizeof(float *));
    train.y.vals = calloc(train.y.rows, sizeof(float *));
    test.y.vals = calloc(test.y.rows, sizeof(float *));

    for (i = 0; i < start; i ++)
    {
        train.x.vals[i] = d.x.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }

    for (i = start; i < end; i ++)
    {
        test.x.vals[i - start] = d.x.vals[i];
        test.y.vals[i - start] = d.y.vals[i];
    }

    for (i = end; i < d.x.rows; i ++)
    {
        train.x.vals[i - (end - start)] = d.x.vals[i];
        train.y.vals[i - (end - start)] = d.y.vals[i];
    }

    split[0] = train;
    split[1] = test;

    return split;
}
