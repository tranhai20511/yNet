//	File        :   YnVoc.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   31-08-2016
//	Author      :   haittt

#include "../include/YnVoc.h"

#ifdef YN_OPENCV
#include "opencv2/highgui/highgui_c.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */
static float **probs;
static tYnBBox *boxes;
static tYnNetwork net;
static tYnImage in   ;
static tYnImage in_s ;
static tYnImage det  ;
static tYnImage det_s;
static tYnImage disp ;
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;

/**************** Global variables */
char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
tYnImage voc_labels[YN_VOC_NUMCLASS];

/**************** Local Implement */
YN_STATIC
void * _YnVocThreadFetch(void *ptr)
YN_ALSWAY_INLINE;

YN_STATIC
void * _YnVocThreadDetect(void *ptr)
YN_ALSWAY_INLINE;

/**************** Implement */
void YnVocTrain(char *cfgfile,
        char *weightfile)
{
    int imgs;
    int i;
    char *base;
    tYnData train, buffer;
    tYnLayer layer;
    int side;
    int classes;
    float jitter;
    tYnList *plist;
    char **paths;
    tYnNetwork net;
    pthread_t load_thread;
    clock_t time;
    float loss;
    char buff[256];

    char *train_images = "/data/voc/train.txt";
    char *backup_directory = "/home/backup/";
    float avg_loss = -1;
    tYnDataLoadArgs args = {0};

    srand(time(0));
    YndataSeedSet(time(0));

    base = YnUtilFindBaseConfig(cfgfile);
    printf("%s\n", base);

    net = YnParserNetworkCfg(cfgfile);

    if(weightfile)
        YnParserWeightsLoad(&net, weightfile);

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learningRate, net.momentum, net.decay);

    imgs = net.batch*net.subdivisions;
    i = *net.seen/imgs;

    layer = net.layers[net.n - 1];

    side = layer.side;
    classes = layer.classes;
    jitter = layer.jitter;

    plist = YnDataPathsGet(train_images);
    paths = (char **)YnListToArr(plist);

    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.numBoxes = side;
    args.d = &buffer;
    args.type = cYnDataRegion;

    load_thread = YnDataLoadInThread(args);
    while (YnNetworkGetCurrentBatch(net) < net.max_batches)
    {
        i += 1;
        time = clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = YnDataLoadInThread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time = clock();

        loss = YnNetworkTrain(net, train);
        if (avg_loss < 0)
            avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n",
                i,
                loss,
                avg_loss,
                YnNetworkCurrentRateget(net),
                sec(clock() - time),
                i * imgs);

        if(i % 1000 == 0 || i == 600)
        {
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            YnParserWeightsSave(net, buff);
        }

        YnDataFree(train);
    }

    sprintf(buff, "%s/%s_finalayer.weights", backup_directory, base);
    YnParserWeightsSave(net, buff);
}

void YnVocConvertDetections(float *predictions,
        int classes,
        int num,
        int square,
        int side,
        int w,
        int h,
        float thresh,
        float **probs,
        tYnBBox *boxes,
        int only_objectness)
{
    int i, j, n;
    int row;
    int col;
    int index;
    int p_index;
    float scale;
    int box_index;
    int class_index;
    float prob;

    for (i = 0; i < side * side; i ++)
    {
        row = i / side;
        col = i % side;

        for (n = 0; n < num; n ++)
        {
            index = i * num + n;
            p_index = side * side * classes + i * num + n;
            scale = predictions[p_index];
            box_index = side * side * (classes + num) + (i * num + n) * 4;

            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].width = pow(predictions[box_index + 2], (square ? 2 : 1)) * w;
            boxes[index].height = pow(predictions[box_index + 3], (square ? 2 : 1)) * h;

            for (j = 0; j < classes; j ++)
            {
                class_index = i * classes;
                prob = scale * predictions[class_index + j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }

            if(only_objectness)
                probs[index][0] = scale;
        }
    }
}

void YnVocTest(char *cfgfile,
        char *weightfile,
        char *filename,
        float thresh)
{
    clock_t time;
    char buff[256];
    int j;
    char *input;
    float nms;
    tYnLayer layer;
    tYnBBox *boxes;
    float **probs;
    tYnImage im;
    tYnImage sized;
    float *X;
    float *predictions;

    tYnNetwork net = YnParserNetworkCfg(cfgfile);
    if(weightfile)
        YnParserWeightsLoad(&net, weightfile);

    layer = net.layers[net.n - 1];
    YnNetworkBatchSet(&net, 1);
    srand(YN_RANDPSEUDO);

    input = buff;
    nms = .5;
    boxes = calloc(layer.side * layer.side * layer.n, sizeof(tYnBBox));

    probs = calloc(layer.side * layer.side * layer.n, sizeof(float *));

    for(j = 0; j < layer.side * layer.side * layer.n; j ++)
        probs[j] = calloc(layer.classes, sizeof(float *));

    while (1)
    {
        if(filename)
        {
            strncpy(input, filename, 256);
        }
        else
        {
            printf("Enter Image Path: ");
            fflush(stdout);

            input = fgets(input, 256, stdin);
            if (!input)
                return;
            strtok(input, "\n");
        }

        im = YnImageLoadColor(input, 0, 0);
        sized = YnImageResize(im, net.w, net.h);
        X = sized.data;
        time = clock();
        predictions = YnNetworkPredict(net, X);

        printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));

        YnConvertVocDetections(predictions, layer.classes, layer.n, layer.sqrt, layer.side, 1, 1, thresh, probs, boxes, 0);
        if (nms)
            YnBBoxNmsSort(boxes, probs, layer.side * layer.side * layer.n, layer.classes, nms);

        YnImageDrawDetections(im, layer.side * layer.side * layer.n, thresh, boxes, probs, voc_names, 0, YN_VOC_NUMCLASS);

        YnImageShow(im, "predictions");
        YnImageShow(sized, "resized");

        YnImageFree(im);
        YnImageFree(sized);

        cvWaitKey(0);
        cvDestroyAllWindows();

        if (filename)
            break;
    }
}

YN_STATIC
void * _YnVocThreadFetch(void *ptr)
{
    in = YnImageFromStreamGet(cap);
    in_s = YnImageResize(in, net.w, net.h);
    return 0;
}

YN_STATIC
void *_YnVocThreadDetect(void *ptr)
{
    float nms = .4;
    tYnLayer layer = net.layers[net.n - 1];
    float *X = det_s.data;
    float *predictions = YnNetworkPredict(net, X);
    YnImageFree(det_s);
    YnConvertVocDetections(predictions, layer.classes, layer.n, layer.sqrt, layer.side, 1, 1, demo_thresh, probs, boxes, 0);

    if (nms > 0)
        YnBBoxNmsSort(boxes, probs, layer.side * layer.side * layer.n, layer.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.0f\n",fps);
    printf("Objects:\n\n");

    YnImageDrawDetections(det, layer.side * layer.side * layer.n, demo_thresh, boxes, probs, voc_names, voc_labels, YN_VOC_NUMCLASS);
    return 0;
}

void YnVocCpuDemo(char *cfgfile,
        char *weightfile,
        float thresh,
        int cam_index,
        char *filename)
{
    int j;
    float curr;
    struct timeval tval_before, tval_after, tval_result;
    tYnLayer layer;
    pthread_t fetch_thread;
    pthread_t detect_thread;

    demo_thresh = thresh;
    printf("VOC demo\n");
    net = YnParserNetworkCfg(cfgfile);

    if (weightfile)
        YnParserWeightsLoad(&net, weightfile);

    YnNetworkBatchSet(&net, 1);
    srand(YN_RANDPSEUDO);

    if(filename)
        cap = cvCaptureFromFile(filename);
    else
        cap = cvCaptureFromCAM(cam_index);

    if(!cap)
        YnUtilError("Couldn't connect to webcam.\n");

    cvNamedWindow("VOC", CV_WINDOW_NORMAL);
    cvResizeWindow("VOC", 512, 512);

    layer = net.layers[net.n - 1];
    boxes = (tYnBBox *)calloc(layer.side * layer.side * layer.n, sizeof(tYnBBox));
    probs = (float **)calloc(layer.side * layer.side * layer.n, sizeof(float *));

    for(j = 0; j < layer.side * layer.side * layer.n; j ++)
        probs[j] = (float *)calloc(layer.classes, sizeof(float *));

    _YnVocThreadFetch(0);
    det = in;
    det_s = in_s;

    _YnVocThreadFetch(0);
    _YnVocThreadDetect(0);
    disp = det;
    det = in;
    det_s = in_s;

    while(1)
    {
        gettimeofday(&tval_before, NULL);

        if (pthread_create(&fetch_thread, 0, _YnVocThreadFetch, 0))
            YnUtilError("Thread creation failed");

        if (pthread_create(&detect_thread, 0, _YnVocThreadDetect, 0))
            YnUtilError("Thread creation failed");

        YnImageShow(disp, "VOC");
        YnImageFree(disp);

        cvWaitKey(1);
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);

        disp  = det;
        det   = in;
        det_s = in_s;

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        curr = 1000000.f / ((long int)tval_result.tv_usec);
        fps = .9 * fps + .1*curr;
    }
}

void YnVocDemo(char *cfgfile,
        char *weightfile,
        float thresh,
        int cam_index,
        const char *filename)
{
#ifdef YN_GPU
    YnVocGpuDemo(cfgfile, weightfile, thresh, cam_index, filename);
#else
    YnVocCpuDemo(cfgfile, weightfile, thresh, cam_index, filename);
#endif
}

void YnVocRun(int argc,
        char **argv)
{
    int i;
    char buff[256];
    float thresh;
    int cam_index;
    char *cfg;
    char *weights;
    char *filename;

    for(i = 0; i < YN_VOC_NUMCLASS; i ++)
    {
        sprintf(buff, "data/labels/%s.png", voc_names[i]);
        voc_labels[i] = YnImageLoadColor(buff, 0, 0);
    }

    thresh = YnUtilFindFloatArg(argc, argv, "-thresh", .2);
    cam_index = YnUtilFindIntArg(argc, argv, "-c", 0);

    if(argc < 4)
    {
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    cfg = argv[3];
    weights = (argc > 4) ? argv[4] : 0;
    filename = (argc > 5) ? argv[5] : 0;

    if (0 == strcmp(argv[2], "test"))
        YnVocTest(cfg, weights, filename, thresh);
    else if (0 == strcmp(argv[2], "train"))
        YnVocTrain(cfg, weights);
    else if (0 == strcmp(argv[2], "demo"))
        YnVocDemo(cfg, weights, thresh, cam_index, filename);
}

#endif
