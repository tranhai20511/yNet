//	File        :   YnVocGpu.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   31-08-2016
//	Author      :   haittt

extern "C" {
#include "../include/YnVoc.h"
}

#ifdef YN_OPENCV
extern "C" {
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
}
#endif

/**************** Define */
//#define USE_CAMERA

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
static cv::VideoCapture cap;
static float fps = 0;
static float demo_thresh = 0;

static uchar *vdisp = NULL;
static CvMat *tmp = NULL;
static CvMat *s = NULL;

static uchar *vdisp2 = NULL;
static CvMat *s2 = NULL;
static CvMat *tmp2 = NULL;

static cv::Mat im_gray;
static CvCapture* cv_cap;

#define MAX_BOX     (49*2)

static tYnBBoxSend* boxesShare = NULL;
static unsigned char * numBoxSend;
static int isDone = 0;
static int processing = 0;

/**************** Global variables */
YN_EXTERN_C char *voc_names[];

/**************** Local Implement */
YN_STATIC_INLINE
int _YnVocSizeofCvMat(CvMat *mat)
YN_ALSWAY_INLINE;

/**************** Implement */
YN_STATIC_INLINE
int _YnVocSizeofCvMat(CvMat *mat)
{
    return mat->rows * mat->step;
}

void * _YnVocThreadFetch(void *ptr)
{

    //cvWaitKey(20);

#ifdef USE_CAMERA
    IplImage* color_img;
    color_img = cvQueryFrame(cv_cap);
    cvWaitKey(1);
    in = YnImageCvIplToimage(color_img);

    while (*vdisp == 1)
    {
        cvWaitKey(1);
    }

    /*CvSize imageSize = cvSize(640, 480);
    IplImage *gray = cvCreateImage(imageSize, 8, 1);
    CvMat stub;

    // Convert color frame to grayscale
    cvCvtColor(color_img, gray, CV_BGR2GRAY);

    // Get matrix from the gray frame and write the matrix in shared memory
    tmp2 = cvGetMat(gray, &stub, 0, 0);*/

    CvMat stub;
    tmp2 = cvGetMat(color_img, &stub, 0, 0);

    for (int row = 0; row < tmp2->rows; row++)
    {
        const uchar* ptr = (const uchar*) (tmp2->data.ptr + row * tmp2->step);
        memcpy((uchar*)(s2->data.ptr + row * s2->step), ptr, tmp2->step);
    }

    //cvReleaseImage(&gray);



#else
    while (*vdisp == 1)
    {
        cvWaitKey(1);
    }

    cv::Mat frame_m;
    cap >> frame_m;
    IplImage frame = frame_m;
    in = YnImageCvIplToimage(&frame);
#endif


    YnImageRgbgr(in);
    in_s = YnImageResize(in, net.w, net.h);

    return 0;
}

void *_YnVocThreadDetect(void *ptr)
{
    float nms = .4;

    tYnLayer layer = net.layers[net.n-1];
    float *X = det_s.data;
    float *predictions = YnNetworkPredict(net, X);

    YnImageFree(det_s);

    YnVocConvertDetections(predictions, layer.classes, layer.n, layer.sqrt, layer.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0)
        YnBBoxNmsSort(boxes, probs, layer.side * layer.side * layer.n, layer.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.0f\n",fps);
    printf("Objects:\n\n");

    YnImageDrawDetections1(det, layer.side * layer.side * layer.n, demo_thresh, boxes, probs, voc_names, NULL, YN_VOC_NUMCLASS, boxesShare, numBoxSend);

    *vdisp = 1;
    return 0;
}

YN_EXTERN_C
void YnVocGpuDemo(char *cfgfile,
        char *weightfile,
        float thresh,
        int cam_index,
        char *filename)
{
    int shmid;
    int j;
    float curr;
    tYnLayer layer;
    key_t key = 1234123;
    pthread_t sync_thread;
    pthread_t fetch_thread;
    pthread_t detect_thread;
    struct timeval tval_before, tval_after, tval_result;

    const size_t vdispsize = (1 + 1 + (MAX_BOX * sizeof(tYnBBoxSend)));

    /* Create the segment */
    if ((shmid = shmget(key, vdispsize + 1, IPC_CREAT | 0666)) < 0)
    {
        perror("shmget");
        exit(1);
    }

    /* Attach the segment to our data space */
    if ((vdisp = (uchar *) shmat(shmid, NULL, 0)) == (uchar *) -1)
    {
        perror("shmat");
        exit(1);
    }

#ifdef USE_CAMERA
    int shmid2;
    key_t key2 = 12341234;

    s2 = cvCreateMat(480, 640, CV_8UC3);
    tmp2 = cvCreateMat(480, 640, CV_8UC3);
    const size_t vdispsize2 = sizeofmat(s2);

    /* Create the segment */
    if ((shmid2 = shmget(key2, vdispsize2, IPC_CREAT | 0666)) < 0)
    {
        perror("shmget");
        exit(1);
    }

    /* Attach the segment to our data space */
    if ((vdisp2 = (uchar *) shmat(shmid2, NULL, 0)) == (uchar *) -1)
    {
        perror("shmat");
        exit(1);
    }

    s2->data.ptr = vdisp2;
#endif

    boxesShare = (tYnBBoxSend*)(vdisp + 2);
    numBoxSend = (unsigned char *)(vdisp + 1);
    *(vdisp + 1) = 0;
    *vdisp = 0;

    printf("Starting VOC kernel \n");
    cvWaitKey(1000);
    demo_thresh = thresh;
    printf("VOC demo\n");

    net = YnParserNetworkCfg(cfgfile);
    if(weightfile)
        YnParserWeightsLoad(&net, weightfile);

    YnNetworkBatchSet(&net, 1);

    srand(YN_RANDPSEUDO);

#ifdef USE_CAMERA
    cv_cap = cvCaptureFromCAM(0);
    cvWaitKey(1000);

    //cap.open(cam_index);
    //if(!cap.isOpened()) error("Couldn't connect to webcam.\n");
#else
    cap.open(filename);
    if(!cap.isOpened())
        YnUtilError("Couldn't open video.\n");
#endif

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
/*        if(pthread_create(&fetch_thread, 0, _YnVocThreadFetch, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, _YnVocThreadDetect, 0)) error("Thread creation failed");*/
        show_image(disp, "CNN");
        free_image(disp);
        cvWaitKey(1);
        _YnVocThreadFetch(0);
        _YnVocThreadDetect(0);
        /*pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);*/


        disp  = det;
        det   = in;
        det_s = in_s;

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        curr = 1000000.f / ((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
}
