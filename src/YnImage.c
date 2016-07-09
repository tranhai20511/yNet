//	File        :   YnImage.c
//	Brief       :   Implement methods.
//	DD-MM_YYYY  :   04-07-2016
//	Author      :   haittt

#include "../YnImage.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#endif

#include "stb_image.h"
#include "stb_image_write.h"

/**************** Define */
#define class_test_car      (6)
#define class_test_person   (14)
#define class_test_bike     (1)
#define class_test_motor    (13)
#define class_test_bus      (5)
#define box_num             (7*7*2)

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */
float colors[6][3] = { {1,0,1},{0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

/**************** Local Implement */

/**************** Implement */

