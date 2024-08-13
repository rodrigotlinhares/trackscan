#ifndef _ROIFinder_h
#define _ROIFinder_h


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <omp.h>

#include "macros.h"
#include "Control.h"


class ROIFinder 
{
	friend class DetectorAux;

	friend class Detector;

	friend class Control;

	friend class Tracker;

private:
	
    void    LoadParameters();

public:    

    Control  *control_internal;

    ~ROIFinder();

    int Initialize(cv::Mat *input);

    int Process(cv::Mat *input);

private:

    int kernel_dilation,
        kernel_erosion,
		start_stop[N_PROCS*2];

    cv::Mat rot_mat;
	
	unsigned int width, height, part;
	
	unsigned int acc_x[N_PROCS], acc_y[N_PROCS], total[N_PROCS];

	cv::Rect ok[N_PROCS];

    cv::Mat kernel;

public: 

    int center_x,
        center_y;

    cv::Mat image_out, image_display, image_big_rotated;
	
    cv::Mat mask_img, mask_final;
};



#endif // _ROIFinder_h

