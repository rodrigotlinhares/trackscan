#ifndef _thread_structs
#define _thread_structs

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "MOSAIC.h"
#include "video_fetcher.h"
#include "keypoint_handler.h"

typedef struct{

    int  *flag_start,
         *flag_refine_mosaic,
         *flag_quit,
         *flag_tracking,
         *flag_save_mosaic,
         *flag_review,
         *flag_reset;

    int  *coords,
         grid_x,
         grid_y,
         size_template_x,
         size_template_y,
         offset_templates;

	float *tracking_param,
		  *confidence,
		  *mosaic_coords,
		  scale_factor;

	double *fps;

    cv::Mat  *DisplayMain;
    cv::Mat  *DisplayMosaic;
    cv::Mat  *DisplayMask;
    VideoFetcher* video_feed;

    MOSAIC  *refine_mosaic;
    std::vector<cv::KeyPoint>* detected_keypoints;
    cv::Point active_template_tl;
	
}ThreadPointersDisplay; 


typedef struct{

    int  *flag_running;

    int  *flag_process;

    // ...

    int     max_features,
            *active_template,
            grid_x,
            grid_y,
            offset_templates,
            *n_ref_features,
            size_template_x,
            size_template_y,
            min_feat_distance;

    float   *tracking_param;

    std::vector<cv::KeyPoint> *keypoints;

    cv::Mat *ICur,
            *Input_mask,
            *ref_keypoint_storage,
            *ref_descriptor_storage,
            *Ti,
            *Tf,
            *Tr,
            *Feat,
            *FeatW;

    KeyPointHandler* keypoint_handler;


}ThreadPointersFeatureDetector;


void    *BuildDisplay(void *param);

void*    ThreadedProcess(void *param_in);

#endif // _macros_h
