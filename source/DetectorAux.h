#ifndef _DetectorAux_h
#define _DetectorAux_h

#include <pthread.h>


#include "opencv2/nonfree/nonfree.hpp"
#include "Control.h"
#include "macros.h"
#include "pThreadStructs.h"



class DetectorAux
{

public:

    int hessianThreshold;

    int	*flag_tracking,
          *flag_reset_detector,
          *flag_start,
          //*flag_detected,
          flag_process,
          flag_thread_isrunning;

    int	*active_template;


public:

  ~DetectorAux();
  
  // vars from initializer
  
  Control	*control_internal;

  int   Initialize(cv::Mat *input);

  int   Process(cv::Mat *input, cv::Mat *input_mask);


private:
	
	void	LoadParameters(void);
	
	// Clock
	double	clock_start,
		    clock_stop,
			tick;

	int		counter,
		    size_template_x,
			size_template_y,
			grid_x,
			grid_y,
			offset_templates,
			max_features; 

	float	min_feat_distance;

    // Thread stuff
    pthread_t   ThreadFeatDetection;

    ThreadPointersFeatureDetector  pointersThread;

	// Working image
    cv::Mat		ICur,
                Input_mask;

	// Feat stuff
	// Detection parameters
  KeyPointHandler keypoint_handler;
	std::vector<cv::KeyPoint> keypoints;

	float ft_param1,
		  ft_param2,
		  ft_param3,
		  ft_param4;

	// Aux matrices
	int		*n_ref_features;

    cv::Mat	*ref_keypoint_storage;
    cv::Mat	*ref_descriptor_storage;
	cv::Mat	Ti,
		    Tf,
			Tr,
			Feat,
			FeatW;
};

#endif 











