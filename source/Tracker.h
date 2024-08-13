#ifndef _Tracker_h
#define _Tracker_h

#include "Control.h"
#include "omp.h"
#include "naya.h"


class Tracker
{
	friend class Control;

public:
	
    Control  *control_internal;

    ~Tracker();


    int     Initialize(cv::Mat *input, cv::Mat *mask_input);

    int     Process();

	
private:

    bool	TestVisibility(float threshold, naya *best_tracker);

    void	ComputeTransform(float *param_final, float *param_initial);

    void	LoadParameters(void);

    cv::Mat extract_piece(cv::Mat image, cv::Size size, cv::Point center);

	
private:

    naya    tracker_forward,
            tracker_backward;

    int		*flag_reset_tracker,
            *flag_tracking,
            *flag_start;
	
    int size_template_x,
		size_template_y,
		n_ctrl_pts_x_i,
		n_ctrl_pts_y_i,
		size_bins,
		n_bins,
		n_max_iters;

    int isfirst;

    int percentage_active,
		n_active_pixels;

    float confidence_forward,
          confidence_backward,
		  confidence_thres,
          epsilon,
          visibility_thres,
          rotation_thres;

    float *transf_update;

    cv::Mat *ICur,
            Buffer,
            *Mask,
            Mask_buffer;

	cv::Mat Tf,
			Ti,
			Tr;

    float parameters_ref[4],
          parameters_forward[4],
          parameters_backward[4],
          parameters_forward_temp[9];
};



#endif // _Tracker_h

