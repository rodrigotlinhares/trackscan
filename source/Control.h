#ifndef _Control_h
#define _Control_h

#include "opencv2/nonfree/nonfree.hpp"

#include "pthread.h"

#include "pThreadStructs.h"
#include "macros.h"
#include "MOSAIC.h"
#include "naya.h"
#include "omp.h"

#include "video_fetcher.h"

class Control
{
    friend class Tracker;

	friend class Detector;

	friend class DetectorAux;

    friend class ROIFinder;

public:

  ~Control();

	// Public flags
    int     flag_quit,
            flag_start,
            flag_tracking,
            flag_reset,
            flag_refine_mosaic,
            flag_save_mosaic,
            flag_launch_display,
            flag_review,
            flag_saveworkspace,
            flag_loadworkspace;

	// ROI finder
	int		coords[2];		

	// Display
    cv::Mat DisplayImage;
    std::vector<cv::KeyPoint> detected_keypoints;

    // Misc
    void    SaveWorkspace();


    // Base fcts
    int     Initialize(cv::Mat *input, cv::Mat *input_big, cv::Mat *mask_input);
    int     Initialize(cv::Mat *input, cv::Mat *input_big, cv::Mat *mask_input, VideoFetcher *capture);

    int     Process();


private:

	// Main fcts	
	void	ResetApplication();

	void	BuildMosaic();

    void	Display();


	//	Aux
	void	MosaicSetup();
	
	void	Warp();	
	
	void	LoadParameters();
	
    float    ComputeEntropy(cv::Mat *img, cv::Mat *mask);

	void	ComputeContrast();
	
    void    SaveStats();

    void    LoadWorkspace();

    void    RestartTracking();


  void RestartTracking(cv::Point position); //TODO remove
	
private:	

	// Internal flags
	int		 flag_resett,
			 flag_resetd,
			 flag_new,
             flag_hold,
			 flag_visibility;

	// Detection
	cv::Mat  ref_keypoint_storage,
			 ref_descriptor_storage;
	
    int		 n_ref_features,
			 max_features;
	
	// Active template
    naya	tracker;

	int		 n_active_pixels,
			 *cur_active_pixels_r,
			 *cur_active_pixels_g,
			 **active_pixels_r,
			 **active_pixels_g;

	// Tracking
	int		size_roi_x,
			size_roi_y,
			size_template_x,
			size_template_y,
			percentage_active;

	int		iterations,
			counter,
			counter_active,
			counter_inactive;
	
	float	current_entropy,
			current_contrast;
	
	float	coef, confidence;

	// Non-rigid illumination compensation
	int		n_ctrl_ptsxi,
			n_ctrl_ptsyi;
	
	// Transformation manager
	cv::Mat Ti, Tf, Tr;
	
	float	mosaic_coords[4], // [cos(theta), sin(theta), translation x, translation y]
            mosaic_coords_display[4],
            tracking_param[4],
            tracking_param_display[4],
            illum_param[100],
			transf_update[6];


	// Telestration aux
	bool	acquisition_start;

	float	*target;
	
    int		n_points,
			length_line;	
	
	
	// Mosaic
	int		offset_templates,
			grid_x,
			grid_y,
			elements_mosaic,
			min_tracked_frames,
			active_template[2],
			active_template_buffer[2];

	int		fundus_x, fundus_y;

	float	thresh_include;

    cv::Mat *ICur,
            *ICur_big,
            *Mask_input,
            *Mask,
            *Template,
			*TemplateHD,
            *MaskHD,
			Mosaic_mask,
			Image_from_file, 
			Edgemap;	

	cv::Mat *Template_ref,
			*Mask_ref,
			*Template_cur,
			*Mask_cur,
			*Template_curHD,
			*Mask_curHD;

	cv::Mat Mosaic_mapx,
			Mosaic_mapy,
			Patch_x,
			Patch_y,
			Visibility_map;

	cv::Mat Entropy_map,
			Contrast_map;

	cv::Mat gradx_tmplt,
			grady_tmplt;

	cv::Mat *Template_set,
			*Template_setHD,
			*Mask_set,
			*Mask_setHD;
	
	cv::Mat dummy_mapx,
			dummy_mapy;

	cv::Mat dummy_mapxHD,
			dummy_mapyHD;


	// Clock
    double	clock_start_roi,
            clock_start_global,
            clock_stop_pre,
            clock_stop_pos,
            clock_stop_control,
            clock_stop_global,
			tick,
			fps;

	
    // MISC
    int		save_stats;
	
	struct  tm *current;

	time_t	now;		
	
	FILE	*fp;

    float   *storage;


    // Display
    ThreadPointersDisplay pointersDisplay;

    cv::Mat *DisplayImageFromROIFinder;

public:

    pthread_t ThreadDisplay;
    pthread_attr_t thAttr;
    int policy;
    int min_prio_for_policy;

    int     repo_pos;

    cv::Mat Mosaic;

    MOSAIC	refine_mosaic,
            refine_mosaicHD;
};


#endif // _Control_h

