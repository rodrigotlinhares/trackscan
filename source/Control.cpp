#include "Control.h"


/****************************************/
/*** Control class ***/
/****************************************/

Control::~Control()
{
    if(pthread_join(ThreadDisplay, NULL))
    {
        printf("Error joining thread\n");
        getchar();
    }

    // Dumping to file
    for(int i=0;i<2000;i++)
    {
        for (int j=0; j<12; j++)
        {
            fprintf(fp, "%f \t", storage[12*i+j]);
        }

        fprintf(fp, "\n");
    }

    // Saving workspace
    if(flag_saveworkspace)
        SaveWorkspace();
}


// Base Fcts

int	 Control::Initialize(cv::Mat *input, cv::Mat *input_big, cv::Mat *mask_input)
{
    printf("> Initializing Tracker Control \n");

    size_roi_x = input->cols;
    size_roi_y = input->rows;

    // Input/Output
    ICur = input;
    ICur_big = input_big;
    Mask_input = mask_input;

    // Init clock
    tick = cvGetTickFrequency();
    time(&now);
    current = localtime(&now);

    // Initialize variables
    // Public
    flag_start = 0;
    flag_quit = 0;
    flag_reset = 0;
    flag_save_mosaic = 0;
    flag_refine_mosaic = 0;
    flag_tracking = 0;
    flag_launch_display = 0;
    flag_review = 0;

    // Internal
    flag_hold = 0;
    flag_resetd = 0;
    flag_resett = 0;
    flag_new = 0;
    flag_visibility = 1;

    // Feature map
    n_ref_features = 0;

    // Stats
    counter_active = 0;
    counter_inactive = 0;

    // Entropy update
    current_entropy = 0;
    //current_contrast = 0;

    // ROI center init
    coords[0] = cvRound(size_roi_x/2);
    coords[1] = cvRound(size_roi_y/2);

    // Loading parameters
    LoadParameters();

    // Ordering vectors	& active pixels
    n_active_pixels = cvRound((float)(size_template_x*size_template_y)*(float)(percentage_active)/100);
    gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
    grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);

    dummy_mapx.create(size_template_y, size_template_x, CV_32FC1);
    dummy_mapy.create(size_template_y, size_template_x, CV_32FC1);

    dummy_mapxHD.create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_32FC1);
    dummy_mapyHD.create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_32FC1);

    // Initialize structures
    Mosaic_mapy.create(grid_y, grid_x, CV_32FC1);
    Mosaic_mapx.create(grid_y, grid_x, CV_32FC1);

    Patch_x.create(grid_y, grid_x, CV_32FC1);
    Patch_y.create(grid_y, grid_x, CV_32FC1);

    Tr.create(2, 3, CV_32FC1);
    Tf.create(2, 3, CV_32FC1);
    Ti.create(3, 3, CV_32FC1);

    // Feature map
    ref_keypoint_storage.create(max_features, 4, CV_32FC1);
    ref_descriptor_storage.create(max_features, 128, CV_32FC1);

    // Visibility and entropy maps
    Visibility_map.create(grid_y, grid_x, CV_8UC1);
    Visibility_map.setTo(cv::Scalar(0));
    Entropy_map.create(grid_y, grid_x, CV_32FC1);
    Entropy_map.setTo(cv::Scalar(0));

    // Template set and respective masks
    Template_set = new cv::Mat [elements_mosaic];
    Template_setHD = new cv::Mat [elements_mosaic];
    Mask_set = new cv::Mat [elements_mosaic];
    Mask_setHD = new cv::Mat [elements_mosaic];

    for(int i=0; i<grid_x*grid_y; i++)
    {
        Template_set[i].create(size_template_y, size_template_x, CV_8UC3);
        Template_setHD[i].create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_8UC3);

        Mask_set[i].create(size_template_y, size_template_x, CV_8UC1);
        Mask_setHD[i].create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_8UC1);
    }

    // Active pixels
    active_pixels_r = (int**) malloc(grid_x*grid_y*sizeof(int*));
    active_pixels_g = (int**) malloc(grid_x*grid_y*sizeof(int*));

    for(int j=0;j<grid_x*grid_y; j++)
    {
        active_pixels_r[j] = (int*) malloc(n_active_pixels*sizeof(int));
        active_pixels_g[j] = (int*) malloc(n_active_pixels*sizeof(int));
    }

    // Should I load saved workspace?
    if(flag_loadworkspace)
    {
        LoadWorkspace();
        flag_refine_mosaic = 1;
    }

    // Tracking structures for active pixel selection
    tracker.Initialize3DOFx(size_template_x,
                            size_template_y,
                            n_active_pixels,
                            256,
                            1,
                            0,
                            0,
                            0,
                            1);

    // Setup mosaic
    MosaicSetup();

    // Definitions
    Template = &Template_set[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];
    Mask = &Mask_set[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];

    MaskHD = &Mask_setHD[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];
    TemplateHD = &Template_setHD[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];

    // Mosaic refinement setup (I need initialized template structures)
    refine_mosaic.Initialize(size_template_x, size_template_y, grid_x, grid_y, offset_templates, Template_set, Mask_set, &Visibility_map);

    Mosaic = refine_mosaic.Mosaic;
    Mosaic_mask = refine_mosaic.Mosaic_mask;

    refine_mosaicHD.Initialize(cvFloor(size_template_x/SCALE_FACTOR), cvFloor(size_template_y/SCALE_FACTOR), grid_x, grid_y, cvFloor(offset_templates/SCALE_FACTOR), Template_setHD, Mask_setHD, &Visibility_map);

    // Saving stats
    if(save_stats)
    {
        char temp_text[100];
        if(repo_pos == -1)
            repo_pos = 0;

        sprintf(temp_text, "../storage/stats_%d.txt", repo_pos);
        fp = fopen(temp_text, "w");

        storage = (float*) malloc(12*2000*sizeof(float));
    }


    // Initializing display thread structure
    // Tracking flags which I'll get from cvWaitKey
    pointersDisplay.flag_reset = &flag_reset;
    pointersDisplay.flag_start = &flag_start;
    pointersDisplay.flag_save_mosaic = &flag_save_mosaic;
    pointersDisplay.flag_review = &flag_review;
    pointersDisplay.flag_quit = &flag_quit;
    pointersDisplay.flag_refine_mosaic = &flag_refine_mosaic;
    pointersDisplay.flag_tracking = &flag_launch_display;
    //pointersDisplay.flag_tracking = &flag_tracking;
    pointersDisplay.detected_keypoints = &detected_keypoints;

    // Pointer to images to display
    pointersDisplay.DisplayMain = DisplayImageFromROIFinder;
    pointersDisplay.DisplayMosaic = &Mosaic;
    pointersDisplay.DisplayMask = mask_input;
    pointersDisplay.video_feed = 0;

    // Pointer to info I'll display
    pointersDisplay.coords = coords;
    pointersDisplay.fps = &fps;
    pointersDisplay.tracking_param = tracking_param_display;
    pointersDisplay.mosaic_coords = mosaic_coords_display;
    pointersDisplay.size_template_x = size_template_x;
    pointersDisplay.size_template_y = size_template_y;
    pointersDisplay.grid_x = grid_x;
    pointersDisplay.grid_y = grid_y;
    pointersDisplay.offset_templates = offset_templates;
    pointersDisplay.confidence = &confidence;
    pointersDisplay.scale_factor = SCALE_FACTOR;

    // Pointer to mosaic refinement class
    pointersDisplay.refine_mosaic = &refine_mosaic;

    if(pthread_create(&ThreadDisplay, NULL, BuildDisplay, &pointersDisplay))
    {
        fprintf(stderr, "Error creating display thread\n");
        getchar();
        return 1;
    }

    return 0;
}


int	 Control::Initialize(cv::Mat *input, cv::Mat *input_big, cv::Mat *mask_input, VideoFetcher *capture)
{
    printf("> Initializing Tracker Control \n");

    size_roi_x = input->cols;
    size_roi_y = input->rows;

    // Input/Output
    ICur = input;
    ICur_big = input_big;
    Mask_input = mask_input;

    // Init clock
    tick = cvGetTickFrequency();
    time(&now);
    current = localtime(&now);

    // Initialize variables
    // Public
    flag_start = 0;
    flag_quit = 0;
    flag_reset = 0;
    flag_save_mosaic = 0;
    flag_refine_mosaic = 0;
    flag_tracking = 0;
    flag_launch_display = 0;
    flag_review = 0;

    // Internal
    flag_hold = 0;
    flag_resetd = 0;
    flag_resett = 0;
    flag_new = 0;
    flag_visibility = 1;

    // Feature map
    n_ref_features = 0;

    // Stats
    counter_active = 0;
    counter_inactive = 0;

    // Entropy update
    current_entropy = 0;
    //current_contrast = 0;

    // ROI center init
    coords[0] = cvRound(size_roi_x/2);
    coords[1] = cvRound(size_roi_y/2);

    // Loading parameters
    LoadParameters();

    // Ordering vectors	& active pixels
    n_active_pixels = cvRound((float)(size_template_x*size_template_y)*(float)(percentage_active)/100);
    gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
    grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);

    dummy_mapx.create(size_template_y, size_template_x, CV_32FC1);
    dummy_mapy.create(size_template_y, size_template_x, CV_32FC1);

    dummy_mapxHD.create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_32FC1);
    dummy_mapyHD.create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_32FC1);

    // Initialize structures
    Mosaic_mapy.create(grid_y, grid_x, CV_32FC1);
    Mosaic_mapx.create(grid_y, grid_x, CV_32FC1);

    Patch_x.create(grid_y, grid_x, CV_32FC1);
    Patch_y.create(grid_y, grid_x, CV_32FC1);

    Tr.create(2, 3, CV_32FC1);
    Tf.create(2, 3, CV_32FC1);
    Ti.create(3, 3, CV_32FC1);

    // Feature map
    ref_keypoint_storage.create(max_features, 4, CV_32FC1);
    ref_descriptor_storage.create(max_features, 128, CV_32FC1);

    // Visibility and entropy maps
    Visibility_map.create(grid_y, grid_x, CV_8UC1);
    Visibility_map.setTo(cv::Scalar(0));
    Entropy_map.create(grid_y, grid_x, CV_32FC1);
    Entropy_map.setTo(cv::Scalar(0));

    // Template set and respective masks
    Template_set = new cv::Mat [elements_mosaic];
    Template_setHD = new cv::Mat [elements_mosaic];
    Mask_set = new cv::Mat [elements_mosaic];
    Mask_setHD = new cv::Mat [elements_mosaic];

    for(int i=0; i<grid_x*grid_y; i++)
    {
        Template_set[i].create(size_template_y, size_template_x, CV_8UC3);
        Template_setHD[i].create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_8UC3);

        Mask_set[i].create(size_template_y, size_template_x, CV_8UC1);
        Mask_setHD[i].create(cvFloor(size_template_y/SCALE_FACTOR), cvFloor(size_template_x/SCALE_FACTOR), CV_8UC1);
    }

    // Active pixels
    active_pixels_r = (int**) malloc(grid_x*grid_y*sizeof(int*));
    active_pixels_g = (int**) malloc(grid_x*grid_y*sizeof(int*));

    for(int j=0;j<grid_x*grid_y; j++)
    {
        active_pixels_r[j] = (int*) malloc(n_active_pixels*sizeof(int));
        active_pixels_g[j] = (int*) malloc(n_active_pixels*sizeof(int));
    }

    // Should I load saved workspace?
    if(flag_loadworkspace)
    {
        LoadWorkspace();
        flag_refine_mosaic = 1;
    }

    // Tracking structures for active pixel selection
    tracker.Initialize3DOFx(size_template_x,
                            size_template_y,
                            n_active_pixels,
                            256,
                            1,
                            0,
                            0,
                            0,
                            1);

    // Setup mosaic
    MosaicSetup();

    // Definitions
    Template = &Template_set[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];
    Mask = &Mask_set[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];

    MaskHD = &Mask_setHD[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];
    TemplateHD = &Template_setHD[cvFloor(grid_y/2) + grid_y*cvFloor(grid_x/2)];

    // Mosaic refinement setup (I need initialized template structures)
    refine_mosaic.Initialize(size_template_x, size_template_y, grid_x, grid_y, offset_templates, Template_set, Mask_set, &Visibility_map);

    Mosaic = refine_mosaic.Mosaic;
    Mosaic_mask = refine_mosaic.Mosaic_mask;

    refine_mosaicHD.Initialize(cvFloor(size_template_x/SCALE_FACTOR), cvFloor(size_template_y/SCALE_FACTOR), grid_x, grid_y, cvFloor(offset_templates/SCALE_FACTOR), Template_setHD, Mask_setHD, &Visibility_map);

    // Saving stats
    if(save_stats)
    {
        char temp_text[100];
        if(repo_pos == -1)
            repo_pos = 0;

        sprintf(temp_text, "../storage/stats_%d.txt", repo_pos);
        fp = fopen(temp_text, "w");

        storage = (float*) malloc(12*2000*sizeof(float));
    }


    // Initializing display thread structure
    // Tracking flags which I'll get from cvWaitKey
    pointersDisplay.flag_reset = &flag_reset;
    pointersDisplay.flag_start = &flag_start;
    pointersDisplay.flag_save_mosaic = &flag_save_mosaic;
    pointersDisplay.flag_review = &flag_review;
    pointersDisplay.flag_quit = &flag_quit;
    pointersDisplay.flag_refine_mosaic = &flag_refine_mosaic;
    pointersDisplay.flag_tracking = &flag_launch_display;
    pointersDisplay.detected_keypoints = &detected_keypoints;

    // Pointer to images to display
    pointersDisplay.DisplayMain = DisplayImageFromROIFinder;
    pointersDisplay.DisplayMosaic = &Mosaic;
    pointersDisplay.DisplayMask = mask_input;
    pointersDisplay.video_feed = capture;

    // Pointer to info I'll display
    pointersDisplay.coords = coords;
    pointersDisplay.fps = &fps;
    pointersDisplay.tracking_param = tracking_param_display;
    pointersDisplay.mosaic_coords = mosaic_coords_display;
    pointersDisplay.size_template_x = size_template_x;
    pointersDisplay.size_template_y = size_template_y;
    pointersDisplay.grid_x = grid_x;
    pointersDisplay.grid_y = grid_y;
    pointersDisplay.offset_templates = offset_templates;
    pointersDisplay.confidence = &confidence;
    pointersDisplay.scale_factor = SCALE_FACTOR;

    // Pointer to mosaic refinement class
    pointersDisplay.refine_mosaic = &refine_mosaic;

    if(pthread_create(&ThreadDisplay, NULL, BuildDisplay, &pointersDisplay))
    {
        fprintf(stderr, "Error creating display thread\n");
        getchar();
        return 1;
    }

//    policy = 0;
//    min_prio_for_policy = 0;
//    pthread_attr_init(&thAttr);
//    pthread_attr_getschedpolicy(&thAttr, &policy);
//    min_prio_for_policy = sched_get_priority_min(policy);
//    int error = pthread_setschedprio(ThreadDisplay, min_prio_for_policy);
//      if(error)
//        printf("Error %d !\n",error);

    return 0;
}


int	 Control::Process()
{	
	// Main Initialization
    #pragma omp master
    {
		// In case we reset the application (when space bar is pressed)	
        if(flag_start && flag_reset)
            ResetApplication();
    }

    #pragma omp barrier

    // Mosaicking
    BuildMosaic();

    #pragma omp master
    {
        // Display
        Display();

        if(flag_save_mosaic)
        {
            // Calcula mosaico refinado em HD
            refine_mosaicHD.Process();

            // Printa o mosaico refinado
            refine_mosaicHD.PrintMosaic(repo_pos);

            // E termina a escritura
            flag_save_mosaic = 0;
        }

        // Saving stats to file
        if(save_stats)
            SaveStats();
	}

    return 0;
}


// Main fcts

void	Control::ResetApplication()
{
	// Resetting tracking confidence 
	confidence = 1.0f;
	
	// Resetting tracking parameters
	mosaic_coords[0] = 1;
	mosaic_coords[1] = 0;
	mosaic_coords[2] = (float)coords[0];
	mosaic_coords[3] = (float)coords[1];
	
	transf_update[0] = 1;
	transf_update[1] = 0;
	transf_update[2] = 0;
	transf_update[3] = 0;
	transf_update[4] = 1;
	transf_update[5] = 0;

	// Resets mosaic 
	n_ref_features = 0;
	Visibility_map.setTo(cv::Scalar(0));
	Entropy_map.setTo(cv::Scalar(0));
	refine_mosaic.ResetMosaic();
    refine_mosaicHD.ResetMosaic();

	active_template[0] = cvFloor(grid_x/2);
	active_template[1] = cvFloor(grid_y/2);

	// Starts tracking
	flag_tracking = 1;
	flag_visibility = 1;
	flag_reset = 0;
    counter_active = min_tracked_frames;
	counter_inactive = 0;
	
    // Now that new template is defined, reset tracker and detector
    flag_resett = 1;
    flag_resetd = 1;
}

void	Control::Display()
{
    if(flag_tracking == 1 && flag_start == 1 && counter_active > min_tracked_frames)
    {
        flag_launch_display = 1;

        // Copy current tracking parameters to display parameters
        for(int i=0; i<4; i++)
        {
            mosaic_coords_display[i] = mosaic_coords[i];
            tracking_param_display[i] = tracking_param[i];
        }
    }
    else
    {
        flag_launch_display = 0;
    }
}

void	Control::BuildMosaic()
{
	if(flag_start)
	{
		// Runs only when tracking is active
		if(flag_tracking)
		{
            #pragma omp master
            {
                // Updates number of successfully tracked frames
                counter_active++;
                counter_inactive = 0;

                // Update mosaic coords (current position of mosaic in intra-op image)
                // with latest tracking results. This is done by multiplying the mosaic coords
                // with the transformation update vector from the template tracker
                float mosaic_coords_orig[4];
                memcpy(mosaic_coords_orig, mosaic_coords, 4*sizeof(float));

                mosaic_coords[0] = transf_update[0]*mosaic_coords_orig[0] + transf_update[1]*mosaic_coords_orig[1];
                mosaic_coords[1] = transf_update[3]*mosaic_coords_orig[0] + transf_update[4]*mosaic_coords_orig[1];
                mosaic_coords[2] = transf_update[0]*mosaic_coords_orig[2] + transf_update[1]*mosaic_coords_orig[3] + transf_update[2];
                mosaic_coords[3] = transf_update[3]*mosaic_coords_orig[2] + transf_update[4]*mosaic_coords_orig[3] + transf_update[5];

                int min_pos[2] = {0,0},
                        min_vis_pos[2] = {0,0};

                float	current_distance,
                        min_distance = 10000,
                        min_vis_distance = 10000;

                // For all patches in mosaic
                for(int j=0; j<grid_x; j++)
                {
                    for(int i=0; i<grid_y; i++)
                    {
                        // Computes current patch location
                        Patch_x.at<float>(i, j) = mosaic_coords[0]*Mosaic_mapx.at<float>(i, j) - mosaic_coords[1]*Mosaic_mapy.at<float>(i, j) + mosaic_coords[2];
                        Patch_y.at<float>(i, j) = mosaic_coords[1]*Mosaic_mapx.at<float>(i, j) + mosaic_coords[0]*Mosaic_mapy.at<float>(i, j) + mosaic_coords[3];

                        // Checking distance to image center
                        current_distance = std::sqrt((float)(((int)Patch_x.at<float>(i, j) - coords[0])*((int)Patch_x.at<float>(i, j) - coords[0]) + ((int)Patch_y.at<float>(i, j) - coords[1])*((int)Patch_y.at<float>(i, j) - coords[1])));

                        if(current_distance < min_distance)
                        {
                            min_distance = current_distance;
                            min_pos[0] = j;
                            min_pos[1] = i;
                        }

                        if(current_distance < min_vis_distance && Visibility_map.at<uchar>(i,j))
                        {
                            min_vis_distance = current_distance;
                            min_vis_pos[0] = j;
                            min_vis_pos[1] = i;
                        }
                    }
                }

                // Jumps to a known position or if it is an unseen posiiton,
                // a new active template if tracking confidence is high and
                // a certain number of frames have been successfully tracked and
                // a suficient portion of the current tracked template is visible
                if( confidence > thresh_include && counter_active > min_tracked_frames && flag_visibility == 1 && Visibility_map.at<uchar>(min_pos[1], min_pos[0]) == 0)
                {
                    active_template_buffer[0] = active_template[0];
                    active_template_buffer[1] = active_template[1];

                    active_template[0] = min_pos[0];
                    active_template[1] = min_pos[1];
                }
                else
                {
                    active_template_buffer[0] = active_template[0];
                    active_template_buffer[1] = active_template[1];

                    active_template[0] = min_vis_pos[0];
                    active_template[1] = min_vis_pos[1];
                }

                // Active template might have changed!
                // Defining tracking parameters for tracker filter
                float u = Mosaic_mapx.at<float>(active_template[1], active_template[0]);
                float v = Mosaic_mapy.at<float>(active_template[1], active_template[0]);

                tracking_param[0] = mosaic_coords[0];
                tracking_param[1] = mosaic_coords[1];
                tracking_param[2] = mosaic_coords[0]*u - mosaic_coords[1]*v + mosaic_coords[2];
                tracking_param[3] = mosaic_coords[1]*u + mosaic_coords[0]*v + mosaic_coords[3];

                // If now tracking a new template, reset illumination parameters (not sure this helps!)
                if(active_template[0] != active_template_buffer[0] && active_template[1] != active_template_buffer[1])
                {
                    flag_resett = 1;
                }

                // Now set pointers to current template
                int dummy_pos = active_template[1] + grid_y*active_template[0];
                Template = &Template_set[dummy_pos];
                Mask = &Mask_set[dummy_pos];

                TemplateHD = &Template_setHD[dummy_pos];
                MaskHD = &Mask_setHD[dummy_pos];

                // and to currently active pixels
                cur_active_pixels_r = active_pixels_r[dummy_pos];
                cur_active_pixels_g = active_pixels_g[dummy_pos];
            }

#pragma omp barrier

			// If patch has not been seen before... 
            if(Visibility_map.at<uchar>(active_template[1], active_template[0]) == 0)
            {
                // Copy new template into template set
                Warp();

                #pragma omp master
                {
                    // Incorporating features from new patch into mosaic
                    if(n_ref_features < max_features)
                        flag_resetd = 1;
                }

                #pragma omp parallel
                {
                    unsigned int ch =  omp_get_thread_num();
                    tracker.WarpGrad(ch, Template);
                }

                #pragma omp master
                {
                    // Computing active pixels
                    int pos_active = active_template[1] + grid_y*active_template[0];

                    //tracker.FastComputeActive3DOFx(Template, Mask, active_pixels_r[pos_active], active_pixels_g[pos_active]);
                    tracker.ComputeActive3DOFx(Template, Mask, active_pixels_r[pos_active], active_pixels_g[pos_active]);

                    // Calcula entropia/contraste do template
                    Entropy_map.at<float>(active_template[1],active_template[0]) = ComputeEntropy(Template, Mask);
                    //ComputeContrast();

                    // Ativar refino do mosaico (esta flag é desativada dentro do thread de display)
                    flag_refine_mosaic = 1;

                    // Uma flag para a estatistica, só pra dizer se um template novo foi adicionado (ela é desativada no SaveStats() )
                    flag_new = 1;

                    // It is visible from now on...
                    Visibility_map.at<uchar>(active_template[1], active_template[0]) = 1;

                    // Para resetar parametros de iluminacao no filtro de rastreamento
                    flag_resett = 1;
                }
			}	
		}
		else // if flag tracking == 0
		{
            // RestartTracking(); -> this is executed inside usrgFeatTrackerFilter
		}
	}

#pragma omp barrier

    #pragma omp master
    {
        // Measuring and displaying elapsed time
        clock_stop_control = ((cvGetTickCount() - clock_start_roi)/(1000*tick));

        // This is the fps that will be displayed on the screen
        fps = 1000/clock_stop_global;
    }
}

// Misc

void Control::RestartTracking(cv::Point template_position) {
  pointersDisplay.active_template_tl = template_position * offset_templates;
  // Updating tracking counter
  counter_active = 0;
  counter_inactive++;

  active_template[0] = template_position.x;
  active_template[1] = template_position.y;

  // Defining current tracking parameters for tracker filter
  float x_offset = Mosaic_mapx.at<float>(template_position);
  float y_offset = Mosaic_mapy.at<float>(template_position);

  mosaic_coords[0] = 1;
  mosaic_coords[1] = 0;
  mosaic_coords[2] = coords[0] - x_offset;
  mosaic_coords[3] = coords[1] - y_offset;

  tracking_param[0] = mosaic_coords[0];
  tracking_param[1] = mosaic_coords[1];
  tracking_param[2] = mosaic_coords[0] * x_offset - mosaic_coords[1] * y_offset + mosaic_coords[2];
  tracking_param[3] = mosaic_coords[1] * x_offset + mosaic_coords[0] * y_offset + mosaic_coords[3];

  // Now set pointers to current template
  int dummy_position = template_position.x * grid_y + template_position.y;
  Template = &Template_set[dummy_position];
  Mask = &Mask_set[dummy_position];
  cur_active_pixels_r = active_pixels_r[dummy_position];
  cur_active_pixels_g = active_pixels_g[dummy_position];

  // Turning tracking back on again
  flag_tracking = 1;
  flag_resett = 1;
}

void Control::RestartTracking()
{
    // Updating tracking counter
    counter_active = 0;
    counter_inactive++;

    // Based on current detection results
    //if(flag_detected)
    {
        // Chosing closest template
        float min_distance = 1e10;

        // Define current active template
        for(int j=0; j<grid_x; j++)
        {
            for(int i=0; i<grid_y; i++)
            {
                // Computes current patch location
                Patch_x.at<float>(i, j) = mosaic_coords[0]*Mosaic_mapx.at<float>(i, j) - mosaic_coords[1]*Mosaic_mapy.at<float>(i, j) + mosaic_coords[2];
                Patch_y.at<float>(i, j) = mosaic_coords[1]*Mosaic_mapx.at<float>(i, j) + mosaic_coords[0]*Mosaic_mapy.at<float>(i, j) + mosaic_coords[3];

                // Checking distance to image center
                float current_distance = pow(cvRound(Patch_x.at<float>(i, j)) - coords[0], 2.0f) + pow(cvRound(Patch_y.at<float>(i, j)) - coords[1], 2.0f);

                if(current_distance < min_distance && Visibility_map.at<uchar>(i, j) == 1)
                {
                    min_distance = current_distance;
                    active_template[0] = j;   active_template[1] = i;
                }
            }
        }

        // Defining current tracking parameters for tracker filter
        float u = Mosaic_mapx.at<float>(active_template[1], active_template[0]);
        float v = Mosaic_mapy.at<float>(active_template[1], active_template[0]);

        tracking_param[0] = mosaic_coords[0];
        tracking_param[1] = mosaic_coords[1];
        tracking_param[2] = mosaic_coords[0]*u - mosaic_coords[1]*v + mosaic_coords[2];
        tracking_param[3] = mosaic_coords[1]*u + mosaic_coords[0]*v + mosaic_coords[3];

        // Now set pointers to current template
        int dummy_pos = active_template[1] + grid_y*active_template[0];

        Template = &Template_set[dummy_pos];
        Mask = &Mask_set[dummy_pos];

        cur_active_pixels_r = active_pixels_r[dummy_pos];
        cur_active_pixels_g = active_pixels_g[dummy_pos];

        // Turning tracking back on again
        flag_tracking = 1;
        flag_resett = 1;

    }
}

void	Control::SaveStats()
{
	static int counter = 0;

    clock_stop_global = ((cvGetTickCount() - clock_start_global)/(1000*tick));
    clock_start_global = cvGetTickCount();

    //fprintf(fp, "%d \t %d \t %f \t %f \t %f \t %f \t %f \t %d \t %d \t %d \t %f \t %f \n ", ++counter, iterations+1, confidence, clock_stop_pre, clock_stop_pos, clock_stop_control, clock_stop_global,
    //flag_new, counter_active, counter_inactive, mosaic_coords[2], mosaic_coords[3]);

    if(counter<1999)
    {
        storage[12*counter] = ++counter;
        storage[12*counter+1] =  iterations;
        storage[12*counter+2] = confidence;
        storage[12*counter+3] = clock_stop_pre;
        storage[12*counter+4] = clock_stop_pos;
        storage[12*counter+5] = clock_stop_control;
        storage[12*counter+6] = clock_stop_global;
        storage[12*counter+7] = flag_new;
        storage[12*counter+8] =  counter_active;
        storage[12*counter+9] = counter_inactive;
        storage[12*counter+10] = mosaic_coords[2];
        storage[12*counter+11] = mosaic_coords[3];
    }

	// Uma flag só pra dizer se um template novo foi adicionado
	flag_new = 0;
}


// Auxiliary fcts

void	Control::MosaicSetup()
{
	// Defining position of each template in mosaic coordinates
	for(int i=0; i<grid_y; i++)
	{
		for(int j=0; j<grid_x; j++)
		{
			Mosaic_mapx.at<float>(i, j) = (float)(j-cvFloor(grid_x/2))*offset_templates;
			Mosaic_mapy.at<float>(i, j) = (float)(i-cvFloor(grid_y/2))*offset_templates;
		}
  }
}

void	Control::Warp()
{
    int ch = omp_get_thread_num();

    //#pragma omp parallel
    {
        int offx = cvCeil((double)size_template_x/2);
        int offy = cvCeil((double)size_template_y/2);

        // Multiplying matrices
        //	for(int i=-offy;i<offy;i++)
        for(int i=tracker.start_stop[2*ch]-offy;i<=tracker.start_stop[2*ch+1]-offy;i++)
        {
            for(int j=-offx;j<offx;j++)
            {
                dummy_mapx.at<float>(i+offy,j+offx) = tracking_param[0]*(float)j - tracking_param[1]*(float)i + tracking_param[2];
                dummy_mapy.at<float>(i+offy,j+offx) = tracking_param[1]*(float)j + tracking_param[0]*(float)i + tracking_param[3];
            }
        }

        // Remapping
        cv::remap(*ICur, (*Template)(tracker.ok[ch]), dummy_mapx(tracker.ok[ch]), dummy_mapy(tracker.ok[ch]), CV_INTER_LINEAR, 0, cv::Scalar(0));
        cv::remap(*Mask_input, (*Mask)(tracker.ok[ch]), dummy_mapx(tracker.ok[ch]), dummy_mapy(tracker.ok[ch]), 0, 0, cv::Scalar(0));
    }

    #pragma omp barrier

    #pragma omp master
    {
        // High res part
        int offx = cvCeil(TemplateHD->cols/2);
        int offy = cvCeil(TemplateHD->rows/2);

        float scaled_tracking_param[4];
        scaled_tracking_param[0] = tracking_param[0];
        scaled_tracking_param[1] = tracking_param[1];
        scaled_tracking_param[2] = tracking_param[2]/SCALE_FACTOR;
        scaled_tracking_param[3] = tracking_param[3]/SCALE_FACTOR;

        // Multiplying matrices
        for(int i=-offy;i<offy;i++)
        {
            for(int j=-offx;j<offx;j++)
            {
                dummy_mapxHD.at<float>(i+offy,j+offx) = scaled_tracking_param[0]*(float)j - scaled_tracking_param[1]*(float)i + scaled_tracking_param[2];
                dummy_mapyHD.at<float>(i+offy,j+offx) = scaled_tracking_param[1]*(float)j + scaled_tracking_param[0]*(float)i + scaled_tracking_param[3];
            }
        }

        cv::remap(*ICur_big, (*TemplateHD), dummy_mapxHD, dummy_mapyHD, CV_INTER_LINEAR, 0, cv::Scalar(0));
        cv::resize(*Mask, *MaskHD, MaskHD->size(), 0, 0, 1);
    }
}

void	Control::LoadParameters()
{
    cv::FileStorage fs("../settings/parameters.yml", cv::FileStorage::READ);

	size_template_x = fs["size_template_x"];
	size_template_y = fs["size_template_y"];
	offset_templates = fs["offset_templates"];
	grid_x = fs["grid_x"];
	grid_y = fs["grid_y"];
	elements_mosaic = grid_x*grid_y;
	thresh_include = fs["thresh_include"];
	length_line = fs["length_line"];
	max_features = fs["max_features"];
	percentage_active = fs["percentage_active"];
	min_tracked_frames = fs["min_tracked_frames"];
    save_stats = fs["save_stats"];
}


void Control::SaveWorkspace()
{
    // Stops tracking
    flag_start = 0;

    // Dumps workspace to file
    cv::FileStorage fs("../storage/workspace.yml", cv::FileStorage::WRITE);
    refine_mosaicHD.Process();
    refine_mosaicHD.PrintMosaic(-1);

    fs << "NRefFeatures" << n_ref_features;
    //fs << "Mosaic" << Mosaic;
    fs << "VisibilityMap" << Visibility_map;
    fs << "EntropyMap" << Entropy_map;
    fs << "RefDescriptorStorage" << ref_descriptor_storage;
    fs << "RefKeypointStorage" << ref_keypoint_storage;

    cv::Mat temp(n_active_pixels, 1, CV_32SC1);

    for(int u=0; u<grid_y; u++)
    {
        for(int v=0; v<grid_x; v++)
        {
            std::ostringstream text;

            if(Visibility_map.at<uchar>(u,v))
            {
                // active pixls
                text.str("");
                text << "ActivePxlG" << u+grid_y*v;
                memcpy(temp.data, active_pixels_g[u+grid_y*v], n_active_pixels*sizeof(int));

                fs << text.str() << temp;

                text.str("");
                text << "ActivePxlR" << u+grid_y*v;
                memcpy(temp.data, active_pixels_r[u+grid_y*v], n_active_pixels*sizeof(int));

                fs << text.str() << temp;

                // saving templates
                text.str("");
                text << "Template" << u+grid_y*v;

                fs << text.str() << Template_set[u+grid_y*v];

                text.str("");
                text << "Mask" << u+grid_y*v;

                fs << text.str() << Mask_set[u+grid_y*v];

//                text.str("");
//                text << "TemplateHD" << u+grid_y*v;
//
//                fs << text.str() << Template_setHD[u+grid_y*v];
//
//                text.str("");
//                text << "MaskHD" << u+grid_y*v;
//
//                fs << text.str() << Mask_setHD[u+grid_y*v];
            }
        }
    }

    // Closing file
    fs.release();
}


void Control::LoadWorkspace()
{
    // Loads workspace from file
    cv::FileStorage fs("../storage/workspace.yml", cv::FileStorage::READ);

    if(!fs.isOpened())
    {
        printf("\n Could not load workspace.yml!!!");
        exit(1);
    }

    fs["NRefFeatures"] >> n_ref_features;
    //fs["Mosaic"] >> Mosaic;
    fs["VisibilityMap"] >> Visibility_map;
    fs["EntropyMap"] >> Entropy_map;
    fs["RefDescriptorStorage"] >> ref_descriptor_storage;
    fs["RefKeypointStorage"] >> ref_keypoint_storage;

    cv::Mat temp(n_active_pixels, 1, CV_32SC1);

    for(int u=0; u<grid_y; u++)
    {
        for(int v=0; v<grid_x; v++)
        {
            std::ostringstream text;

            if(Visibility_map.at<uchar>(u,v))
            {
                // Loading active pixels
                text.str("");
                text << "ActivePxlG" << u+grid_y*v;
                fs[text.str()] >> temp;
                memcpy(active_pixels_g[u+grid_y*v], temp.data, n_active_pixels*sizeof(int));

                text.str("");
                text << "ActivePxlR" << u+grid_y*v;
                fs[text.str()] >> temp;
                memcpy(active_pixels_r[u+grid_y*v], temp.data, n_active_pixels*sizeof(int));

                // Loading templates
                text.str("");
                text << "Template" << u+grid_y*v;
                fs[text.str()] >> Template_set[u+grid_y*v];

                text.str("");
                text << "Mask" << u+grid_y*v;
                fs[text.str()] >> Mask_set[u+grid_y*v];

                text.str("");
                text << "TemplateHD" << u+grid_y*v;
                fs[text.str()] >> Template_setHD[u+grid_y*v];

                text.str("");
                text << "MaskHD" << u+grid_y*v;
                fs[text.str()] >> Mask_setHD[u+grid_y*v];

//                cv::imshow("teste1", Template_setHD[u+grid_y*v]);
//                cv::waitKey(0);
            }
        }
    }

    // Launch tracking
    flag_start = 1;
    flag_tracking = 0;
}



// Computing the current warp entropy


//float	Control::ComputeEntropy(cv::Mat *img)
//{
//    int u, v, acc = 0;
//    float p_cur_r[256], p_cur_g[256], p_cur_b[256], entropy_tmplt;

//    entropy_tmplt = 0;

//    for(u=0; u<256; u++)
//    {
//        p_cur_r[u] = 0;
//        p_cur_g[u] = 0;
//        p_cur_b[u] = 0;
//    }

//    for(u=0; u<size_template_x; u++)
//    {
//        for(v=0; v<size_template_y; v++)
//        {
//            if(Mask->at<uchar>(v,u) == 255)
//            {
//                p_cur_r[Template->ptr<uchar>(v)[3*u+2]]++ ;
//                p_cur_g[Template->ptr<uchar>(v)[3*u+1]]++ ;
//                p_cur_b[Template->ptr<uchar>(v)[3*u]]++ ;

//                acc++;
//            }
//        }
//    }

//    for(u=0; u<256; u++)
//    {
//        float a = p_cur_r[u]/acc;
//        float b = p_cur_g[u]/acc;
//        float c = p_cur_b[u]/acc;

//        /*if(a>0)
//            entropy_tmplt += -u*a;

//        if(b>0)
//            entropy_tmplt += u*b;

//        if(c>0)
//            entropy_tmplt += u*c;*/

//        if(a>0)
//            entropy_tmplt += a*log(a);

//        if(b>0)
//            entropy_tmplt += b*log(b);

//        if(c>0)
//            entropy_tmplt += c*log(c);
//    }

//    Entropy_map.at<float>(active_template[1],active_template[0]) = -entropy_tmplt;
//}


float	Control::ComputeEntropy(cv::Mat *img, cv::Mat *mask)
{
    float current_entropy;

    int histogram_r[256], histogram_g[256], counter = 0;

    memset(histogram_r, 0, 256*sizeof(int));
    memset(histogram_g, 0, 256*sizeof(int));

    // First we compute the current warp histogram
    for(int i=0;i<img->cols*img->rows;i++)
    {
        if(mask->ptr<uchar>(0)[i]>0)
        {
            histogram_r[(img->ptr<uchar>(0))[3*i+2]]++;
            histogram_g[(img->ptr<uchar>(0))[3*i+1]]++;
            counter++;
        }
    }

    // Then we compute entropy using the blue channel
    current_entropy = 0;

    for(int i=0;i<256;i++)
    {
        if(histogram_r[i] > 0)
        {
            float bin_value = static_cast<float>(histogram_r[i])/static_cast<float>(counter);

            current_entropy -= (bin_value)*log(bin_value);
        }
    }

    // and then the entropy using the red channel
    float temp_entropy = 0;

    for(int i=0;i<256;i++)
    {
        if(histogram_g[i] > 0)
        {
            float bin_value = static_cast<float>(histogram_g[i])/static_cast<float>(counter);

            temp_entropy -= (bin_value)*log(bin_value);
        }
    }

    // and the entropy result is the average of red and blue channel entropies...
    current_entropy = (current_entropy+temp_entropy)/2;

    return current_entropy;
}

