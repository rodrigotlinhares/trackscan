#include "DetectorAux.h"



/***********************************/
/*** DetectorAux class ***/
/***********************************/

DetectorAux::~DetectorAux()
{
    flag_thread_isrunning = 0;

    if(pthread_join(ThreadFeatDetection, NULL))
    {
        printf("Error joining feature detection thread\n");
        getchar();
    }
}


int DetectorAux::Initialize(cv::Mat *input)
{
    printf("> Initializing Detection Buddy \n");

	// Load tracking parameters
	LoadParameters();

	// Initialize structures
	counter = 0;
    flag_process = 0;
    flag_thread_isrunning = 1;

    ICur.create(input->rows, input->cols, CV_8UC1);
    Input_mask.create(input->rows, input->cols, CV_8UC1);

	Tr.create(2, 3, CV_32FC1);
	Tf.create(2, 3, CV_32FC1);
	Ti.create(3, 3, CV_32FC1);
	Feat.create(3, 1, CV_32FC1);
	FeatW.create(2, 1, CV_32FC1);
	Feat.at<float>(2, 0) = 1;

	// Clock init
	tick = cvGetTickFrequency();    

	// Variables from control filter
	flag_start = &(control_internal->flag_start);
	flag_reset_detector = &(control_internal->flag_resetd);
	active_template = control_internal->active_template;
	n_ref_features = &(control_internal->n_ref_features);

    ref_keypoint_storage = &(control_internal->ref_keypoint_storage);
    ref_descriptor_storage = &(control_internal->ref_descriptor_storage);


	// Initializing misterious OpenCV fct
    cv::initModule_nonfree();

    pointersThread.flag_process = &flag_process;
    pointersThread.flag_running = &flag_thread_isrunning;

    pointersThread.ICur = &ICur;
    pointersThread.Input_mask = &Input_mask;
    pointersThread.tracking_param  = control_internal->tracking_param;
    pointersThread.max_features = max_features;
    pointersThread.n_ref_features = n_ref_features;
    pointersThread.keypoints = &keypoints;
    pointersThread.Ti = &Ti;
    pointersThread.Tf = &Tf;
    pointersThread.Tr = &Tr;
    pointersThread.Feat = &Feat;
    pointersThread.FeatW = &FeatW;
    pointersThread.ref_keypoint_storage = ref_keypoint_storage;
    pointersThread.ref_descriptor_storage = ref_descriptor_storage;
    pointersThread.min_feat_distance = min_feat_distance;
    pointersThread.keypoint_handler = &keypoint_handler;
    pointersThread.active_template = active_template;
    pointersThread.size_template_x = size_template_x;
    pointersThread.size_template_y = size_template_y;
    pointersThread.grid_x = grid_x;
    pointersThread.grid_y = grid_y;
    pointersThread.offset_templates = offset_templates;

    if(pthread_create(&ThreadFeatDetection, NULL, ThreadedProcess, &pointersThread))
    {
        fprintf(stderr, "Error creating feat detection thread\n");
        getchar();
        return 1;
    }

    return 0;
}

int DetectorAux::Process(cv::Mat *input, cv::Mat *input_mask)
{
    // Main Program
    #pragma omp master
    {
        // System is active
        if(*flag_start)
        {
            // New template was defined by user, redefine bag of features...
            if(*flag_reset_detector)
            {
                // Copying current image and mask for async treatment
                cv::cvtColor(*input, ICur, CV_RGB2GRAY);
                input_mask->copyTo(Input_mask);

                flag_process = 1;

                *flag_reset_detector = 0;
            }
        }
    }

    return 0;
}

void	DetectorAux::LoadParameters()
{
    cv::FileStorage fs("../settings/parameters.yml", cv::FileStorage::READ);

	size_template_x = fs["size_template_x"];
	size_template_y = fs["size_template_y"];
	offset_templates = fs["offset_templates"];
	grid_x = fs["grid_x"];
	grid_y = fs["grid_y"];
	max_features = fs["max_features"];
	min_feat_distance = fs["min_feat_distance"];

	ft_param1 = fs["ft_param1"];
	ft_param2 = fs["ft_param2"];
	ft_param3 = fs["ft_param3"];
	ft_param4 = fs["ft_param4"];
}


