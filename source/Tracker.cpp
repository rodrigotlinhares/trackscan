#include "Tracker.h"

/***************************/
/*** Tracker class ***/
/***************************/


// Main fcts

int	 Tracker::Initialize(cv::Mat *input, cv::Mat *input_mask)
{
    printf("> Initializing Tracker \n");

    // Inits
    isfirst = 1;
    Buffer = input->clone();
    Mask_buffer = input_mask->clone();

    // I/O
    ICur = input;
    Mask = input_mask;

    // Load tracking parameters
    LoadParameters();

    // Active pixels
    n_active_pixels = cvRound((float)(size_template_x*size_template_y)*(float)(percentage_active)/100);

    // Initialize structures
    tracker_forward.Initialize3DOFxi(size_template_x,
                                    size_template_y,
                                    n_ctrl_pts_x_i,
                                    n_ctrl_pts_y_i,
                                    n_active_pixels,
                                    n_bins,
                                    size_bins,
                                    n_max_iters,
                                    epsilon,
                                    0,
                                    1);

    tracker_backward.Initialize3DOFxi(size_template_x,
                                        size_template_y,
                                        n_ctrl_pts_x_i,
                                        n_ctrl_pts_y_i,
                                        n_active_pixels, //size_template_x*size_template_y, // tracking 100% of pixels
                                        n_bins,
                                        size_bins,
                                        n_max_iters,
                                        epsilon,
                                        0,
                                        1);

    Tr.create(2, 3, CV_32FC1);
    Tf.create(2, 3, CV_32FC1);
    Ti.create(3, 3, CV_32FC1);

    // Variables from control
    flag_start = &(control_internal->flag_start);
    flag_tracking = &(control_internal->flag_tracking);
    flag_reset_tracker = &(control_internal->flag_resett);
    transf_update = control_internal->transf_update;

    return 0;
}

int	 Tracker::Process()
{
    // Main program
    if(*flag_start)
    {
        // In case tracking was reset
        if(*flag_reset_tracker)
        {
            #pragma omp master
            {
                // Turning off reset
                *flag_reset_tracker = 0;

                // Resets illumination parameters
                tracker_forward.ResetIlluminationParam3DOFxi(control_internal->illum_param);
            }
        }

        // Tracking starts
        if(*flag_tracking)
        {
            // forward tracking
            #pragma omp master
            {
                // Loading tracking parameters
                parameters_forward[0] = control_internal->tracking_param[0];
                parameters_forward[1] = control_internal->tracking_param[1];
                parameters_forward[2] = control_internal->tracking_param[2];
                parameters_forward[3] = control_internal->tracking_param[3];

                // Timing these functions
                control_internal->clock_stop_pre = (cvGetTickCount() - control_internal->clock_start_roi)/(1000*control_internal->tick);
            }

            // Run Threaded Naya
            tracker_forward.Run3DOFxi_threaded(ICur,
                                                Mask,
                                                control_internal->Template,
                                                control_internal->Mask,
                                                parameters_forward,
                                                control_internal->illum_param,
                                                control_internal->cur_active_pixels_r,
                                                control_internal->cur_active_pixels_g);
            //#pragma omp barrier
            #pragma omp master
            {
                confidence_forward = tracker_forward.ComputeTrackingConfidenceSSDi();
                //cv::imshow("teste1", tracker_forward.compensated_warp);cv::imshow("teste11", tracker_forward.Mask);
            }
            #pragma omp barrier

            cv::Mat roi;
            cv::Mat roi_mask;
            float illum_backward[50];

            // If forward tracking is not doing super well, try backward tracking
            if(!isfirst && confidence_forward  < 0.96)
            {
                #pragma omp master
                {
                    // Crop input image
                    cv::Size roi_size(size_template_x, size_template_y);
                    cv::Point roi_center(control_internal->coords[0], control_internal->coords[1]);
                    roi = extract_piece(*ICur, roi_size, roi_center); //TODO optimize
                    roi_mask = extract_piece(*Mask, roi_size, roi_center); //TODO optimize

                    // Setting ref tracking parameters
                    parameters_backward[0] = parameters_ref[0] = 1;
                    parameters_backward[1] = parameters_ref[1] = 0;
                    parameters_backward[2] = parameters_ref[2] = (float)control_internal->coords[0];
                    parameters_backward[3] = parameters_ref[3] = (float)control_internal->coords[1];

                    // Erasing illumination parameters
                    tracker_backward.ResetIlluminationParam3DOFxi(illum_backward);
                }

                // Run Threaded Naya
                tracker_backward.Run3DOFxi_threaded(&Buffer,
                                                    &Mask_buffer,
                                                    &roi,
                                                    &roi_mask,
                                                    parameters_backward,
                                                    illum_backward,
                                                    control_internal->cur_active_pixels_r,
                                                    control_internal->cur_active_pixels_g);
                #pragma omp barrier
                #pragma omp master
                {
                    confidence_backward = tracker_backward.ComputeTrackingConfidenceSSDi();
                }

//                    cv::Mat Tfinal(3,3,CV_32FC1), Tinitial(3,3,CV_32FC1), Transfer(3,3,CV_32FC1);

//                    Tfinal.at<float>(0, 0) = parameters_ref[0];
//                    Tfinal.at<float>(0, 1) = -parameters_ref[1];
//                    Tfinal.at<float>(0, 2) = parameters_ref[2];
//                    Tfinal.at<float>(1, 0) = parameters_ref[1];
//                    Tfinal.at<float>(1, 1) = parameters_ref[0];
//                    Tfinal.at<float>(1, 2) = parameters_ref[3];
//                    Tfinal.at<float>(2, 0) = 0;
//                    Tfinal.at<float>(2, 1) = 0;
//                    Tfinal.at<float>(2, 2) = 1;

//                    Tinitial.at<float>(0, 0) = parameters_backward[0];
//                    Tinitial.at<float>(0, 1) = -parameters_backward[1];
//                    Tinitial.at<float>(0, 2) = parameters_backward[2];
//                    Tinitial.at<float>(1, 0) = parameters_backward[1];
//                    Tinitial.at<float>(1, 1) = parameters_backward[0];
//                    Tinitial.at<float>(1, 2) = parameters_backward[3];
//                    Tinitial.at<float>(2, 0) = 0;
//                    Tinitial.at<float>(2, 1) = 0;
//                    Tinitial.at<float>(2, 2) = 1;

//                    cv::Mat dummy_mat(3,3,CV_32FC1);

//                    // Achando transf inversa
//                    Transfer = Tfinal*Tinitial.inv();

//                    // Copiar resultado no vetor transf_update
//                    dummy_mat.at<float>(0, 0) = control_internal->tracking_param[0];
//                    dummy_mat.at<float>(0, 1) = -control_internal->tracking_param[1];
//                    dummy_mat.at<float>(0, 2) = control_internal->tracking_param[2];
//                    dummy_mat.at<float>(1, 0) = control_internal->tracking_param[1];
//                    dummy_mat.at<float>(1, 1) = control_internal->tracking_param[0];
//                    dummy_mat.at<float>(1, 2) = control_internal->tracking_param[3];
//                    dummy_mat.at<float>(2, 0) = 0;
//                    dummy_mat.at<float>(2, 1) = 0;
//                    dummy_mat.at<float>(2, 2) = 1;

//                    dummy_mat = Transfer*dummy_mat;

//                    parameters_forward_temp[0] = dummy_mat.at<float>(0, 0);
//                    parameters_forward_temp[1] = dummy_mat.at<float>(0, 1);
//                    parameters_forward_temp[2] = dummy_mat.at<float>(0, 2);
//                    parameters_forward_temp[3] = dummy_mat.at<float>(1, 0);
//                    parameters_forward_temp[4] = dummy_mat.at<float>(1, 1);
//                    parameters_forward_temp[5] = dummy_mat.at<float>(1, 2);
//                    parameters_forward_temp[6] = dummy_mat.at<float>(2, 0);
//                    parameters_forward_temp[7] = dummy_mat.at<float>(2, 1);
//                    parameters_forward_temp[8] = dummy_mat.at<float>(2, 2);
//                }

//                #pragma omp barrier
//                unsigned int ch = omp_get_thread_num();
//                tracker_forward.Warp3DOFAux(ch, parameters_forward_temp);

//                #pragma omp barrier
//                tracker_forward.OcclusionMap(ch);

//                #pragma omp barrier
//                #pragma omp master
//                {
//                    confidence_backward = tracker_forward.ComputeTrackingConfidenceSSDi();
////                    std::cout << confidence_forward << " " <<  confidence_backward << std::endl;
////                    cv::imshow("teste2", tracker_forward.compensated_warp);
////                    cv::imshow("teste22", tracker_forward.Mask);
////                    cv::imshow("teste3", *control_internal->Template);
////                    cv::waitKey(0);
//                }

//                #pragma omp master
//                {
//                    cv::imshow("teste", Buffer);
//                    cv::imshow("teste2", Mask_buffer);
//                    cv::waitKey(0);
//                        std::cout << tracker_forward.ComputeTrackingConfidenceSSDi() << " " <<  tracker_backward.ComputeTrackingConfidenceSSDi() << std::endl;
//                }

            }
            else
                confidence_backward = 0;

            // Now let's sort out the results
            #pragma omp master
            {
                // Selecionar aqui o melhor rastreamento (lembrar de copiar os parametros pro vetor parameters[])
                naya *best_tracker;

                //if(confidence_forward > 0.93 && confidence_backward > 0.97 && control_internal->counter_active > 100)
                if(confidence_forward < confidence_backward && control_internal->counter_active > 100 && control_internal->flag_hold == 0)
                {
                    best_tracker = &tracker_backward;

                    // Finding relative transformation between previous and current tracking parameters
                    ComputeTransform(parameters_ref, parameters_backward);

                    // Setting confidence level
                    control_internal->confidence = confidence_backward;

                }
                else
                {
                    best_tracker = &tracker_forward;

                    // Finding relative transformation between previous and current tracking parameters
                    ComputeTransform(parameters_forward, control_internal->tracking_param);

                    // Setting confidence level
                    control_internal->confidence = confidence_forward;

                    // Lets tracker use backward tracker
                    if(confidence_forward>0.96)
                        control_internal->flag_hold = 0;
                }
                //std::cout <<  control_internal->flag_hold << " " << control_internal->confidence << " " << confidence_forward << " " << confidence_backward <<std::endl;

                if(control_internal->confidence < confidence_thres)
                {
                    printf("\nConfidence below threshold : %f !              \n", control_internal->confidence);
                    *flag_tracking = 0;
                }

                // Checks consistency
                if(best_tracker->CheckConsistency3DOF(rotation_thres))
                    *flag_tracking = 0;

                // Check entropy and updates template if necessary
                //control_internal->current_entropy = control_internal->ComputeEntropy(&(best_tracker->compensated_warp), &(best_tracker->Mask));

//                if(control_internal->current_entropy > control_internal->Entropy_map.at<float>(control_internal->active_template[1],control_internal->active_template[0]))
//                {
//                    control_internal->Entropy_map.at<float>(control_internal->active_template[1],control_internal->active_template[0]) = control_internal->current_entropy;

//                    best_tracker->compensated_warp.copyTo(*control_internal->Template);
//                    best_tracker->Mask.copyTo(*control_internal->Mask);
//                    std::cout << " updated                                  ! " << std::endl;
//                }


                // Running security checks
                control_internal->flag_visibility = TestVisibility(visibility_thres, best_tracker); 				// teste de visibilidade

                // Computes Entropy and records number of iterations,
                control_internal->iterations = best_tracker->iters;

//                std::cout << tracker.ctrl_pts_wi << std::endl;

//                cv::imshow("rf", tracker.current_warp);
//                cv::imshow("teste", tracker_forward.compensated_warp); cv::imshow("teste2", *control_internal->Template);
//                cv::imshow("teste1", tracker.Mask); cv::imshow("teste3", *control_internal->Mask);
//                cv::waitKey(1);

                // Resetting tracking parameters (soon to be obsolete if I do a good job with the new template class)
                if(!*flag_tracking)
                {
                    // Resets illumination parameters
                    tracker_forward.ResetIlluminationParam3DOFxi(control_internal->illum_param);
                }

                // Clock stop
                control_internal->clock_stop_pos = (cvGetTickCount() - control_internal->clock_start_roi)/(1000*control_internal->tick);
            }
            #pragma omp barrier

            // Copying current image to buffer
            #pragma omp master
            {
                ICur->copyTo(Buffer);
                Mask->copyTo(Mask_buffer);
                isfirst = 0;
            }
        }
    }

    return 0;
}

// Aux functions
cv::Mat Tracker::extract_piece(cv::Mat image, cv::Size size, cv::Point center) {
    cv::Point top_left(center.x - size.width / 2, center.y - size.height / 2);
    if(top_left.x < 0)
        top_left.x = 0;
    if(top_left.y < 0)
        top_left.y = 0;
    if(top_left.x + size.width > image.cols)
        top_left.x = image.cols - size.width;
    if(top_left.y + size.height > image.rows)
        top_left.y = image.rows - size.height;
    cv::Rect bounding_box(top_left, size);
    return cv::Mat(image, bounding_box);
}



bool    Tracker::TestVisibility(float threshold, naya *best_tracker)
{
    int sum = 0;

    unsigned char *value = best_tracker->Mask.data;

    for(int i=0; i<size_template_x*size_template_y; i++, value++)
    {
        sum += (*value>0);
    }

    if((float)sum/(float)(size_template_x*size_template_y) > threshold){
        return 1;
    }
    else
    {
        std::cout<< "Low Visibility Warning!" << std::endl;

        return 0;
    }
}

void	Tracker::ComputeTransform(float *param_final, float *param_initial)
{
    // This function finds Tx, where T_final = Tx*T_initial

    // Defines T_initial(3x3)
    Ti.at<float>(0, 0) = param_initial[0];
    Ti.at<float>(0, 1) = -param_initial[1];
    Ti.at<float>(0, 2) = param_initial[2];
    Ti.at<float>(1, 0) = param_initial[1];
    Ti.at<float>(1, 1) = param_initial[0];
    Ti.at<float>(1, 2) = param_initial[3];
    Ti.at<float>(2, 0) = 0;
    Ti.at<float>(2, 1) = 0;
    Ti.at<float>(2, 2) = 1;

    // Defines T_final
    Tf.at<float>(0, 0) = param_final[0];
    Tf.at<float>(0, 1) = -param_final[1];
    Tf.at<float>(0, 2) = param_final[2];
    Tf.at<float>(1, 0) = param_final[1];
    Tf.at<float>(1, 1) = param_final[0];
    Tf.at<float>(1, 2) = param_final[3];

    // Achando transf inversa
    Tr = Tf*Ti.inv();

    // Copiar resultado no vetor transf_update
    transf_update[0] = Tr.at<float>(0, 0);
    transf_update[1] = Tr.at<float>(0, 1);
    transf_update[2] = Tr.at<float>(0, 2);
    transf_update[3] = Tr.at<float>(1, 0);
    transf_update[4] = Tr.at<float>(1, 1);
    transf_update[5] = Tr.at<float>(1, 2);
}

void	Tracker::LoadParameters()
{
    cv::FileStorage fs("../settings/parameters.yml", cv::FileStorage::READ);

    size_template_x = fs["size_template_x"];
    size_template_y = fs["size_template_y"];
    n_ctrl_pts_x_i = fs["n_ctrl_pts_x_i"];
    n_ctrl_pts_y_i = fs["n_ctrl_pts_y_i"];
    size_bins = fs["size_bins"];
    epsilon = fs["epsilon"];
    n_bins = fs["n_bins"];
    n_max_iters = fs["n_max_iters"];
    confidence_thres = fs["confidence_thres"];
    percentage_active = fs["percentage_active"];
    visibility_thres = fs["visibility_thres"];
    rotation_thres = fs["rotation_thres"];
}

Tracker::~Tracker()
{
}
