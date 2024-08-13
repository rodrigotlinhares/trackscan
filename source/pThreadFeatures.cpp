#include "pThreadStructs.h"
#include "unistd.h"



void ComputeTransform(cv::Mat *Ti, cv::Mat *Tf, cv::Mat *Tr, float *tracking_param)
{
    // The goal here is to find the transformation that aligns the current image ICur to the mosaic coordinates.
    // This tranformation is the inverse of the mosaic coordinates
    // This function finds Tx, where T_final = Tx*T_initial

    Ti->at<float>(0, 0) = tracking_param[0];
    Ti->at<float>(0, 1) = -tracking_param[1];
    Ti->at<float>(0, 2) = tracking_param[2];
    Ti->at<float>(1, 0) = tracking_param[1];
    Ti->at<float>(1, 1) = tracking_param[0];
    Ti->at<float>(1, 2) = tracking_param[3];
    Ti->at<float>(2, 0) = 0;
    Ti->at<float>(2, 1) = 0;
    Ti->at<float>(2, 2) = 1;

    // Defines T_final
    Tf->at<float>(0, 0) = 1;
    Tf->at<float>(0, 1) = 0;
    Tf->at<float>(0, 2) = 0;
    Tf->at<float>(1, 0) = 0;
    Tf->at<float>(1, 1) = 1;
    Tf->at<float>(1, 2) = 0;

    // Achando transf inversa
    *Tr = *Tf*Ti->inv();
}

void DetectFeaturesTemplate(cv::Mat *ICur,
                            cv::Mat *Input_mask,
                            std::vector<cv::KeyPoint> *keypoints,
                            cv::Mat *Ti,
                            cv::Mat *Tf,
                            cv::Mat *Tr,
                            float *tracking_param,
                            int *n_ref_features,
                            int max_features,
                            cv::Mat *Feat,
                            cv::Mat *FeatW,
                            int size_template_x,
                            int size_template_y,
                            int min_feat_distance,
                            cv::Mat *ref_keypoint_storage,
                            cv::Mat *ref_descriptor_storage,
                            KeyPointHandler* keypoint_handler,
                            int *active_template,
                            int grid_x,
                            int grid_y,
                            int offset_templates)
{
    // This is gonna hold the descriptors ( I don't know yet how to optimize this.. )
    cv::Mat descriptors;

    //Detecting features on current image
    *keypoints = keypoint_handler->extract_keypoints(*ICur, *Input_mask, METHOD_SURF);
    descriptors = keypoint_handler->generate_descriptors(*ICur, *keypoints, METHOD_SURF);

    // Finding inverse of current transformation
    ComputeTransform(Ti, Tf, Tr, tracking_param);

    // Loop through detected features
    for(int i = 0; i< (int)keypoints->size() && (*n_ref_features) < max_features; i++)
    {
        // Transforming detected keypoints into mosaic coords...
        Feat->at<float>(0,0) = (*keypoints)[i].pt.x;
        Feat->at<float>(1,0) = (*keypoints)[i].pt.y;
        *FeatW = (*Tr)*(*Feat);

        // Test if current feature is within template boundaries
        if(FeatW->at<float>(0, 0) > -size_template_x/2 && FeatW->at<float>(0, 0) < size_template_x/2 &&
            FeatW->at<float>(1, 0) > -size_template_y/2 && FeatW->at<float>(1, 0) < size_template_y/2)
        {
            ref_keypoint_storage->at<float>(*n_ref_features, 0) = FeatW->at<float>(0, 0) + (float)(active_template[0]-cvFloor(grid_x/2))*offset_templates;
            ref_keypoint_storage->at<float>(*n_ref_features, 1) = FeatW->at<float>(1, 0) + (float)(active_template[1]-cvFloor(grid_y/2))*offset_templates;
            ref_keypoint_storage->at<float>(*n_ref_features, 2) = ((*keypoints)[i]).size;

            // Checking if new feature is really new
            bool is_valid_new_feature = true;

            for(int j=0;j<*n_ref_features;j++)
            {
                //comparar feature atual com todas outras da storage
                float distance = pow(ref_keypoint_storage->at<float>(*n_ref_features, 0) - ref_keypoint_storage->at<float>(j, 0), 2.0f)
                               + pow(ref_keypoint_storage->at<float>(*n_ref_features, 1) - ref_keypoint_storage->at<float>(j, 1), 2.0f);

                // Here I compare feature distance, size and laplacian values
                if(pow(distance, 0.5f) < min_feat_distance && abs(ref_keypoint_storage->at<float>(*n_ref_features, 2) - ref_keypoint_storage->at<float>(j, 2)) < 15 )
                    is_valid_new_feature = false;
            }

            // Another valid feature detected, increase counter
            if(is_valid_new_feature)
            {
                // Getting pointer to descriptor matrices
                float *ptr1 = ref_descriptor_storage->ptr<float>(*n_ref_features);
                float *ptr2 = descriptors.ptr<float>(i);

                //// Copying to descriptor matrix
                memcpy(ptr1, ptr2, 128*sizeof(float));

                // Incrementing number of detected features
                (*n_ref_features)++;
            }
        }
    }
}



// Processing function
void*    ThreadedProcess(void *param_in)
{
    // I/O
    ThreadPointersFeatureDetector *param = (ThreadPointersFeatureDetector*) param_in;

    // While system is running
    while(*(param->flag_running) == 1)
    {
        // New template was chosen by user, redefine bag of features...
        if(*(param->flag_process))
        {
            *(param->flag_process) = 0;

            // Detecting features
            DetectFeaturesTemplate(param->ICur,
                                   param->Input_mask,
                                   param->keypoints,
                                   param->Ti,
                                   param->Tf,
                                   param->Tr,
                                   param->tracking_param,
                                   param->n_ref_features,
                                   param->max_features,
                                   param->Feat,
                                   param->FeatW,
                                   param->size_template_x,
                                   param->size_template_y,
                                   param->min_feat_distance,
                                   param->ref_keypoint_storage,
                                   param->ref_descriptor_storage,
                                   param->keypoint_handler,
                                   param->active_template,
                                   param->grid_x,
                                   param->grid_y,
                                   param->offset_templates);

            // Display
            printf("> Currently %d features in Mosaic\n", *param->n_ref_features);
        }

        // Sleep
        usleep(100000);
    }

    std::cout << "Feature detection thread will now terminate" << std::endl;
}



