#include "Detector.h"

int Detector::Initialize(cv::Mat *input, cv::Mat *input_mask)
{
    printf("> Initializing Feature Detector \n");

    // I/O
    ICur = input;
    Input_mask = input_mask;

    // Load tracking parameters
    LoadParameters();

    // Initialize structures
    //ICur_gray.create(input->rows, input->cols, CV_8UC1);

    Tr.create(2, 3, CV_32FC1);
    Tf.create(2, 3, CV_32FC1);
    Ti.create(3, 3, CV_32FC1);
    Feat.create(3, 1, CV_32FC1);
    FeatW.create(2, 1, CV_32FC1);
    Feat.at<float>(2, 0) = 1;

    target_keypoint_storage.create(max_features, 4, CV_32FC1);
    target_descriptor_storage.create(max_features, 128, CV_32FC1);

    matches = new int [max_features*2];

    // Clock init
    tick = cvGetTickFrequency();

    // Variables from initializer
    flag_start = &(control_internal->flag_start);
    flag_reset = &(control_internal->flag_reset);
    flag_tracking = &(control_internal->flag_tracking);
    n_ref_features = &control_internal->n_ref_features;

    // RANSAC
    // transform.Initialize3DOF(max_features, d_ransac_iterations, d_ransac_inliers, d_ransac_error,
    //                                                    d_ransac_min_consensus);
    transform.Initialize4DOF(max_features, d_ransac_iterations, d_ransac_inliers, d_ransac_error,
                                                     d_ransac_min_consensus);

    // Initializing misterious OpenCV fct
    cv::initModule_nonfree();

    return 0;
}

int Detector::ProcessWithNCC() {
    // Main Program
    #pragma omp master
    {
        // System is active
        if(*flag_start) {
            // In case of tracking fail...
            if(!*flag_tracking && !*flag_reset) {
                cv::Point roi_center(control_internal->coords[0], control_internal->coords[1]);
                cv::Size roi_size(size_template_x, size_template_y);
                cv::Rect roi = region_of_interest(roi_center, roi_size, ICur->size());
                cv::Point most_similar_position = find_most_similar_template(cv::Mat(*ICur, roi),
                cv::Mat(*Input_mask, roi));
                control_internal->RestartTracking(most_similar_position);

                // New flag to avoid using backward tracking when template is re-initialized
                control_internal->flag_hold = 1;
            }
        }
    }
    return 1;
}

int Detector::Process()
{
    // Main Program
    #pragma omp master
    {
        // System is active
        if(*flag_start)
        {
            // In case of tracking fail...
            if(!*flag_tracking && !*flag_reset)
            {
                // Starts clock
                clock_start = (double) cvGetTickCount();

                // Convert image to grayscale
                //cv::cvtColor(*ICur, ICur_gray, CV_RGB2GRAY);

                // Detecting features & creating descriptors from current image
                detect_features_target();

                // Matching features with those from the mosaic
                if(*n_ref_features > 4 && keypoints.size() > 4) {
                    // Get pointers from control (due to sync issues, I have to do this at runtime)
                    ref_descriptor_storage = control_internal->ref_descriptor_storage;
                    ref_keypoint_storage = control_internal->ref_keypoint_storage;

                    // Generating feature matches
                    generate_matches_flann();

                    // Find transformation between current image and mosaic using RANSAC
                    if(n_matches > 4)
                    {
                        // Runs detection
                        //transform.Run3DOF(ref_keypoint_storage, target_keypoint_storage, n_matches, matches);
                        //
                        //// Copying results into transformation vector from control filter
                        //control_internal->mosaic_coords[0] = std::cos(transform.estim_transf_3dof[0]);
                        //control_internal->mosaic_coords[1] = std::sin(transform.estim_transf_3dof[0]);
                        //control_internal->mosaic_coords[2] = transform.estim_transf_3dof[1];
                        //control_internal->mosaic_coords[3] = transform.estim_transf_3dof[2];

                        transform.Run4DOF(ref_keypoint_storage, target_keypoint_storage, n_matches, matches, 1.6f, 0.7f, 0.4f);

                        // Copying results into transformation vector from control filter
                        //float scale = std::sqrt(std::pow(transform.estim_transf_4dof[0],2) + transform.estim_transf_4dof[1]);
                        control_internal->mosaic_coords[0] = transform.estim_transf_4dof[0];
                        control_internal->mosaic_coords[1] = transform.estim_transf_4dof[1];
                        control_internal->mosaic_coords[2] = transform.estim_transf_4dof[2];
                        control_internal->mosaic_coords[3] = transform.estim_transf_4dof[3];

                        // Detector found something
                        control_internal->RestartTracking();

                        //*flag_detected = 1;

                        // Displays estimated transformation (Mosaic origin)
                        /*CvPoint src_corners[4] = {{-size_template_x/2, -size_template_y/2}, {size_template_x/2, -size_template_y/2}, {size_template_x/2, size_template_y/2}, {-size_template_x/2, size_template_y/2}};
                        CvPoint dst_corners[4];

                        for(int i = 0; i < 4; i++ )
                        {
                            double X = src_corners[i].x + transform.estim_transf_2dof[0];
                            double Y = src_corners[i].y + transform.estim_transf_2dof[1];
                            dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
                        }

                        for(int i = 0; i < 4; i++)
                        {
                            CvPoint r1 = dst_corners[i%4];
                            CvPoint r2 = dst_corners[(i+1)%4];
                            cvLine(input->IplImageRef(0), cvPoint(r1.x, r1.y), cvPoint(r2.x, r2.y), CV_RGB( 255, 255, 0 ) );
                        }*/
                    }
                    else
                    {
                        // In case not many feature matches are found, do nothing
                        printf("Not enough matches found!\n");
                    }
                }
                else
                {
                    // Not many features available on reference/mosaic
                    printf("Only %d feats on mosaic... %d feats on current img!\n", *n_ref_features, (int)keypoints.size());
                }

                // Clock stop - How long did it take?
                clock_stop = ((double) cvGetTickCount() - clock_start) / (1000*tick);
                printf("\n> Feat Detection - Elapsed time: %f ms                                                             \r", clock_stop);

            } // end *flag_tracking
        } // end *flag_start
    } // end onsinglethread

    return 1;
}

void Detector::detect_features_target()
{
    // This is gonna hold the descriptors ( I don't know how to optimize this.. )
    cv::Mat descriptors;

    //Detecting features on current image
    keypoints = keypoint_handler.extract_keypoints(*ICur, *Input_mask, METHOD_SURF);
    control_internal->detected_keypoints = keypoints; //TODO remove
    descriptors = keypoint_handler.generate_descriptors(*ICur, keypoints, METHOD_SURF);

    // Loop through detected features
    for(int i = 0; i < (int)keypoints.size() && i < max_features; i++)
    {
        // Handling keypoints
        cv::KeyPoint keypoint = keypoints[i];
        target_keypoint_storage.at<float>(i, 0) = keypoint.pt.x;
        target_keypoint_storage.at<float>(i, 1) = keypoint.pt.y;
        target_keypoint_storage.at<float>(i, 2) = keypoint.size;

        // Copies keypoints into storage matrix
        // Getting pointer to descriptor matrices
        float *ptr1 = target_descriptor_storage.ptr<float>(i);
        float *ptr2 = descriptors.ptr<float>(i);

        // Copying to descriptor matrix
        memcpy(ptr1, ptr2, 128 * sizeof(float));
    }
}

void Detector::generate_matches_flann(void)
{
    // Defining ROI in matrices
    cv::Mat ref_d = ref_descriptor_storage(cv::Rect(0, 0, 128, *n_ref_features));
    cv::Mat target_d = target_descriptor_storage(cv::Rect(0, 0, 128, MIN(keypoints.size(), max_features)));

    // find nearest neighbors using FLANN
    cv::Mat m_indices((*n_ref_features), 2, CV_32S);
    cv::Mat m_dists((*n_ref_features), 2, CV_32F);

    cv::flann::Index flann_index(target_d, cv::flann::KDTreeIndexParams(4)); // using 4 randomized kdtrees
    flann_index.knnSearch(ref_d, m_indices, m_dists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked

    n_matches = 0;    // Important! resetting number of matches

    int* indices_ptr = m_indices.ptr<int>(0);
    float* dists_ptr = m_dists.ptr<float>(0);

    for(int i=0;i<m_indices.rows;++i)
        if(dists_ptr[2*i]<flann_thres*dists_ptr[2*i+1])
        {
            matches[2*n_matches] = i;
            matches[1+2*n_matches] = indices_ptr[2*i];
            n_matches++;
        }
}

cv::Point Detector::find_most_similar_template(cv::Mat reference, cv::Mat reference_mask)
{
    cv::Mat little_reference, little_mask, little_template, little_template_mask;
    cv::Size little_size(reference.cols * ncc_scale, reference.rows * ncc_scale);
    resize(reference, little_reference, little_size);
    resize(reference_mask, little_mask, little_size);
    cv::Point result;
    float similarity, max_similarity = 0;
    cv::Mat* _template, *mask;
    for(int row = 0; row < control_internal->grid_y; row++)
        for(int col = 0; col < control_internal->grid_x; col++)
            if(control_internal->Visibility_map.at<uchar>(row, col))
            {
                int array_index = col * control_internal->grid_y + row;
                _template = &control_internal->Template_set[array_index];
                mask = &control_internal->Mask_set[array_index];
                resize(*_template, little_template, little_size);
                resize(*mask, little_template_mask, little_size);
                similarity = ncc_similarity(little_reference, little_mask,
                                            little_template, little_template_mask);
                if(similarity > max_similarity)
                {
                    max_similarity = similarity;
                    result.x = col;
                    result.y = row;
                }
            }
    return result;
}

float Detector::ncc_similarity(cv::Mat image1, cv::Mat mask1, cv::Mat image2, cv::Mat mask2)
{
    int counter = 0;
    int I1_bar[3] = {0}, I2_bar[3] = {0}, I1sq[3] = {0}, I2sq[3] = {0};
    long erro[3] = {0};
    float ncc = 0, stds[3] = {0};

    for(int k = 1; k < 3; k++)
    {
        for(int row = 0; row < image1.rows; row++)
        {
            for(int col = 0; col < image1.cols; col++)
            {
                if(mask1.at<uchar>(row, col) == 255 && mask2.at<uchar>(row, col) == 255)
                {
                    I1_bar[k] += (int)image1.ptr<uchar>(row)[3*col+k];
                    I2_bar[k] += (int)image2.ptr<uchar>(row)[3*col+k];
                    counter++;
                }
            }
        }

        if(counter < 1)
        {
            std::cout << "No active pixels to compute NCC!!\n";
            counter = 1;
        }

        I1_bar[k] = I1_bar[k] / (counter);
        I2_bar[k] = I2_bar[k] / (counter);
    }

    for(int k = 1; k < 3; k++)
    {
        for(int i = 0; i < image2.rows; i++)
        {
            for(int j = 0; j < image2.cols; j++)
            {
                if(mask1.at<uchar>(i, j) == 255 && mask2.at<uchar>(i, j) == 255)
                {
                    I1sq[k] += (image1.ptr<uchar>(i)[3*j+k] - I1_bar[k]) *
                                         (image1.ptr<uchar>(i)[3*j+k] - I1_bar[k]);
                    I2sq[k] += (image2.ptr<uchar>(i)[3*j+k] - I2_bar[k]) *
                                         (image2.ptr<uchar>(i)[3*j+k] - I2_bar[k]);
                    erro[k] += (image1.ptr<uchar>(i)[3*j+k] - I1_bar[k]) *
                                         (image2.ptr<uchar>(i)[3*j+k] - I2_bar[k]);
                }
            }
        }

        stds[k] = std::sqrt((float)I1sq[k] * (float)I2sq[k]);
    }

    for(int k = 1; k < 3; k++)
        if(stds[k] > 0)
            ncc += (float)erro[k] / stds[k];

    if(ncc < 0)
    {
        printf("Something went wrong when computing NCC coef! \n");
        return 0;
    }

    return ncc / 2;
}

cv::Rect Detector::region_of_interest(cv::Point center, cv::Size size, cv::Size image_size)
{
    cv::Point top_left(center.x - size.width / 2, center.y - size.height / 2);
    cv::Rect result(top_left, size);
    if(result.x < 0)
        result.x = 0;
    if(result.y < 0)
        result.y = 0;
    if(result.br().x > image_size.width)
        result.x = image_size.width - size.width;
    if(result.br().y > image_size.height)
        result.y = image_size.height - size.height;
    return result;
}

void Detector::LoadParameters()
{
    cv::FileStorage fs("../settings/parameters.yml", cv::FileStorage::READ);

    ncc_scale = fs["ncc_scale"];
    size_template_x = fs["size_template_x"];
    size_template_y = fs["size_template_y"];
    max_features = fs["max_features"];
    flann_thres = fs["flann_thres"];
    d_ransac_iterations = fs["d_ransac_iterations"];
    d_ransac_inliers = fs["d_ransac_inliers"];
    d_ransac_error = fs["d_ransac_error"];
    d_ransac_min_consensus = fs["d_ransac_min_consensus"];
    ft_param1 = fs["ft_param1"];
    ft_param2 = fs["ft_param2"];
    ft_param3 = fs["ft_param3"];
    ft_param4 = fs["ft_param4"];
}
