#ifndef _Detector_h
#define _Detector_h

#include "Control.h"
#include "ROIFinder.h"
#include "macros.h"
#include "RANSAC.h"
#include "opencv2/nonfree/nonfree.hpp"
#include "keypoint_handler.h"
#include "config.h"

class Detector
{
public:
    int Initialize(cv::Mat *input, cv::Mat *input_mask);
    int Process();
    int ProcessWithNCC(); //TODO remove

    Control *control_internal;
    int *flag_tracking, *flag_reset, *flag_start;
    int hessianThreshold;

private:
    void detect_features_target();
    void generate_matches();
    void generate_matches_flann();
    void compute_transform();
    void LoadParameters();
    float ncc_similarity(cv::Mat image1, cv::Mat mask1, cv::Mat image2, cv::Mat mask2);
    cv::Point find_most_similar_template(cv::Mat reference, cv::Mat reference_mask);
    cv::Rect region_of_interest(cv::Point center, cv::Size size, cv::Size image_size);

    // Clock
    double clock_start, clock_stop, tick;

    // Working image
    cv::Mat *ICur, ICur_gray, *Input_mask;

    int size_template_x, size_template_y, max_features;

    KeyPointHandler keypoint_handler;
    KeyPointVector keypoints;

    float ncc_scale;
    float ft_param1, ft_param2, ft_param3, ft_param4; //TODO remove

    // Aux matrices
    int n_matches, *n_ref_features, *matches;
    int d_ransac_iterations, d_ransac_inliers, d_ransac_error, d_ransac_min_consensus;
    float flann_thres;

    cv::Mat target_keypoint_storage, ref_keypoint_storage;
    cv::Mat target_descriptor_storage, ref_descriptor_storage;
    cv::Mat Ti, Tf, Tr, Feat, FeatW;

    // RANSAC class
    RANSAC    transform;
};

#endif 
