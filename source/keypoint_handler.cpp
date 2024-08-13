#include "keypoint_handler.h"

KeyPointHandler::~KeyPointHandler() {
  delete detector;
}

KeyPointVector KeyPointHandler::extract_keypoints(cv::Mat image, cv::Mat mask, std::string method) {
  //TODO test mask
  KeyPointVector result;
  cv::FileStorage fs("../settings/keypoint_handler.yml", cv::FileStorage::READ); //TODO magic string
  if(method.compare(METHOD_FAST) == 0)
    cv::FAST(image, result, fs["fast_threshold"]);

  else if(method.compare(METHOD_ORB) == 0) {
    cv::ORB orb(fs["orb_number_of_features"],
                fs["orb_scale_factor"],
                fs["orb_number_of_levels"],
                fs["orb_edge_threshold"],
                fs["orb_first_level"],
                fs["orb_wta_k"]);
    orb(image, cv::Mat(), result, cv::noArray());
  }

  else if(method.compare(METHOD_BRISK) == 0) {
    cv::BRISK brisk(fs["brisk_threshold"],
                    fs["brisk_octaves"],
                    fs["brisk_pattern_scale"]);
    cv::Mat gambi; // Workaround. cv::noArray() should work, but it doesn't.
    brisk(image, cv::Mat(), result, gambi);
  }

  else if(method.compare(METHOD_SIFT) == 0) {
    cv::SIFT sift(fs["sift_number_of_features"],
                  fs["sift_number_of_octave_layers"],
                  fs["sift_contrast_threshold"],
                  fs["sift_edge_threshold"],
                  fs["sift_sigma"]);
    sift(image, cv::Mat(), result, cv::noArray());
  }

  else if(method.compare(METHOD_SURF) == 0) {
    cv::SURF surf(fs["surf_hessian_threshold"],
                  fs["surf_number_of_octaves"],
                  fs["surf_number_of_octave_layers"]);
    surf(image, cv::Mat(), result, cv::noArray());
  }

  else if(method.compare(METHOD_FERNS) == 0) {
    cv::imwrite("/home/rodrigolinhares/Desktop/output.png", image);
    affine_transformation_range range;
    if(!detector) {
      detector = planar_pattern_detector_builder::build_with_cache("../src/ferns/model.bmp", //TODO change
                   &range,
                   400, // maxNumPointsOnModel
                   5000, // 5000, // numGeneratedImagesToFindStablePoints
                   0.4, // 0.0, // minNumViewsRate
                   32, 7, 4, // patchSize, yapeRadius, numOctaves
                   30, 12, // 30, 12, // numFerns, numTestPerFern
                   10000, 200); // 10000, 200); // numSamplesForRefinement, numSamplesForTest
    }
    cv::Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    IplImage ipl_image = gray_image;
    detector->detect(&ipl_image);
    for(int i = 0; i < detector->number_of_detected_points; i++)
      result.push_back(detector->detected_points[i].to_cv_keypoint());
  }
  return result;
}

cv::Mat KeyPointHandler::generate_descriptors(cv::Mat image, KeyPointVector key_points,
                                              std::string method) {
  cv::Mat result;
  if(method.compare(METHOD_ORB) == 0) {
    cv::ORB orb;
    orb(image, cv::Mat(), key_points, result, true);
  }

  else if(method.compare(METHOD_BRISK) == 0) {
    cv::BRISK brisk;
    brisk(image, cv::Mat(), key_points, result, true);
  }

  else if(method.compare(METHOD_SIFT) == 0) {
    cv::SIFT sift;
    sift(image, cv::Mat(), key_points, result, true);
  }

  else if(method.compare(METHOD_SURF) == 0) {
    cv::SURF surf;
    surf(image, cv::Mat(), key_points, result, true);
  }
  return result;
}
