#ifndef KEYPOINTHANDLER_H_
#define KEYPOINTHANDLER_H_

#include <cv.h>
#include <highgui.h>
#include <opencv2/nonfree/features2d.hpp> // Workaround. Without this, OpenCV can't find SURF.
#include "config.h"
#include "ferns/mcv.h"
#include "ferns/planar_pattern_detector_builder.h"
#include "ferns/template_matching_based_tracker.h"

class KeyPointHandler {
public:
  ~KeyPointHandler();
  //TODO These methods were separated for testing purposes. Merge them.
  KeyPointVector extract_keypoints(cv::Mat image, cv::Mat mask, std::string method);
  cv::Mat generate_descriptors(cv::Mat image,KeyPointVector key_points, std::string method);

private:
  planar_pattern_detector* detector;
  void ferns_detection(IplImage* image, std::vector<cv::KeyPoint>& keypoints);
};

#endif
