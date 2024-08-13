#ifndef CONFIG_H_
#define CONFIG_H_

#include <cv.h>
#include <vector>

#define METHOD_FAST "fast"
#define METHOD_MSER "mser"
#define METHOD_ORB "orb"
#define METHOD_BRISK "brisk"
#define METHOD_SIFT "sift"
#define METHOD_SURF "surf"
#define METHOD_FLANN "flann"
#define METHOD_BRUTE_FORCE "brute"
#define METHOD_FERNS "ferns"

#define BLACK cv::Scalar(0, 0, 0)
#define BLUE cv::Scalar(255, 0, 0)
#define GREEN cv::Scalar(0, 255, 0)
#define RED cv::Scalar(0, 0, 255)
#define WHITE cv::Scalar(255, 255, 255)

typedef std::vector<cv::DMatch> MatchVector;
typedef std::vector<cv::KeyPoint> KeyPointVector;

#endif
