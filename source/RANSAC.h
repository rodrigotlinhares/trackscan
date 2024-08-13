#ifndef RANSAC_H
#define RANSAC_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

class RANSAC
{

public:

    void	Initialize2DOF(int input_maxfeatures, int iterations, int inliers, int error, int consensus);

	void	Initialize3DOF(int input_maxfeatures, int iterations, int inliers, int error, int consensus);

	void	Initialize4DOF(int input_maxfeatures, int iterations, int inliers, int error, int consensus);

	
	void	Run2DOF(cv::Mat sourceKeypoints, cv::Mat targetKeypoints, int n_matches, int *matches);

	void	Run3DOF(cv::Mat sourceKeypoints, cv::Mat targetKeypoints, int n_matches, int *matches);

	void	Run4DOF(cv::Mat sourceKeypoints, cv::Mat targetKeypoints, int n_matches, int *matches, float max_scale, float min_scale, float max_angle);
	

	void	Deallocate(void);

	float   estim_transf_2dof[2], estim_transf_3dof[3], estim_transf_4dof[4], estim_transf_6dof[6], estim_transf_12dof[12];

	// In this code I assume 
	// a = estim_transf_4dof[0]
	// b = estim_transf_4dof[1]
	// c = estim_transf_4dof[2]
	// d = estim_transf_4dof[3]
	//
	// where
	// T = [a -b c
	//		b  a d]
	//
	// and for the 12dof case
	//
	// T = [estim_transf_12dof[0] estim_transf_12dof[1] estim_transf_12dof[2] ...]

    
private:

	bool *pair_chosen;

	int max_features;
	
	int count;

	int numRANSACIterations; 

	int numRANSACInliers;

	int numRANSACError; 

	int numRANSACMinConsensusSet; 

	double bestTransformationError;
	
	double currentTransformation_2dof[2], bestTransformation_2dof[2];
	
	double currentTransformation_3dof[3], bestTransformation_3dof[3];

	double currentTransformation_4dof[6], bestTransformation_4dof[6];
	
	cv::Mat A, B, delta;
};

#endif
