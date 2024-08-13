#include "RANSAC.h"

/********************/
/*** RANSAC class ***/
/********************/


// Initialization

void RANSAC::Initialize2DOF(int input_maxfeatures, int iterations, int inliers, int error, int consensus)
{
	pair_chosen = new bool [input_maxfeatures];

	// Initialize estimated transformations
	estim_transf_2dof[0] = 0;
	estim_transf_2dof[1] = 0;
	
	// Input parameters
	numRANSACIterations = iterations; // 1000 iterations for RANSAC
	numRANSACInliers = inliers; // 4 inliers
	numRANSACError = error; // 10 pixels
	numRANSACMinConsensusSet = consensus; // at least 2 matching pts (10pixels max difference)

	// 2 dof
	A.create(2*numRANSACInliers, 2, CV_32FC1);
	B.create(2*numRANSACInliers, 1, CV_32FC1);
	delta.create(2, 1, CV_32FC1);
}

void RANSAC::Initialize3DOF(int input_maxfeatures, int iterations, int inliers, int error, int consensus)
{
	pair_chosen = new bool [input_maxfeatures];

	// Initialize estimated transformations
	estim_transf_3dof[0] = 0;
	estim_transf_3dof[1] = 0;
	estim_transf_3dof[2] = 0;
	
	// Input parameters
	numRANSACIterations = iterations; // 1000 iterations for RANSAC
	numRANSACInliers = inliers; // 4 inliers
	numRANSACError = error; // 10 pixels
	numRANSACMinConsensusSet = consensus; // at least 2 matching pts (10pixels max difference)

	// 3 dof
	A.create(2*numRANSACInliers, 3, CV_32FC1);
	B.create(2*numRANSACInliers, 1, CV_32FC1);
	delta.create(3, 1, CV_32FC1);
}

void RANSAC::Initialize4DOF(int input_maxfeatures, int iterations, int inliers, int error, int consensus)
{
	pair_chosen = new bool [input_maxfeatures];

	// Initialize estimated transformations
	estim_transf_4dof[0] = 1;
	estim_transf_4dof[1] = 0;
	estim_transf_4dof[2] = 0;
	estim_transf_4dof[3] = 0;
	
	// Input parameters
	numRANSACIterations = iterations; // 1000 iterations for RANSAC
	numRANSACInliers = inliers; // 4 inliers
	numRANSACError = error; // 10 pixels
	numRANSACMinConsensusSet = consensus; // at least 2 matching pts (10pixels max difference)

	// 4 dof
	A.create(2*numRANSACInliers, 4, CV_32FC1);
	B.create(2*numRANSACInliers, 1, CV_32FC1);
	delta.create(4, 1, CV_32FC1);
}


// Deconstructor

void	RANSAC::Deallocate(void)
{
	delete pair_chosen;
}


// Runs RANSAC 

void RANSAC::Run2DOF(cv::Mat sourceKeypoints, cv::Mat targetKeypoints, int n_matches, int *matches) 
{
	bool flag_go = true;

	// Reset params
	bestTransformationError = 1e10;

	// Seed random number
	srand( (int) time(0) ); 
	
	// RANSAC start
	if(n_matches > 3)
	{
		// If we have enough matches (>2), start RANSAC estimation
		for(int i=0; i < numRANSACIterations; i++)
		{
			// Generate set of non-repeated integers
			count = 0;

			// Set flags to zero
			memset(pair_chosen, 0, n_matches*sizeof(bool)); 

			while(count < numRANSACInliers)
			{
				int pair = rand() % (n_matches); // range 0 to n_matches (exclusive)

				// Only pick pairs I haven't used before (avoid repetition)...
				if(pair_chosen[pair] == false)
				{					
					// Update bool array
					pair_chosen[pair] = true;
															
					A.at<float>(count*2,0) = 1;
					A.at<float>(count*2,1) = 0;

					A.at<float>(count*2+1,0) = 0;
					A.at<float>(count*2+1,1) = 1;

					B.at<float>(count*2,0) = targetKeypoints.at<float>(matches[pair*2+1], 0);
					B.at<float>(count*2+1,0) = targetKeypoints.at<float>(matches[pair*2+1], 1);

					// increment counter since we have chosen another pair
					count++; 
				}
			}

			// Solve Ax = B ...  
			delta = (A.t()*A).inv(CV_LU)*(A.t()*B);

			// Construct Transformation
			currentTransformation_2dof[0] = delta.at<float>(0,0);
			currentTransformation_2dof[1] = delta.at<float>(1,0);

			// Testing consistency of random transformation
			flag_go = true;
			/*float scale = (float) (delta.at<float>(0,0)*delta.at<float>(0,0) + delta.at<float>(1,0)*delta.at<float>(1,0));
			float angle = (float) abs(atan2(delta.at<float>(1,0),delta.at<float>(0,0)));

			if( scale > max_scale || scale < min_scale || angle > max_angle) 
				flag_go = false;*/

			// Calculate projection error if transformation is consistent
			if(flag_go)
			{			
				int consensus = 0;
				double totalError = 0;

				for(int j = 0; j < n_matches; j++)
				{
					float ref[2], cur[2];

					ref[0] = sourceKeypoints.at<float>(matches[j*2], 0);
					ref[1] = sourceKeypoints.at<float>(matches[j*2], 1);

					cur[0] = targetKeypoints.at<float>(matches[j*2+1], 0);
					cur[1] = targetKeypoints.at<float>(matches[j*2+1], 1);

					double x = currentTransformation_2dof[0] - (double)cur[0];
					double y = currentTransformation_2dof[1] - (double)cur[1];
					double error = sqrt(x*x + y*y);

					totalError += error;

					if(error < numRANSACError)
						consensus++; // this item agrees with transformation within error bounds
				}
				
				//printf("%d %d %f\n", n_matches, consensus, totalError);

				// Checks if consensus is bigger than the required minimum
				if(consensus > numRANSACMinConsensusSet)
				{
					// Check if we have a better transformation
					if(bestTransformationError > totalError)
					{ 
						// update bestTransformation_4dof and error
						bestTransformationError = totalError;

						for(int j = 0; j < 2; j++)
							bestTransformation_2dof[j] = currentTransformation_2dof[j];
					}
				}
			}
		}

		// Copying results into transformation array
		if(!(bestTransformation_2dof[0] == 0 && bestTransformation_2dof[1] == 0))
		{
			estim_transf_2dof[0] = (float) bestTransformation_2dof[0];
			estim_transf_2dof[1] = (float) bestTransformation_2dof[1];
		}

		//printf("Transformation T: %f %f %f %f \n", estim_transf_4dof[0], estim_transf_4dof[1], estim_transf_4dof[2], estim_transf_4dof[3]);
	}
}

void RANSAC::Run3DOF(cv::Mat sourceKeypoints, cv::Mat targetKeypoints, int n_matches, int *matches) 
{
	bool flag_go = true;

	// Reset params
	bestTransformationError = 1e10;

	// Seed random number
	srand( (int) time(0) ); 
	
	// RANSAC start
	if(n_matches > 3)
	{
		// If we have enough matches (>2), start RANSAC estimation
		for(int i=0; i < numRANSACIterations; i++)
		{
			// Generate set of non-repeated integers
			count = 0;

			// Set flags to zero
			memset(pair_chosen, 0, n_matches*sizeof(bool)); 

			while(count < numRANSACInliers)
			{
				int pair = rand() % (n_matches); // range 0 to n_matches (exclusive)

				// Only pick pairs I haven't used before (avoid repetition)...
				if(pair_chosen[pair] == false)
				{					
					// Update bool array
					pair_chosen[pair] = true;
					
					// Populate matrices
					float ref[2], cur[2];

					ref[0] = sourceKeypoints.at<float>(matches[pair*2], 0);
					ref[1] = sourceKeypoints.at<float>(matches[pair*2], 1);

					cur[0] = targetKeypoints.at<float>(matches[pair*2+1], 0);
					cur[1] = targetKeypoints.at<float>(matches[pair*2+1], 1);
										
					A.at<float>(count*2,0) = ref[0];
					A.at<float>(count*2,1) = 1;
					A.at<float>(count*2,2) = 0;

					A.at<float>(count*2+1,0) = ref[1];
					A.at<float>(count*2+1,1) = 0;
					A.at<float>(count*2+1,2) = 1;

					B.at<float>(count*2,0) = cur[0];
					B.at<float>(count*2+1,0) = cur[1];

					// increment counter since we have chosen another pair
					count++; 
				}
			}

			// Solve Ax = B ...  
			delta = (A.t()*A).inv(CV_LU)*(A.t()*B);

			// Construct Transformation
			currentTransformation_3dof[0] = delta.at<float>(0,0);
			currentTransformation_3dof[1] = delta.at<float>(1,0);
			currentTransformation_3dof[2] = delta.at<float>(2,0);
			
			// Testing consistency of random transformation
			flag_go = true;
			/*float scale = (float) (delta.at<float>(0,0)*delta.at<float>(0,0) + delta.at<float>(1,0)*delta.at<float>(1,0));
			float angle = (float) abs(atan2(delta.at<float>(1,0),delta.at<float>(0,0)));

			if( scale > max_scale || scale < min_scale || angle > max_angle) 
				flag_go = false;*/

			// Calculate Projection Error if transformation is consistent
			if(flag_go)
			{			
				int consensus = 0;
				double totalError = 0;

				for(int j = 0; j < n_matches; j++)
				{
					float ref[2], cur[2];

					ref[0] = sourceKeypoints.at<float>(matches[j*2], 0);
					ref[1] = sourceKeypoints.at<float>(matches[j*2], 1);

					cur[0] = targetKeypoints.at<float>(matches[j*2+1], 0);
					cur[1] = targetKeypoints.at<float>(matches[j*2+1], 1);

					float coseno = std::cos(currentTransformation_3dof[0]);
					float seno = std::sin(currentTransformation_3dof[0]);

					double x = coseno * ref[0] - seno * ref[1] + currentTransformation_3dof[1] - (double)cur[0];
					double y = seno * ref[0] + coseno * ref[1] + currentTransformation_3dof[2] - (double)cur[1];
					double error = sqrt(x*x + y*y);

					totalError += error;

					if(error < numRANSACError)
						consensus++; // this item agrees with transformation within error bounds
				}
				
				//printf("%d %d %f\n", n_matches, consensus, totalError);

				// Checks if consensus is bigger than the required minimum
				if(consensus > numRANSACMinConsensusSet)
				{
					// Check if we have a better transformation
					if(bestTransformationError > totalError)
					{ 
						// update bestTransformation and error
						bestTransformationError = totalError;

						for(int j = 0; j < 3; j++)
							bestTransformation_3dof[j] = currentTransformation_3dof[j];
					}
				}
			}
		}

		// Copying results into transformation array
		if(!(bestTransformation_3dof[0] == 0 && bestTransformation_3dof[2] == 0))
		{
			estim_transf_3dof[0] = (float) bestTransformation_3dof[0];
			estim_transf_3dof[1] = (float) bestTransformation_3dof[1];
			estim_transf_3dof[2] = (float) bestTransformation_3dof[2];
		}

		//printf("Transformation T: %f %f %f %f \n", estim_transf_4dof[0], estim_transf_4dof[1], estim_transf_4dof[2], estim_transf_4dof[3]);
	}
}

void RANSAC::Run4DOF(cv::Mat sourceKeypoints, cv::Mat targetKeypoints, int n_matches, int *matches, float max_scale, float min_scale, float max_angle) 
{
	bool flag_go = true;

	// Reset params
	bestTransformationError = 1e10;

	// Seed random number
	srand( (int) time(0) ); 
	
	// RANSAC start
	if(n_matches > 3)
	{
		// If we have enough matches (>2), start RANSAC estimation
		for(int i=0; i < numRANSACIterations; i++)
		{
			// Generate set of non-repeated integers
			count = 0;

			// Set flags to zero
			memset(pair_chosen, 0, n_matches*sizeof(bool)); 

			while(count < numRANSACInliers)
			{
				int pair = rand() % (n_matches); // range 0 to n_matches (exclusive)

				// Only pick pairs I haven't used before (avoid repetition)...
				if(pair_chosen[pair] == false)
				{					
					// Update bool array
					pair_chosen[pair] = true;
					
					// Populate matrices
					float ref[2], cur[2];

					ref[0] = sourceKeypoints.at<float>(matches[pair*2], 0);
					ref[1] = sourceKeypoints.at<float>(matches[pair*2], 1);

					cur[0] = targetKeypoints.at<float>(matches[pair*2+1], 0);
					cur[1] = targetKeypoints.at<float>(matches[pair*2+1], 1);
										
					A.at<float>(count*2,0) = ref[0];
					A.at<float>(count*2,1) = -ref[1];
					A.at<float>(count*2,2) = 1;
					A.at<float>(count*2,3) = 0;

					A.at<float>(count*2+1,0) = ref[1];
					A.at<float>(count*2+1,1) = ref[0];
					A.at<float>(count*2+1,2) = 0;
					A.at<float>(count*2+1,3) = 1;

					B.at<float>(count*2,0) = cur[0];
					B.at<float>(count*2+1,0) = cur[1];

					// increment counter since we have chosen another pair
					count++; 
				}
			}

			// Solve Ax = B ...  
			delta = (A.t()*A).inv(CV_LU)*(A.t()*B);

			// Construct Transformation
			currentTransformation_4dof[0] = delta.at<float>(0,0);
			currentTransformation_4dof[1] = -delta.at<float>(1,0);
			currentTransformation_4dof[2] = delta.at<float>(2,0);
			currentTransformation_4dof[3] = delta.at<float>(1,0);
			currentTransformation_4dof[4] = delta.at<float>(0,0);
			currentTransformation_4dof[5] = delta.at<float>(3,0);

			// Testing consistency of random transformation
			flag_go = true;
			float scale = (float) (delta.at<float>(0,0)*delta.at<float>(0,0) + delta.at<float>(1,0)*delta.at<float>(1,0));
			float angle = (float) abs(atan2(delta.at<float>(1,0),delta.at<float>(0,0)));

			if( scale > max_scale || scale < min_scale || angle > max_angle) 
				flag_go = false;

			// Calculate Projection Error if transformation is consistent
			if(flag_go)
			{			
				int consensus = 0;
				double totalError = 0;

				for(int j = 0; j < n_matches; j++)
				{
					float ref[2], cur[2];

					ref[0] = sourceKeypoints.at<float>(matches[j*2], 0);
					ref[1] = sourceKeypoints.at<float>(matches[j*2], 1);

					cur[0] = targetKeypoints.at<float>(matches[j*2+1], 0);
					cur[1] = targetKeypoints.at<float>(matches[j*2+1], 1);

					double x = currentTransformation_4dof[0] * ref[0] + currentTransformation_4dof[1] * ref[1] + currentTransformation_4dof[2] - (double)cur[0];
					double y = currentTransformation_4dof[3] * ref[0] + currentTransformation_4dof[4] * ref[1] + currentTransformation_4dof[5] - (double)cur[1];
					double error = sqrt(x*x + y*y);

					totalError += error;

					if(error < numRANSACError)
						consensus++; // this item agrees with transformation within error bounds
				}
				
				//printf("%d %d %f\n", n_matches, consensus, totalError);

				// Checks if consensus is bigger than the required minimum
				if(consensus > numRANSACMinConsensusSet)
				{
					// Check if we have a better transformation
					if(bestTransformationError > totalError)
					{ 
						// update bestTransformation_4dof and error
						bestTransformationError = totalError;

						for(int j = 0; j < 6; j++)
							bestTransformation_4dof[j] = currentTransformation_4dof[j];
					}
				}
			}
		}

		// Copying results into transformation array
		if(!(bestTransformation_4dof[0] == 0 && bestTransformation_4dof[2] == 0 && bestTransformation_4dof[3] == 0))
		{
			estim_transf_4dof[0] = (float) bestTransformation_4dof[0];
			estim_transf_4dof[1] = (float) bestTransformation_4dof[3];
			estim_transf_4dof[2] = (float) bestTransformation_4dof[2];
			estim_transf_4dof[3] = (float) bestTransformation_4dof[5];
		}

		//printf("Transformation T: %f %f %f %f \n", estim_transf_4dof[0], estim_transf_4dof[1], estim_transf_4dof[2], estim_transf_4dof[3]);
	}
}
