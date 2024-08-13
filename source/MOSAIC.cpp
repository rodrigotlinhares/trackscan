#include "MOSAIC.h"

/********************/
/*** MOSAIC class ***/
/********************/


// Initialization

void	MOSAIC::Initialize(int size_template_x_, int size_template_y_, int grid_x_, int grid_y_, int offset_templates_, cv::Mat *Template_set_ptr, cv::Mat *Mask_set_ptr, cv::Mat *New_visibility_map_)
{
	// Inits
    this->size_template_x = size_template_x_;
    this->size_template_y = size_template_y_;
    this->grid_x = grid_x_;
    this->grid_y = grid_y_;
    this->offset_templates = offset_templates_;

	switch_buffer = 0;

    // Some more I/O
    Template_set = Template_set_ptr;
    Mask_set = Mask_set_ptr;
    this->New_visibility_map = New_visibility_map_;

	// Creating Gaussian kernel (for weighting each piece of the mosaic)
    cv::Mat kernelX = cv::getGaussianKernel(size_template_x_, size_template_x_/5.);
    cv::Mat kernelY = cv::getGaussianKernel(size_template_y_, size_template_y_/5.);

	GKernel = kernelY * kernelX.t(); 

	//for(int i=0; i<size_template_y; i++)
	//{
	//	for(int j=0; j<size_template_x; j++)
	//	{
	//		printf("%f ", GKernel.at<double>(i,j) ); getchar();
	//	}
	//	printf("\n");
	//}

	// Creating final image
    Visibility_map.create(grid_y_, grid_x_, CV_8UC1);
    Visibility_map.setTo(cv::Scalar(0));

    Mosaic.create((grid_y_-1)*offset_templates_ + size_template_y_, (grid_x_-1)*offset_templates_ + size_template_x_, CV_8UC3);
    Mosaic_mask.create((grid_y_-1)*offset_templates_ + size_template_y_, (grid_x_-1)*offset_templates_ + size_template_x_, CV_8UC1);
		
	Mosaic.setTo(0);
	Mosaic_mask.setTo(0);

    Weights.create((grid_y_-1)*offset_templates_ + size_template_y_, (grid_x_-1)*offset_templates_ + size_template_x_, CV_32FC1);
    Acc_red.create((grid_y_-1)*offset_templates_ + size_template_y_, (grid_x_-1)*offset_templates_ + size_template_x_, CV_32FC1);
    Acc_green.create((grid_y_-1)*offset_templates_ + size_template_y_, (grid_x_-1)*offset_templates_ + size_template_x_, CV_32FC1);
    Acc_blue.create((grid_y_-1)*offset_templates_ + size_template_y_, (grid_x_-1)*offset_templates_ + size_template_x_, CV_32FC1);
				
	// Zero weights and etc
	Weights.setTo(0);
	Acc_red.setTo(0);
	Acc_green.setTo(0);
	Acc_blue.setTo(0);
}

// Process

void	MOSAIC::Process()
{
	// Switch to unused buffer 
	//displayMosaic = &Mosaic[switch_buffer];
    //switch_buffer = !switch_buffer;

	// For all patches in mosaic
	for(int v=0; v<grid_x; v++)
	{
		for(int u=0; u<grid_y; u++)
		{
            if(Visibility_map.at<uchar>(u,v) == 0 && New_visibility_map->at<uchar>(u,v) == 1)
			{
				// Now set pointers to current template
				cv::Mat	Template = Template_set[u + grid_y*v];
				cv::Mat	Mask = Mask_set[u + grid_y*v];

				// Reading template
                for(int j=0; j<size_template_x; j++)
				{
                    for(int i=0; i<size_template_y; i++)
					{
						if(Mask.at<uchar>(i,j) != 0)
						{
                            Weights.at<float>(i + u*offset_templates, j + v*offset_templates) += static_cast<float>(GKernel.at<double>(i,j));
                            Acc_red.at<float>(i + u*offset_templates, j + v*offset_templates) += Template.ptr<uchar>(i)[3*j+2]*static_cast<float>(GKernel.at<double>(i,j));
                            Acc_green.at<float>(i + u*offset_templates, j + v*offset_templates) += Template.ptr<uchar>(i)[3*j+1]*static_cast<float>(GKernel.at<double>(i,j));
                            Acc_blue.at<float>(i + u*offset_templates, j + v*offset_templates) += Template.ptr<uchar>(i)[3*j]*static_cast<float>(GKernel.at<double>(i,j));
						}
					}
				}
			}
		}
	}

	// Now we'll get the resulting mosaic
		
	// Going through all patches in mosaic
	for(int v=0; v<grid_x; v++)
	{
		for(int u=0; u<grid_y; u++)
		{
            if(Visibility_map.at<uchar>(u,v) == 0 && New_visibility_map->at<uchar>(u,v) == 1)
            {
                for(int i=u*offset_templates; i<size_template_y + u*offset_templates; i++)
                {
                    for(int j=v*offset_templates; j<size_template_x + v*offset_templates; j++)
					{
						if(Weights.at<float>(i,j) > 0)
						{	
							// Updating mosaic image
                            Mosaic.ptr<uchar>(i)[3*j + 2] = static_cast<uchar>(cvFloor(Acc_red.at<float>(i,j)/Weights.at<float>(i,j)));
                            Mosaic.ptr<uchar>(i)[3*j + 1] = static_cast<uchar>(cvFloor(Acc_green.at<float>(i,j)/Weights.at<float>(i,j)));
                            Mosaic.ptr<uchar>(i)[3*j] =  static_cast<uchar>(cvFloor(Acc_blue.at<float>(i,j)/Weights.at<float>(i,j)));

							// Setting mosaic mask to 255
							Mosaic_mask.at<uchar>(i,j) = 255;
						}
					}
				}
			}
		}
	}

	// Copy new entries to local Visiblity_map
	for(int v=0; v<grid_x; v++)
	{
		for(int u=0; u<grid_y; u++)
		{
            if(Visibility_map.at<uchar>(u,v) == 0 && New_visibility_map->at<uchar>(u,v) == 1)
			{
				Visibility_map.at<uchar>(u,v) = 1;
			}
		}
	}

	// End Process
}


//void	MOSAIC::Process(std::vector<cv::Mat> Template_set_, std::vector<cv::Mat> Mask_set_, cv::Mat New_Visibility_map)
//{
//	// Switch to unused buffer
//	//displayMosaic = &Mosaic[switch_buffer];
//	//switch_buffer = !switch_buffer;

//	// For all patches in mosaic
//	int counter = 0;

//	for(int v=0; v<grid_x; v++)
//	{
//		for(int u=0; u<grid_y; u++)
//		{
//			if(Visibility_map.at<uchar>(u,v) == 0 && New_Visibility_map.at<uchar>(u,v) == 1)
//			{
//                std::cout << " got it.." << std::endl;

//				// Now set pointers to current template
//                cv::Mat	Template = Template_set_[counter];
//                cv::Mat	Mask = Mask_set_[counter];
//				counter ++;

//				// Reading template
//                for(int j=0; j<size_template_x; j++)
//				{
//                    for(int i=0; i<size_template_y; i++)
//					{
//						if(Mask.at<uchar>(i,j) != 0)
//						{
//                            Weights.at<float>(i + u*offset_templates, j + v*offset_templates) += static_cast<float>(GKernel.at<double>(i,j));
//                            Acc_red.at<float>(i + u*offset_templates, j + v*offset_templates) += Template.ptr<uchar>(i)[3*j+2]*static_cast<float>(GKernel.at<double>(i,j));
//                            Acc_green.at<float>(i + u*offset_templates, j + v*offset_templates) += Template.ptr<uchar>(i)[3*j+1]*static_cast<float>(GKernel.at<double>(i,j));
//                            Acc_blue.at<float>(i + u*offset_templates, j + v*offset_templates) += Template.ptr<uchar>(i)[3*j]*static_cast<float>(GKernel.at<double>(i,j));
//						}
//					}
//				}
//			}
//		}
//	}

//	// Now we'll get the resulting mosaic
		
//	// Going through all patches in mosaic
//	for(int v=0; v<grid_x; v++)
//	{
//		for(int u=0; u<grid_y; u++)
//		{
//			if(Visibility_map.at<uchar>(u,v) == 0 && New_Visibility_map.at<uchar>(u,v) == 1)
//            {
//                for(int i=u*offset_templates; i<size_template_y + u*offset_templates; i++)
//                {
//                    for(int j=v*offset_templates; j<size_template_x + v*offset_templates; j++)
//					{
//						if(Weights.at<float>(i,j) > 0)
//						{
//							// Updating mosaic image
//                            Mosaic.ptr<uchar>(i)[3*j + 2] = static_cast<uchar>(cvFloor(Acc_red.at<float>(i,j)/Weights.at<float>(i,j)));
//                            Mosaic.ptr<uchar>(i)[3*j + 1] = static_cast<uchar>(cvFloor(Acc_green.at<float>(i,j)/Weights.at<float>(i,j)));
//                            Mosaic.ptr<uchar>(i)[3*j] =  static_cast<uchar>(cvFloor(Acc_blue.at<float>(i,j)/Weights.at<float>(i,j)));

//							// Setting mosaic mask to 255
//							Mosaic_mask.at<uchar>(i,j) = 255;
//						}
//					}
//				}
//			}
//		}
//	}

//	// Copy new entries to local Visiblity_map
//	for(int v=0; v<grid_x; v++)
//	{
//		for(int u=0; u<grid_y; u++)
//		{
//			if(Visibility_map.at<uchar>(u,v) == 0 && New_Visibility_map.at<uchar>(u,v) == 1)
//			{
//				Visibility_map.at<uchar>(u,v) = 1;
//			}
//		}
//	}

//	// End Process
//}

void	MOSAIC::ResetMosaic()
{
	Mosaic.setTo(0);
	Mosaic_mask.setTo(0);
    Visibility_map.setTo(0);
	
	// Zero weights and etc
	Weights.setTo(0);
	Acc_red.setTo(0);
	Acc_green.setTo(0);
	Acc_blue.setTo(0);
}


void	MOSAIC::PrintMosaic(int counter)
{
    if(counter == -1)
    {
        cv::imwrite("../storage/refined_mosaic_final.png", Mosaic);
        return;
    }
        
    counter++;
    std::stringstream text;
    text << "../storage/refined_mosaic_" << counter << ".png";
    cv::imwrite(text.str(), Mosaic);
}

//void	MOSAIC::PrintMosaic(std::string fileName)
//{
//    cv::imwrite(fileName, Mosaic);
//}

// Deconstructor

void	MOSAIC::Deallocate(void)
{
}
