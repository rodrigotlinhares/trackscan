#ifndef MOSAIC_H
#define MOSAIC_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

class MOSAIC
{

public:

    void	Initialize(int size_template_x, int size_template_y, int grid_x, int grid_y, int offset_templates, cv::Mat *Template_set, cv::Mat *Mask_set, cv::Mat *Visibility_map);
    
    void	Process();
	
    void	Process(cv::Mat Template_set[], cv::Mat Mask_set[], cv::Mat Visibility_map);
	
    void	PrintMosaic(int current);
	
	void	ResetMosaic();

	void	Deallocate(void);

private:

	bool	switch_buffer;

    int grid_x,
        grid_y,
        size_template_x,
        size_template_y,
        offset_templates;

    cv::Mat GKernel,
            Weights,
            Acc_red,
            Acc_green,
            Acc_blue;

    cv::Mat *Template_set,
            *Mask_set,
            *New_visibility_map;

public:

    cv::Mat Mosaic,
            Mosaic_mask,
            Visibility_map;

};

#endif
