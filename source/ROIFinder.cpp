#include "ROIFinder.h"

/***************************/
/*** ROIFinder  ***/
/***************************/

ROIFinder::~ROIFinder()
{
}

int	 ROIFinder::Initialize(cv::Mat *input)
{
    printf("> Initializing AdvancedROIFinder \n");

    // Creating resized and rotated input image
#ifndef USE_VIDEO
        // Creating rotated input image
        image_big_rotated.create(input->cols, input->rows, CV_8UC3);

        // Creating rotated downsampled output image
        image_out.create(cvFloor(image_big_rotated.rows*SCALE_FACTOR), cvFloor(image_big_rotated.cols*SCALE_FACTOR), CV_8UC3);

        // Display img
        image_display.create(cvFloor(input->cols), cvFloor(input->rows), CV_8UC3);
#else
    // Creating rotated input image
    image_big_rotated.create(input->rows, input->cols, CV_8UC3);

    // Creating rotated downsampled output image
    image_out.create(cvFloor(image_big_rotated.rows*SCALE_FACTOR), cvFloor(image_big_rotated.cols*SCALE_FACTOR), CV_8UC3);
#endif

    // To control filter for display
    control_internal->DisplayImageFromROIFinder = &image_display;

    // Creating mask image
    mask_img.create(image_out.size(), CV_8UC1);
    mask_final.create(image_out.size(), CV_8UC1);

    // Centers
    center_x = cvCeil((double)image_out.cols/2)-1;
    center_y = cvCeil((double)image_out.rows/2)-1;

    // Rotation matrix
    rot_mat.create(2,3,CV_32FC1);
    rot_mat.at<float>(0,0) = 0;
    rot_mat.at<float>(0,1) = -1;
    rot_mat.at<float>(0,2) = image_big_rotated.cols;
    rot_mat.at<float>(1,0) = 1;
    rot_mat.at<float>(1,1) = 0;
    rot_mat.at<float>(1,2) = 0;

    // Load parameters
    LoadParameters();

    // Morphological ops
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

    // Parallel stuff
    int	part = cvFloor((float)image_out.rows/N_PROCS);

    for(int i=0; i<N_PROCS-1; i++)
    {
        start_stop[2*i] = i*part;
        start_stop[2*i+1] = i*part + part - 1;

        ok[i] = cv::Rect(0, i*part, image_out.cols, part);
    }

    start_stop[2*(N_PROCS-1)] = (N_PROCS-1)*part;
    start_stop[2*(N_PROCS-1)+1] = image_out.rows-1;

    ok[N_PROCS-1] = cv::Rect(0, (N_PROCS-1)*part, image_out.cols, image_out.rows - (N_PROCS-1)*part);

    // Done
    return 0;
}

cv::Mat apply_mask(cv::Mat image, cv::Mat mask) {
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());
    for(int r = 0; r < image.rows; r++)
        for(int c = 0; c < image.cols; c++)
            if(mask.at<uchar>(r, c) != 0)
                result.at<cv::Vec3b>(r, c) = image.at<cv::Vec3b>(r, c);
    return result;
}

int	 ROIFinder::Process(cv::Mat *input)
{	
    #pragma omp master
    {
        // We start by resizing the image

        // Clock start
        control_internal->clock_start_roi = (double) cvGetTickCount();

#ifndef  USE_VIDEO
            // Rotating image
            cv::warpAffine(*input, image_big_rotated, rot_mat, image_big_rotated.size(), cv::INTER_NEAREST);

            // Resizing image
            cv::resize(image_big_rotated, image_out, image_out.size(), 0, 0, 0);
#else
            cv::resize(*input, image_out, image_out.size(), 0, 0, 0);
#endif

#ifndef USE_VIDEO
        // Resize image for display
        cv::resize(image_big_rotated, image_display, image_display.size(), 0, 0, 0);
#endif

        width = image_out.cols;
        height = image_out.rows;
        part = cvFloor((float)height/N_PROCS);

        // Cleaning up some arrays
        for(int i=0;i<N_PROCS;i++)
        {
            acc_x[i] = 0;
            total[i] = 0;
        }
    }

    // Synch threads
    #pragma omp barrier

    // Começando processamento em paralelo
    unsigned char *img_in, *mask_ptr;
    unsigned int red, green, blue;

    int procInfo = omp_get_thread_num();

    img_in = (unsigned char *) (image_out.ptr<uchar>(0) + procInfo*part*width*3);
    mask_ptr = (unsigned char *) (mask_img.ptr<uchar>(0) + procInfo*part*width);

    // Scanning thru image
    for (unsigned int i = start_stop[2*procInfo]; i <= start_stop[2*procInfo+1]; i ++)
    {
        for (unsigned int j = 0; j < width; j ++)
        {
            blue = *img_in;  img_in++;
            green = *img_in; img_in++;
            red = *img_in;   img_in++;

            // color test mask
           if((int)red-(int)(green+30)>0)  // 40 funciona bem
            {
                *mask_ptr = 255; mask_ptr++;
                acc_x[procInfo] += j;
                total[procInfo] ++;
            }
            else
            {
                *mask_ptr = 0; mask_ptr++;
            }
        }
    }

    #pragma omp barrier
    #pragma omp master
    {
        cv::dilate(mask_img, mask_img, kernel, cv::Point(-1,-1), kernel_dilation);
        cv::erode(mask_img, mask_img, kernel, cv::Point(-1,-1), kernel_erosion);

        // Calculando centro de massa
        for(int i=1;i<N_PROCS;i++)
        {
            acc_x[0] += acc_x[i];
            total[0] += total[i];
        }

        if(total[0] > 1)
        {
            center_x = (int)((float)center_x*0.2) + cvCeil(0.8*(float)acc_x[0]/(float)total[0])-1;
            //center_y = (int)((float)center_y*0.35) + cvCeil(0.65*(float)acc_y/(float)total)-1;

            // Copying ROI center to control
            control_internal->coords[0] = center_x;
        }

        //mask_img.copyTo(mask_final);

        cv::imshow("c", image_out);
        cv::imshow("m", mask_img);
        cv::Mat result = apply_mask(image_out, mask_img);
        cv::imshow("r", result);
        char key = cv::waitKey();
        if(key == 'c')
        {
            cv::imwrite("/home/rodrigolinhares/Dropbox/papers/master_thesis/figures/color.png", image_out);
            cv::imwrite("/home/rodrigolinhares/Dropbox/papers/master_thesis/figures/mask.png", mask_img);
            cv::imwrite("/home/rodrigolinhares/Dropbox/papers/master_thesis/figures/result.png", result);
        }
    }

    return 0;
}


void	ROIFinder::LoadParameters()
{
    cv::FileStorage fs("../settings/parameters.yml", cv::FileStorage::READ);

    kernel_erosion = fs["kernel_erosion"];
    kernel_dilation = fs["kernel_dilation"];

}
