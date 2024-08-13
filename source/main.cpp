// Retina tracker

#include <stdio.h>
#include <omp.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "blackmagic_thread.h"
#include "video_fetcher.h"

#include "macros.h"

#include "ROIFinder.h"
#include "Tracker.h"
#include "Detector.h"
#include "DetectorAux.h"
#include "Control.h"


void	GetFrame(cv::VideoCapture *capture, cv::Mat *ICur, cv::Mat *ICur_raw, int isgrayscale)
{
    *capture >> *ICur_raw;
	
	if(ICur_raw->channels() > 1 && isgrayscale == 1)
	{
		cv::cvtColor(*ICur_raw, *ICur, CV_BGR2GRAY);
	}
	else
		*ICur = *ICur_raw;
}

void	GetFrame(VideoFetcher *blackmagic, cv::Mat *ICur, cv::Mat *ICur_raw, int isgrayscale)
{
    blackmagic->copyTo(ICur_raw);

    if(ICur_raw->channels() > 1 && isgrayscale == 1)
    {
        cv::cvtColor(*ICur_raw, *ICur, CV_BGR2GRAY);
    }
    else
        *ICur = *ICur_raw;
}

int main(void)
{
    // Welcome screen
    printf("--------------------------------------------\n");
    printf(" Retina Tracker ----------------------------\n");
    printf("--------------------------------------------\n\n");

	cv::Mat ICur,
            ICur_raw;

#ifdef USE_VIDEO

    BlackmagicThread capture;

    ICur.create(capture.size().second, capture.size().first, CV_8UC3);
    ICur_raw.create(capture.size().second, capture.size().first, CV_8UC3);

#else

	// video path
    //std::string	filename = "/media/roger/3TB/Quantel/Seance3/OD1m.avi";
//    std::string	filename = "/media/roger/3TB/Quantel/Seance4/bOD2cut.avi";
    //std::string	filename = "/media/richa/3TB/Quantel/Seance7/2013-10-02-P1-OD.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OG1m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/bOD2cut.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/od1_cropped.avi";

    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OD1m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OD2m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OD3m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OD4m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OD5m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OD7m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OG1m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OG2m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OG3m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OG4m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OG5m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance3/OG7m.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/bOD2cut.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/od10.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/od3.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/od5.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/od6.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/od8.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/og1.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/og2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/og3.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/og5.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/og6.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance4/og9.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance5/p1odffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance5/p2odffdshow10x.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance5/p2ogffdshow10x.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance5/p4odffdshow10x.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance5/p4ogffdshow10x1_rotated.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance5/p6ffdshow10x.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance5/p6ffdshow16x.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P1-OD-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P1-OG-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P2-OD-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P2-OG-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P3-OD-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P3-OG-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P4-OD-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P4-OG-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P5-OG-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P6-OD-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance6/2013-07-10-P6-OG-ffdshow.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P1-OD.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P1-OG.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P2-OD.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P2-OG.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P3-OG.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P4-OD.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P4-OD.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P5-OG.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P6-OD.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance7/2013-10-02-P6-OG.mp4";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P1-OD-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P1-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P2-OD-Camera-2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P2-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P3-OD-Camera-2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P3-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P4-OD-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P4-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P5-OD-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P5-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P6-OD-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P6-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P7-OD-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P7-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P8-OD-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P8-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P9-OD-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance8/camera2/2013-12-09-P9-OG-Camera 2.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P1-OG.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P2-OD.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P3-OD.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P4-OD.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P4-OD_Single use lens.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P5-OD.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P6-OD.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P6-OG.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P7-OD.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P7-OD_single use lens.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P8-OG.avi";
//    std::string filename = "/home/rodrigolinhares/videos/retina/seance9/2014-02-24-P8-OG_single use lens.avi";


	// Loading video
	printf("> Loading video from file ...\n");
    std::cout << filename << std::endl;
    cv::VideoCapture capture(filename);
    //capture.set(CV_CAP_PROP_POS_MSEC, 86000);
	if(!capture.isOpened())
	{
        printf("* Ops! what happened to the video file?\n");
		exit(0);
	}
    printf("> Video loaded...\n");
#endif    

    // Grabs image from video as first sample
    GetFrame(&capture, &ICur, &ICur_raw, 0);

	// Mosaicking function declarations
    ROIFinder       roi_finder;
    Tracker         tracker;
    Control         control;
    Detector        detector;
    //DetectorAux     detector_buddy;

    // Passing some pointersusrgAdvancedROIFinder
    roi_finder.control_internal = &control;
    detector.control_internal = &control;
    //detector_buddy.control_internal = &control;
    tracker.control_internal = &control;

    // In this version, I'm not processing a batch, so I have to set this variable inside the control filter to -1 so I can save the stats file with a correct file name
    control.repo_pos = -1;
    control.flag_loadworkspace = 0;
    control.flag_saveworkspace = 0;

    // Setup workspace saving device
    printf("\n # Load workspace? (y/n - default: n) ");
    char input_keyboard;
    scanf("%s", &input_keyboard); fflush(stdin);

    if(input_keyboard == 'y' || input_keyboard == 'Y' )
    {
        control.flag_loadworkspace = 1;
        printf("\n\nWorkspace will be loaded from \"workspace.yml\" \n ");
    }
    else
    {
        printf("No \n ");
        control.flag_loadworkspace = 0;

        printf("\n # Save workspace? (y/n - default: n) ");
        scanf("%s", &input_keyboard);

        if(input_keyboard == 'y' || input_keyboard == 'Y' )
        {
            control.flag_saveworkspace = 1;
            printf("\n\nWorkspace will be saved to \"workspace.yml\" \n\n");
        }
        else
        {
            printf("No \n\n");
            control.flag_saveworkspace = 0;
        }
    }

	// Mosaicking function initializations
    roi_finder.Initialize(&ICur);
    tracker.Initialize(&roi_finder.image_out, &roi_finder.mask_img);
    detector.Initialize(&roi_finder.image_out, &roi_finder.mask_img);
    //detector_buddy.Initialize(&roi_finder.image_out);

#ifdef  USE_VIDEO
    control.Initialize(&roi_finder.image_out, &roi_finder.image_big_rotated, &roi_finder.mask_img, &capture);
#else
    control.Initialize(&roi_finder.image_out, &roi_finder.image_big_rotated, &roi_finder.mask_img);
#endif

	// OpenMP stuff
    omp_set_dynamic(0);
    omp_set_num_threads(N_PROCS);

	// Tracking loop

    // Display menu
    printf("> Program start ...\n");

    printf("\nAvailable commands: \n");
    printf("Press 'space' to start/reset tracking \n");
    printf("'m' to save mosaic snapshot \n");
    printf("'r' to start review mode \n");
    printf("'p' to stop tracking    \n");
    printf("'q' to quit application \n\n");
    printf("\n");
    printf("Status:\n");

    cv::namedWindow("c", CV_WINDOW_NORMAL);
    cv::namedWindow("m", CV_WINDOW_NORMAL);
    cv::namedWindow("r", CV_WINDOW_NORMAL);

    while(!control.flag_quit)
    {
		// Grabs image from video
		GetFrame(&capture, &ICur, &ICur_raw, 0);

        // Running filters
        #pragma omp parallel
        {
            #pragma omp barrier

            roi_finder.Process(&ICur);

            #pragma omp barrier

            detector.ProcessWithNCC();

            #pragma omp barrier

            tracker.Process();

            #pragma omp barrier

            control.Process();

            //detector_buddy.Process(&roi_finder.image_out, &roi_finder.mask_img);
        }
	}

	return 0;
}
