#include "pThreadStructs.h"
#include "macros.h"
#include "show_image.hpp"


// Processing function
void* BuildDisplay(void *param_in)
{
    // Casting input
    ThreadPointersDisplay *param = (ThreadPointersDisplay*) param_in;

    int n_points_display = 100;
    cv::Mat pos(3, n_points_display, CV_32FC1), projected_pos(3, n_points_display, CV_32FC1), image_display;
    pos.setTo(1);

    int draw_pos = 0;

    cv::Mat Ti(3,3,CV_32FC1);

    // Display image
#ifdef     USE_VIDEO

    unsigned    frameCount;

    image_display.create(cvFloor(param->video_feed->size().second), cvFloor(param->video_feed->size().first), CV_8UC3);

    // OpenGL display
    ShowImage showImage(SIZE_WIN_X, SIZE_WIN_Y);
    Texture frameTexture = showImage.allocateTexture(param->video_feed->size().first,param->video_feed->size().second);
    Texture otherTexture = showImage.allocateTexture(param->refine_mosaic->Mosaic.cols, param->refine_mosaic->Mosaic.rows);

#else

    image_display = *(param->DisplayMain);

    // OpenGL display
    ShowImage showImage(SIZE_WIN_X, SIZE_WIN_Y);
    Texture frameTexture = showImage.allocateTexture(param->DisplayMain->cols,param->DisplayMain->rows);
    Texture otherTexture = showImage.allocateTexture(param->refine_mosaic->Mosaic.cols, param->refine_mosaic->Mosaic.rows);

#endif

    float prop_x = (image_display.cols*(SIZE_WIN_Y/image_display.rows))/SIZE_WIN_X;
    float prop_mx, prop_my;

    float mosaic_aspect_ratio = param->refine_mosaic->Mosaic.cols/param->refine_mosaic->Mosaic.rows;

    if(mosaic_aspect_ratio > 1)
    {
        prop_mx = 1-prop_x;
        prop_my = (param->refine_mosaic->Mosaic.rows*((SIZE_WIN_X-image_display.cols/(image_display.rows/SIZE_WIN_Y))/param->refine_mosaic->Mosaic.cols))/SIZE_WIN_Y;
    }
    else
    {
        prop_mx = (param->refine_mosaic->Mosaic.cols*(SIZE_WIN_Y/param->refine_mosaic->Mosaic.rows))/(SIZE_WIN_X);
        prop_my = 1;
    }

    // While program active
    while(!*param->flag_quit)
    {
        // Building mosaic image
        if(*param->flag_refine_mosaic)
        {
            param->refine_mosaic->Process();
            *param->flag_refine_mosaic = 0;
        }

#ifdef USE_VIDEO
        // Fetch next frame
        param->video_feed->copyTo(&image_display,frameCount);
#endif

        // Drawing on image
        if(!*param->flag_tracking)
        {
            //TODO remove this whole block
            for(int i = 0; i < param->detected_keypoints->size(); i++) {
              cv::KeyPoint point = param->detected_keypoints->at(i);
              float scaledRow = point.pt.y / SCALE_FACTOR;
              float scaledCol = point.pt.x / SCALE_FACTOR;
              circle(image_display, cv::Point(scaledCol, scaledRow), 5, cv::Scalar(0, 255, 0), 2);
            }

            // Text
            cv::putText(image_display, "Tracking Lost!", cvPoint(10,30), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255));

            // Cross in the middle of the ROI
            cv::line(image_display, cvPoint((1/param->scale_factor)*param->coords[0], (1/param->scale_factor)*param->coords[1]+10), cvPoint((1/param->scale_factor)*param->coords[0], (1/param->scale_factor)*param->coords[1]-10),
                   cvScalar(0,0,255), 3);

            cv::line(image_display, cvPoint((1/param->scale_factor)*param->coords[0]+10, (1/param->scale_factor)*param->coords[1]), cvPoint((1/param->scale_factor)*param->coords[0]-10, (1/param->scale_factor)*param->coords[1]),
                   cvScalar(0,0,255), 3);
        }
        else
        {
            // Text
            char text[30];
            sprintf(text,"Tracking at %.2f fps", *param->fps);
            cv::putText(image_display, text, cvPoint(10,30), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255));

            sprintf(text,"Confidence %.1f%% ", *param->confidence*100);
            cv::putText(image_display, text, cvPoint(10,60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255));

            CvScalar color = cvScalar(0,0,255);

            // Box representing tracked template on main display
            cv::line(image_display, cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*-param->size_template_x/2-param->tracking_param[1]*-param->size_template_y/2), (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*-param->size_template_x/2+param->tracking_param[0]*-param->size_template_y/2)),
                   cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*-param->size_template_x/2-param->tracking_param[1]*param->size_template_y/2),  (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*-param->size_template_x/2+param->tracking_param[0]*+param->size_template_y/2)),
                   color, 3);


            cv::line(image_display, cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*-param->size_template_x/2-param->tracking_param[1]*param->size_template_y/2),  (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*-param->size_template_x/2+param->tracking_param[0]*+param->size_template_y/2)),
                   cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*param->size_template_x/2-param->tracking_param[1]*param->size_template_y/2), (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*param->size_template_x/2+param->tracking_param[0]*param->size_template_y/2)),
                   color, 3);

            cv::line(image_display, cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*param->size_template_x/2-param->tracking_param[1]*-param->size_template_y/2),  (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*param->size_template_x/2+param->tracking_param[0]*-param->size_template_y/2)),
                   cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*param->size_template_x/2-param->tracking_param[1]*param->size_template_y/2), (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*param->size_template_x/2+param->tracking_param[0]*param->size_template_y/2)),
                   color, 3);

            cv::line(image_display, cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*-param->size_template_x/2-param->tracking_param[1]*-param->size_template_y/2), (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*-param->size_template_x/2+param->tracking_param[0]*-param->size_template_y/2)),
                   cvPoint((1/param->scale_factor)*(param->tracking_param[2]+param->tracking_param[0]*param->size_template_x/2-param->tracking_param[1]*-param->size_template_y/2), (1/param->scale_factor)*(param->tracking_param[3]+param->tracking_param[1]*param->size_template_x/2+param->tracking_param[0]*-param->size_template_y/2)),
                   color, 3);

            // Plotting points on screen
            if(draw_pos>0)
            {
                // Inverting current tracking transformation
                Ti.at<float>(0, 0) = (1/param->scale_factor)*param->mosaic_coords[0];
                Ti.at<float>(0, 1) = -(1/param->scale_factor)*param->mosaic_coords[1];
                Ti.at<float>(0, 2) = (1/param->scale_factor)*param->mosaic_coords[2];
                Ti.at<float>(1, 0) = (1/param->scale_factor)*param->mosaic_coords[1];
                Ti.at<float>(1, 1) = (1/param->scale_factor)*param->mosaic_coords[0];
                Ti.at<float>(1, 2) = (1/param->scale_factor)*param->mosaic_coords[3];
                Ti.at<float>(2, 0) = 0;
                Ti.at<float>(2, 1) = 0;
                Ti.at<float>(2, 2) = 1;

                // Achando transf inversa
                projected_pos = Ti*pos;

                for(int i=0; i<draw_pos; i++)
                    cv::circle(image_display, cv::Point(projected_pos.at<float>(0,i)/projected_pos.at<float>(2,i), projected_pos.at<float>(1,i)/projected_pos.at<float>(2,i)), 3, cv::Scalar(0,255,255), 3, 8, 0);
            }
        }

        // OpenGL display CAMERA FEED
        showImage.start();
        showImage.show(&image_display,frameTexture,0,0,prop_x,1);

        // Draws box around most similar template during detection
        //cv::Mat mosaic_clone = param->refine_mosaic->Mosaic.clone();
//        if(!*param->flag_tracking) {
//          cv::Rect template_box(param->active_template_tl,
//                                cv::Size(param->size_template_x, param->size_template_y));
//          rectangle(mosaic_clone, template_box, cv::Scalar(0, 255, 0), 3);
//        }
//        showImage.show(&mosaic_clone, otherTexture, prop_x, 0, prop_mx, prop_my);

        showImage.show(&param->refine_mosaic->Mosaic,otherTexture,prop_x,0,prop_mx,prop_my);
        showImage.draw();

        ShowImage::Event* evt;
        while( (evt = showImage.getEvent()) != NULL )
        {
            if( evt->type == ShowImage::Event::KEYBOARD )
            {
                ShowImage::KeyboardEvent* keyEvt = static_cast<ShowImage::KeyboardEvent*>(evt);
                if( keyEvt->pressed == true ) //Only key release matters
                    continue;

                ShowImage::KeyboardEvent::Key key1 = keyEvt->key;
                if(key1 == ShowImage::KeyboardEvent::KEY_SPACE )
                {
                    *param->flag_tracking = 0;
                    *param->flag_reset = 1;
                    *param->flag_start = 1;
                }
                if(key1 == ShowImage::KeyboardEvent::KEY_m )
                {
                    *param->flag_save_mosaic = 1;
                }
                if(key1 == ShowImage::KeyboardEvent::KEY_r )
                {
                    *param->flag_review = 1;
                    std::cout << "Review Mode On!" << std::endl;
                }
                if(key1 == ShowImage::KeyboardEvent::KEY_q )
                {
                    *param->flag_quit = 1;
                }
                if(key1 == ShowImage::KeyboardEvent::KEY_p )
                {
                    *param->flag_start = 0;
                }
                if(key1 == ShowImage::KeyboardEvent::KEY_d )
                {
                    draw_pos = 0;
                }
                if(key1 == ShowImage::KeyboardEvent::KEY_o )
                {
                    *param->flag_tracking = 0;
                }
            }
            else
                if( evt->type == ShowImage::Event::QUIT )
                {
                    *param->flag_quit = 1;
                }
                else
                    if( evt->type == ShowImage::Event::MOUSE )
                    {
                        ShowImage::MouseEvent* mouseEvt = static_cast<ShowImage::MouseEvent*>(evt);
                        if( mouseEvt->pressed || mouseEvt->button != ShowImage::MouseEvent::LEFT ) //Only left button release matters
                            continue;

//						if( mouseEvt->dragging ) {
//							Mouse dragging
//						}

                        //Mapping to Mosaic coordinates:
                        std::pair<float,float> mosaicPos = mouseEvt->get(prop_x,0,prop_mx,prop_my);
                        //std::cout << mosaicPos.first << " " << mosaicPos.second << std::endl;

                        if(draw_pos<n_points_display and mosaicPos.first > 0 and mosaicPos.second > 0)
                        {
                            float x = mosaicPos.first*param->DisplayMosaic->cols;
                            float y = mosaicPos.second*param->DisplayMosaic->rows;

                            pos.at<float>(0,draw_pos) = x - param->DisplayMosaic->cols/2;
                            pos.at<float>(1,draw_pos) = y - (param->DisplayMosaic->rows)/(2*prop_my);

                            draw_pos ++;
                        }
                    }
        }
    }

    std::cout << "Display thread will now terminate" << std::endl;
}









