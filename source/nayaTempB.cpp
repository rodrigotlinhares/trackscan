
/**********************/
/*** naya class ***/
/**********************/

#include "naya.h"



// Initialization

void naya::Initialize3DOFxi_tempB(int size_template_x,
                                int size_template_y,
                                int n_ctrl_pts_xi,
                                int n_ctrl_pts_yi,
                                int n_active_pixels,
                                int n_bins,
                                int size_bins,
                                int n_max_iters,
                                float epsilon,
                                int isgrayscale,
                                int interp)
{
    // Setup
    this->size_template_x = size_template_x;
    this->size_template_y = size_template_y;
    this->n_ctrl_pts_xi = n_ctrl_pts_xi;
    this->n_ctrl_pts_yi = n_ctrl_pts_yi;
    this->n_active_pixels = n_active_pixels;
    this->epsilon = epsilon;
    this->n_bins = n_bins;
    this->size_bins = size_bins;
    this->n_max_iters = n_max_iters;
    this->isgrayscale = isgrayscale;
    this->interp = interp;

    // Activel pixel stuff
    visited_r = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));
    visited_g = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));

    for(int i=0; i<4; i++)
    {
        pair_r[i].resize(size_template_x*size_template_y);
        pair_g[i].resize(size_template_x*size_template_y);
    }

    // Reference pixel list when n_active_pixels = size_template_x*size_template_y
    std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
    for(int i=0;i<size_template_x*size_template_y;i++)
        std_pixel_list[i] = i;

    // Allocates control point vectors
    total_n_ctrl_ptsi = n_ctrl_pts_yi*n_ctrl_pts_xi;
    DefineCtrlPtsIllum();

    // TPS Precomputations
    TPSPrecomputationsIllum();
    gain.create(size_template_x*size_template_y, 1, CV_32FC1);

    // Rest of allocations
    dummy_mapx.create(size_template_y, size_template_x, CV_32FC1);
    dummy_mapy.create(size_template_y, size_template_x, CV_32FC1);
    delta.create(4+total_n_ctrl_ptsi+1, 1, CV_32FC1);
    Mask.create(size_template_y, size_template_x, CV_8UC1);
    Mask.setTo(255);

    if(isgrayscale)
    {
//        dif.create(n_active_pixels, 1, CV_32FC1);
//        SD.create(n_active_pixels, 2+total_n_ctrl_ptsi+1, CV_32FC1);

//        gradx.create(size_template_y, size_template_x, CV_32FC1);
//        grady.create(size_template_y, size_template_x, CV_32FC1);
//        gradx_tmplt.create(size_template_y, size_template_x, CV_32FC1);
//        grady_tmplt.create(size_template_y, size_template_x, CV_32FC1);
//        current_warp.create(size_template_y, size_template_x, CV_8UC1);
//        compensated_warp.create(size_template_y, size_template_x, CV_8UC1);

//        expected = (float*)malloc(n_bins*sizeof(float));
    }
    else
    {
        dif.create(n_active_pixels*2, 1, CV_32FC1);
        SD.create(n_active_pixels*2, 4+total_n_ctrl_ptsi+1, CV_32FC1);

        gradx.create(size_template_y, size_template_x, CV_32FC3);
        grady.create(size_template_y, size_template_x, CV_32FC3);
        gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
        grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);
        current_warp.create(size_template_y, size_template_x, CV_8UC3);
        compensated_warp.create(size_template_y, size_template_x, CV_8UC3);

        expected = (float*)malloc(3*n_bins*sizeof(float));
    }

    // Joint histogram
    correction = (float*)malloc(n_bins*sizeof(float));
    p_joint =  (float*)malloc(n_bins*n_bins*sizeof(float));

    ResetExpected();

    // Definindo cv Rects
    int part = cvFloor((float)size_template_y/N_PROCS);
    int part2 = cvFloor((float)(size_template_y*size_template_x)/N_PROCS);
    int part3 = cvFloor((float)(n_active_pixels)/N_PROCS);
    int part4 = cvFloor((float)(2*n_active_pixels)/N_PROCS);

    for(int i=0; i<N_PROCS-1; i++)
    {
        // Start stops
        start_stop[2*i] = i*part;
        start_stop[2*i+1] = i*part + part-1;

        start_stop2[2*i] = i*part2;
        start_stop2[2*i+1] = i*part2 + part2-1;

        start_stop3[2*i] = i*part3;
        start_stop3[2*i+1] = i*part3 + part3 - 1;

        start_stop4[2*i] = i*part4;
        start_stop4[2*i+1] = i*part4 + part4 - 1;

        // Oks
        ok[i] = cvRect(0, i*part, size_template_x, part);
        ok2[i] = cvRect(0, i*part2, total_n_ctrl_ptsi, part2);
        ok3[i] = cvRect(0, i*part2, 1, part2);
        ok_hess[i] = cvRect(0, i*part4, 4+total_n_ctrl_ptsi+1, part4);
    }

    start_stop[2*(N_PROCS-1)] = (N_PROCS-1)*part;
    start_stop[2*(N_PROCS-1)+1] = size_template_y-1;

    start_stop2[2*(N_PROCS-1)] = (N_PROCS-1)*part2;
    start_stop2[2*(N_PROCS-1)+1] = size_template_y*size_template_x-1;

    start_stop3[2*(N_PROCS-1)] = (N_PROCS-1)*part3;
    start_stop3[2*(N_PROCS-1)+1] = n_active_pixels-1;

    start_stop4[2*(N_PROCS-1)] = (N_PROCS-1)*part4;
    start_stop4[2*(N_PROCS-1)+1] = 2*n_active_pixels-1;

    ok[N_PROCS-1] = cvRect(0, (N_PROCS-1)*part, size_template_x, size_template_y - (N_PROCS-1)*part);
    ok2[N_PROCS-1] = cvRect(0, (N_PROCS-1)*part2, total_n_ctrl_ptsi, (size_template_y*size_template_x) - (N_PROCS-1)*part2);
    ok3[N_PROCS-1] = cvRect(0, (N_PROCS-1)*part2, 1, (size_template_y*size_template_x) - (N_PROCS-1)*part2);
    ok_hess[N_PROCS-1] = cvRect(0, (N_PROCS-1)*part4, 4+total_n_ctrl_ptsi+1, 2*n_active_pixels - (N_PROCS-1)*part4);

    // More crazy inits
    pre_SD = new cv::Mat [N_PROCS];
    pre_hess = new cv::Mat [N_PROCS];

    for(int i=0;i<N_PROCS;i++)
    {
        pre_SD[i].create(4+total_n_ctrl_ptsi+1, 1, CV_32FC1);
        pre_hess[i].create(4+total_n_ctrl_ptsi+1, 4+total_n_ctrl_ptsi+1, CV_32FC1);
    }
}



void	naya::pWarp3DOF1_tempB(unsigned int ch)
{
    int offx = cvCeil((double)size_template_x/2);
    int offy = cvCeil((double)size_template_y/2);

    // Multiplying matrices
    for(int i=start_stop[2*ch]-offy;i<=start_stop[2*ch+1]-offy;i++)
    {
        for(int j=-offx;j<offx;j++)
        {
            dummy_mapx.at<float>(i+offy,j+offx) = parameters[0]*(float)j - parameters[1]*(float)i + parameters[2];
            dummy_mapy.at<float>(i+offy,j+offx) = parameters[1]*(float)j + parameters[0]*(float)i + parameters[3];
        }
    }

    // Remapping
    cv::remap(*ICur, current_warp(ok[ch]), dummy_mapx(ok[ch]), dummy_mapy(ok[ch]), interp, 0, cv::Scalar(0));

    if(using_masks)
        cv::remap(*Mask_roi, Mask(ok[ch]), dummy_mapx(ok[ch]), dummy_mapy(ok[ch]), 0, 0, cv::Scalar(0));

    // Computes gain map
    gain(ok3[ch]) = MKinvi(ok2[ch])*ctrl_pts_wi;
}




void	naya::MountJacobian3DOFColorxi_tempB(unsigned int ch)
{
    int i1, j1, i2, j2;
    int offx = cvCeil((double)size_template_x/2);
    int offy = cvCeil((double)size_template_y/2);
    float sum_gradx, sum_grady, temp[30];
    unsigned char current_pxl, current_comp_pxl, current_tmplt;

    pre_hess[ch].setTo(0);
    pre_SD[ch].setTo(0);

    // Mounting matrix
    for(int k=start_stop3[2*ch];k<=start_stop3[2*ch+1];k++)
    {
        // Active green pixel positions
        i1 = cvFloor((float)active_pixels_g[k]/size_template_x);
        j1 = active_pixels_g[k] - i1*size_template_x;

        if(Mask.at<uchar>(i1,j1) != 0)
        {
            // Current intensities
            current_pxl = current_warp.ptr<uchar>(i1)[3*j1+1];
            current_comp_pxl = compensated_warp.ptr<uchar>(i1)[3*j1+1];
            current_tmplt = Template->ptr<uchar>(i1)[3*j1+1];

            // gradients
            sum_gradx = (gradx.ptr<float>(i1)[3*j1+1] + gradx_tmplt.ptr<float>(i1)[3*j1+1]);
            sum_grady = (grady.ptr<float>(i1)[3*j1+1] + grady_tmplt.ptr<float>(i1)[3*j1+1]);

            // img difference
            //dif.at<float>(k, 0) = (float)(current_comp_pxl - current_tmplt);
            float dif = (float)(current_comp_pxl - current_tmplt);

            // gradient
//			SD.at<float>(k, 0) = ((float)(j1+1-offx))*sum_grady - ((float)(i1+1-offy))*sum_gradx;
//			SD.at<float>(k, 1) = sum_gradx;
//			SD.at<float>(k, 2) = sum_grady;

//			for(int i=0; i<total_n_ctrl_ptsi; i++)
//				SD.at<float>(k, i+3) = MKinvi.at<float>(active_pixels_g[k], i)*(current_pxl + current_tmplt);

//			SD.at<float>(k, 3+total_n_ctrl_ptsi) = 2;

            temp[0] = ((float)(j1+1-offx))*sum_gradx + ((float)(i1+1-offy))*sum_grady;
            temp[1] = ((float)(-(i1+1-offy)))*sum_gradx + ((float)(j1+1-offx))*sum_grady;
            temp[2] = sum_grady;
            temp[3] = sum_grady;

            for(int i=0; i<total_n_ctrl_ptsi; i++)
                temp[i+4] = MKinvi.at<float>(active_pixels_g[k], i)*(current_pxl + current_tmplt);

            temp[4+total_n_ctrl_ptsi] = 2;

            for(int i=0;i<=(total_n_ctrl_ptsi+4);i++)
            {
                for(int j=i;j<=(total_n_ctrl_ptsi+4);j++)
                {
                    pre_hess[ch].at<float>(j,i) += temp[i]*temp[j];
                }

                pre_SD[ch].at<float>(i,0) += temp[i]*dif;
            }
        }
        else
        {
//			// img difference
//			dif.at<float>(k,0) = 0;

//			// gradient
//			SD.at<float>(k, 0) = 0;
//			SD.at<float>(k, 1) = 0;
//			SD.at<float>(k, 2) = 0;

//			for(int i=0; i<total_n_ctrl_ptsi; i++)
//				SD.at<float>(k, i+3) = 0;

//			SD.at<float>(k, 3+total_n_ctrl_ptsi) = 0;
        }

        // Active red pixel positions
        i2 = cvFloor((float)active_pixels_r[k]/size_template_x);
        j2 = active_pixels_r[k] - i2*size_template_x;

        if(Mask.at<uchar>(i2,j2) != 0)
        {
            // Current intensities
            current_pxl = current_warp.ptr<uchar>(i2)[3*j2+2];
            current_comp_pxl = compensated_warp.ptr<uchar>(i2)[3*j2+2];
            current_tmplt = Template->ptr<uchar>(i2)[3*j2+2];

            // gradients
            sum_gradx = (gradx.ptr<float>(i2)[3*j2+2] + gradx_tmplt.ptr<float>(i2)[3*j2+2]);
            sum_grady = (grady.ptr<float>(i2)[3*j2+2] + grady_tmplt.ptr<float>(i2)[3*j2+2]);

            // img difference
            //dif.at<float>(k + n_active_pixels, 0) = (float)(current_comp_pxl - current_tmplt);
            float dif = (float)(current_comp_pxl - current_tmplt);

            // gradient
//			SD.at<float>(k + n_active_pixels, 0) = ((float)(j2+1-offx))*sum_grady - ((float)(i2+1-offy))*sum_gradx;
//			SD.at<float>(k + n_active_pixels, 1) = sum_gradx;
//			SD.at<float>(k + n_active_pixels, 2) = sum_grady;

//			for(int i=0; i<total_n_ctrl_ptsi; i++)
//				SD.at<float>(k+ n_active_pixels, i+3) = MKinvi.at<float>(active_pixels_r[k], i)*(current_pxl + current_tmplt);

//			SD.at<float>(k+ n_active_pixels, 3+total_n_ctrl_ptsi) = 2;

            temp[0] = ((float)(j2+1-offx))*sum_gradx + ((float)(i2+1-offy))*sum_grady;
            temp[1] = ((float)(-(i2+1-offy)))*sum_gradx + ((float)(j2+1-offx))*sum_grady;
            temp[2] = sum_grady;
            temp[3] = sum_grady;

            for(int i=0; i<total_n_ctrl_ptsi; i++)
                temp[i+4] = MKinvi.at<float>(active_pixels_r[k], i)*(current_pxl + current_tmplt);

            temp[4+total_n_ctrl_ptsi] = 2;

            for(int i=0;i<=(total_n_ctrl_ptsi+4);i++)
            {
                for(int j=i;j<=(total_n_ctrl_ptsi+4);j++)
                {
                    pre_hess[ch].at<float>(j,i) += temp[i]*temp[j];
                }

                pre_SD[ch].at<float>(i,0) += temp[i]*dif;
            }
        }
        else
        {
//			// img difference
//			dif.at<float>(k + n_active_pixels, 0) = 0;

//			// gradient
//			SD.at<float>(k + n_active_pixels, 0) = 0;
//			SD.at<float>(k + n_active_pixels, 1) = 0;
//			SD.at<float>(k + n_active_pixels, 2) = 0;

//			for(int i=0; i<total_n_ctrl_ptsi; i++)
//				SD.at<float>(k+ n_active_pixels, i+3) = 0;

//			SD.at<float>(k+ n_active_pixels, 3+total_n_ctrl_ptsi) = 0;
        }
    }
}

int		naya::Update3DOFix_tempB()
{
    float sum = 0;

    for(int i=1;i<N_PROCS;i++)
    {
        pre_SD[0] += pre_SD[i];
        pre_hess[0] += pre_hess[i];
    }

    // Flipping Hessian
    for(int i=0; i<SD.cols; i++)
    {
        for(int j=i; j<SD.cols; j++)
        {
            pre_hess[0].at<float>(i,j) = pre_hess[0].at<float>(j,i);
        }
    }

    // ESM update
    delta = 2*((pre_hess[0]).inv(CV_SVD)*(pre_SD[0]));

    // Update
    parameters[0] -= delta.at<float>(0, 0); sum += fabs(delta.at<float>(0, 0));
    parameters[1] -= delta.at<float>(1, 0); sum += fabs(delta.at<float>(1, 0));
    parameters[2] -= delta.at<float>(2, 0); sum += fabs(delta.at<float>(2, 0));
    parameters[3] -= delta.at<float>(3, 0); sum += fabs(delta.at<float>(3, 0));

    for(int i=0; i<total_n_ctrl_ptsi; i++)
    {
        ctrl_pts_wi.at<float>(i, 0) -= delta.at<float>(4+i, 0);
        sum += fabs(delta.at<float>(4+i, 0));
    }

    bias -= delta.at<float>(total_n_ctrl_ptsi+4, 0);

    // Now checking to see if gain diverged
//    for(int i=0; i<total_n_ctrl_ptsi; i++)
//    {
//        if(	ctrl_pts_wi.at<float>(i,0) < 0.7 )
//            ctrl_pts_wi.at<float>(i,0) = 1;
//    }

    return sum < epsilon;
}

void		naya::Run3DOFxi_step1_tempB(cv::Mat *ICur,
                                cv::Mat *Mask_roi,
                                cv::Mat *Template,
                                cv::Mat *Mask_template,
                                float *parameters,
                                float *parameters_illum,
                                int *active_pixels_r,
                                int *active_pixels_g)
{
    // Taking in input arguments
    this->ICur = ICur;
    this->Mask_roi = Mask_roi;
    this->Mask_template = Mask_template;
    this->Template = Template;
    this->parameters = parameters;
    this->parameters_illum = parameters_illum;

    if(active_pixels_r == 0)
    {
        this->active_pixels_r = std_pixel_list;
        this->active_pixels_g = std_pixel_list;
    }
    else
    {
        this->active_pixels_r = active_pixels_r;
        this->active_pixels_g = active_pixels_g;
    }

    // Are we using masks?
    (Mask_roi == 0 || Mask_template == 0) ? using_masks = 0 : using_masks = 1;

    // Must copy input illumination parameters to internal structure
    ParameterIOIllum(1);

    iters = 0;
}


void	naya::Run3DOFxi3_tempB()
{
    // Converting back to original form


    // copying internal structure back to output params
    ParameterIOIllum(0);
}

void 	naya::Run3DOFxi_threaded_tempB(cv::Mat *ICur,
                            cv::Mat *Mask_roi,
                            cv::Mat *Template,
                            cv::Mat *Mask_template,
                            float *parameters,
                            float *parameters_illum,
                           int *active_pixels_r,
                           int *active_pixels_g)
{

    int procInfo =  omp_get_thread_num();

    #pragma omp master
    {
        // Primeiro passo do rastreamento
        Run3DOFxi_step1_tempB(ICur,
                            Mask_roi,
                            Template,
                            Mask_template,
                            parameters,
                            parameters_illum,
                            active_pixels_r,
                            active_pixels_g);

        flag_done = 0;
    }

    // Tracking loop here
    for(int i=0; i<n_max_iters; i++)
    {
        #pragma omp barrier
        if(!flag_done)
        {
            pWarp3DOF1_tempB(procInfo);

            #pragma omp barrier
            pWarp3DOF2(procInfo);

            #pragma omp barrier
            Run3DOFxi2(procInfo);

            #pragma omp barrier
            MountJacobian3DOFColorxi_tempB(procInfo);

            #pragma omp barrier

            #pragma omp master
            {
                iters++;
                flag_done = Update3DOFix_tempB();
            }
        }
    }

    // Finalizando
    #pragma omp barrier
    #pragma omp master
        Run3DOFxi3_tempB();
}

