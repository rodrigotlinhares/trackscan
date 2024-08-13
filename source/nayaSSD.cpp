
/**********************/
/*** naya class ***/
/**********************/

#include "naya.h"


// Initialization

void naya::Initialize3DOFxi(int size_template_x, 
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
	
	for(int i=0; i<3; i++)
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
	delta.create(3+total_n_ctrl_ptsi+1, 1, CV_32FC1);
	Mask.create(size_template_y, size_template_x, CV_8UC1);
	Mask.setTo(255);

	if(isgrayscale)
	{
		dif.create(n_active_pixels, 1, CV_32FC1);
		SD.create(n_active_pixels, 3+total_n_ctrl_ptsi+1, CV_32FC1);

		gradx.create(size_template_y, size_template_x, CV_32FC1);
		grady.create(size_template_y, size_template_x, CV_32FC1);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		current_warp.create(size_template_y, size_template_x, CV_8UC1);
		compensated_warp.create(size_template_y, size_template_x, CV_8UC1);

		expected = (float*)malloc(n_bins*sizeof(float));
	}
	else
	{
		dif.create(n_active_pixels*2, 1, CV_32FC1);
		SD.create(n_active_pixels*2, 3+total_n_ctrl_ptsi+1, CV_32FC1);

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
		ok_hess[i] = cvRect(0, i*part4, 3+total_n_ctrl_ptsi+1, part4);
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
	ok_hess[N_PROCS-1] = cvRect(0, (N_PROCS-1)*part4, 3+total_n_ctrl_ptsi+1, 2*n_active_pixels - (N_PROCS-1)*part4);
	
	// More crazy inits
	pre_SD = new cv::Mat [N_PROCS];
	pre_hess = new cv::Mat [N_PROCS];

	for(int i=0;i<N_PROCS;i++)
	{
		pre_SD[i].create(3+total_n_ctrl_ptsi+1, 1, CV_32FC1);
		pre_hess[i].create(3+total_n_ctrl_ptsi+1, 3+total_n_ctrl_ptsi+1, CV_32FC1);
	}
}


// naya Run


void 	naya::Run3DOFxi_threaded(cv::Mat *ICur,
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
        Run3DOFxi_step1(ICur,
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
            pWarp3DOF1(procInfo);

            #pragma omp barrier
            pWarp3DOF2(procInfo);

            #pragma omp barrier
            Run3DOFxi2(procInfo);

            #pragma omp barrier
            MountJacobian3DOFColorxi(procInfo);

            #pragma omp barrier

            #pragma omp master
            {
                iters++;
                flag_done = Update3DOFix();
            }
        }
    }

    // Finalizando
    #pragma omp barrier
    #pragma omp master
        Run3DOFxi3();
}



void		naya::Run3DOFxi_step1(cv::Mat *ICur,
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

    // Converting parameters
    angle_3DOF = std::atan2(parameters[1],parameters[0]);

    // Must copy input illumination parameters to internal structure
    ParameterIOIllum(1);

    iters = 0;
}


void	naya::Run3DOFxi2(unsigned int ch)
{
    // Computes image gradients
	WarpGradi(ch);

    // Computes occlusion map
	if(using_masks)
        OcclusionMap(ch);
}



void	naya::OcclusionMap(unsigned int ch)
{
	// Scans image
	for(int i=start_stop[2*ch];i<=start_stop[2*ch+1];i++)
	{
		for(int j=0;j<size_template_x;j++)
		{
			if(Mask_template->at<uchar>(i,j) == 0)
				Mask.at<unsigned char>(i,j) = 0;
		}
	}
}

void	naya::pWarp3DOF1(unsigned int ch)
{
    int offx = cvCeil((double)size_template_x/2);
    int offy = cvCeil((double)size_template_y/2);
	
    float coseno = std::cos(angle_3DOF);
    float seno = std::sin(angle_3DOF);
	
    // Multiplying matrices
    for(int i=start_stop[2*ch]-offy;i<=start_stop[2*ch+1]-offy;i++)
    {
        for(int j=-offx;j<offx;j++)
        {
            dummy_mapx.at<float>(i+offy,j+offx) = coseno*(float)j - seno*(float)i + parameters[2];
            dummy_mapy.at<float>(i+offy,j+offx) = seno*(float)j + coseno*(float)i + parameters[3];
        }
    }
	
    // Remapping
    cv::remap(*ICur, current_warp(ok[ch]), dummy_mapx(ok[ch]), dummy_mapy(ok[ch]), interp, 0, cv::Scalar(0));

    if(using_masks)
        cv::remap(*Mask_roi, Mask(ok[ch]), dummy_mapx(ok[ch]), dummy_mapy(ok[ch]), 0, 0, cv::Scalar(0));
	
    // Computes gain map
    gain(ok3[ch]) = MKinvi(ok2[ch])*ctrl_pts_wi;
}

void    naya::pWarp3DOF2(unsigned int ch)
{	
    // Computes compensated image
    NonRigidCompensation(ch);
}

void		naya::Update3DOFi1(unsigned int ch)
{	
    pre_SD[ch] = (SD(ok_hess[ch])).t()*dif(cv::Rect(0,start_stop4[2*ch], 1, start_stop4[2*ch+1]-start_stop4[2*ch]+1));

    cv::Mat SDt = (SD(ok_hess[ch])).t();

    // fazer loop de multiplicaçao do tamanho do n cols de SD
    for(int i=0; i<SD.cols; i++)
    {
        // fazer multiplicaçao de cada linha da matriz de ponteiros vezes as colunas de SD, excluindo colunas ate no de linha atual da mat de ponteiros
        pre_hess[ch](cv::Rect(i, i, SD.cols-i, 1)) = SDt(cv::Rect(0,i,ok_hess[ch].height, 1))*SD(cv::Rect(i,ok_hess[ch].y, ok_hess[ch].width-i, ok_hess[ch].height));
    }
}

int		naya::Update3DOFix()
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
    angle_3DOF -= delta.at<float>(0, 0); sum += fabs(delta.at<float>(0, 0));
    parameters[2] -= delta.at<float>(1, 0); sum += fabs(delta.at<float>(1, 0));
    parameters[3] -= delta.at<float>(2, 0); sum += fabs(delta.at<float>(2, 0));

	for(int i=0; i<total_n_ctrl_ptsi; i++)
    {
		ctrl_pts_wi.at<float>(i, 0) -= delta.at<float>(3+i, 0);
        sum += fabs(delta.at<float>(3+i, 0));
    }

    bias -= delta.at<float>(total_n_ctrl_ptsi+3, 0);

    // Now checking to see if gain diverged
//    for(int i=0; i<total_n_ctrl_ptsi; i++)
//    {
//        if(	ctrl_pts_wi.at<float>(i,0) < 0.7 )
//            ctrl_pts_wi.at<float>(i,0) = 1;
//    }

	return sum < epsilon;
}


void	naya::Run3DOFxi3()
{
    // Converting back to original form
    parameters[0] = std::cos(angle_3DOF);
    parameters[1] = std::sin(angle_3DOF);

    // copying internal structure back to output params
    ParameterIOIllum(0);
}


// naya Run

int		naya::Run3DOFxi(cv::Mat *ICur,
						cv::Mat *Mask_roi,
						cv::Mat *Template,
						cv::Mat *Mask_template,
						float *parameters,
						float *parameters_illum,
						int *active_pixels_r,
						int *active_pixels_g)
{
	int flag_tracking = 1;

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
	
	// Converting parameters
	angle_3DOF = std::atan2(parameters[1],parameters[0]);

	// Must copy input illumination parameters to internal structure
    ParameterIOIllum(1);

    // Fun begins
    for(iters=0;iters<n_max_iters;iters++)
	{	
		// Computing mapped positions in parallel
		Warp3DOF();
		
		// Computes gain map
		gain = MKinvi*ctrl_pts_wi;
		
		// Computes compensated image
		NonRigidCompensation();

		// Computes image gradients
		WarpGradi();
		
		// Computes occlusion map
		if(using_masks)
			OcclusionMap();

		// Mounts Jacobian
		isgrayscale ? MountJacobian3DOFGrayxi() : MountJacobian3DOFColorxi();

		// Updates naya parameters
        if(Update3DOFi())
            break;
    }

	// Converting back to original form
	parameters[0] = std::cos(angle_3DOF);
	parameters[1] = std::sin(angle_3DOF);
	
	// copying internal structure back to output params
	ParameterIOIllum(0);

	return flag_tracking;
}

// Non-rigid illumination compensation core

void	naya::WarpGradi()
{
	cv::Sobel(compensated_warp, gradx, CV_32F, 1, 0, 1);
	cv::Sobel(compensated_warp, grady, CV_32F, 0, 1, 1);
}

void	naya::WarpGradi(unsigned int ch)
{
    cv::Sobel(compensated_warp(ok[ch]), gradx(ok[ch]), CV_32F, 1, 0, 1);
    cv::Sobel(compensated_warp(ok[ch]), grady(ok[ch]), CV_32F, 0, 1, 1);
}

void	naya::WarpGrad(unsigned int ch, cv::Mat *input)
{
    cv::Sobel((*input)(ok[ch]), (gradx)(ok[ch]), CV_32F, 1, 0, 1);
    cv::Sobel((*input)(ok[ch]), (grady)(ok[ch]), CV_32F, 0, 1, 1);
}

void	naya::ParameterIOIllum(int isinput)
{
	if(isinput)
		for(int i=0; i<total_n_ctrl_ptsi; i++)
		{
			ctrl_pts_wi.at<float>(i, 0) = parameters_illum[i];
			bias = parameters_illum[total_n_ctrl_ptsi];
		}
	else
		for(int i=0; i<total_n_ctrl_ptsi; i++)
		{
			parameters_illum[i] = ctrl_pts_wi.at<float>(i, 0);
			parameters_illum[total_n_ctrl_ptsi] = bias;
		}
}

void	naya::DefineCtrlPtsIllum()
{	
	// Initializing control point vector
	ctrl_pts_xi = (int*) malloc(n_ctrl_pts_xi*n_ctrl_pts_yi*sizeof(int));
	ctrl_pts_yi = (int*) malloc(n_ctrl_pts_xi*n_ctrl_pts_yi*sizeof(int));

	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	for(int j=0; j<n_ctrl_pts_xi; j++)
		for(int i=0; i<n_ctrl_pts_yi; i++)
		{
			ctrl_pts_xi[i + j*n_ctrl_pts_yi] = cvRound((j+1)*size_template_x/(n_ctrl_pts_xi+1) - offx);
			ctrl_pts_yi[i + j*n_ctrl_pts_yi] = cvRound((i+1)*size_template_y/(n_ctrl_pts_yi+1) - offy);
		}
}

void	naya::TPSPrecomputationsIllum()
{
	// TPS specific
	cv::Mat Mi(size_template_x*size_template_y, total_n_ctrl_ptsi+3, CV_32FC1);
	MKinvi.create(size_template_x*size_template_y, total_n_ctrl_ptsi, CV_32FC1);
	cv::Mat Kinvi(total_n_ctrl_ptsi+3, total_n_ctrl_ptsi, CV_32FC1);

	// Vector containing gain values
	ctrl_pts_wi.create(total_n_ctrl_ptsi, 1, CV_32FC1);

	// TPS Precomputations start here - See TPSPrecomputations for more info

	// Mounting Matrix 'K'	
	cv::Mat Ki(total_n_ctrl_ptsi+3, total_n_ctrl_ptsi+3, CV_32FC1);

	for(int j=0;j<total_n_ctrl_ptsi;j++)
	{
		Ki.at<float>(j, total_n_ctrl_ptsi) = 1;
		Ki.at<float>(j, total_n_ctrl_ptsi+1) = (float) ctrl_pts_xi[j];
		Ki.at<float>(j, total_n_ctrl_ptsi+2) = (float) ctrl_pts_yi[j];

		Ki.at<float>(total_n_ctrl_ptsi,   j) = 1;
		Ki.at<float>(total_n_ctrl_ptsi+1, j) = (float) ctrl_pts_xi[j];
		Ki.at<float>(total_n_ctrl_ptsi+2, j) = (float) ctrl_pts_yi[j];		
	}

	for(int i=0;i<total_n_ctrl_ptsi;i++)
		for(int j=0;j<total_n_ctrl_ptsi;j++)
			Ki.at<float>(i, j) = Tps(Norm( (float) (ctrl_pts_xi[i]-ctrl_pts_xi[j]), (float) (ctrl_pts_yi[i]-ctrl_pts_yi[j])));

	for(int i=total_n_ctrl_ptsi; i<total_n_ctrl_ptsi+3; i++)
		for(int j=total_n_ctrl_ptsi; j<total_n_ctrl_ptsi+3; j++)
			Ki.at<float>(i, j) = 0;

	// Inverting Matrix 'K'
	cv::Mat K2i = Ki.inv(CV_LU);

	// Passing result to Kinv
	for(int i=0;i<total_n_ctrl_ptsi+3;i++)
		for(int j=0;j<total_n_ctrl_ptsi;j++)
			Kinvi.at<float>(i, j) = K2i.at<float>(i, j);

	// Creating Matrix 'M'	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	for(int i=0;i<size_template_y;i++)
	{
		for(int j=0;j<size_template_x;j++)
		{
			for(int k=0;k<total_n_ctrl_ptsi;k++)
				Mi.at<float>(j+size_template_x*i, k) = Tps(Norm((float)(j-offx - ctrl_pts_xi[k]), (float)(i-offy - ctrl_pts_yi[k])));

			Mi.at<float>(j+size_template_x*i, total_n_ctrl_ptsi) = 1;
			Mi.at<float>(j+size_template_x*i, total_n_ctrl_ptsi+1) = (float) j-offx;
			Mi.at<float>(j+size_template_x*i, total_n_ctrl_ptsi+2) = (float) i-offy;	
		}
	}

	// Pre-computing M with Kinv
	MKinvi = Mi*Kinvi;
}


// Core stuff

void	naya::MountJacobian3DOFGrayxi()
{	
	int i, j;
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);
	float sum_gradx, sum_grady;

	// Mounting matrix 
	for(int k=0;k<n_active_pixels;k++)
	{
		// Active gray pixel positions
		i = cvFloor((float)active_pixels_r[k]/size_template_x);
		j = active_pixels_r[k] - i*size_template_x;

		if(Mask.at<uchar>(i,j) != 0)
		{
			// gradients
			sum_gradx = gradx.at<float>(i,j) + gradx_tmplt.at<float>(i,j);
			sum_grady = grady.at<float>(i,j) + grady_tmplt.at<float>(i,j);

			// img difference
			dif.at<float>(k, 0) = (float)(current_warp.at<uchar>(i,j) - Template->at<uchar>(i,j));

			// gradient
			SD.at<float>(k, 0) = ((float)(j+1-offx))*sum_grady - ((float)(i+1-offy))*sum_gradx;
			SD.at<float>(k, 1) = sum_gradx;
			SD.at<float>(k, 2) = sum_grady;

		}
		else
		{
			// img difference
			dif.at<float>(k,0) = 0;

			// gradient
			SD.at<float>(k, 0) = 0;
			SD.at<float>(k, 1) = 0;
			SD.at<float>(k, 2) = 0;
		}	
	}
}

void	naya::MountJacobian3DOFColorxi(unsigned int ch)
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
            temp[0] = ((float)(j1+1-offx))*sum_grady - ((float)(i1+1-offy))*sum_gradx;
            temp[1] = sum_gradx;
            temp[2] = sum_grady;

            for(int i=0; i<total_n_ctrl_ptsi; i++)
                temp[i+3] = MKinvi.at<float>(active_pixels_g[k], i)*(current_pxl + current_tmplt);

            temp[3+total_n_ctrl_ptsi] = 2;

            for(int i=0;i<=(total_n_ctrl_ptsi+3);i++)
            {
                for(int j=i;j<=(total_n_ctrl_ptsi+3);j++)
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
            temp[0] = ((float)(j2+1-offx))*sum_grady - ((float)(i2+1-offy))*sum_gradx;
            temp[1] = sum_gradx;
            temp[2] = sum_grady;

            for(int i=0; i<total_n_ctrl_ptsi; i++)
                temp[i+3] = MKinvi.at<float>(active_pixels_r[k], i)*(current_pxl + current_tmplt);

            temp[3+total_n_ctrl_ptsi] = 2;

            for(int i=0;i<=(total_n_ctrl_ptsi+3);i++)
            {
                for(int j=i;j<=(total_n_ctrl_ptsi+3);j++)
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


void	naya::MountJacobian3DOFColorxi()
{	
	int i1, j1, i2, j2;
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);
	float sum_gradx, sum_grady;
	unsigned char current_pxl, current_comp_pxl, current_tmplt;

    // Mounting matrix
	for(int k=0;k<n_active_pixels;k++)
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
			dif.at<float>(k, 0) = (float)(current_comp_pxl - current_tmplt);

			// gradient
			SD.at<float>(k, 0) = ((float)(j1+1-offx))*sum_grady - ((float)(i1+1-offy))*sum_gradx;
			SD.at<float>(k, 1) = sum_gradx;
			SD.at<float>(k, 2) = sum_grady;
			
			for(int i=0; i<total_n_ctrl_ptsi; i++)
				SD.at<float>(k, i+3) = MKinvi.at<float>(active_pixels_g[k], i)*(current_pxl + current_tmplt);

			SD.at<float>(k, 3+total_n_ctrl_ptsi) = 2;
		}
		else
		{
			// img difference
			dif.at<float>(k,0) = 0;

			// gradient
			SD.at<float>(k, 0) = 0;
			SD.at<float>(k, 1) = 0;
			SD.at<float>(k, 2) = 0;

			for(int i=0; i<total_n_ctrl_ptsi; i++)
				SD.at<float>(k, i+3) = 0;

			SD.at<float>(k, 3+total_n_ctrl_ptsi) = 0;
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
			dif.at<float>(k + n_active_pixels, 0) = (float)(current_comp_pxl - current_tmplt);

			// gradient
			SD.at<float>(k + n_active_pixels, 0) = ((float)(j2+1-offx))*sum_grady - ((float)(i2+1-offy))*sum_gradx;
			SD.at<float>(k + n_active_pixels, 1) = sum_gradx;
			SD.at<float>(k + n_active_pixels, 2) = sum_grady;
			
			for(int i=0; i<total_n_ctrl_ptsi; i++)
				SD.at<float>(k+ n_active_pixels, i+3) = MKinvi.at<float>(active_pixels_r[k], i)*(current_pxl + current_tmplt);
			
			SD.at<float>(k+ n_active_pixels, 3+total_n_ctrl_ptsi) = 2;
		}
		else
		{
			// img difference
			dif.at<float>(k + n_active_pixels, 0) = 0;

			// gradient
			SD.at<float>(k + n_active_pixels, 0) = 0;
			SD.at<float>(k + n_active_pixels, 1) = 0;
			SD.at<float>(k + n_active_pixels, 2) = 0;
			
			for(int i=0; i<total_n_ctrl_ptsi; i++)
				SD.at<float>(k+ n_active_pixels, i+3) = 0;

			SD.at<float>(k+ n_active_pixels, 3+total_n_ctrl_ptsi) = 0;
		}	
	}
}


int		naya::Update3DOFi()
{	
	float sum = 0;

	// ESM update
	delta = 2*((SD.t()*SD).inv(CV_SVD)*(SD.t()*dif));

	// Update
    angle_3DOF -= delta.at<float>(0, 0); sum += fabs(delta.at<float>(0, 0));
    parameters[2] -= delta.at<float>(1, 0); sum += fabs(delta.at<float>(1, 0));
    parameters[3] -= delta.at<float>(2, 0); sum += fabs(delta.at<float>(2, 0));

	for(int i=0; i<total_n_ctrl_ptsi; i++)
    {
		ctrl_pts_wi.at<float>(i, 0) -= delta.at<float>(3+i, 0);
        sum += fabs(delta.at<float>(3+i, 0));
    }

    bias -= delta.at<float>(total_n_ctrl_ptsi+3, 0);

	return sum < epsilon;
}

void	naya::NonRigidCompensation(unsigned int ch)
{
	// Computes compensated warped image 
	if(isgrayscale)
	{
		/*for(int i=0; i<size_template_x*size_template_y; i++)
		{
			*(compensated_warp.ptr<uchar>(0)+i) = cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+i)) + bias);
		}*/
	}
	else
	{
		for(int i=start_stop2[2*ch]; i<=start_stop2[2*ch+1]; i++)
		{
			*(compensated_warp.ptr<uchar>(0)+3*i) =   cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+3*i))   + bias);
			*(compensated_warp.ptr<uchar>(0)+3*i+1) = cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+3*i+1)) + bias);
			*(compensated_warp.ptr<uchar>(0)+3*i+2) = cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+3*i+2)) + bias);
		}
	}
}


void	naya::NonRigidCompensation()
{
	// Computes compensated warped image 
	if(isgrayscale)
	{
		for(int i=0; i<size_template_x*size_template_y; i++)
		{
			*(compensated_warp.ptr<uchar>(0)+i) = cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+i)) + bias);
		}
	}
	else
	{
		for(int i=0; i<size_template_x*size_template_y; i++)
		{
			*(compensated_warp.ptr<uchar>(0)+3*i) =   cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+3*i))   + bias);
			*(compensated_warp.ptr<uchar>(0)+3*i+1) = cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+3*i+1)) + bias);
			*(compensated_warp.ptr<uchar>(0)+3*i+2) = cv::saturate_cast<uchar>(gain.at<float>(i,0)*(*(current_warp.ptr<uchar>(0)+3*i+2)) + bias);
		}
	}
}

float	naya::ComputeTrackingConfidenceSSDi()
{
	int i,j, counter=0;

	int   I1_bar[3] = {0},
		I2_bar[3] = {0},
		I1sq[3] = {0},
		I2sq[3] = {0};

	long  erro[3] = {0};

	float ncc = 0,
		stds[3] = {0};

	for(int k=1;k<3;k++)
	{
		for(i=0;i<current_warp.rows;i++)
		{
			for(j=0;j<current_warp.cols;j++)
			{
				if(Mask.at<uchar>(i,j) == 255)
				{
					I1_bar[k] += (int)compensated_warp.ptr<uchar>(i)[3*j+k];
                    I2_bar[k] += (int)Template->ptr<uchar>(i)[3*j+k];
					counter++;
				}
			}
		}

		if(counter<1)
		{
			printf("No active pixels to cmpute NCC!! \n");
			counter = 1;
		}

		I1_bar[k] = I1_bar[k]/(counter);
		I2_bar[k] = I2_bar[k]/(counter);

	}

	for(int k=1;k<3;k++)
	{
		for(i=0;i<Template->rows;i++)
		{
			for(j=0;j<Template->cols;j++)
			{	
				if(Mask.at<uchar>(i,j) == 255)
				{
					I1sq[k] += (compensated_warp.ptr<uchar>(i)[3*j+k] - I1_bar[k])*(compensated_warp.ptr<uchar>(i)[3*j+k] - I1_bar[k]);
					I2sq[k] += (Template->ptr<uchar>(i)[3*j+k] - I2_bar[k])*(Template->ptr<uchar>(i)[3*j+k] - I2_bar[k]);

					erro[k] += (compensated_warp.ptr<uchar>(i)[3*j+k] - I1_bar[k])*(Template->ptr<uchar>(i)[3*j+k] - I2_bar[k]);
				}
			}
		}

		stds[k] = std::sqrt((float)I1sq[k]*(float)I2sq[k]);
	}


	for(int k=1;k<3;k++)
	{
		if(stds[k]>0)
			ncc += (float)erro[k]/stds[k];
	}

	if(ncc<0)
	{
		printf("Somthing went wrong when computing NCC coef! \n");
		return 0;
	}

	return ncc/2;
}

// Display

void	naya::Display3DOFxi(cv::Mat *ICur, int delay)
{
	cv::line(*ICur,
			cvPoint( (parameters[0]*-size_template_x/2 - parameters[1]*-size_template_y/2 + parameters[2]), (parameters[1]*-size_template_x/2 + parameters[0]*-size_template_y/2 + parameters[3])),
			cvPoint( (parameters[0]*size_template_x/2 - parameters[1]*-size_template_y/2 + parameters[2]), (parameters[1]*size_template_x/2 + parameters[0]*-size_template_y/2 + parameters[3])),
			CV_RGB( 255, 255, 255), 2, 8, 0);
	
	cv::line(*ICur,
			cvPoint( (parameters[0]*-size_template_x/2 - parameters[1]*-size_template_y/2 + parameters[2]), (parameters[1]*-size_template_x/2 + parameters[0]*-size_template_y/2 + parameters[3])),
			cvPoint( (parameters[0]*-size_template_x/2 - parameters[1]*size_template_y/2 + parameters[2]), (parameters[1]*-size_template_x/2 + parameters[0]*size_template_y/2 + parameters[3])),
			CV_RGB( 255, 255, 255), 2, 8, 0);
	
	cv::line(*ICur,
			cvPoint( (parameters[0]*size_template_x/2 - parameters[1]*size_template_y/2 + parameters[2]), (parameters[1]*size_template_x/2 + parameters[0]*size_template_y/2 + parameters[3])),
			cvPoint( (parameters[0]*-size_template_x/2 - parameters[1]*size_template_y/2 + parameters[2]), (parameters[1]*-size_template_x/2 + parameters[0]*size_template_y/2 + parameters[3])),
			CV_RGB( 255, 255, 255), 2, 8, 0);
	
	cv::line(*ICur,
			cvPoint( (parameters[0]*size_template_x/2 - parameters[1]*size_template_y/2 + parameters[2]), (parameters[1]*size_template_x/2 + parameters[0]*size_template_y/2 + parameters[3]) ),
			cvPoint( (parameters[0]*size_template_x/2 - parameters[1]*-size_template_y/2 + parameters[2]), (parameters[1]*size_template_x/2 + parameters[0]*-size_template_y/2 + parameters[3]) ),
			CV_RGB( 255, 255, 255), 2, 8, 0);


	cv::imshow("Current Image", *ICur);
	cv::imshow("Template", *Template); 
	cv::imshow("Current Warp NC", current_warp);
	cv::imshow("Current Warp", compensated_warp);
	cv::waitKey(delay);
	
}


void	naya::ResetIlluminationParam3DOFxi(float *illum_param)
{
	for(int i=0; i<total_n_ctrl_ptsi; i++)
		illum_param[i]  = 1;

	illum_param[total_n_ctrl_ptsi] = 0;
}


void	naya::Warp3DOFAux(unsigned int ch, float *parameters)
{
    int offx = cvCeil((double)size_template_x/2);
    int offy = cvCeil((double)size_template_y/2);

    // Multiplying matrices
    for(int i=start_stop[2*ch]-offy;i<=start_stop[2*ch+1]-offy;i++)
    {
        for(int j=-offx;j<offx;j++)
        {
            dummy_mapx.at<float>(i+offy,j+offx) = (parameters[0]*(float)j + parameters[1]*(float)i + parameters[2])/(parameters[6]*(float)j + parameters[7]*(float)i + parameters[8]);
            dummy_mapy.at<float>(i+offy,j+offx) = (parameters[3]*(float)j + parameters[4]*(float)i + parameters[5])/(parameters[6]*(float)j + parameters[7]*(float)i + parameters[8]);
        }
    }

    // Remapping
    cv::remap(*ICur, compensated_warp(ok[ch]), dummy_mapx(ok[ch]), dummy_mapy(ok[ch]), interp, 0, cv::Scalar(0));

    if(using_masks)
        cv::remap(*Mask_roi, Mask(ok[ch]), dummy_mapx(ok[ch]), dummy_mapy(ok[ch]), 0, 0, cv::Scalar(0));
}








