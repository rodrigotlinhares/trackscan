
/**********************/
/*** naya class ***/
/**********************/

#include "naya.h"


// Initialization

void naya::Initialize2DOF(int size_template_x, 
							int size_template_y, 
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
	this->epsilon = epsilon;
	this->n_bins = n_bins;
	this->size_bins = size_bins;
	this->n_max_iters = n_max_iters;
	this->isgrayscale = isgrayscale;
	this->interp = interp;
	
	// Activel pixel stuff
	visited_r = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));
	visited_g = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));
	visited_b = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));
		
	for(int i=0; i<2; i++)
	{
		pair_r[i].resize(size_template_x*size_template_y);
		pair_g[i].resize(size_template_x*size_template_y);
	}

	// Reference pixel list when n_active_pixels = size_template_x*size_template_y
	std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
	for(int i=0;i<size_template_x*size_template_y;i++)
		std_pixel_list[i] = i;

	// Rest of allocations
	dummy_mapx.create(size_template_y, size_template_x, CV_32FC1);
	dummy_mapy.create(size_template_y, size_template_x, CV_32FC1);
	delta.create(2, 1, CV_32FC1);
	Mask.create(size_template_y, size_template_x, CV_8UC1);
	Mask.setTo(255);

	if(isgrayscale)
	{
		dif.create(size_template_x*size_template_y, 1, CV_32FC1);
		SD.create(size_template_x*size_template_y, 2, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC1);

		gradx.create(size_template_y, size_template_x, CV_32FC1);
		grady.create(size_template_y, size_template_x, CV_32FC1);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		current_warp.create(size_template_y, size_template_x, CV_8UC1);

		expected = (float*)malloc(n_bins*sizeof(float));
	}
	else
	{
		dif.create(size_template_x*size_template_y*3, 1, CV_32FC1);
		SD.create(size_template_x*size_template_y*3, 2, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC3);

		gradx.create(size_template_y, size_template_x, CV_32FC3);
		grady.create(size_template_y, size_template_x, CV_32FC3);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		current_warp.create(size_template_y, size_template_x, CV_8UC3);

		expected = (float*)malloc(3*n_bins*sizeof(float));
	}

	// Joint histogram
	correction = (float*)malloc(n_bins*sizeof(float));
	p_joint =  (float*)malloc(n_bins*n_bins*sizeof(float));

	ResetExpected();
}

void naya::Initialize3DOFx(int size_template_x, 
							int size_template_y, 
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

    fast_pair_r.resize(size_template_x*size_template_y);
    fast_pair_g.resize(size_template_x*size_template_y);

	// Reference pixel list when n_active_pixels = size_template_x*size_template_y
	std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
	for(int i=0;i<size_template_x*size_template_y;i++)
		std_pixel_list[i] = i;

	// Rest of allocations
	dummy_mapx.create(size_template_y, size_template_x, CV_32FC1);
	dummy_mapy.create(size_template_y, size_template_x, CV_32FC1);
	delta.create(3, 1, CV_32FC1);
	Mask.create(size_template_y, size_template_x, CV_8UC1);
	Mask.setTo(255);

	if(isgrayscale)
	{
		dif.create(n_active_pixels, 1, CV_32FC1);
		SD.create(n_active_pixels, 3, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC1);

		gradx.create(size_template_y, size_template_x, CV_32FC1);
		grady.create(size_template_y, size_template_x, CV_32FC1);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		current_warp.create(size_template_y, size_template_x, CV_8UC1);

		expected = (float*)malloc(n_bins*sizeof(float));
	}
	else
	{
		dif.create(n_active_pixels*2, 1, CV_32FC1);
		SD.create(n_active_pixels*2, 3, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC3);

		gradx.create(size_template_y, size_template_x, CV_32FC3);
		grady.create(size_template_y, size_template_x, CV_32FC3);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		current_warp.create(size_template_y, size_template_x, CV_8UC3);

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

}

void naya::Initialize4DOF(int size_template_x, 
							int size_template_y, 
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
	visited_b = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));
		
	for(int i=0; i<4; i++)
	{
		pair_r[i].resize(size_template_x*size_template_y);
		pair_g[i].resize(size_template_x*size_template_y);
		pair_b[i].resize(size_template_x*size_template_y);
	}
	
	// Reference pixel list when n_active_pixels = size_template_x*size_template_y
	std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
	for(int i=0;i<size_template_x*size_template_y;i++)
		std_pixel_list[i] = i;

	// Rest of allocations
	dummy_mapx.create(size_template_y, size_template_x, CV_32FC1);
	dummy_mapy.create(size_template_y, size_template_x, CV_32FC1);
	delta.create(4, 1, CV_32FC1);
	Mask.create(size_template_y, size_template_x, CV_8UC1);
	Mask.setTo(255);

	if(isgrayscale)
	{
		dif.create(n_active_pixels, 1, CV_32FC1);
		SD.create(n_active_pixels, 4, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC1);

		gradx.create(size_template_y, size_template_x, CV_32FC1);
		grady.create(size_template_y, size_template_x, CV_32FC1);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		current_warp.create(size_template_y, size_template_x, CV_8UC1);

		expected = (float*)malloc(n_bins*sizeof(float));
	}
	else
	{
		dif.create(n_active_pixels*3, 1, CV_32FC1);
		SD.create(n_active_pixels*3, 4, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC3);

		gradx.create(size_template_y, size_template_x, CV_32FC3);
		grady.create(size_template_y, size_template_x, CV_32FC3);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		current_warp.create(size_template_y, size_template_x, CV_8UC3);

		expected = (float*)malloc(3*n_bins*sizeof(float));
	}

	// Joint histogram
	correction = (float*)malloc(n_bins*sizeof(float));
	p_joint =  (float*)malloc(n_bins*n_bins*sizeof(float));

	ResetExpected();
}

void naya::Initialize8DOF(int size_template_x, 
								int size_template_y, 
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
	visited_b = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));	

	for(int i=0; i<8; i++)
	{
		pair_r[i].resize(size_template_x*size_template_y);
		pair_g[i].resize(size_template_x*size_template_y);
		pair_b[i].resize(size_template_x*size_template_y);
	}
	
	// Reference pixel list when n_active_pixels = size_template_x*size_template_y
	std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
	for(int i=0;i<size_template_x*size_template_y;i++)
		std_pixel_list[i] = i;

	dummy_mapx.create(size_template_y, size_template_x, CV_32FC1);
	dummy_mapy.create(size_template_y, size_template_x, CV_32FC1);
	delta.create(8, 1, CV_32FC1);
	Mask.create(size_template_y, size_template_x, CV_8UC1);
	Mask.setTo(255);

	if(isgrayscale)
	{
		dif.create(n_active_pixels, 1, CV_32FC1);
		SD.create(n_active_pixels, 8, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC1);

		gradx.create(size_template_y, size_template_x, CV_32FC1);
		grady.create(size_template_y, size_template_x, CV_32FC1);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		current_warp.create(size_template_y, size_template_x, CV_8UC1);

		expected = (float*)malloc(n_bins*sizeof(float));
	}
	else
	{
		dif.create(n_active_pixels*3, 1, CV_32FC1);
		SD.create(n_active_pixels*3, 8, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC3);

		gradx.create(size_template_y, size_template_x, CV_32FC3);
		grady.create(size_template_y, size_template_x, CV_32FC3);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		current_warp.create(size_template_y, size_template_x, CV_8UC3);

		expected = (float*)malloc(3*n_bins*sizeof(float));
	}

	// Aux Lie
	update_auxA.create(3, 3, CV_32FC1);	
	update_auxH.create(3, 3, CV_32FC1);	
	aux1.create(3, 3, CV_32FC1);
	aux2.create(3, 3, CV_32FC1);
	aux3.create(3, 3, CV_32FC1);
	aux4.create(3, 3, CV_32FC1);
	aux5.create(3, 3, CV_32FC1);

	// Joint histogram
	correction = (float*)malloc(n_bins*sizeof(float));
	p_joint =  (float*)malloc(n_bins*n_bins*sizeof(float));

	ResetExpected();
}

void naya::InitializeTPS(int size_template_x, 
							int size_template_y, 
							int n_active_pixels, 
							int n_ctrl_pts_x,
							int n_ctrl_pts_y,
							float lambda,
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
	this->n_active_pixels = n_active_pixels;
	this->n_ctrl_pts_x = n_ctrl_pts_x;
	this->n_ctrl_pts_y = n_ctrl_pts_y;
	this->lambda = lambda;
	this->epsilon = epsilon;
	this->n_bins = n_bins;
	this->size_bins = size_bins;
	this->n_max_iters = n_max_iters;
	this->isgrayscale = isgrayscale;
	this->interp = interp;

	// Reference pixel list when n_active_pixels = size_template_x*size_template_y
	std_pixel_list = (int*)malloc(size_template_x*size_template_y*sizeof(int));
	for(int i=0;i<size_template_x*size_template_y;i++)
		std_pixel_list[i] = i;

	// Allocates control point vectors
	total_n_ctrl_pts = n_ctrl_pts_y*n_ctrl_pts_x;
	DefineCtrlPts();

	// TPS Precomputations
	TPSPrecomputations();
	
	// Activel pixel stuff
	visited_r = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));
	visited_g = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));
	visited_b = (bool*) malloc(size_template_x*size_template_y*sizeof(bool));	

	for(int i=0; i<total_n_ctrl_pts; i++)
	{
		pair_r[i].resize(size_template_x*size_template_y);
		pair_g[i].resize(size_template_x*size_template_y);
	}

	// The dummy mapping matrices' shape are different in the TPS code
	dummy_mapx.create(size_template_y*size_template_x, 1, CV_32FC1);
	dummy_mapy.create(size_template_y*size_template_x, 1, CV_32FC1);

	// The rest is pretty much standard
	delta.create(2*total_n_ctrl_pts, 1, CV_32FC1);
	Mask.create(size_template_y, size_template_x, CV_8UC1);
	Mask.setTo(255);

	if(isgrayscale)
	{
		dif.create(n_active_pixels, 1, CV_32FC1);
		SD.create(n_active_pixels, 2*total_n_ctrl_pts, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC1);

		gradx.create(size_template_y, size_template_x, CV_32FC1);
		grady.create(size_template_y, size_template_x, CV_32FC1);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC1);
		current_warp.create(size_template_y, size_template_x, CV_8UC1);

		expected = (float*)malloc(n_bins*sizeof(float));
	}
	else
	{
		dif.create(n_active_pixels*3, 1, CV_32FC1);
		SD.create(n_active_pixels*3, 2*total_n_ctrl_pts, CV_32FC1);
		Template_comp.create(size_template_y, size_template_x, CV_8UC3);

		gradx.create(size_template_y, size_template_x, CV_32FC3);
		grady.create(size_template_y, size_template_x, CV_32FC3);
		gradx_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		grady_tmplt.create(size_template_y, size_template_x, CV_32FC3);
		current_warp.create(size_template_y, size_template_x, CV_8UC3);

		expected = (float*)malloc(3*n_bins*sizeof(float));
	}

	// Joint histogram
	correction = (float*)malloc(n_bins*sizeof(float));
	p_joint =  (float*)malloc(n_bins*n_bins*sizeof(float));

	ResetExpected();
}

// naya Run

int		naya::Run2DOF(cv::Mat *ICur, 
						cv::Mat *Mask_roi,
						cv::Mat *Template,
						cv::Mat *Mask_template,
						float *parameters)
{
	int flag_tracking = 1;

	// Taking in input arguments
	this->ICur = ICur;
	this->Mask_roi = Mask_roi;
	this->Mask_template = Mask_template;
	this->Template = Template;
	this->parameters = parameters;

	// Computing expected Template
	ComputeExpectedImg();

	// Are we using masks?
	(Mask_roi == 0 || Mask_template == 0) ? using_masks = 0 : using_masks = 1;

	// Fun begins
	for(iters=0;iters<n_max_iters;iters++)
	{		
		// Computing mapped positions in parallel
		Warp2DOF();

		// Computes image gradients
		WarpGrad();
		
		// Computes occlusion map
		if(using_masks)
			OcclusionMap();

		// Mounts Jacobian
		isgrayscale ? MountJacobian2DOFGray() : MountJacobian2DOFColor();

		// Updates naya parameters
		if(Update2DOF())
			break;	
	}	
	
	// Computes joint histogram
	if(isgrayscale)
	{
		if(ComputeJointHistogramGray())
			flag_tracking = 0;
	}
	else
	{
		if(ComputeJointHistogramColor())
			flag_tracking = 0;
	}	

	return flag_tracking;
}


int		naya::Run3DOFx(cv::Mat *ICur, 
						cv::Mat *Mask_roi,
						cv::Mat *Template,
						cv::Mat *Mask_template,
						float *parameters,
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
	// Computing expected Template
	ComputeExpectedImg();

	// Converting parameters
	angle_3DOF = std::atan2(parameters[1],parameters[0]);

	// Fun begins
	for(iters=0;iters<n_max_iters;iters++)
	{	
		// Computing mapped positions in parallel
		Warp3DOF();

		// Computes image gradients
		WarpGrad();
		
		// Computes occlusion map
		if(using_masks)
			OcclusionMap();

		// Mounts Jacobian
		isgrayscale ? MountJacobian3DOFGrayx() : MountJacobian3DOFColorx();

		// Updates naya parameters
		if(Update3DOF())
			break;	
	}
		
	// Computes joint histogram
	if(isgrayscale)
	{
		if(ComputeJointHistogramGray())
			flag_tracking = 0;
	}
	else
	{
		if(ComputeJointHistogramColor())
			flag_tracking = 0;
	}	

	// Converting back to original form
	parameters[0] = std::cos(angle_3DOF);
	parameters[1] = std::sin(angle_3DOF);

	return flag_tracking;
}


int		naya::Run4DOF(cv::Mat *ICur, 
						cv::Mat *Mask_roi,
						cv::Mat *Template,
						cv::Mat *Mask_template,
						float *parameters,
						int *active_pixels_r,
						int *active_pixels_g,
						int *active_pixels_b)
{
	int flag_tracking = 1;

	// Taking in input arguments
	this->ICur = ICur;
	this->Mask_roi = Mask_roi;
	this->Mask_template = Mask_template;
	this->Template = Template;
	this->parameters = parameters;

	if(active_pixels_r == 0)
	{
		this->active_pixels_r = std_pixel_list;
		this->active_pixels_g = std_pixel_list;
		this->active_pixels_b = std_pixel_list;
	}
	else
	{
		this->active_pixels_r = active_pixels_r;
		this->active_pixels_g = active_pixels_g;
		this->active_pixels_b = active_pixels_b;
	}

	// Are we using masks?
	(Mask_roi == 0 || Mask_template == 0) ? using_masks = 0 : using_masks = 1;

	// Computing expected Template
	ComputeExpectedImg();

	// Fun begins
	for(iters=0; iters<n_max_iters; iters++)
	{				
		// Computing mapped positions in parallel
		Warp4DOF();

		// Computes image gradients
		WarpGrad();

		// Computes occlusion map
		if(using_masks)
			OcclusionMap();

		// Mounts Jacobian
		isgrayscale ? MountJacobian4DOFGray() : MountJacobian4DOFColor();

		// Updates naya parameters
		if(Update4DOF())
			break;	
	}
	
	// Computes joint histogram
	if(isgrayscale)
	{
		if(ComputeJointHistogramGray())
			flag_tracking = 0;
	}
	else
	{
		if(ComputeJointHistogramColor())
			flag_tracking = 0;
	}	
	
	return flag_tracking;
}

int		naya::Run8DOF(cv::Mat *ICur, 
						cv::Mat *Mask_roi,
						cv::Mat *Template,
						cv::Mat *Mask_template,
						float *parameters,
						int *active_pixels_r,
						int *active_pixels_g,
						int *active_pixels_b)
{
	int flag_tracking = 1;

	// Taking in input arguments
	this->ICur = ICur;	
	this->Template = Template;
	this->parameters = parameters;
	this->active_pixels_b = active_pixels_b;
	this->active_pixels_g = active_pixels_g;
	this->active_pixels_r = active_pixels_r;
	this->Mask_roi = Mask_roi;
	this->Mask_template = Mask_template;

	if(active_pixels_r == 0)
	{
		this->active_pixels_r = std_pixel_list;
		this->active_pixels_g = std_pixel_list;
		this->active_pixels_b = std_pixel_list;
	}
	else
	{
		this->active_pixels_r = active_pixels_r;
		this->active_pixels_g = active_pixels_g;
		this->active_pixels_b = active_pixels_b;
	}

	// Are we using masks?
	(Mask_roi == 0 || Mask_template == 0) ? using_masks = 0 : using_masks = 1;

	// Computing expected Template
	ComputeExpectedImg();

	// Fun begins
	for(iters=0; iters<n_max_iters; iters++)
	{				
		// Computing mapped positions in parallel
		Warp8DOF();

		// Computes image gradients
		WarpGrad();

		// Computes occlusion map
		if(using_masks)
			OcclusionMap();

		// Mounts Jacobian
		isgrayscale ? MountJacobian8DOFGray() : MountJacobian8DOFColor();
		
		// Updates naya parameters
		if(Update8DOF())
			break;	
	}

	// Computes joint histogram
	if(isgrayscale)
	{
		if(ComputeJointHistogramGray())
			flag_tracking = 0;
	}
	else
	{
		if(ComputeJointHistogramColor())
			flag_tracking = 0;
	}	

	return flag_tracking;
}

int		naya::RunTPS(cv::Mat *ICur, 
						cv::Mat *Mask_roi,
						cv::Mat *Template,
						cv::Mat *Mask_template,
						float *parameters,
						int *active_pixels_r,
						int *active_pixels_g,
						int *active_pixels_b)
{
	int flag_tracking = 1;

	// Taking in input arguments
	this->ICur = ICur;	
	this->Template = Template;
	this->parameters = parameters;
	this->active_pixels_b = active_pixels_b;
	this->active_pixels_g = active_pixels_g;
	this->active_pixels_r = active_pixels_r;
	this->Mask_roi = Mask_roi;
	this->Mask_template = Mask_template;

	if(active_pixels_r == 0)
	{
		this->active_pixels_r = std_pixel_list;
		this->active_pixels_g = std_pixel_list;
		this->active_pixels_b = std_pixel_list;
	}
	else
	{
		this->active_pixels_r = active_pixels_r;
		this->active_pixels_g = active_pixels_g;
		this->active_pixels_b = active_pixels_b;
	}

	// In the TPS code, I must copy the input parameter vector to another structure
	ParameterIO(1);

	// Are we using masks?
	(Mask_roi == 0 || Mask_template == 0) ? using_masks = 0 : using_masks = 1;

	// Computing expected Template
	ComputeExpectedImg();

	// Fun begins
	for(iters=0; iters<n_max_iters; iters++)
	{				
		// Computing mapped positions in parallel
		WarpTPS();

		// Computes image gradients
		WarpGrad();

		// Computes occlusion map
		if(using_masks)
			OcclusionMap();

		// Mounts Jacobian
		isgrayscale ? MountJacobianTPSGray() : MountJacobianTPSColor();

		// Updates naya parameters
		if(UpdateTPS())
			break;	
	}

	// Computes joint histogram
	if(isgrayscale)
	{
		if(ComputeJointHistogramGray())
			flag_tracking = 0;
	}
	else
	{
		if(ComputeJointHistogramColor())
			flag_tracking = 0;
	}	

	// Output parameters
	ParameterIO(0);

	return flag_tracking;
}


// Warping images

void	naya::Warp2DOF()
{
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	// Multiplying matrices
	for(int i=-offy;i<offy;i++)
	{
		for(int j=-offx;j<offx;j++)
		{
			dummy_mapx.at<float>(i+offy,j+offx) = (float)j + parameters[2];
			dummy_mapy.at<float>(i+offy,j+offx) = (float)i + parameters[3];
		}
	}

	// Remapping 
	cv::remap(*ICur, current_warp, dummy_mapx, dummy_mapy, interp, 0, cv::Scalar(0));
	
	if(using_masks)
		cv::remap(*Mask_roi, Mask, dummy_mapx, dummy_mapy, 0, 0, cv::Scalar(0));
}

void	naya::Warp3DOF()
{
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	float coseno = std::cos(angle_3DOF);
	float seno = std::sin(angle_3DOF);
	
    // Multiplying matrices
    for(int i=-offy;i<offy;i++)
    {
        for(int j=-offx;j<offx;j++)
        {
            dummy_mapx.at<float>(i+offy,j+offx) = coseno*(float)j - seno*(float)i + parameters[2];
            dummy_mapy.at<float>(i+offy,j+offx) = seno*(float)j + coseno*(float)i + parameters[3];
        }
    }

	// Remapping 
	cv::remap(*ICur, current_warp, dummy_mapx, dummy_mapy, interp, 0, cv::Scalar(0));
	
	if(using_masks)
		cv::remap(*Mask_roi, Mask, dummy_mapx, dummy_mapy, 0, 0, cv::Scalar(0));
}


void	naya::Warp4DOF()
{
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	// Multiplying matrices
	for(int i=-offy;i<offy;i++)
	{
		for(int j=-offx;j<offx;j++)
		{
			dummy_mapx.at<float>(i+offy,j+offx) = parameters[0]*(float)j - parameters[1]*(float)i + parameters[2];
			dummy_mapy.at<float>(i+offy,j+offx) = parameters[1]*(float)j + parameters[0]*(float)i + parameters[3];
		}
	}

	// Remapping 
	cv::remap(*ICur, current_warp, dummy_mapx, dummy_mapy, interp, 0, cv::Scalar(0));
	
	if(using_masks)
		cv::remap(*Mask_roi, Mask, dummy_mapx, dummy_mapy, 0, 0, cv::Scalar(0));
}

void	naya::Warp8DOF()
{
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	// Multiplying matrices
	for(float i=-(float)offy;i<(float)offy;i++)
	{
		for(float j=-(float)offx;j<(float)offx;j++)
		{
			float z = parameters[6]*j + parameters[7]*i + parameters[8];

			dummy_mapx.at<float>(i+offy,j+offx) = (parameters[0]*j + parameters[1]*i + parameters[2])/z;
			dummy_mapy.at<float>(i+offy,j+offx) = (parameters[3]*j + parameters[4]*i + parameters[5])/z;
		}
	}

	// Remapping 
	cv::remap(*ICur, current_warp, dummy_mapx, dummy_mapy, interp, 0, cv::Scalar(0));

	if(using_masks)
		cv::remap(*Mask_roi, Mask, dummy_mapx, dummy_mapy, 0, 0, cv::Scalar(0));
}

void	naya::WarpTPS()
{
	// Mapping pixel positions
	dummy_mapx = MKinv*ctrl_pts_x_w;
	dummy_mapy = MKinv*ctrl_pts_y_w;

	// Remapping 
	cv::remap(*ICur, current_warp, dummy_mapx.reshape(1, size_template_y), dummy_mapy.reshape(1, size_template_y), interp, 0, cvScalar(0));

	if(using_masks)
		cv::remap(*Mask_roi, Mask, dummy_mapx.reshape(1, size_template_y), dummy_mapy.reshape(1, size_template_y), 0, 0, cv::Scalar(0));
}


// Computing gradients

void	naya::WarpGrad()
{
	cv::Sobel(current_warp, gradx, CV_32F, 1, 0, 1);
	cv::Sobel(current_warp, grady, CV_32F, 0, 1, 1);
}

void	naya::WarpGrad(cv::Mat *Input)
{
	cv::Sobel(*Input, gradx, CV_32F, 1, 0, 1);
	cv::Sobel(*Input, grady, CV_32F, 0, 1, 1);
}

// Computing occlusion mask

void	naya::OcclusionMap()
{
	// Scans image
	for(int i=0;i<size_template_y;i++)
	{
		for(int j=0;j<size_template_x;j++)
		{
			if(Mask_template->at<uchar>(i,j) == 0)
				Mask.at<unsigned char>(i,j) = 0;
		}
	}
}



// Mounting Jacobian matrix

void	naya::MountJacobian2DOFGray()
{	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	// Mounting matrix 
	int k = 0;

	for(int i1=0; i1<size_template_y; i1++)
	{
		for(int j1=0; j1<size_template_x; j1++)
		{
			if(Mask.at<uchar>(i1,j1) != 0)
			{
				// gradients
				float sum_gradx = gradx.at<float>(i1,j1) + gradx_tmplt.at<float>(i1,j1);
				float sum_grady = grady.at<float>(i1,j1) + grady_tmplt.at<float>(i1,j1);

				// img difference
				dif.at<float>(k, 0) = (float)(current_warp.at<uchar>(i1,j1) - Template_comp.at<uchar>(i1,j1));

				// gradient
				SD.at<float>(k, 0) = sum_gradx;
				SD.at<float>(k, 1) = sum_grady;

			}
			else
			{
				// img difference
				dif.at<float>(k,0) = 0;

				// gradient
				SD.at<float>(k, 0) = 0;
				SD.at<float>(k, 1) = 0;
			}

			// Counter incremenet
			k++;
		}
	}	
}

void	naya::MountJacobian2DOFColor()
{	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	// Mounting matrix 
	int k = 0;
	for(int i1=0; i1<size_template_y; i1++)
	{
		for(int j1=0; j1<size_template_x; j1++)
		{
			if(Mask.at<uchar>(i1,j1) != 0)
			{
				// img difference
				dif.at<float>(3*k, 0) = (float)(current_warp.ptr<uchar>(i1)[3*j1] - Template_comp.ptr<uchar>(i1)[3*j1]);
				dif.at<float>(3*k+1, 0) = (float)(current_warp.ptr<uchar>(i1)[3*j1+1] - Template_comp.ptr<uchar>(i1)[3*j1+1]);
				dif.at<float>(3*k+2, 0) = (float)(current_warp.ptr<uchar>(i1)[3*j1+2] - Template_comp.ptr<uchar>(i1)[3*j1+2]);
				
				// Jacobian
				SD.at<float>(3*k, 0) = gradx.ptr<float>(i1)[3*j1] + gradx_tmplt.ptr<float>(i1)[3*j1];
				SD.at<float>(3*k, 1) = grady.ptr<float>(i1)[3*j1] + grady_tmplt.ptr<float>(i1)[3*j1];
				SD.at<float>(3*k+1, 0) = gradx.ptr<float>(i1)[3*j1+1] + gradx_tmplt.ptr<float>(i1)[3*j1+1];
				SD.at<float>(3*k+1, 1) = grady.ptr<float>(i1)[3*j1+1] + grady_tmplt.ptr<float>(i1)[3*j1+1];
				SD.at<float>(3*k+2, 0) = gradx.ptr<float>(i1)[3*j1+2] + gradx_tmplt.ptr<float>(i1)[3*j1+2];
				SD.at<float>(3*k+2, 1) = grady.ptr<float>(i1)[3*j1+2] + grady_tmplt.ptr<float>(i1)[3*j1+2];

			}
			else
			{
				// img difference
				dif.at<float>(3*k,0) = 0;
				dif.at<float>(3*k+1,0) = 0;
				dif.at<float>(3*k+2,0) = 0;

				// gradient
				SD.at<float>(3*k, 0) = 0;
				SD.at<float>(3*k, 1) = 0;
				SD.at<float>(3*k+1, 0) = 0;
				SD.at<float>(3*k+1, 1) = 0;
				SD.at<float>(3*k+2, 0) = 0;
				SD.at<float>(3*k+2, 1) = 0;
			}

			// Counter increment
			k++;
		}
	}	
}

void	naya::MountJacobian3DOFGrayx()
{	
	int i1, j1, i2, j2;
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);
	float sum_gradx, sum_grady;

	// Mounting matrix 
	for(int k=0;k<n_active_pixels;k++)
	{
		// Active red pixel positions
		i2 = cvFloor((float)active_pixels_r[k]/size_template_x);
		j2 = active_pixels_r[k] - i2*size_template_x;

		if(Mask.at<uchar>(i2,j2) != 0)
		{
			// gradients
			sum_gradx = gradx.at<float>(i2,j2) + gradx_tmplt.at<float>(i2,j2);
			sum_grady = grady.at<float>(i2,j2) + grady_tmplt.at<float>(i2,j2);

			// img difference
			dif.at<float>(k, 0) = (float)(current_warp.at<uchar>(i2,j2) - Template_comp.at<uchar>(i2,j2));

			// gradient
			SD.at<float>(k, 0) = ((float)(j2+1-offx))*sum_grady - ((float)(i2+1-offy))*sum_gradx;
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

void	naya::MountJacobian3DOFColorx()
{	
	int i1, j1, i2, j2;
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);
	float sum_gradx, sum_grady;

	// Mounting matrix 
	for(int k=0;k<n_active_pixels;k++)
	{
		// Active green pixel positions
		i1 = cvFloor((float)active_pixels_g[k]/size_template_x);
		j1 = active_pixels_g[k] - i1*size_template_x;

		if(Mask.at<uchar>(i1,j1) != 0)
		{
			// gradients
			sum_gradx = gradx.ptr<float>(i1)[3*j1+1] + gradx_tmplt.ptr<float>(i1)[3*j1+1];
			sum_grady = grady.ptr<float>(i1)[3*j1+1] + grady_tmplt.ptr<float>(i1)[3*j1+1];

			// img difference
			dif.at<float>(k, 0) = (float)(current_warp.ptr<uchar>(i1)[3*j1+1] - Template_comp.ptr<uchar>(i1)[3*j1+1]);

			// gradient
			SD.at<float>(k, 0) = ((float)(j1+1-offx))*sum_grady - ((float)(i1+1-offy))*sum_gradx;
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

		// Active red pixel positions
		i2 = cvFloor((float)active_pixels_r[k]/size_template_x);
		j2 = active_pixels_r[k] - i2*size_template_x;

		if(Mask.at<uchar>(i2,j2) != 0)
		{
			// gradients
			sum_gradx = gradx.ptr<float>(i2)[3*j2+2] + gradx_tmplt.ptr<float>(i2)[3*j2+2];
			sum_grady = grady.ptr<float>(i2)[3*j2+2] + grady_tmplt.ptr<float>(i2)[3*j2+2];

			// img difference
			dif.at<float>(k + n_active_pixels, 0) = (float)(current_warp.ptr<uchar>(i2)[3*j2+2] - Template_comp.ptr<uchar>(i2)[3*j2+2]);

			// gradient
			SD.at<float>(k + n_active_pixels, 0) = ((float)(j2+1-offx))*sum_grady - ((float)(i2+1-offy))*sum_gradx;
			SD.at<float>(k + n_active_pixels, 1) = sum_gradx;
			SD.at<float>(k + n_active_pixels, 2) = sum_grady;

		}
		else
		{
			// img difference
			dif.at<float>(k + n_active_pixels,0) = 0;

			// gradient
			SD.at<float>(k + n_active_pixels, 0) = 0;
			SD.at<float>(k + n_active_pixels, 1) = 0;
			SD.at<float>(k + n_active_pixels, 2) = 0;
		}	
	}
}


void	naya::MountJacobian4DOFGray()
{	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);
	float sum_gradx, sum_grady;

	// Mounting matrix 
	for(int k=0;k<n_active_pixels;k++)
	{
		// Active red pixel positions
		int i3 = cvFloor((float)active_pixels_r[k]/size_template_x);
		int j3 = active_pixels_r[k] - i3*size_template_x;

		if(Mask.at<uchar>(i3,j3) != 0)
		{
			// gradients
			sum_gradx = gradx.at<float>(i3,j3) + gradx_tmplt.at<float>(i3,j3);
			sum_grady = grady.at<float>(i3,j3) + grady_tmplt.at<float>(i3,j3);

			// img difference
			dif.at<float>(k, 0) = (float)(current_warp.at<uchar>(i3,j3) - Template_comp.at<uchar>(i3,j3));

			// gradient
			SD.at<float>(k, 0) = ((float)(j3+1-offx))*sum_gradx + ((float)(i3+1-offy))*sum_grady;
			SD.at<float>(k, 1) = ((float)(-(i3+1-offy)))*sum_gradx + ((float)(j3+1-offx))*sum_grady;
			SD.at<float>(k, 2) = sum_gradx;
			SD.at<float>(k, 3) = sum_grady;
		}
		else
		{
			// img difference
			dif.at<float>(k,0) = 0;

			// gradient
			SD.at<float>(k, 0) = 0;
			SD.at<float>(k, 1) = 0;
			SD.at<float>(k, 2) = 0;
			SD.at<float>(k, 3) = 0;
		}	
	}
}

void	naya::MountJacobian4DOFColor()
{	
	int i1, j1, i2, j2, i3, j3;
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);
	float sum_gradx, sum_grady;

	// Mounting matrix 
	for(int k=0;k<n_active_pixels;k++)
	{
		// Active blue pixel positions
		i1 = cvFloor((float)active_pixels_b[k]/size_template_x);
		j1 = active_pixels_b[k] - i1*size_template_x;

		if(Mask.at<uchar>(i1,j1) != 0)
		{
			// gradients
			sum_gradx = gradx.ptr<float>(i1)[3*j1+1] + gradx_tmplt.ptr<float>(i1)[3*j1+1];
			sum_grady = grady.ptr<float>(i1)[3*j1+1] + grady_tmplt.ptr<float>(i1)[3*j1+1];

			// img difference
			dif.at<float>(k, 0) = (float)(current_warp.ptr<uchar>(i1)[3*j1+1] - Template_comp.ptr<uchar>(i1)[3*j1+1]);

			// gradient
			SD.at<float>(k, 0) = ((float)(j1+1-offx))*sum_gradx + ((float)(i1+1-offy))*sum_grady;
			SD.at<float>(k, 1) = ((float)(-(i1+1-offy)))*sum_gradx + ((float)(j1+1-offx))*sum_grady;
			SD.at<float>(k, 2) = sum_gradx;
			SD.at<float>(k, 3) = sum_grady;
		}
		else
		{
			// img difference
			dif.at<float>(k,0) = 0;

			// gradient
			SD.at<float>(k, 0) = 0;
			SD.at<float>(k, 1) = 0;
			SD.at<float>(k, 2) = 0;
			SD.at<float>(k, 3) = 0;
		}

		// Active green pixel positions
		i2 = cvFloor((float)active_pixels_g[k]/size_template_x);
		j2 = active_pixels_g[k] - i2*size_template_x;

		if(Mask.at<uchar>(i2,j2) != 0)
		{
			// gradients
			sum_gradx = gradx.ptr<float>(i2)[3*j2+2] + gradx_tmplt.ptr<float>(i2)[3*j2+2];
			sum_grady = grady.ptr<float>(i2)[3*j2+2] + grady_tmplt.ptr<float>(i2)[3*j2+2];

			// img difference
			dif.at<float>(k + n_active_pixels, 0) = (float)(current_warp.ptr<uchar>(i2)[3*j2+2] - Template_comp.ptr<uchar>(i2)[3*j2+2]);

			// gradient
			SD.at<float>(k + n_active_pixels, 0) = ((float)(j2+1-offx))*sum_gradx + ((float)(i2+1-offy))*sum_grady;
			SD.at<float>(k + n_active_pixels, 1) = ((float)(-(i2+1-offy)))*sum_gradx + ((float)(j2+1-offx))*sum_grady;
			SD.at<float>(k + n_active_pixels, 2) = sum_gradx;
			SD.at<float>(k + n_active_pixels, 3) = sum_grady;

		}
		else
		{
			// img difference
			dif.at<float>(k + n_active_pixels,0) = 0;

			// gradient
			SD.at<float>(k + n_active_pixels, 0) = 0;
			SD.at<float>(k + n_active_pixels, 1) = 0;
			SD.at<float>(k + n_active_pixels, 2) = 0;
			SD.at<float>(k + n_active_pixels, 3) = 0;
		}			

		// Active red pixel positions
		i3 = cvFloor((float)active_pixels_r[k]/size_template_x);
		j3 = active_pixels_r[k] - i3*size_template_x;

		if(Mask.at<uchar>(i3,j3) != 0)
		{
			// gradients
			sum_gradx = gradx.ptr<float>(i3)[3*j3+2] + gradx_tmplt.ptr<float>(i3)[3*j3+2];
			sum_grady = grady.ptr<float>(i3)[3*j3+2] + grady_tmplt.ptr<float>(i3)[3*j3+2];

			// img difference
			dif.at<float>(k + 2*n_active_pixels, 0) = (float)(current_warp.ptr<uchar>(i3)[3*j3+2] - Template_comp.ptr<uchar>(i3)[3*j3+2]);

			// gradient
			SD.at<float>(k + 2*n_active_pixels, 0) = ((float)(j3+1-offx))*sum_gradx + ((float)(i3+1-offy))*sum_grady;
			SD.at<float>(k + 2*n_active_pixels, 1) = ((float)(-(i3+1-offy)))*sum_gradx + ((float)(j3+1-offx))*sum_grady;
			SD.at<float>(k + 2*n_active_pixels, 2) = sum_gradx;
			SD.at<float>(k + 2*n_active_pixels, 3) = sum_grady;
		}
		else
		{
			// img difference
			dif.at<float>(k + 2*n_active_pixels,0) = 0;

			// gradient
			SD.at<float>(k + 2*n_active_pixels, 0) = 0;
			SD.at<float>(k + 2*n_active_pixels, 1) = 0;
			SD.at<float>(k + 2*n_active_pixels, 2) = 0;
			SD.at<float>(k + 2*n_active_pixels, 3) = 0;
		}	
	}
}


void	naya::MountJacobian8DOFGray()
{	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	// Mounting matrix 
	for(int k=0; k<n_active_pixels; k++)
	{
		// Active pixel positions
		int i = cvFloor((float)active_pixels_r[k]/size_template_x);
		int j = active_pixels_r[k] - i*size_template_x;

		if(Mask.at<uchar>(i,j) != 0)
		{
			// gradients
			float sum_gradx = gradx.at<float>(i,j) + gradx_tmplt.at<float>(i,j);
			float sum_grady = grady.at<float>(i,j) + grady_tmplt.at<float>(i,j);

			// img difference
			dif.at<float>(k, 0) = (float)(current_warp.at<uchar>(i,j) - Template_comp.at<uchar>(i,j));

			// gradient
			float i1 = ((float)(i+1-offy));
			float j1 = ((float)(j+1-offx));

			float temp = -j1*sum_gradx - i1*sum_grady;

			SD.at<float>(k, 0) = sum_gradx;
			SD.at<float>(k, 1) = sum_grady;
			SD.at<float>(k, 2) = i1*sum_gradx;
			SD.at<float>(k, 3) = j1*sum_grady;
			SD.at<float>(k, 4) = -i1*sum_grady + j1*sum_gradx;
			SD.at<float>(k, 5) = temp - i1*sum_grady;
			SD.at<float>(k, 6) = j1*temp;
			SD.at<float>(k, 7) = i1*temp;						
		}
		else
		{
			// img difference
			dif.at<float>(k,0) = 0;

			// gradient
			SD.at<float>(k, 0) = 0;
			SD.at<float>(k, 1) = 0;
			SD.at<float>(k, 2) = 0;
			SD.at<float>(k, 3) = 0;
			SD.at<float>(k, 4) = 0;
			SD.at<float>(k, 5) = 0;
			SD.at<float>(k, 6) = 0;
			SD.at<float>(k, 7) = 0;
		}
	}
}

void	naya::MountJacobian8DOFColor()
{	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	// Mounting matrix 
	for(int k=0; k<n_active_pixels; k++)
	{		
		// blue 			
		// Active pixel positions
		int i = cvFloor((float)active_pixels_b[k]/size_template_x);
		int j = active_pixels_b[k] - i*size_template_x;

		if(Mask.at<uchar>(i,j) != 0)
		{
			float i1 = (float)(i+1-offy);
			float j1 = (float)(j+1-offx);

			// Residue
			dif.at<float>(3*k, 0) = (float)(current_warp.ptr<uchar>(i)[3*j] - Template_comp.ptr<uchar>(i)[3*j]);

			// Gradients
			float sum_gradx = gradx.ptr<float>(i)[3*j] + gradx_tmplt.ptr<float>(i)[3*j];
			float sum_grady = grady.ptr<float>(i)[3*j] + grady_tmplt.ptr<float>(i)[3*j];

			// Jacobian
			float temp = -j1*sum_gradx - i1*sum_grady;

			SD.at<float>(3*k, 0) = sum_gradx;
			SD.at<float>(3*k, 1) = sum_grady;
			SD.at<float>(3*k, 2) = i1*sum_gradx;
			SD.at<float>(3*k, 3) = j1*sum_grady;
			SD.at<float>(3*k, 4) = -i1*sum_grady + j1*sum_gradx;
			SD.at<float>(3*k, 5) = temp - i1*sum_grady;
			SD.at<float>(3*k, 6) = j1*temp;
			SD.at<float>(3*k, 7) = i1*temp;	
		}
		else
		{
			SD.at<float>(3*k, 0) = 0;
			SD.at<float>(3*k, 1) = 0;
			SD.at<float>(3*k, 2) = 0;
			SD.at<float>(3*k, 3) = 0;
			SD.at<float>(3*k, 4) = 0;
			SD.at<float>(3*k, 5) = 0;
			SD.at<float>(3*k, 6) = 0;
			SD.at<float>(3*k, 7) = 0;	
		}

		// green
		// Active pixel positions
		i = cvFloor((float)active_pixels_g[k]/size_template_x);
		j = active_pixels_g[k] - i*size_template_x;

		if(Mask.at<uchar>(i,j) != 0)
		{
			float i1 = (float)(i+1-offy);
			float j1 = (float)(j+1-offx);

			// Residue
			dif.at<float>(3*k+1, 0) = (float)(current_warp.ptr<uchar>(i)[3*j+1] - Template_comp.ptr<uchar>(i)[3*j+1]);

			// Gradients
			float sum_gradx = gradx.ptr<float>(i)[3*j+1] + gradx_tmplt.ptr<float>(i)[3*j+1];
			float sum_grady = grady.ptr<float>(i)[3*j+1] + grady_tmplt.ptr<float>(i)[3*j+1];

			// Jacobian
			float temp = -j1*sum_gradx - i1*sum_grady;

			SD.at<float>(3*k+1, 0) = sum_gradx;
			SD.at<float>(3*k+1, 1) = sum_grady;
			SD.at<float>(3*k+1, 2) = i1*sum_gradx;
			SD.at<float>(3*k+1, 3) = j1*sum_grady;
			SD.at<float>(3*k+1, 4) = -i1*sum_grady + j1*sum_gradx;
			SD.at<float>(3*k+1, 5) = temp - i1*sum_grady;
			SD.at<float>(3*k+1, 6) = j1*temp;
			SD.at<float>(3*k+1, 7) = i1*temp;	
		}			
		else
		{
			SD.at<float>(3*k+1, 0) = 0;
			SD.at<float>(3*k+1, 1) = 0;
			SD.at<float>(3*k+1, 2) = 0;
			SD.at<float>(3*k+1, 3) = 0;
			SD.at<float>(3*k+1, 4) = 0;
			SD.at<float>(3*k+1, 5) = 0;
			SD.at<float>(3*k+1, 6) = 0;
			SD.at<float>(3*k+1, 7) = 0;	
		}

		// red
		// Active pixel positions
		i = cvFloor((float)active_pixels_r[k]/size_template_x);
		j = active_pixels_r[k] - i*size_template_x;

		if(Mask.at<uchar>(i,j) != 0)
		{
			float i1 = (float)(i+1-offy);
			float j1 = (float)(j+1-offx);

			// Residue
			dif.at<float>(3*k+2, 0) = (float)(current_warp.ptr<uchar>(i)[3*j+2] - Template_comp.ptr<uchar>(i)[3*j+2]);

			// Gradients
			float sum_gradx = gradx.ptr<float>(i)[3*j+2] + gradx_tmplt.ptr<float>(i)[3*j+2];
			float sum_grady = grady.ptr<float>(i)[3*j+2] + grady_tmplt.ptr<float>(i)[3*j+2];

			float temp = -j1*sum_gradx - i1*sum_grady;

			SD.at<float>(3*k+2, 0) = sum_gradx;
			SD.at<float>(3*k+2, 1) = sum_grady;
			SD.at<float>(3*k+2, 2) = i1*sum_gradx;
			SD.at<float>(3*k+2, 3) = j1*sum_grady;
			SD.at<float>(3*k+2, 4) = -i1*sum_grady + j1*sum_gradx;
			SD.at<float>(3*k+2, 5) = temp - i1*sum_grady;
			SD.at<float>(3*k+2, 6) = j1*temp;
			SD.at<float>(3*k+2, 7) = i1*temp;	
		}
		else
		{
			SD.at<float>(3*k+2, 0) = 0;
			SD.at<float>(3*k+2, 1) = 0;
			SD.at<float>(3*k+2, 2) = 0;
			SD.at<float>(3*k+2, 3) = 0;
			SD.at<float>(3*k+2, 4) = 0;
			SD.at<float>(3*k+2, 5) = 0;
			SD.at<float>(3*k+2, 6) = 0;
			SD.at<float>(3*k+2, 7) = 0;	
		}		
	}
}


void	naya::MountJacobianTPSGray()
{	
	// Going through all pixels
	for(int index=0; index<n_active_pixels; index++)
	{
		// Active pixel positions
		int i = cvFloor((float)active_pixels_r[index]/size_template_x);
		int j = active_pixels_r[index] - i*size_template_x;

		// Pointer to J and M is updated at every increment of index
		float *ptr_j = SD.ptr<float>(index);
		float *ptr_m = MKinv.ptr<float>(active_pixels_r[index]);

		// Computing residue
		if(Mask.at<uchar>(i,j) != 0)
		{
			dif.at<float>(index,0) = (float)(current_warp.at<uchar>(i,j)-Template_comp.at<uchar>(i,j));

			// Build Jacobian 
			// 'x' derivative
			for(int k=0; k<total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k]*(gradx.at<float>(i,j)+gradx_tmplt.at<float>(i,j));

			// 'y' derivative
			for(int k=total_n_ctrl_pts; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k-total_n_ctrl_pts]*(grady.at<float>(i,j)+grady_tmplt.at<float>(i,j));
		}
		else
		{
			dif.at<float>(index,0) = 0;

			// Setting all elements to 0
			for(int k=0; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = 0;
		}
	}
}

void	naya::MountJacobianTPSColor()
{	
	// Going through all blue pixels
	for(int index=0; index<n_active_pixels; index++)
	{
		// Active pixel positions
		int i = cvFloor((float)active_pixels_b[index]/size_template_x);
		int j = active_pixels_b[index] - i*size_template_x;

		// Pointer to J and M is updated at every increment of index
		float *ptr_j = SD.ptr<float>(index);
		float *ptr_m = MKinv.ptr<float>(active_pixels_b[index]);

		// Computing residue
		if(Mask.at<uchar>(i,j) != 0)
		{
			dif.at<float>(index,0) = (float)(current_warp.ptr<uchar>(i)[3*j] - Template_comp.ptr<uchar>(i)[3*j]);

			// Build Jacobian 
			// 'x' derivative
			for(int k=0; k<total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k]*(gradx.ptr<float>(i)[3*j]+gradx_tmplt.ptr<float>(i)[3*j]);

			// 'y' derivative
			for(int k=total_n_ctrl_pts; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k-total_n_ctrl_pts]*(grady.ptr<float>(i)[3*j]+grady_tmplt.ptr<float>(i)[3*j]);
		}
		else
		{
			dif.at<float>(index,0) = 0;

			// Setting all elements to 0
			for(int k=0; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = 0;
		}
	}
	
	// Going through all green pixels
	for(int index=0; index<n_active_pixels; index++)
	{
		// Active pixel positions
		int i = cvFloor((float)active_pixels_g[index]/size_template_x);
		int j = active_pixels_g[index] - i*size_template_x;

		// Pointer to J and M is updated at every increment of index
		float *ptr_j = SD.ptr<float>(index+n_active_pixels);
		float *ptr_m = MKinv.ptr<float>(active_pixels_g[index]);

		// Computing residue
		if(Mask.at<uchar>(i,j) != 0)
		{
			dif.at<float>(index+n_active_pixels,0) = (float)(current_warp.ptr<uchar>(i)[3*j+1] - Template_comp.ptr<uchar>(i)[3*j+1]);

			// Build Jacobian 
			// 'x' derivative
			for(int k=0; k<total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k]*(gradx.ptr<float>(i)[3*j+1]+gradx_tmplt.ptr<float>(i)[3*j+1]);

			// 'y' derivative
			for(int k=total_n_ctrl_pts; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k-total_n_ctrl_pts]*(grady.ptr<float>(i)[3*j+1]+grady_tmplt.ptr<float>(i)[3*j+1]);
		}
		else
		{
			dif.at<float>(index+n_active_pixels,0) = 0;

			// Setting all elements to 0
			for(int k=0; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = 0;
		}
	}
	
	// Going through all red pixels
	for(int index=0; index<n_active_pixels; index++)
	{
		// Active pixel positions
		int i = cvFloor((float)active_pixels_r[index]/size_template_x);
		int j = active_pixels_r[index] - i*size_template_x;

		// Pointer to J and M is updated at every increment of index
		float *ptr_j = SD.ptr<float>(index+2*n_active_pixels);
		float *ptr_m = MKinv.ptr<float>(active_pixels_r[index]);

		// Computing residue
		if(Mask.at<uchar>(i,j) != 0)
		{
			dif.at<float>(index+2*n_active_pixels,0) = (float)(current_warp.ptr<uchar>(i)[3*j+2] - Template_comp.ptr<uchar>(i)[3*j+2]);

			// Build Jacobian 
			// 'x' derivative
			for(int k=0; k<total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k]*(gradx.ptr<float>(i)[3*j+2]+gradx_tmplt.ptr<float>(i)[3*j+2]);

			// 'y' derivative
			for(int k=total_n_ctrl_pts; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = ptr_m[k-total_n_ctrl_pts]*(grady.ptr<float>(i)[3*j+2]+grady_tmplt.ptr<float>(i)[3*j+2]);
		}
		else
		{
			dif.at<float>(index+2*n_active_pixels,0) = 0;

			// Setting all elements to 0
			for(int k=0; k<2*total_n_ctrl_pts; k++)
				ptr_j[k] = 0;
		}
	}
}


// Updating naya parameters

int		naya::Update2DOF()
{	
	float sum = 0;

	// ESM update
	delta = 2*((SD.t()*SD).inv(CV_LU)*(SD.t()*dif));

	// Update
	parameters[2] -= delta.at<float>(0, 0); sum += abs(delta.at<float>(0, 0));
	parameters[3] -= delta.at<float>(1, 0); sum += abs(delta.at<float>(1, 0));

	return sum < epsilon;
}

int		naya::Update3DOF()
{	
	float sum = 0;

	// ESM update
	delta = 2*((SD.t()*SD).inv(CV_LU)*(SD.t()*dif));

	// Update
	angle_3DOF -= delta.at<float>(0, 0); sum += abs(delta.at<float>(0, 0));
	parameters[2] -= delta.at<float>(1, 0); sum += abs(delta.at<float>(1, 0));
	parameters[3] -= delta.at<float>(2, 0); sum += abs(delta.at<float>(2, 0));

	return sum < epsilon;
}

int		naya::Update4DOF()
{	
	float sum = 0;

	// ESM update
	delta = 2*((SD.t()*SD).inv(CV_LU)*(SD.t()*dif));

	// Update
	parameters[0] -= delta.at<float>(0, 0); sum += abs(delta.at<float>(0, 0));
	parameters[1] -= delta.at<float>(1, 0); sum += abs(delta.at<float>(1, 0));
	parameters[2] -= delta.at<float>(2, 0); sum += abs(delta.at<float>(2, 0));
	parameters[3] -= delta.at<float>(3, 0); sum += abs(delta.at<float>(3, 0));

	return sum < epsilon;
}

int		naya::Update8DOF()
{	
	// ESM update
	delta = -2*((SD.t()*SD).inv(CV_LU)*(SD.t()*dif));

	// Lie update
	update_auxA.at<float>(0,0) = delta.at<float>(4,0);
	update_auxA.at<float>(0,1) = delta.at<float>(2,0);
	update_auxA.at<float>(0,2) = delta.at<float>(0,0);

	update_auxA.at<float>(1,0) = delta.at<float>(3,0);
	update_auxA.at<float>(1,1) = -delta.at<float>(4,0) - delta.at<float>(5,0);
	update_auxA.at<float>(1,2) = delta.at<float>(1,0);

	update_auxA.at<float>(2,0) = delta.at<float>(6,0);
	update_auxA.at<float>(2,1) = delta.at<float>(7,0);
	update_auxA.at<float>(2,2) = delta.at<float>(5,0);

	MyExpm(&update_auxA);

	update_auxH.at<float>(0,0) = parameters[0];
	update_auxH.at<float>(0,1) = parameters[1];
	update_auxH.at<float>(0,2) = parameters[2];

	update_auxH.at<float>(1,0) = parameters[3];
	update_auxH.at<float>(1,1) = parameters[4];
	update_auxH.at<float>(1,2) = parameters[5];

	update_auxH.at<float>(2,0) = parameters[6];
	update_auxH.at<float>(2,1) = parameters[7];
	update_auxH.at<float>(2,2) = parameters[8];

	// H = H*A
	update_auxH = update_auxH*update_auxA;

	// Update
	float sum = 0;
	parameters[0] = update_auxH.at<float>(0, 0); sum += abs(delta.at<float>(0, 0));
	parameters[1] = update_auxH.at<float>(0, 1); sum += abs(delta.at<float>(1, 0));
	parameters[2] = update_auxH.at<float>(0, 2); sum += abs(delta.at<float>(2, 0));
	parameters[3] = update_auxH.at<float>(1, 0); sum += abs(delta.at<float>(3, 0));
	parameters[4] = update_auxH.at<float>(1, 1); sum += abs(delta.at<float>(4, 0));
	parameters[5] = update_auxH.at<float>(1, 2); sum += abs(delta.at<float>(5, 0));
	parameters[6] = update_auxH.at<float>(2, 0); sum += abs(delta.at<float>(6, 0));
	parameters[7] = update_auxH.at<float>(2, 1); sum += abs(delta.at<float>(7, 0));
	parameters[8] = update_auxH.at<float>(2, 2); 

	return sum < epsilon;
}

int		naya::UpdateTPS()
{	
	float sum = 0;

	if(lambda != 0)
	{
		// Populating Ksw
		for(int i=0; i<total_n_ctrl_pts; i++)
		{
			list_ctrl_pts.at<float>(i,0) = ctrl_pts_x_w.at<float>(i,0);
			list_ctrl_pts.at<float>(i+total_n_ctrl_pts,0) = ctrl_pts_y_w.at<float>(i,0);
		}

		Ksw = lambda*Ks*list_ctrl_pts;

		// ESM update
		delta = 2*((SD.t()*SD + lambda*Ks).inv(CV_LU)*(SD.t()*dif + Ksw));

		// Update
		for(int i=0; i<total_n_ctrl_pts; i++)
		{
			ctrl_pts_x_w.ptr<float>(i)[0] -= delta.at<float>(i, 0); sum += abs(delta.at<float>(i, 0));
			ctrl_pts_y_w.ptr<float>(i)[0] -= delta.at<float>(i+total_n_ctrl_pts, 0); sum += abs(delta.at<float>(i+total_n_ctrl_pts, 0));
		}
	}
	else
	{
		// ESM update
		delta = 2*((SD.t()*SD).inv(CV_LU)*(SD.t()*dif));

		// Update
		for(int i=0; i<total_n_ctrl_pts; i++)
		{
			ctrl_pts_x_w.ptr<float>(i)[0] -= delta.at<float>(i, 0); sum += abs(delta.at<float>(i, 0));
			ctrl_pts_y_w.ptr<float>(i)[0] -= delta.at<float>(i+total_n_ctrl_pts, 0); sum += abs(delta.at<float>(i+total_n_ctrl_pts, 0));
		}
	}

	return sum < epsilon;
}


// Resetting expected image

void	naya::ResetExpected()
{
	for(int u=0; u<n_bins; u++)
	{
		expected[u] = (float)u;

		if(!isgrayscale)
		{
			expected[u+n_bins] = (float)u;
			expected[u+n_bins*2] = (float)u;
		}
	}
}


// Computing expected image

void	naya::ComputeExpectedImg()
{
	if(isgrayscale)
	{
		// Calculates intensity value to be added to each intensity value in reference image
		for(int u=0;u<n_bins;u++)
		{
			correction[u] = size_bins*(expected[u] - u);
		}

		// Correcting template
		for(int v=0;v<size_template_y;v++)
		{
			for(int u=0;u<size_template_x;u++)
			{
				Template_comp.at<uchar>(v,u) = Template->at<uchar>(v,u) + cvRound(correction[cvRound((float)Template->at<uchar>(v,u)/size_bins)]);
			}
		}
	}
	else // iscolor
	{
		// Calculates intensity value to be added to each intensity value in reference image
		for(int k=0;k<3;k++)
		{
			for(int u=0;u<n_bins;u++)
			{
				correction[u] = size_bins*(expected[k*n_bins + u] - u);
			}
		}
		// Correcting template
		for(int v=0;v<size_template_y;v++)
		{
			for(int u=0;u<size_template_x;u++)
			{
				for(int k=0;k<3;k++)
				{
					Template_comp.ptr<uchar>(v)[3*u+k] = Template->ptr<uchar>(v)[3*u+k] + cvRound(correction[cvRound((float)Template->ptr<uchar>(v)[3*u+k]/size_bins)]);
				}
			}
		}
	}

	// Re-computing Gradient
	cv::Sobel(Template_comp, gradx_tmplt, CV_32F, 1, 0, 1);
	cv::Sobel(Template_comp, grady_tmplt, CV_32F, 0, 1, 1);
}


// Computing joint histogram

int	 naya::ComputeJointHistogramGray()
{
	int u, v, index, sum = 0, flag_error = 0, acc;

	float p_ref,
		p_cur;

	current_entropy = 0;

	// zerando p_joint e acc
	for(u=0; u<n_bins*n_bins; u++)
		p_joint[u] = 0;

	acc = 0;

	// computing p_joint entre 'current_warp' e 'Template'
	if(using_masks)
	{
		for(u=0; u<size_template_x; u++)
		{
			for(v=0; v<size_template_y; v++)
			{
				if(Mask.at<uchar>(v,u) == 255)
				{
					index = ((current_warp.at<uchar>(v,u) + 1)/size_bins - 1) + n_bins*((Template->at<uchar>(v,u) + 1)/size_bins - 1);

					p_joint[index]++ ;

					acc++;
				}
			}
		}
	}
	else
	{
		for(u=0; u<size_template_x; u++)
		{
			for(v=0; v<size_template_y; v++)
			{
				index = ((current_warp.at<uchar>(v,u) + 1)/size_bins - 1) + n_bins*((Template->at<uchar>(v,u) + 1)/size_bins - 1);

				p_joint[index]++ ;

				acc++;
			}
		}
	}

	// Normalizing the histogram
	if(acc > 0)
	{
		for(u=0; u<n_bins*n_bins; u++)
			p_joint[u] = p_joint[u]/acc;

		// Calculando a entropia
		for(v=0; v<n_bins; v++)
		{
			p_cur = 0;

			for(u=0; u<n_bins; u++)
				p_cur += p_joint[v + n_bins*u];

			if(p_cur > 0)
				current_entropy -= p_cur*log(p_cur);
			//control_internal->current_entropy += v*p_cur; this is std 
		}

		// computing expected intensity values
		for(u=0; u<n_bins; u++)
		{
			// calcula p_ref
			p_ref = 0;

			for(v=0; v<n_bins; v++)
				p_ref += p_joint[v + n_bins*u];

			// expected value
			expected[u] = 0;

			if(p_ref > 0)
			{
				for(v=0; v<n_bins; v++)
					expected[u] += ((v+1)*p_joint[v + n_bins*u]/p_ref);

				expected[u]--;

			}
			else
				expected[u] = (float)u;
		}
	}
	else
	{
		current_entropy = 1000000;

		for(u=0; u<n_bins; u++)
			expected[u] = (float)u;
	}

	return flag_error;
}

int	 naya::ComputeJointHistogramColor()
{
	int u, v, index, sum = 0, flag_error = 0, acc;
	float p_ref, p_cur;

	current_entropy = 0;

	// Correcting for all color channels
	for(int k=0; k<3; k++)
	{
		// zerando p_joint e acc
		for(u=0; u<n_bins*n_bins; u++)
			p_joint[u] = 0;

		acc = 0;

		// computing p_joint entre 'current_warp' e 'Template'
		if(using_masks)
		{
			for(u=0; u<size_template_x; u++)
			{
				for(v=0; v<size_template_y; v++)
				{
					if(Mask.at<uchar>(v,u) == 255)
					{
						index = ((current_warp.ptr<uchar>(v)[3*u+k] + 1)/size_bins - 1) + n_bins*((Template->ptr<uchar>(v)[3*u+k] + 1)/size_bins - 1);

						p_joint[index]++ ;

						acc++;
					}
				}
			}
		}
		else
		{
			for(u=0; u<size_template_x; u++)
			{
				for(v=0; v<size_template_y; v++)
				{
					index = ((current_warp.ptr<uchar>(v)[3*u+k] + 1)/size_bins - 1) + n_bins*((Template->ptr<uchar>(v)[3*u+k] + 1)/size_bins - 1);

					p_joint[index]++ ;

					acc++;
				}
			}
		}

		// Normalizing the histogram
		if(acc>0)
		{
			for(u=0; u<n_bins*n_bins; u++)
				p_joint[u] = p_joint[u]/acc;


			// Calculando a entropia
			for(v=0; v<n_bins; v++)
			{
				p_cur = 0;

				for(u=0; u<n_bins; u++)
					p_cur += p_joint[v + n_bins*u];

				if(p_cur > 0)
					current_entropy -= p_cur*log(p_cur);
				//control_internal->current_entropy += v*p_cur; this is std 
			}

			// computing expected intensity values
			for(u=0; u<n_bins; u++)
			{
				// calcula p_ref
				p_ref = 0;

				for(v=0; v<n_bins; v++)
					p_ref += p_joint[v + n_bins*u];

				// expected value
				expected[u + n_bins*k] = 0;

				if(p_ref > 0)
				{
					for(v=0; v<n_bins; v++)
						expected[u + n_bins*k] += ((v+1)*p_joint[v + n_bins*u]/p_ref);

					expected[u + n_bins*k]--;

				}
				else
					expected[u + n_bins*k] = (float)u;
			}
		}
		else
		{
			current_entropy = 1000000;

			for(u=0; u<n_bins; u++)
				expected[u + n_bins*k] = (float)u;
		}
	}

	return flag_error;
}


// Computing tracking confidence

float	naya::ComputeTrackingConfidenceSCV()
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
					I1_bar[k] += current_warp.ptr<uchar>(i)[3*j+k];
					I2_bar[k] += Template_comp.ptr<uchar>(i)[3*j+k];

					counter++;
				}
			}
		}

		if(counter<1)
			counter = 1;

		I1_bar[k] = I1_bar[k]/(counter);
		I2_bar[k] = I2_bar[k]/(counter);

	}

	for(int k=1;k<3;k++)
	{
		for(i=0;i<Template_comp.rows;i++)
		{
			for(j=0;j<Template_comp.cols;j++)
			{	
				if(Mask.at<uchar>(i,j) == 255)
				{
					I1sq[k] += (current_warp.ptr<uchar>(i)[3*j+k] - I1_bar[k])*(current_warp.ptr<uchar>(i)[3*j+k] - I1_bar[k]);
					I2sq[k] += (Template_comp.ptr<uchar>(i)[3*j+k] - I2_bar[k])*(Template_comp.ptr<uchar>(i)[3*j+k] - I2_bar[k]);

					erro[k] += (current_warp.ptr<uchar>(i)[3*j+k] - I1_bar[k])*(Template_comp.ptr<uchar>(i)[3*j+k] - I2_bar[k]);
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
		return 0;

	return ncc/2;
}


// Checking naya consistency

bool	naya::CheckConsistency2DOF()
{
	if(parameters[2]<0 || parameters[2]>ICur->cols ||
		parameters[3]<0 || parameters[3]>ICur->rows)
		return 1;
	else
		return 0;
}

bool	naya::CheckConsistency3DOF(float rotation_thres)
{
    if( std::abs(angle_3DOF)>rotation_thres ||
		parameters[2]<0 || parameters[2]>ICur->cols ||
		parameters[3]<0 || parameters[3]>ICur->rows)
		return 1;
	else
		return 0;
}

bool	naya::CheckConsistency4DOF()
{
	float a = (float)pow((double)parameters[0]*parameters[0] + parameters[1]*parameters[1], 0.5);

	if(parameters[2]<0 || parameters[2]>ICur->cols ||
		parameters[3]<0 || parameters[3]>ICur->rows ||
		a < 0.8 || a > 1.2)
		return 1;
	else
		return 0;
}




// Lie update

double	naya::Max(cv::Mat *M)
{
	int i,j;
	double maximum = 0, sum;

	for ( i = 0; i < M->rows; i++)
	{
		sum = 0;
		for ( j = 0; j < M->cols; j++)
		{
			sum += M->at<float>(i, j);
		}	
		if (sum < 0)
		{
			sum = sum * -1;
		}
		if (sum > maximum)
		{
			maximum = sum;
			sum = 0;
		}
	}
	return maximum;
}

void	naya::MyExpm(cv::Mat *input)
{
	int e;

	double n = Max(input);
	double f = frexp(n, &e);
	double s = (e + 1) > 0 ? (e + 1) : 0;

	aux1.setTo(pow(2,s));	
	cv::divide(*input, aux1, aux3);

	// Pade approx
	double c = 0.5;
	aux2.setTo(c);
	cv::multiply(aux3, aux2, aux1);

	aux2 = aux2.eye(3,3,CV_32F);

	aux4 = aux2 + aux1;
	aux5 = aux2 - aux1;

	double q = 6;
	char tf = 1;

	for (int k = 2; k < q; k++)
	{
		c = c * (q-k + 1) / (k*(2*q-k+1));
		aux1 = aux3*aux3;
		aux1.copyTo(aux3);		
		aux1.setTo(c);
		cv::multiply(aux3, aux1, aux2);
		aux1 = aux2 + aux4;
		aux1.copyTo(aux4);

		if (tf)
		{
			aux1 = aux5 + aux2;
			aux1.copyTo(aux5);
			tf = 0;
		}
		else 
		{
			aux1 = aux5-aux2;
			aux1.copyTo(aux5);
			tf = 1;
		}
	}

	cv::invert(aux5, aux1, CV_LU);
	aux1.copyTo(aux5);

	aux1 = aux5*aux4;
	aux1.copyTo(aux4);

	// undoing scaling
	for (int k = 1; k <= s; k++)
	{
		aux4.copyTo(aux1);
		aux2 = aux4*aux1;
		aux2.copyTo(aux4);
	}

	aux4.copyTo(*input);

}


// MISC

void	naya::Display2DOF(cv::Mat *ICur, int delay)
{
	cv::line(*ICur,
			cvPoint( -size_template_x/2 + parameters[2], (-size_template_y/2 + parameters[3])),
			cvPoint( size_template_x/2 + parameters[2], (-size_template_y/2 + parameters[3])),
			CV_RGB( 255, 255, 255), 2, 8, 0);

	cv::line(*ICur,
			cvPoint( size_template_x/2 + parameters[2], (-size_template_y/2 + parameters[3])),
			cvPoint( size_template_x/2 + parameters[2], (size_template_y/2 + parameters[3])),
			CV_RGB( 255, 255, 255), 2, 8, 0);

	
	cv::line(*ICur,
			cvPoint( size_template_x/2 + parameters[2], (size_template_y/2 + parameters[3])),
			cvPoint( -size_template_x/2 + parameters[2], (size_template_y/2 + parameters[3])),
			CV_RGB( 255, 255, 255), 2, 8, 0);

	cv::line(*ICur,
			cvPoint( -size_template_x/2 + parameters[2], (-size_template_y/2 + parameters[3])),
			cvPoint( -size_template_x/2 + parameters[2], (size_template_y/2 + parameters[3])),
			CV_RGB( 255, 255, 255), 2, 8, 0);


	cv::imshow("Current Image", *ICur);
	cv::imshow("SCV Template", Template_comp); 
	cv::imshow("Current Warp", current_warp);
	cv::waitKey(delay);
	
}

void	naya::Display4DOF(cv::Mat *ICur, int delay)
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
	cv::imshow("SCV Template", Template_comp); 
	cv::imshow("Current Warp", current_warp);
	cv::waitKey(delay);
	
}

void	naya::Display8DOF(cv::Mat *ICur, int delay)
{
	float display_z1 = (parameters[6]*-size_template_x/2 - parameters[7]*size_template_y/2 + parameters[8]);
	float display_z2 =  (parameters[6]*size_template_x/2 - parameters[7]*size_template_y/2 + parameters[8]);
	float display_z3 =  (parameters[6]*size_template_x/2 + parameters[7]*size_template_y/2 + parameters[8]);
	float display_z4 =  (parameters[6]*-size_template_x/2 + parameters[7]*size_template_y/2 + parameters[8]);

	cv::line(*ICur,
			cvPoint( (parameters[0]*-size_template_x/2 + parameters[1]*-size_template_y/2 + parameters[2])/display_z1, (parameters[3]*-size_template_x/2 + parameters[4]*-size_template_y/2 + parameters[5]) / display_z1),
			cvPoint( (parameters[0]*size_template_x/2 + parameters[1]*-size_template_y/2 + parameters[2])/display_z2, (parameters[3]*size_template_x/2 + parameters[4]*-size_template_y/2 + parameters[5]) / display_z2),
			CV_RGB( 255, 255, 255), 2, 8, 0);
	
	cv::line(*ICur,
			cvPoint( (parameters[0]*-size_template_x/2 + parameters[1]*-size_template_y/2 + parameters[2])/display_z1, (parameters[3]*-size_template_x/2 + parameters[4]*-size_template_y/2 + parameters[5]) / display_z1),
			cvPoint( (parameters[0]*-size_template_x/2 + parameters[1]*size_template_y/2 + parameters[2])/display_z4, (parameters[3]*-size_template_x/2 + parameters[4]*size_template_y/2 + parameters[5]) / display_z4),
			CV_RGB( 255, 255, 255), 2, 8, 0);
	
	cv::line(*ICur,
			cvPoint( (parameters[0]*size_template_x/2 + parameters[1]*size_template_y/2 + parameters[2])/display_z3, (parameters[3]*size_template_x/2 + parameters[4]*size_template_y/2 + parameters[5]) / display_z3),
			cvPoint( (parameters[0]*-size_template_x/2 + parameters[1]*size_template_y/2 + parameters[2])/display_z4, (parameters[3]*-size_template_x/2 + parameters[4]*size_template_y/2 + parameters[5]) / display_z4),
			CV_RGB( 255, 255, 255), 2, 8, 0);
	
	cv::line(*ICur,
			cvPoint( (parameters[0]*size_template_x/2 + parameters[1]*size_template_y/2 + parameters[2])/display_z3, (parameters[3]*size_template_x/2 + parameters[4]*size_template_y/2 + parameters[5]) / display_z3),
			cvPoint( (parameters[0]*size_template_x/2 + parameters[1]*-size_template_y/2 + parameters[2])/display_z2, (parameters[3]*size_template_x/2 + parameters[4]*-size_template_y/2 + parameters[5]) / display_z2),
			CV_RGB( 255, 255, 255), 2, 8, 0);


	cv::imshow("Current Image", *ICur);
	cv::imshow("SCV Template", Template_comp); 
	cv::imshow("Current Warp", current_warp);

	cv::waitKey(delay);	
}

void	naya::DisplayTPS(cv::Mat *ICur, int delay)
{
	int step_x = cvFloor(size_template_x/(n_ctrl_pts_x+1));
	int step_y = cvFloor(size_template_y/(n_ctrl_pts_y+1));

	// Draws horizontally		
	for(int j=1; j<=(n_ctrl_pts_x+1); j++)
	{
		float x1, x2, y1, y2;

		// Computes warped positions of points on grid 
		for(int i=0; i<=(n_ctrl_pts_y+1); i++)
		{
			GetPos((j-1)*step_x, i*step_y-1, &x1, &y1);	
			GetPos(j*step_x-1, i*step_y-1, &x2, &y2);
			cv::line(*ICur, cv::Point(cvRound(x1), cvRound(y1)), cv::Point(cvRound(x2), cvRound(y2)), cv::Scalar(255,255,255), 2, 8, 0);
		}
	}

	// Draws vertically
	for(int i=1; i<=(n_ctrl_pts_y+1); i++)
	{
		float x1, x2, y1, y2;

		// Computes warped positions of points on grid 
		for(int j=0; j<=(n_ctrl_pts_x+1); j++)
		{
			GetPos(j*step_x-1, (i-1)*step_y, &x1, &y1);	
			GetPos(j*step_x-1, i*step_y-1, &x2, &y2);
			cv::line(*ICur, cv::Point(cvRound(x1), cvRound(y1)), cv::Point(cvRound(x2), cvRound(y2)), cv::Scalar(255,255,255), 2, 8, 0);
		}
	}

	cv::imshow("Current Image", *ICur);
	cv::imshow("SCV Template", Template_comp); 
	cv::imshow("Current Warp", current_warp);

	cv::waitKey(delay);		
}


// TPS stuff

void	naya::DefineCtrlPts()
{	
	// Initializing control point vector
	ctrl_pts_x = (int*) malloc(n_ctrl_pts_x*n_ctrl_pts_y*sizeof(int));
	ctrl_pts_y = (int*) malloc(n_ctrl_pts_x*n_ctrl_pts_y*sizeof(int));

	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	for(int j=0; j<n_ctrl_pts_x; j++)
		for(int i=0; i<n_ctrl_pts_y; i++)
		{
			ctrl_pts_x[i + j*n_ctrl_pts_y] = cvRound((j+1)*size_template_x/(n_ctrl_pts_x+1) - offx);
			ctrl_pts_y[i + j*n_ctrl_pts_y] = cvRound((i+1)*size_template_y/(n_ctrl_pts_y+1) - offy);
		}
}

void	naya::TPSPrecomputations()
{
	// TPS specific
	cv::Mat M(size_template_x*size_template_y, total_n_ctrl_pts+3, CV_32FC1);
	MKinv.create(size_template_x*size_template_y, total_n_ctrl_pts, CV_32FC1);
	cv::Mat Kinv(total_n_ctrl_pts+3, total_n_ctrl_pts, CV_32FC1);
	Ks.create(2*total_n_ctrl_pts, 2*total_n_ctrl_pts, CV_32FC1);
	Ksw.create(2*total_n_ctrl_pts, 1, CV_32FC1);

	Ks.setTo(0);

	ctrl_pts_x_w.create(total_n_ctrl_pts, 1, CV_32FC1);
	ctrl_pts_y_w.create(total_n_ctrl_pts, 1, CV_32FC1);
	list_ctrl_pts.create(2*total_n_ctrl_pts, 1, CV_32FC1);

	// TPS Precomputations start here - I will precompute MKinv, which is the matrix 
	// I will multiply with ctrl_points_w to get the warped pixel positions

	// Mounting Matrix 'K'	
	cv::Mat K(total_n_ctrl_pts+3, total_n_ctrl_pts+3, CV_32FC1);

	for(int j=0;j<total_n_ctrl_pts;j++)
	{
		K.at<float>(j, total_n_ctrl_pts) = 1;
		K.at<float>(j, total_n_ctrl_pts+1) = (float) ctrl_pts_x[j];
		K.at<float>(j, total_n_ctrl_pts+2) = (float) ctrl_pts_y[j];

		K.at<float>(total_n_ctrl_pts,   j) = 1;
		K.at<float>(total_n_ctrl_pts+1, j) = (float) ctrl_pts_x[j];
		K.at<float>(total_n_ctrl_pts+2, j) = (float) ctrl_pts_y[j];		
	}

	for(int i=0;i<total_n_ctrl_pts;i++)
		for(int j=0;j<total_n_ctrl_pts;j++)
			K.at<float>(i, j) = Tps(Norm( (float) (ctrl_pts_x[i]-ctrl_pts_x[j]), (float) (ctrl_pts_y[i]-ctrl_pts_y[j])));

	for(int i=total_n_ctrl_pts; i<total_n_ctrl_pts+3; i++)
		for(int j=total_n_ctrl_pts; j<total_n_ctrl_pts+3; j++)
			K.at<float>(i, j) = 0;

	// Inverting Matrix 'K'
	cv::Mat K2 = K.inv(CV_LU);

	// Passing result to Kinv
	for(int i=0;i<total_n_ctrl_pts+3;i++)
	{
		for(int j=0;j<total_n_ctrl_pts;j++)
		{
			Kinv.at<float>(i, j) = K2.at<float>(i, j);

			if(i<total_n_ctrl_pts)
			{
				Ks.at<float>(i, j) = Kinv.at<float>(i, j);
				Ks.at<float>(i+total_n_ctrl_pts, j+total_n_ctrl_pts) = Ks.at<float>(i, j);
			}
		}
	}

	// Creating Matrix 'M'	
	int offx = cvCeil((double)size_template_x/2);
	int offy = cvCeil((double)size_template_y/2);

	for(int i=0;i<size_template_y;i++)
	{
		for(int j=0;j<size_template_x;j++)
		{
			for(int k=0;k<total_n_ctrl_pts;k++)
				M.at<float>(j+size_template_x*i, k) = Tps(Norm((float)(j-offx - ctrl_pts_x[k]), (float)(i-offy - ctrl_pts_y[k])));

			M.at<float>(j+size_template_x*i, total_n_ctrl_pts) = 1;
			M.at<float>(j+size_template_x*i, total_n_ctrl_pts+1) = (float) j-offx;
			M.at<float>(j+size_template_x*i, total_n_ctrl_pts+2) = (float) i-offy;	
		}
	}

	// Pre-computing M with Kinv
	MKinv = M*Kinv;
}

float	naya::Tps(float r)
{
	float ans;

	if(r != 0)
		ans = r*r*log10(r*r);
	else
		ans = 0;

	return ans;
}

float	naya::Norm(float x, float y)
{
	float ans;

	ans = pow(x*x+y*y, 0.5f);

	return ans;
}

void	naya::ParameterIO(int isinput)
{
	if(isinput)
	{
		for(int i=0; i<total_n_ctrl_pts; i++)
		{
			ctrl_pts_x_w.at<float>(i, 0) = parameters[i];
			ctrl_pts_y_w.at<float>(i, 0) = parameters[i+total_n_ctrl_pts];
		}
	}
	else
	{	
		for(int i=0; i<total_n_ctrl_pts; i++)
		{
			parameters[i] = ctrl_pts_x_w.at<float>(i, 0);
			parameters[i+total_n_ctrl_pts] = ctrl_pts_y_w.at<float>(i, 0);
		}
	}
}

void	naya::InitTPSParameters(float *parameters, int *coords)
{
	for(int i=0; i<total_n_ctrl_pts; i++)
	{
		parameters[i] = (float) (ctrl_pts_x[i]+coords[0]);
		parameters[i+total_n_ctrl_pts] = (float) (ctrl_pts_y[i]+coords[1]);
	}
}

void 	naya::GetPos(int x, int y, float *wx, float *wy)
{	
	if(x < 0)
		x = 0;

	if(y <0)
		y = 0;

	cv::Mat out_x(1, 1, CV_32F, wx);
	cv::Mat out_y(1, 1, CV_32F, wy);

	out_x = MKinv(cv::Rect(0, x+size_template_x*y, total_n_ctrl_pts, 1))*ctrl_pts_x_w;
	out_y = MKinv(cv::Rect(0, x+size_template_x*y, total_n_ctrl_pts, 1))*ctrl_pts_y_w;
}

cv::Mat* naya::GetPtrMapx()
{
	return &dummy_mapx;
}

cv::Mat* naya::GetPtrMapy()
{
	return &dummy_mapy;
}

/*cv::imshow("teste1", Mask);
cv::imshow("teste2", current_warp);
cv::imshow("teste3", Template_comp);
cv::Mat maskedr (Template_comp.rows, Template_comp.cols, CV_8UC1);
cv::Mat maskedg (Template_comp.rows, Template_comp.cols, CV_8UC1);
cv::Mat red (Template_comp.rows, Template_comp.cols, CV_8UC3);
cv::Mat green (Template_comp.rows, Template_comp.cols, CV_8UC3);

maskedr.setTo(0);
maskedg.setTo(0);
red.setTo(0);
green.setTo(0);

for(int k=0; k<n_active_pixels; k++)
{
int i1 = cvFloor((float)active_pixels_r[k]/size_template_x);
int j1 = active_pixels_r[k] - i1*size_template_x;

int i2 = cvFloor((float)active_pixels_g[k]/size_template_x);
int j2 = active_pixels_g[k] - i2*size_template_x;

maskedr.at<uchar>(i1,j1) = 255;
maskedg.at<uchar>(i2,j2) = 255;

red.ptr<uchar>(i1)[3*j1+1] = Template_comp.ptr<uchar>(i1)[3*j1+1];
red.ptr<uchar>(i1)[3*j1+2] = Template_comp.ptr<uchar>(i1)[3*j1+2];
red.ptr<uchar>(i1)[3*j1] = Template_comp.ptr<uchar>(i1)[3*j1];

green.ptr<uchar>(i2)[3*j2+1] = Template_comp.ptr<uchar>(i2)[3*j2+1];
green.ptr<uchar>(i2)[3*j2+2] = Template_comp.ptr<uchar>(i2)[3*j2+2];
green.ptr<uchar>(i2)[3*j2] = Template_comp.ptr<uchar>(i2)[3*j2];
}

cv::imshow("megatest1", maskedr);
cv::imshow("megatest2", maskedg);
cv::imshow("megatest3", red);
cv::imshow("megatest4", green);
cv::waitKey(0);*/

/*clock_stop = (cvGetTickCount()-clock_start)/(1000*tick);
printf(">TR: %f ms", clock_stop);*/

/*clock_stop = (cvGetTickCount()-clock_start)/(1000*tick);
printf(">ncc: %f ms\r", clock_stop);*/
