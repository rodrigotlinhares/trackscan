#ifndef naya_H
#define naya_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "omp.h"

#include "macros.h"

class naya
{
    // Parallel implementation
public:
	
    void    Warp3DOFAux(unsigned int ch, float *parameters);

    void    pWarp3DOF1(unsigned int ch);

    void    pWarp3DOF1_tempA(unsigned int ch);

    void    pWarp3DOF1_tempB(unsigned int ch);

    void    pWarp3DOF2(unsigned int ch);

    void 	Run3DOFxi_step1(cv::Mat *ICur,
                    cv::Mat *Mask_roi,
                    cv::Mat *Template,
                    cv::Mat *Mask_template,
                    float *parameters,
                    float *parameters_illum,
                   int *active_pixels_r,
                   int *active_pixels_g);

    void 	Run3DOFxi_step1_tempB(cv::Mat *ICur,
                    cv::Mat *Mask_roi,
                    cv::Mat *Template,
                    cv::Mat *Mask_template,
                    float *parameters,
                    float *parameters_illum,
                   int *active_pixels_r,
                   int *active_pixels_g);

    void 	Run3DOFxi_threaded(cv::Mat *ICur,
                                cv::Mat *Mask_roi,
                                cv::Mat *Template,
                                cv::Mat *Mask_template,
                                float *parameters,
                                float *parameters_illum,
                               int *active_pixels_r,
                               int *active_pixels_g);

    void 	Run3DOFxi_threaded_tempA(cv::Mat *ICur,
                                    cv::Mat *Mask_roi,
                                    cv::Mat *Template,
                                    cv::Mat *Mask_template,
                                    float *parameters,
                                    float *parameters_illum,
                                   int *active_pixels_r,
                                   int *active_pixels_g);

    void 	Run3DOFxi_threaded_tempB(cv::Mat *ICur,
                                    cv::Mat *Mask_roi,
                                    cv::Mat *Template,
                                    cv::Mat *Mask_template,
                                    float *parameters,
                                    float *parameters_illum,
                                   int *active_pixels_r,
                                   int *active_pixels_g);


    void 	Run3DOFxi2(unsigned int ch);
	
    void 	Run3DOFxi3();

    void    Run3DOFxi3_tempA();

    void    Run3DOFxi3_tempB();

	void	Update3DOFi1(unsigned int ch);

    int		Update3DOFix();

    int     Update3DOFix_tempA();

    int     Update3DOFix_tempB();

    void	MountJacobian3DOFColorxi(unsigned int ch);

    void	MountJacobian3DOFColorxi_tempA(unsigned int ch);

    void	MountJacobian3DOFColorxi_tempB(unsigned int ch);

    cv::Mat dummy_mapx,
            dummy_mapy;

	cv::Mat *pre_SD, *pre_hess;

	int		flag_done;

    // New
    int start_stop[2*N_PROCS],
        start_stop2[2*N_PROCS],
        start_stop3[2*N_PROCS],
        start_stop4[2*N_PROCS];

    cv::Rect ok[N_PROCS],
			 ok2[N_PROCS],
			 ok3[N_PROCS],
			 ok_hess[N_PROCS];
	
	void	NonRigidCompensation(unsigned int ch);
	
	void	WarpGradi(unsigned int ch);
	
	void	OcclusionMap(unsigned int ch);

	// Pixel selection part
	
public:

    void	ComputeActive3DOFx(cv::Mat *Input, cv::Mat *MaskInput, int *active_r, int *active_g);

    void	FastComputeActive3DOFx(cv::Mat *Input, cv::Mat *MaskInput, int *active_r, int *active_g);
	
	void	ComputeActive4DOF(cv::Mat *Input, cv::Mat *MaskInput, int *active_r, int *active_g, int *active_b);
	
	void	ComputeActive8DOF(cv::Mat *Input, cv::Mat *MaskInput, int *active_r, int *active_g, int *active_b);

private:

    void	SuperFastJacobian3DOFx(int isgrayscale);

    void	FastJacobian3DOFx(int isgrayscale);
	
	void	FastJacobian4DOF(int isgrayscale);

	void	FastJacobian8DOF(int isgrayscale);


	// Tracking part

public:
	
    void	Initialize2DOF(int size_template_x,
						   int size_template_y,
						   int n_bins,
						   int size_bins,
						   int n_max_iters,
						   float epsilon,
						   int isgrayscale,
						   int interp);   
	
	void	Initialize3DOFx(int size_template_x,
						   int size_template_y,
						   int n_active_pixels,
						   int n_bins,
						   int size_bins,
						   int n_max_iters,
						   float epsilon,
						   int isgrayscale,
                           int interp);

    void	Initialize3DOFxi(int size_template_x,
                           int size_template_y,
                            int n_ctrl_pts_xi,
                            int n_ctrl_pts_yi,
                           int n_active_pixels,
                           int n_bins,
                           int size_bins,
                           int n_max_iters,
                           float epsilon,
                           int isgrayscale,
                           int interp);


    void	Initialize3DOFxi_tempA(int size_template_x,
                           int size_template_y,
                            int n_ctrl_pts_xi,
                            int n_ctrl_pts_yi,
                           int n_active_pixels,
                           int n_bins,
                           int size_bins,
                           int n_max_iters,
                           float epsilon,
                           int isgrayscale,
                           int interp);

    void	Initialize3DOFxi_tempB(int size_template_x,
                           int size_template_y,
                            int n_ctrl_pts_xi,
                            int n_ctrl_pts_yi,
                           int n_active_pixels,
                           int n_bins,
                           int size_bins,
                           int n_max_iters,
                           float epsilon,
                           int isgrayscale,
                           int interp);
		
	void	Initialize4DOF(int size_template_x,
						   int size_template_y,
						   int n_active_pixels,
						   int n_bins,
						   int size_bins,
						   int n_max_iters,
						   float epsilon,
						   int isgrayscale,
						   int interp);
	
	void	Initialize8DOF(int size_template_x,
						   int size_template_y,
						   int n_active_pixels,
						   int n_bins,
						   int size_bins,
						   int n_max_iters,
						   float epsilon,
						   int isgrayscale,
						   int interp);
	
	void	InitializeTPS(int size_template_x,
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
						  int interp);

	
	int 	Run2DOF(cv::Mat *ICur, 
					cv::Mat *Mask_roi,
					cv::Mat *Template,
					cv::Mat *Mask_template,
					float *parameters);
	
	int 	Run3DOFx(cv::Mat *ICur, 
					cv::Mat *Mask_roi,
					cv::Mat *Template,
					cv::Mat *Mask_template,
					float *parameters,
				   int *active_pixels_r,
				   int *active_pixels_g);

    int 	Run3DOFxi(cv::Mat *ICur,
                    cv::Mat *Mask_roi,
                    cv::Mat *Template,
                    cv::Mat *Mask_template,
                    float *parameters,
                    float *parameters_illum,
                   int *active_pixels_r,
                   int *active_pixels_g);
		
	int 	Run4DOF(cv::Mat *ICur, 
					cv::Mat *Mask_roi,
					cv::Mat *Template,
					cv::Mat *Mask_template,
					float *parameters,
				   int *active_pixels_r,
				   int *active_pixels_g,
				   int *active_pixels_b);
	
	int 	Run8DOF(cv::Mat *ICur, 
					cv::Mat *Mask_roi,
					cv::Mat *Template,
					cv::Mat *Mask_template,
					float *parameters,
					int *active_pixels_r,
					int *active_pixels_g,
					int *active_pixels_b);	

	int 	RunTPS(cv::Mat *ICur, 
				   cv::Mat *Mask_roi,
				   cv::Mat *Template,
				   cv::Mat *Mask_template,
				   float *parameters,
				   int *active_pixels_r,
				   int *active_pixels_g,
				   int *active_pixels_b);	
	
	
	void	Display2DOF(cv::Mat *ICur, int delay);
	
	void	Display3DOFxi(cv::Mat *ICur, int delay);

	void	Display4DOF(cv::Mat *ICur, int delay);

	void	Display8DOF(cv::Mat *ICur, int delay);

	void	DisplayTPS(cv::Mat *ICur, int delay);
	
	
	void	InitTPSParameters(float *parameters, int *coords);
	
	void	ResetIlluminationParam3DOFxi(float *illum_param);


	void	ResetExpected();

    float	ComputeTrackingConfidenceSSDi();

    float	ComputeTrackingConfidenceSCV();
		

	bool	CheckConsistency2DOF();
	
    bool	CheckConsistency3DOF(float rotation_thres);
	
    bool	CheckConsistency4DOF();


	cv::Mat* GetPtrMapx();

	cv::Mat* GetPtrMapy();


    void	WarpGrad(unsigned int ch, cv::Mat *input);
	
private:
	
	void	Warp2DOF();
	
    void	Warp3DOF();

    void	Warp4DOF();
	
    void	Warp8DOF();

    void	WarpTPS();
	
	
	void	WarpGrad();

	void	WarpGradi();

    void	WarpGrad(cv::Mat *Input);


    int		ComputeJointHistogramGray();
	
    int		ComputeJointHistogramColor();


	void	ComputeExpectedImg();
	
	
    void	MountJacobian2DOFGray();

    void	MountJacobian2DOFColor();
	
	
    void	MountJacobian3DOFGrayx();

    void	MountJacobian3DOFColorx();
	
	
    void	MountJacobian3DOFGrayxi();

    void	MountJacobian3DOFColorxi();


    void	MountJacobian4DOFGray();

    void	MountJacobian4DOFColor();


    void	MountJacobian8DOFGray();

    void	MountJacobian8DOFColor();
	
	
	void	MountJacobianTPSGray();

	void	MountJacobianTPSColor();
	
	
    int		Update2DOF();
	
    int		Update3DOF();

	int		Update3DOFi();
	
    int		Update4DOF();
	
    int		Update8DOF();

	int		UpdateTPS();


	void	OcclusionMap();
	
	
	double	Max(cv::Mat *M);
		
	void	MyExpm(cv::Mat *input);

	
	void	DefineCtrlPts();

	void	TPSPrecomputations();

	float	Tps(float r);

	float	Norm(float x, float y);
	
	void	ParameterIO(int isinput);

	void	GetPos(int x, int y, float *wx, float *wy);	


	void	TPSPrecomputationsIllum();

	void	DefineCtrlPtsIllum();

	void	ParameterIOIllum(int isinput);
	
	void	NonRigidCompensation();


private:

	// Are we using masks?
	bool	using_masks;

	// naya parameters
	int		size_template_x,
			size_template_y,
			n_bins,
			size_bins,
			n_active_pixels,
			n_max_iters,
			interp,
			isgrayscale;

	float	epsilon;
	
	// List of active pixels	
	bool	 *visited_r,
			 *visited_g,
			 *visited_b;

	int		*active_pixels_r,
		    *active_pixels_g,
		    *active_pixels_b;

	int		*std_pixel_list;
	
    std::vector<std::pair<int, int> > pair_r[8],
                                     pair_g[8],
                                     pair_b[8];

    std::vector<std::pair<int, int> > fast_pair_r,
                                     fast_pair_g,
                                     fast_pair_b;

	// SCV aux
	float	*correction,
			*p_joint,
			*expected;
	
	// naya parameters
	float	*parameters,
			 angle_3DOF;
	
	// Input images and masks for naya
	cv::Mat	*ICur,
			*Template,
			*Mask_template,
			*Mask_roi;

	// naya core stuff
    cv::Mat SD,
            delta,
			dif,
			gradx,
			grady,
			gradx_tmplt,
			grady_tmplt;

	// 8dof update 
	cv::Mat update_auxA,
			update_auxH,
			aux1,
			aux2,
			aux3,
			aux4,
			aux5;

	// TPS parameters
	int		n_ctrl_pts_x,
		    n_ctrl_pts_y,
			total_n_ctrl_pts;

	int		*ctrl_pts_x,
			*ctrl_pts_y;

	float	lambda;

	cv::Mat	ctrl_pts_x_w,
			ctrl_pts_y_w,
			list_ctrl_pts;

	cv::Mat MKinv,
			Ks,
			Ksw;
	
	cv::Mat weights;

	// Non-rigid illumination compensation 
	float	*parameters_illum,
			bias;

	int		n_ctrl_pts_xi,
			n_ctrl_pts_yi,
			total_n_ctrl_ptsi;

	int		*ctrl_pts_xi,
			*ctrl_pts_yi;


	cv::Mat MKinvi;

public:


    cv::Mat	ctrl_pts_wi,
            gain;

	// Current image and mask
	cv::Mat current_warp,
			compensated_warp,
			Template_comp,
			Mask;

	// naya stats
	int		iters;

	// Current entropy
	float	current_entropy;

};

#endif
