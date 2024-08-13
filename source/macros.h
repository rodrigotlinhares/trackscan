#ifndef _macros_h
#define _macros_h

// Misc
//#define PI							3.14159
//#define PIXEL_GRAY16(img,nl,nc)		((short int*)(img->imageData + img->widthStep*(nl)))[nc]
//#define PIXEL_GRAY32(img,nl,nc)		((float*)(img->imageData + img->widthStep*(nl)))[nc]
//#define MATRIX_ELEMENT(mat,i,j)		((float*)mat->data.fl)[((int)j)+((int)i)*mat->cols]
//#define MATRIX_ELEMENT8U(mat,i,j)	((uchar*)mat->data.ptr)[((int)j)+((int)i)*mat->cols]

// blue = 0, green = 1, red = 2
#define PIXEL_COLOR(img,nl,nc,ch)   ((uchar*)(img->imageData + (nl)*img->widthStep))[(nc)*img->nChannels + ch]
#define PIXEL_GRAY(img,nl,nc)		((uchar*)(img->imageData + img->widthStep*(nl)))[nc]
//#define PIXEL_COLOR_F(img,nl,nc,ch) ((float *)(img->imageData + (nl)*img->widthStep))[(nc)*img->nChannels + ch]

// System defines
//#define     USE_VIDEO
#define     SCALE_FACTOR	     0.25f

#define		N_PROCS  6

#define SIZE_WIN_X  1920.
#define SIZE_WIN_Y  1010.
//#define SIZE_WIN_X  1280.
//#define SIZE_WIN_Y  720.

#endif // _macros_h
