#pragma once
#ifndef  _PHOTOMETRIC_H
#define _PHOTOMETRIC_H
#include <opencv2\opencv.hpp>
using namespace cv;

class Photometric
{
public:
	Photometric();
	~Photometric();
	// init dst and mask, only need calling once
	static void initMask(Mat image, Mat imageMask, uchar unknown = 0, uchar known = 255);
	// modify patch
	static void correct(Mat &patch, int offset_x, int offset_y);
	// implemented with Eigen
	static void correctE(Mat &patch, int offset_x, int offset_y);
	// set SOR parameter
	static void setParam(double ptol, double pomega = 1.7);
	// using mix gradients or not
	static void useMixing(bool toggle);
	// poisson blending
	static Mat blend(Mat dstMat, Mat srcMat, Mat maskMat, int rel_offset_x, int rel_offset_y);
	// poisson blending with Eigen
	static Mat blendE(Mat dstMat, Mat srcMat, Mat maskMat, int rel_offset_x, int rel_offset_y);
	// mask M_BORDER for border, M_DST for base image, M_SRC for src image
	static Mat mask;
	// store the original image
	static Mat dst;
	// may used for poisson blending
	static bool mixing;
	// tolerance for iteration
	static double tol;
	// SOR \omega
	static double omega;
};

#endif 
