#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <malloc.h>
#include <cv.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <Windows.h>
#include "Include_OpenCVHeadFile.h"
#define  LARGER(x,y) (x)>(y)?(x):(y)
#define  SMALLER(x,y) (x)<(y)?(x):(y)
typedef struct EyeInfo{
	short int leftx;
	short int lefty;
	short int rightx;
	short int righty;
}EyeInfo;

//DLL_EXP int Enhance_StandardImage( unsigned char *AlignedData, int w, int h);


class Preprocess
{
public:
	Mat getBilateralFilterImg(Mat&equalizeHistImg);
	void mixEqualizeHist(Mat &warpedImg);
	double PointDistance(Point p1, Point p2);
	bool CropSubImage(unsigned char* pbyteSrcImgData, int nSrcImgWidth, int nSrcImgHeight, unsigned char* pDstImgData, Rect subRect);
	int	Convert2Stride(int	nWidth);
	bool ResizeImage(unsigned char* pSrcImg, int nSrcWidth, int nSrcHeight, unsigned char* pDstImg, int nDstWidth, int nDstHeight);
	bool AlignFaceRegion(unsigned char* pbyteSrcImg, int nWidth, int nHeight, EyeInfo aKeyPnts, unsigned char* pbyteAlignedImg, int nDstImgSize);
	void hMirrorTrans(const Mat &src, Mat &dst);
	Mat getGaborFilter(float lambda, float theta, 
		float sigma2,float gamma, 
		float psi = 0.0f);
	Mat gaborFilter(Mat& img, Mat& filter);
	Preprocess(void);
	~Preprocess(void);

};

