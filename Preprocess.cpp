#include "Preprocess.h"
#include "Include_OpenCVHeadFile.h"
#define PI 3.1416

Preprocess::Preprocess(void)
{
}

Preprocess::~Preprocess(void)
{
}

double Preprocess::PointDistance(Point p1, Point p2)
{
	return (double) sqrt((p2.y-p1.y) * (p2.y-p1.y)*1.0 + (p2.x-p1.x) * (p2.x-p1.x)*1.0);
}

bool Preprocess::CropSubImage(unsigned char* pbyteSrcImgData, int nSrcImgWidth, int nSrcImgHeight, unsigned char* pDstImgData, Rect subRect)
{
	/* */
	int		left	= subRect.x;
	int		top		= subRect.y;
	int		right	= subRect.width + left;
	int		bottom	= subRect.height + top;

	int width  = right - left + 1;
	int height = bottom - top + 1;

#if 0
	int		left	= subRect.left;
	int		top		= subRect.top;
	int		right	= subRect.right;
	int		bottom	= subRect.bottom;

	int width  = right - left + 1;
	int height = bottom - top + 1;
#endif

	int stride = width;
	int	nSrcStride = nSrcImgWidth;

	assert( width > 0 && height > 0 && left >= 0 && right >=0 && top >= 0 && bottom >= 0 && right < nSrcImgWidth && bottom <= nSrcImgHeight);

	int i;
	unsigned char* pS = pbyteSrcImgData+top*nSrcStride+left;
	unsigned char* pD = pDstImgData;
	for( i = 0; i < height; i++)
	{
		CopyMemory( (void*)pD, (void*)pS, width);
		//memcpy((void*)pD, (void*)pS, width);//改为这个函数不知道是否会出错
		pS += nSrcStride;
		pD += stride;
	}
	return true;
}

int Preprocess::Convert2Stride(int nWidth)
{
	return ((nWidth%4)?(nWidth+4-nWidth%4):(nWidth));
}

bool Preprocess::ResizeImage(unsigned char* pSrcImg, int nSrcWidth, int nSrcHeight, unsigned char* pDstImg, int nDstWidth, int nDstHeight)
{
	int		n_x_d, n_y_d;
	int		n_x_s, n_y_s;
	double	lfXscl, lfYScl, lf_x_s, lf_y_s, lfNewGray;
	double	lfWeight_x, lfWeight_y;

	if(nSrcWidth == nDstWidth && nSrcHeight == nDstHeight)
	{
		memcpy(pDstImg, pSrcImg, Convert2Stride(nSrcWidth)*nSrcHeight);
		return true;
	}

	lfXscl = double(nSrcWidth+0.0)/nDstWidth;
	lfYScl = double(nSrcHeight+0.0)/nDstHeight;

	//if the image data is strided, open the following 2 lines
	//nSrcWidth = Convert2Stride(nSrcWidth);
	//nDstWidth = Convert2Stride(nDstWidth);

	for(n_y_d=0; n_y_d<nDstHeight; n_y_d++)
	{
		for(n_x_d=0; n_x_d<nDstWidth; n_x_d++)
		{
			lf_x_s = lfXscl * n_x_d;
			lf_y_s = lfYScl * n_y_d;
			n_x_s = int(lf_x_s);
			n_x_s = SMALLER(n_x_s, nSrcWidth-2);
			n_y_s = int(lf_y_s);
			n_y_s = SMALLER(n_y_s, nSrcHeight-2);
			lfWeight_x = lf_x_s - n_x_s;
			lfWeight_y = lf_y_s - n_y_s;
			lfNewGray = (1-lfWeight_y)*((1-lfWeight_x)*pSrcImg[n_y_s*nSrcWidth+n_x_s]+lfWeight_x*pSrcImg[n_y_s*nSrcWidth+n_x_s+1])+
				lfWeight_y*((1-lfWeight_x)*pSrcImg[(n_y_s+1)*nSrcWidth+n_x_s]+lfWeight_x*pSrcImg[(n_y_s+1)*nSrcWidth+n_x_s+1]);
			pDstImg[n_y_d*nDstWidth+n_x_d] = unsigned char(lfNewGray);
		}
	}
	return true;
}

bool Preprocess::AlignFaceRegion(unsigned char* pbyteSrcImg, int nWidth, int nHeight, EyeInfo aKeyPnts, unsigned char* pbyteAlignedImg, int nDstImgSize)
{
	///////////////nWidth should be the nStride of the source image
	long	t3 = 0;
	int		i, j, w, h, x1, x, y, t;
	double	oldx, oldy, cofX, cofY, FTmp;
	double  d, SinCita, CosCita, t1, t2;
	Point	LeftIrisPos, RightIrisPos;

	//LeftIrisPos  = aKeyPnts[0];
	//RightIrisPos = aKeyPnts[1];
	LeftIrisPos.x = aKeyPnts.leftx, LeftIrisPos.y = aKeyPnts.lefty;
	RightIrisPos.x = aKeyPnts.rightx, RightIrisPos.y = aKeyPnts.righty;
		
	w = nWidth;
	//w = ::Convert2Stride(nWidth);
	h = nHeight;
	d = PointDistance(LeftIrisPos, RightIrisPos);

	SinCita = (-RightIrisPos.y+ LeftIrisPos.y) / d;
	CosCita = (RightIrisPos.x - LeftIrisPos.x) / d;

	Point IrisMidPos = Point((LeftIrisPos.x + RightIrisPos.x) / 2,
								(LeftIrisPos.y + RightIrisPos.y) / 2);

	/////////////Place the original at the center of the face//////
	x = LeftIrisPos.x - IrisMidPos.x;
	y = LeftIrisPos.y - IrisMidPos.y;
	t = x;	x = -y;	y = -t;
	x += IrisMidPos.x;
	y += IrisMidPos.y;
	//CPoint fixpos = CPoint(x, y);
	Point fixpos = Point(x, y);
	
	/*///////////////////rotate the image////////////////////////////*/
	unsigned char* pRotatedImg = new unsigned char[w*h];
	unsigned char* pOriData = pbyteSrcImg;
	unsigned char* pNewData = pRotatedImg;
	for(j=0; j<h; j++)
	{
		t1 = SinCita * (j - fixpos.y);
		t2 = CosCita * (j - fixpos.y);
		t3 = j * w;
		for(i=0; i<w; i++)
		{
			oldx = fixpos.x +  (CosCita * (i - fixpos.x) + t1);
			oldy = fixpos.y +  (t2 - SinCita * (i - fixpos.x));
			if(oldx < w-1 && oldy < h-1 && oldx >= 0 && oldy >= 0)
			{	
				cofX = oldx - int(oldx);
				cofY = oldy - int(oldy);
				x1 = ((int)oldy) * w + (int)oldx;
				FTmp = (1-cofX) * (double)pOriData[x1] + cofX * (double)pOriData[x1 + 1];
				FTmp = (1-cofY) * FTmp + cofY * ((1-cofX) * (double)pOriData[x1 + w] + cofX * (double)pOriData[x1 + w + 1]);
				pNewData[j * w + i] = unsigned char(((FTmp + 0.5)>255)?255:(FTmp + 0.5));
			}
			else 
				pNewData[j * w + i] = 0;
		}
	}
	//cal the new pos of the iris in the new face
	LeftIrisPos.x = fixpos.x + (int)(CosCita * (LeftIrisPos.x - fixpos.x)
						 - SinCita * (LeftIrisPos.y - fixpos.y) + 0.5);
	LeftIrisPos.y = fixpos.y + (int)(SinCita * (LeftIrisPos.x - fixpos.x)
						 + CosCita * (LeftIrisPos.y - fixpos.y) + 0.5);
	
	RightIrisPos.x = fixpos.x + (int)(CosCita * (RightIrisPos.x - fixpos.x)
						- SinCita * (RightIrisPos.y - fixpos.y) + 0.5);
	RightIrisPos.y = fixpos.y + (int)(SinCita * (RightIrisPos.x - fixpos.x)
						+ CosCita * (RightIrisPos.y - fixpos.y) + 0.5);
	
	////////////////crop according to the mean face structure///////////////
	d = PointDistance(LeftIrisPos, RightIrisPos);
	//int	nLeft = int(LeftIrisPos.x-d*0.7);//-0.5-1);
	int nLeft = int(LeftIrisPos.x-d*0.6);
	//int	nTop  = int(LeftIrisPos.y-d*0.5);
	int nTop = int(LeftIrisPos.y-d*0.6);
	nLeft = LARGER(nLeft, 0);
	nTop  = LARGER(nTop, 0);

	int nRight = int(nLeft+2.3*d);
	int nBottom=int(nTop+2.7*d);
	nRight = SMALLER(nRight, nWidth-1);
	nBottom= SMALLER(nBottom, nHeight-1);
	int nSize = SMALLER(nRight-nLeft, nBottom-nTop);
	nRight = nLeft + nSize;
	nBottom= nTop + nSize;

	//Rect	rectFaceRgn = Rect(nLeft, nTop, nRight, nBottom);
	Rect	rectFaceRgn = Rect(nLeft, nTop, nSize, nSize);
	int	nCropWidth = nRight - nLeft+1;
	int nCropHeight= nBottom- nTop+1;
	assert(nCropWidth == nCropHeight);
	int nCropStride= Convert2Stride(nCropWidth);
	
	unsigned char*	pCropImgData = new unsigned char[nCropStride*nCropHeight];
	CropSubImage(pRotatedImg, nWidth, nHeight, pCropImgData, rectFaceRgn);



	/////////////////////Resize to the required image size///////////////
	ResizeImage(pCropImgData, nCropWidth, nCropHeight, pbyteAlignedImg, nDstImgSize, nDstImgSize);
	/////////////////////////////////////////////////////////////////////
	delete pRotatedImg;
	delete pCropImgData;

	return true;
}

Mat Preprocess::getGaborFilter(float lambda, float theta, 
	float sigma2,float gamma, 
	float psi = 0.0f){
		if(abs

			(lambda-0.0f)<1e-6){
				lambda = 1.0f;
		} 
		float sigma_x = sigma2;
		float 

			sigma_y = sigma2/(gamma*gamma);
		int nstds = 3;
		float sqrt_sigma_x = sqrt(sigma_x);
		float 

			sqrt_sigma_y = sqrt(sigma_y);
		int xmax = max(abs(nstds*sqrt_sigma_x*cos(theta)),abs(nstds*sqrt_sigma_y*sin

			(theta)));
		int ymax = max(abs(nstds*sqrt_sigma_x*sin(theta)),abs(nstds*sqrt_sigma_y*cos(theta)));
		int 

			half_filter_size = xmax>ymax ? xmax:ymax;
		int filter_size = 2*half_filter_size+1;
		Mat gaber = 

			Mat::zeros(filter_size,filter_size,CV_32F);
		for(int i=0;i<filter_size;i++){
			float* f = 

				gaber.ptr<float>(i);
			for(int j=0;j<filter_size;j++){
				int x = j-half_filter_size;


				int y = i-half_filter_size;
				float x_theta=x*cos(theta)+y*sin(theta);


				float y_theta=-x*sin(theta)+y*cos(theta);
				f[x] = exp(-.5*
					(x_theta*x_theta/sigma_x+y_theta*y_theta/sigma_y));
				f[x] = f[x]*cos(2*PI*x_theta/lambda+psi);


			};
		}
		return gaber;
}

Mat Preprocess::gaborFilter(Mat& img, Mat& filter){
	int half_filter_size = (max(filter.rows,filter.cols)-1)/2;
	Mat filtered_img(img.rows,img.cols,CV_32F);
	for(int i=0;i<img.rows;i++){
		uchar* img_p = img.ptr<uchar>(i);
		float* img_f = filtered_img.ptr<float>(i);
		for(int j=0;j<img.cols;j++){
			float filter_value = 0.0f;
			for(int fi=0;fi<filter.rows;fi++){
				float* f = filter.ptr<float>(fi);
				int img_i = i+fi-half_filter_size;
				img_i = img_i < 0 ? 0 : img_i;
				img_i = img_i >= img.rows ? (img.rows-1) : img_i;
				uchar* p = img.ptr<uchar>(img_i);
				for(int fj=0;fj<filter.cols;fj++){
					int img_j =j+fj-half_filter_size;
					img_j = img_j < 0 ? 0 : img_j;
					img_j = (img_j >= img.cols) ? (img.cols-1) : img_j;
					float tmp = (float)p[img_j]*f[fj];
					filter_value += tmp;
				}
			}
			img_f[j] = filter_value;
		}
	}
	return filtered_img;
}

Mat normalizeFilterShow(Mat gaber){
	Mat gaber_show = Mat::zeros(gaber.rows,gaber.cols,CV_8UC1);
	float gaber_max = FLT_MIN;
	float gaber_min = FLT_MAX;
	for(int i=0;i<gaber.rows;i++){
		float* f = gaber.ptr<float>(i);
		for(int j=0;j<gaber.cols;j++){
			if(f[j]>gaber_max){
				gaber_max = f[j];
			}
			if(f[j]<gaber_min){
				gaber_min = f[j];
			}
		}
	}
	float gaber_max_min = gaber_max-gaber_min;
	for(int i=0;i<gaber_show.rows;i++){
		uchar* p = gaber_show.ptr<uchar>(i);
		float* f = gaber.ptr<float>(i);
		for(int j=0;j<gaber_show.cols;j++){
			if(gaber_max_min!=0.0f){
				float tmp = (f[j]-gaber_min)*255.0f/gaber_max_min;
				p[j] = (uchar)tmp;
			}
			else{
				p[j] = 255;
			}
		}
	}
	return gaber_show;
}

void LBP(IplImage* src, IplImage* dst)  
{  
	int width=src->width;  
	int height=src->height;  
	for(int j=1;j<width-1;j++)  
	{  
		for(int i=1;i<height-1;i++)  
		{  
			uchar neighborhood[8]={0};  
			neighborhood[7] = CV_IMAGE_ELEM( src, uchar, i-1, j-1);  
			neighborhood[6] = CV_IMAGE_ELEM( src, uchar, i-1, j);  
			neighborhood[5] = CV_IMAGE_ELEM( src, uchar, i-1, j+1);  
			neighborhood[4] = CV_IMAGE_ELEM( src, uchar, i, j-1);  
			neighborhood[3] = CV_IMAGE_ELEM( src, uchar, i, j+1);  
			neighborhood[2] = CV_IMAGE_ELEM( src, uchar, i+1, j-1);  
			neighborhood[1] = CV_IMAGE_ELEM( src, uchar, i+1, j);  
			neighborhood[0] = CV_IMAGE_ELEM( src, uchar, i+1, j+1);  
			uchar center = CV_IMAGE_ELEM( src, uchar, i, j);  
			uchar temp=0;  

			for(int k=0;k<8;k++)  
			{  
				temp+=(neighborhood[k]>=center)<<k;  
			}  
			CV_IMAGE_ELEM( dst, uchar, i, j)=temp;  
		}  
	}  
}  

void Preprocess::hMirrorTrans(const Mat &src, Mat &dst)
{
    CV_Assert(src.depth() == CV_8U);
    dst.create(src.rows, src.cols, src.type());

    int rows = src.rows;
    int cols = src.cols;

    switch (src.channels())
    {
    case 1:
        const uchar *origal;
        uchar *p;
        for (int i = 0; i < rows; i++){
            origal = src.ptr<uchar>(i);
            p = dst.ptr<uchar>(i);
            for (int j = 0; j < cols; j++){
                p[j] = origal[cols - 1 - j];
            }
        }
        break;
    case 3:
        const Vec3b *origal3;
        Vec3b *p3;
        for (int i = 0; i < rows; i++) {
            origal3 = src.ptr<Vec3b>(i);
            p3 = dst.ptr<Vec3b>(i);
            for(int j = 0; j < cols; j++){
                p3[j] = origal3[cols - 1 - j];
            }
        }
        break;
    default:
        break;
    }
    
}

/*如果我们简单地在左半部分应用直方图均衡化，然后应用到右边，
我们会看到在中间有一个很明显的边界，因为平均的亮度在左边和右
边可能不同，因此为了移除边界，我们将应用两个直方图均衡化，逐
渐地从左手边或者右手边朝向中心。用一个全部的人脸直方图来混合。
因此，远离 左手边将使用左直方图均衡化，远离右手边将使用右直方图均衡化，
并且中心使用一个平滑的左右值的和整脸均衡值的混合值。*/
void Preprocess::mixEqualizeHist(Mat &warpedImg)
{
	int w = warpedImg.cols;  
	int h = warpedImg.rows;  
	Mat wholeFace;  
	equalizeHist(warpedImg, wholeFace); //给整张脸均衡 
	int midX = w/2;  
	Mat leftSide = warpedImg(Rect(0,0, midX,h));  
	Mat rightSide = warpedImg(Rect(midX,0, w-midX,h));  
	equalizeHist(leftSide, leftSide);  //半脸均衡
	equalizeHist(rightSide, rightSide);  

	for (int y=0; y<h; y++) 
	{  
		for (int x=0; x<w; x++)
		{  
			int v;  
			if (x < w/4)
			{  
				//左侧25%：仅使用左脸  
				v = leftSide.at<uchar>(y,x);  
			}  
			else if (x < w*2/4)
			{  
				//中-左25%：混合左脸和整个人脸  
				int lv = leftSide.at<uchar>(y,x);  
				int wv = wholeFace.at<uchar>(y,x);  
				// 混合更多的整个人脸，因为它移动远离右侧脸  
				// further right along the face.  
				float f = (x - w*1/4) / (float)(w/4);  
				v = cvRound((1.0f - f) * lv + (f) * wv);  
			}  
			else if (x < w*3/4) 
			{  
				// Mid-right 25%: blend right face & whole face.  
				int rv = rightSide.at<uchar>(y,x-midX);  
				int wv = wholeFace.at<uchar>(y,x);  
				// Blend more of the right-side face as it moves  
				// further right along the face.  
				float f = (x - w*2/4) / (float)(w/4);  
				v = cvRound((1.0f - f) * wv + (f) * rv);  
			}  
			else 
			{  
				// Right 25%: just use the right face.  
				v = rightSide.at<uchar>(y,x-midX);  
			}  
			warpedImg.at<uchar>(y,x) = v;  
		}// end x loop  
	}//end y loop  
}
/*为了减少图像噪声的影响，我们对人脸使用双边滤波器，
因为双边滤波器善于平滑一幅图像的大部分，同时保持边
缘的尖锐。直方图均衡显著地增加了噪声，因此我们用
sigmaColor为20来覆盖更多的噪声，但是使用一个只有两
个像素的邻域,因为我们想大量的平滑少量的噪声，但不是
大的图像区域*/
Mat Preprocess::getBilateralFilterImg(Mat&equalizeHistImg)
{
	Mat filtered = Mat(equalizeHistImg.size(), CV_8U);  
	bilateralFilter(equalizeHistImg, filtered, 0, 20.0, 2.0);  
	return filtered;
}
void Preprocess::ellipseMask(Mat&filteredImg)
{
	//在图像的中心画一用黑色填充的椭圆  
	//首先我们初始化掩码图像到白色（255）  
	Mat mask = Mat(filteredImg.size(), CV_8UC1, Scalar(255));  
	double dw = 64;  
	double dh = 64;  
	Point faceCenter = Point( cvRound(dw * 0.5),  
		cvRound(dh * 0.4) );  
	Size size = Size( cvRound(dw * 0.5), cvRound(dh * 0.7) );  
	ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(0), CV_FILLED);  
	//对人脸应用掩码，去掉角  
	//设置角到灰度，没有接触到内部脸  
	filteredImg.setTo(Scalar(128), mask);
}