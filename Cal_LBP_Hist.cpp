#include "Cal_LBP_Hist.h"

// src为输入图像，dst为输出图像，radius为半径，neighbor为计算当前点LBP所需的邻域像素点数，也就是样本点个数
template <typename _Tp> static // 模板函数，根据不同的原始数据类型得到不同的结果
	inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) 
{
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result因此不用在外部给_dst分配内存空间，输出数据类型都是int
	_dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for(int n=0; n<neighbors; n++) 
	{
		// sample points 获取当前采样点
		float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
		float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
		// relative indices 下取整和上取整
		int fx = static_cast<int>(floor(x)); // 向下取整
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));  // 向上取整
		int cy = static_cast<int>(ceil(y));
		// fractional part 小数部分
		float tx = x - fx;
		float ty = y - fy;
		// set interpolation weights 设置四个点的插值权重
		float w1 = (1 - tx) * (1 - ty);
		float w2 =      tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 =      tx  *      ty;
		// iterate through your data 循环处理图像数据
		for(int i=radius; i < src.rows-radius;i++) 
		{
			for(int j=radius;j < src.cols-radius;j++) 
			{
				// calculate interpolated value 计算插值，t表示四个点的权重和
				float t = w1*src.at<_Tp>(i+fy,j+fx) + 
					w2*src.at<_Tp>(i+fy,j+cx) + 
					w3*src.at<_Tp>(i+cy,j+fx) + 
					w4*src.at<_Tp>(i+cy,j+cx);
				// floating point precision, so check some machine-dependent epsilon
				// std::numeric_limits<float>::epsilon()=1.192092896e-07F
				// 当t>=src(i,j)的时候取1，并进行相应的移位
				dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || 
					(std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

// 外部接口，根据不同的数据类型调用模板函数
static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
	int type = src.type();
	switch (type) {
	case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
	case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
	case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
	case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
	case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
	case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
	case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
	default:
		string error_msg = format("Using Circle Local Binary Patterns for feature extraction only works                                     on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
		CV_Error(CV_StsNotImplemented, error_msg);
		break;
	}
}
Mat elbp(InputArray src, int radius, int neighbors) {
	Mat dst;
	elbp(src, dst, radius, neighbors);
	return dst;
}

// 原始LBP算子只是计算8邻域内的局部二值模式
template <typename _Tp> static
	void olbp_(InputArray _src, OutputArray _dst) {
		// get matrices
		Mat src = _src.getMat();
		// allocate memory for result
		_dst.create(src.rows-2, src.cols-2, CV_8UC1);
		Mat dst = _dst.getMat();
		// zero the result matrix
		dst.setTo(0);
		// calculate patterns
		for(int i=1;i<src.rows-1;i++) 
		{
			for(int j=1;j<src.cols-1;j++)
			{
				_Tp center = src.at<_Tp>(i,j);
				unsigned char code = 0;
				code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
				code |= (src.at<_Tp>(i-1,j) >= center) << 6;
				code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
				code |= (src.at<_Tp>(i,j+1) >= center) << 4;
				code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
				code |= (src.at<_Tp>(i+1,j) >= center) << 2;
				code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
				code |= (src.at<_Tp>(i,j-1) >= center) << 0;
				dst.at<unsigned char>(i-1,j-1) = code;
			}
		}
}
// 外部接口，根据不同的数据类型调用模板函数
void olbp(InputArray src, OutputArray dst) {
	switch (src.getMat().type()) {
	case CV_8SC1:   olbp_<char>(src,dst); break;
	case CV_8UC1:   olbp_<unsigned char>(src,dst); break;
	case CV_16SC1:  olbp_<short>(src,dst); break;
	case CV_16UC1:  olbp_<unsigned short>(src,dst); break;
	case CV_32SC1:  olbp_<int>(src,dst); break;
	case CV_32FC1:  olbp_<float>(src,dst); break;
	case CV_64FC1:  olbp_<double>(src,dst); break;
	default: 
		string error_msg = format("Using Original Local Binary Patterns for feature extraction only works                                     on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
		CV_Error(CV_StsNotImplemented, error_msg);
		break;
	}
}
Mat olbp(InputArray src) {
	Mat dst;
	olbp(src, dst);
	return dst;
}

// 计算LBPM的空间直方图分布，得到一个一维向量
// src为LBPM是通过olbp或者elbp计算得到的
// numPatterns为计算LBP的模式数目，一般为2的幂
// grid_x和grid_y分别为每行或每列的block个数
// normed为是否进行归一化处理
Mat Cal_LBP_Hist::spatial_histogram(InputArray _src, int numPatterns,
	int grid_x, int grid_y, bool normed)
{
	Mat src = _src.getMat();
	// allocate memory for the spatial histogram为LBPH分配内存空间
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given，如果没有输入数据，返回的是0
	if(src.empty())
		return result.reshape(1,1);
	// calculate LBP patch size block的尺寸
	int width = src.cols/grid_x;
	int height = src.rows/grid_y;
	// initial result_row 初始化结果行
	int resultRowIdx = 0;
	float hranges[]={0,255};
	const float *ranges[]={hranges};
	int channel=0;
	int size=numPatterns-1;
	// iterate through grid
	for(int i = 0; i < grid_y; i++) 
	{
		for(int j = 0; j < grid_x; j++)
		{
			// 获取指定区域
			Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
			// 计算指定区域的直方图
			MatND cell_hist;
			calcHist(&src_cell,1,&channel,Mat(),cell_hist,1,&size,ranges);
			// copy to the result matrix 将计算得到的结果拷贝到每一行
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1,1);
}

//从image到LBPH

//int radius = 1;
//int neighbor = 8;
//Mat lbp_image = elbp(cv::cvarrToMat(m_refFaceImg), radius, neighbor); //robust performance
//Mat query = spatial_histogram(
//lbp_image, /* lbp_image */
//m_numPatterns, /* number of possible patterns可为static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors)))*/
//m_grid_x, /* grid size x */
//m_grid_y, /* grid size y */
//true /* normed histograms */
//);

Cal_LBP_Hist::Cal_LBP_Hist(void)
{
}


Cal_LBP_Hist::~Cal_LBP_Hist(void)
{
}
