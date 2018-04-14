#include "Cal_LBP_Hist.h"

// srcΪ����ͼ��dstΪ���ͼ��radiusΪ�뾶��neighborΪ���㵱ǰ��LBP������������ص�����Ҳ�������������
template <typename _Tp> static // ģ�庯�������ݲ�ͬ��ԭʼ�������͵õ���ͬ�Ľ��
	inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) 
{
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result��˲������ⲿ��_dst�����ڴ�ռ䣬����������Ͷ���int
	_dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for(int n=0; n<neighbors; n++) 
	{
		// sample points ��ȡ��ǰ������
		float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
		float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
		// relative indices ��ȡ������ȡ��
		int fx = static_cast<int>(floor(x)); // ����ȡ��
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));  // ����ȡ��
		int cy = static_cast<int>(ceil(y));
		// fractional part С������
		float tx = x - fx;
		float ty = y - fy;
		// set interpolation weights �����ĸ���Ĳ�ֵȨ��
		float w1 = (1 - tx) * (1 - ty);
		float w2 =      tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 =      tx  *      ty;
		// iterate through your data ѭ������ͼ������
		for(int i=radius; i < src.rows-radius;i++) 
		{
			for(int j=radius;j < src.cols-radius;j++) 
			{
				// calculate interpolated value �����ֵ��t��ʾ�ĸ����Ȩ�غ�
				float t = w1*src.at<_Tp>(i+fy,j+fx) + 
					w2*src.at<_Tp>(i+fy,j+cx) + 
					w3*src.at<_Tp>(i+cy,j+fx) + 
					w4*src.at<_Tp>(i+cy,j+cx);
				// floating point precision, so check some machine-dependent epsilon
				// std::numeric_limits<float>::epsilon()=1.192092896e-07F
				// ��t>=src(i,j)��ʱ��ȡ1����������Ӧ����λ
				dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || 
					(std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

// �ⲿ�ӿڣ����ݲ�ͬ���������͵���ģ�庯��
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

// ԭʼLBP����ֻ�Ǽ���8�����ڵľֲ���ֵģʽ
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
// �ⲿ�ӿڣ����ݲ�ͬ���������͵���ģ�庯��
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

// ����LBPM�Ŀռ�ֱ��ͼ�ֲ����õ�һ��һά����
// srcΪLBPM��ͨ��olbp����elbp����õ���
// numPatternsΪ����LBP��ģʽ��Ŀ��һ��Ϊ2����
// grid_x��grid_y�ֱ�Ϊÿ�л�ÿ�е�block����
// normedΪ�Ƿ���й�һ������
Mat Cal_LBP_Hist::spatial_histogram(InputArray _src, int numPatterns,
	int grid_x, int grid_y, bool normed)
{
	Mat src = _src.getMat();
	// allocate memory for the spatial histogramΪLBPH�����ڴ�ռ�
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given�����û���������ݣ����ص���0
	if(src.empty())
		return result.reshape(1,1);
	// calculate LBP patch size block�ĳߴ�
	int width = src.cols/grid_x;
	int height = src.rows/grid_y;
	// initial result_row ��ʼ�������
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
			// ��ȡָ������
			Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
			// ����ָ�������ֱ��ͼ
			MatND cell_hist;
			calcHist(&src_cell,1,&channel,Mat(),cell_hist,1,&size,ranges);
			// copy to the result matrix ������õ��Ľ��������ÿһ��
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1,1);
}

//��image��LBPH

//int radius = 1;
//int neighbor = 8;
//Mat lbp_image = elbp(cv::cvarrToMat(m_refFaceImg), radius, neighbor); //robust performance
//Mat query = spatial_histogram(
//lbp_image, /* lbp_image */
//m_numPatterns, /* number of possible patterns��Ϊstatic_cast<int>(std::pow(2.0, static_cast<double>(_neighbors)))*/
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
